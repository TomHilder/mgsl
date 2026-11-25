"""Misuse, failures and their reporting."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial
from problems import LinGauss, Nonlinear2D, tight_options

import mgsl
from mgsl import _tree
from mgsl.diagnostics import Failures, FitStats, NewtonCounts, Solves
from mgsl.mgvi import HashableStruct, draw_residuals, make_problem, split_options

SCHED = mgsl.Schedule(6, 1.0, 100.0, 3)


def run(lik, mu, opts=None, **kw):
    return mgsl.sample(
        jax.random.key(0),
        lik,
        mu,
        kw.pop("sched", SCHED),
        opts or tight_options(),
        progress=False,
        **kw,
    )


def test_rebuilt_likelihood_sees_changed_data():
    """Bound methods of a mutated object must not reuse code compiled for the old data."""
    p = LinGauss.make(d=4, n=6)
    A = jnp.asarray(p.A)

    class Model:
        def __init__(self, data):
            self.data = jnp.asarray(data)

        def ll(self, x):
            return -0.5 * jnp.sum(((self.data - A @ x) / p.sd) ** 2)

        def fi(self, x, v):
            return A.T @ ((A @ v) / p.sd**2)

        def fs(self, x, e):
            return A.T @ (e / p.sd)

    m = Model(p.data)
    ds = jax.ShapeDtypeStruct((6,), jnp.float64)
    a = run(mgsl.Likelihood(m.ll, m.fi, m.fs, ds), jnp.zeros(4))
    m.data = jnp.asarray(p.data + 50.0)
    b = run(mgsl.Likelihood(m.ll, m.fi, m.fs, ds), jnp.zeros(4))
    assert not np.allclose(a.samples, b.samples)
    ref = run(mgsl.gaussian(lambda x: A @ x, p.data + 50.0, noise_std=p.sd), jnp.zeros(4))
    np.testing.assert_allclose(b.samples, ref.samples, rtol=1e-8, atol=1e-10)


def _scaled_ll(A, data, scale, x):
    return -0.5 * jnp.sum(((data - A @ x) / scale) ** 2)


def _scaled_fisher(A, scale, x, v):
    return A.T @ (A @ v) / scale**2


def _scaled_sqrt(A, scale, x, e):
    return A.T @ e / scale


def test_python_float_in_partial_is_runtime():
    from mgsl import sampler

    p = LinGauss.make(d=4, n=6)
    A, data = jnp.asarray(p.A), jnp.asarray(p.data)

    def lik(scale):
        return mgsl.Likelihood(
            Partial(_scaled_ll, A, data, scale),
            Partial(_scaled_fisher, A, scale),
            Partial(_scaled_sqrt, A, scale),
            jax.ShapeDtypeStruct((6,), jnp.float64),
        )

    run(lik(0.5), jnp.zeros(4))
    n = [f._cache_size() for f in sampler._JITTED.values()]
    run(lik(0.7), jnp.zeros(4))
    assert [f._cache_size() for f in sampler._JITTED.values()] == n


def setup(lik, x, options, diagnostics=True):
    struct = _tree.latent_struct(x)
    _, _, d, dtype = _tree.make_unravel(struct)
    st, rt = split_options(options, d, dtype, diagnostics)
    from mgsl.mgvi import _Precond

    pre = None if options.preconditioner is None else _Precond(options.preconditioner)
    return make_problem(lik, pre, HashableStruct(struct)), st, rt


def test_reported_residual_is_unpreconditioned():
    p = LinGauss.make(d=8, n=10, scale=5.0)
    scales = jnp.asarray(np.geomspace(1e-3, 1e3, 8))
    cg = mgsl.CGOptions(tol=1e-14, miniter=0, maxiter=2)  # stop early: residuals are large
    opts = tight_options(preconditioner=lambda x, tau: lambda v: scales * v)
    opts = mgsl.MGVIOptions(
        n_samples_mgvi=3,
        n_iter_mgvi=1,
        newton=opts.newton,
        draw_cg=cg,
        preconditioner=opts.preconditioner,
    )
    prob, st, rt = setup(p.lik(), jnp.zeros(8), opts)
    xbar, tau = jnp.asarray(p.mean), 2.0
    deltas, zetas, stats, _ = draw_residuals(prob, jax.random.key(0), xbar, tau, 3, st, rt)
    Lam = p.precision + tau * np.eye(8)
    res = np.linalg.norm(np.asarray(zetas) - np.asarray(deltas) @ Lam, axis=1) / np.linalg.norm(
        zetas, axis=1
    )
    np.testing.assert_allclose(float(stats.max_res), res.max(), rtol=1e-6)


def test_singular_preconditioner_is_a_breakdown():
    p = LinGauss.make(d=4, n=6)
    opts = tight_options(preconditioner=lambda x, tau: lambda v: 0.0 * v)
    with pytest.warns(RuntimeWarning, match="cg_breakdowns"):
        res = run(p.lik(), jnp.asarray(p.mean), opts)
    assert res.failures.totals()["cg_breakdowns"] > 0


def test_failure_counts_map_to_fields():
    z = jnp.zeros(3, jnp.int32)
    draws = Solves(z + 4, z, z, z + jnp.array([1, 0, 0]), z + jnp.array([0, 2, 0]), jnp.zeros(3))
    ncg = Solves(z + 4, z, z, z + jnp.array([0, 3, 0]), z + jnp.array([0, 0, 5]), jnp.zeros(3))
    newton = NewtonCounts(
        z, z + jnp.array([7, 0, 0]), z + jnp.array([0, 0, 1]), z + jnp.array([0, 9, 0])
    )
    stats = FitStats(draws, ncg, newton, jnp.zeros(3))
    f = Failures.from_stats(jax.tree_util.tree_map(lambda a: np.asarray(a)[:, None], stats))
    np.testing.assert_array_equal(f.draw_cg_capped, [1, 0, 0])
    np.testing.assert_array_equal(f.newton_cg_capped, [0, 3, 0])
    np.testing.assert_array_equal(f.cg_breakdowns, [0, 2, 5])
    np.testing.assert_array_equal(f.sd_fallbacks, [7, 0, 0])
    np.testing.assert_array_equal(f.line_search_failures, [0, 0, 1])
    np.testing.assert_array_equal(f.nonfinite, [0, 9, 0])
    np.testing.assert_array_equal((f + f).nonfinite, [0, 18, 0])


def test_newton_cg_cap_is_counted():
    p = LinGauss.make(d=8, n=8, seed=2, scale=1e3)
    cg = mgsl.CGOptions(tol=1e-12, miniter=0, maxiter=1)
    opts = mgsl.MGVIOptions(
        n_samples_mgvi=1,
        n_iter_mgvi=1,
        newton=mgsl.NewtonOptions(maxiter=2, adaptive_cg=False, cg=cg),
        draw_cg=mgsl.CGOptions(tol=1e-12, miniter=0, maxiter=500),
    )
    with pytest.warns(RuntimeWarning, match="newton_cg_capped"):
        res = run(p.lik(), jnp.zeros(8), opts)
    assert res.failures.totals()["newton_cg_capped"] > 0


def test_frozen_paths_give_nan_samples_and_a_specific_warning():
    p = LinGauss.make(d=3, n=4)
    good = p.lik()
    bad = mgsl.Likelihood(
        log_likelihood=lambda x: jnp.where(x[0] > 1.5, jnp.nan, good.log_likelihood(x)),
        fisher=good.fisher,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    with pytest.warns(RuntimeWarning, match="model's domain"):
        res = run(bad, jnp.array([2.0, 0.0, 0.0]))
    x = np.asarray(res.samples)
    frozen = np.asarray(res.failures.nonfinite) > 0
    assert frozen.any()
    assert np.all(np.isnan(x[frozen])) and np.all(np.isfinite(x[~frozen]))


def test_check_likelihood_catches_scaled_mistakes():
    p = LinGauss.make(d=5, n=7)
    good = p.lik()
    A, W = jnp.asarray(p.A), jnp.asarray(p.sd**-2)
    no_weights = mgsl.Likelihood(
        good.log_likelihood, lambda x, v: A.T @ (A @ v), lambda x, e: A.T @ e, good.data_shape
    )
    with pytest.warns(UserWarning, match="missing noise weights"):
        mgsl.check_likelihood(no_weights, jnp.zeros(5), jax.random.key(0))
    half = mgsl.Likelihood(
        lambda x: 2.0 * good.log_likelihood(x), good.fisher, good.fisher_sqrt, good.data_shape
    )
    with pytest.warns(UserWarning, match="factor 1/2"):
        mgsl.check_likelihood(half, jnp.zeros(5), jax.random.key(0))
    wrong_op = mgsl.Likelihood(
        good.log_likelihood,
        good.fisher,
        good.fisher_sqrt,
        good.data_shape,
        fisher_operator=lambda x: (Partial(lambda v: A.T @ (A @ v)), Partial(lambda e: A.T @ e)),
    )
    with pytest.raises(ValueError, match="fisher_operator"):
        mgsl.check_likelihood(wrong_op, jnp.zeros(5), jax.random.key(0))
    assert W.shape == (7,)


def test_gaussian_shape_and_dtype_handling():
    p = LinGauss.make(d=3, n=5)
    A = jnp.asarray(p.A)
    column = mgsl.gaussian(lambda x: (A @ x)[:, None], p.data, noise_std=p.sd)
    with pytest.raises(ValueError, match="must match exactly"):
        column.log_likelihood(jnp.zeros(3))
    with pytest.raises(ValueError, match="must match exactly"):
        run(column, jnp.zeros(3))
    A32 = A.astype(jnp.float32)
    lik32 = mgsl.gaussian(lambda x: A32 @ x.astype(jnp.float32), p.data, noise_std=p.sd)
    with warnings.catch_warnings():  # float32 rounding in the objective: expected warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        res = run(lik32, jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(res.samples)))
    with pytest.raises(TypeError, match="floating"):
        mgsl.Likelihood(good_ll, good_ll, good_ll, jax.ShapeDtypeStruct((5,), jnp.int64))


def good_ll(*a):
    return 0.0


def test_mixed_dtype_latent_rejected():
    with pytest.raises(TypeError, match="one floating dtype"):
        _tree.latent_struct({"a": jnp.zeros(2, jnp.float32), "b": jnp.zeros(2)})


def test_likelihood_must_accept_the_state():
    p, q = LinGauss.make(d=4, n=6), LinGauss.make(d=5, n=6)
    res = run(p.lik(), jnp.asarray(p.mean))
    with pytest.raises(ValueError, match="does not accept"):
        mgsl.advance(q.lik(), res.state, mgsl.grid(res.state.tau, 1e3, 2), tight_options())


def test_progress_lines(capsys):
    p = LinGauss.make(d=3, n=4)
    mgsl.sample(jax.random.key(0), p.lik(), jnp.asarray(p.mean), SCHED, tight_options())
    err = capsys.readouterr().err.splitlines()
    assert [ln.split()[1] for ln in err] == ["step", "step", "terminal"]
    assert "step 1/2" in err[0] and "ESS fraction median" in err[0] and "failures 0" in err[0]
    mgsl.sample(
        jax.random.key(0), p.lik(), jnp.asarray(p.mean), SCHED, tight_options(), progress=False
    )
    assert capsys.readouterr().err == ""


def test_vectorize_and_linearize_once_give_the_same_samples():
    p = Nonlinear2D()
    mu = jnp.array([0.35, 0.8])
    ref = run(p.lik(), mu)
    vec = run(p.lik(), mu, vectorize=True)
    lin = mgsl.gaussian(p.forward, np.asarray(p.data), noise_std=p.sd, linearize_once=True)
    once = run(lin, mu)
    np.testing.assert_allclose(vec.samples, ref.samples, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(once.samples, ref.samples, rtol=1e-8, atol=1e-10)
    mgsl.check_likelihood(lin, mu, jax.random.key(0))


def test_golden_samples():
    """Pins the key derivation and step indexing. Regenerate if JAX changes its PRNG."""
    p = LinGauss.make(d=3, n=4)
    res = mgsl.sample(
        jax.random.key(42),
        p.lik(),
        jnp.asarray(p.mean),
        mgsl.Schedule(2, 1.0, 10.0, 3),
        tight_options(),
        progress=False,
    )
    np.testing.assert_allclose(np.asarray(res.samples), GOLDEN, rtol=1e-8)


GOLDEN = [
    [0.1756541718889953, -0.7339916081155624, 0.484299131935991],
    [-0.1659329585633089, -1.1957776107072824, 0.3591745835603169],
]


def test_no_spurious_warnings_on_clean_runs():
    p = LinGauss.make(d=3, n=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(p.lik(), jnp.asarray(p.mean))


def test_default_options_run_cleanly_on_a_nonlinear_problem():
    """Default (NIFTy.re) solver settings, adaptive Newton CG included: no failures, and every
    Newton solve ends with a residual no larger than its right-hand side."""
    p = Nonlinear2D(b=2.0, sd=0.1)
    lik = p.lik()
    opts = mgsl.MGVIOptions(n_samples_mgvi=2, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from dataclasses import replace

        fit = mgsl.mgvi_fit(jax.random.key(0), lik, jnp.zeros(2), replace(opts, n_iter_mgvi=10))
        res = mgsl.sample(
            jax.random.key(1),
            lik,
            fit.mean,
            mgsl.Schedule(64, 1.0, 1e5, 16),
            opts,
            progress=False,
        )
    assert not res.failures.any(), res.failures.totals()
    assert np.nanmax(res.diagnostics.cg_newton.max_rel_residual) <= 1.0


def test_changing_refresh_does_not_recompile():
    from mgsl import sampler

    p = LinGauss.make(d=4, n=6)
    lik = p.lik()  # one object: a new likelihood (new forward function) would recompile
    run(lik, jnp.asarray(p.mean))
    n = [f._cache_size() for f in sampler._JITTED.values()]
    cg = mgsl.CGOptions(tol=1e-12, miniter=0, maxiter=500, refresh=7)
    base = tight_options()
    opts = mgsl.MGVIOptions(
        n_samples_mgvi=base.n_samples_mgvi,
        n_iter_mgvi=base.n_iter_mgvi,
        newton=mgsl.NewtonOptions(maxiter=5, xtol=0.0, adaptive_cg=False, cg=cg),
        draw_cg=cg,
    )
    run(lik, jnp.asarray(p.mean), opts)
    assert [f._cache_size() for f in sampler._JITTED.values()] == n
