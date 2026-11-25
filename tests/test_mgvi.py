from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import LinGauss, LinGaussPytree, Nonlinear2D, assert_standard_normal, tight_options

import mgsl
from mgsl import _tree
from mgsl.mgvi import (
    HashableStruct,
    draw_residuals,
    local_fit,
    make_problem,
    newton,
    split_options,
)

FIXTURE = Path(__file__).parent / "fixtures" / "prototype_nonlinear2d.npz"


def setup(lik, x, options, diagnostics=True):
    struct = _tree.latent_struct(x)
    _, _, d, dtype = _tree.make_unravel(struct)
    st, rt = split_options(options, d, dtype, diagnostics)
    pre = None
    if options.preconditioner is not None:
        from mgsl.mgvi import _Precond

        pre = _Precond(options.preconditioner)
    return make_problem(lik, pre, HashableStruct(struct)), st, rt


def zero_fisher_lik(d):
    return mgsl.Likelihood(
        log_likelihood=lambda x: jnp.zeros((), x.dtype),
        fisher=lambda x, v: jnp.zeros_like(v),
        fisher_sqrt=lambda x, e: jnp.zeros(d),
        data_shape=jax.ShapeDtypeStruct((1,), jnp.float64),
    )


def test_draws_solve_metric_and_have_right_covariance():
    p = LinGauss.make(d=4, n=6)
    opts = tight_options()
    prob, st, rt = setup(p.lik(), jnp.zeros(4), opts)
    tau = 0.8
    xbar = jnp.asarray(p.mean)
    draw = jax.jit(lambda k: draw_residuals(prob, k, xbar, tau, 20000, st, rt))
    deltas, zetas, stats, valid = draw(jax.random.key(3))
    Lam = p.precision + tau * np.eye(4)
    np.testing.assert_allclose(np.asarray(deltas) @ Lam, zetas, rtol=1e-8, atol=1e-10)
    assert bool(valid) and int(stats.capped) == 0
    L = np.linalg.cholesky(Lam)  # delta ~ N(0, Lam^-1)  =>  L^T delta ~ N(0, I)
    assert_standard_normal(np.asarray(deltas) @ L)


def test_zero_fisher_draw_is_likelihood_free():
    opts = tight_options()
    prob, st, rt = setup(zero_fisher_lik(3), jnp.zeros(3), opts)
    tau = 2.0
    deltas, _, stats, _ = draw_residuals(prob, jax.random.key(0), jnp.zeros(3), tau, 3, st, rt)
    assert int(stats.iter_sum) == 0
    k1s = [jax.random.split(k)[0] for k in jax.random.split(jax.random.key(0), 3)]
    e1 = np.stack([jax.random.normal(k, (3,)) for k in k1s])
    np.testing.assert_allclose(deltas, e1 / np.sqrt(1 + tau), rtol=1e-14)
    # same with the default miniter=6: still finite
    prob, st, rt = setup(
        zero_fisher_lik(3),
        jnp.zeros(3),
        mgsl.MGVIOptions(n_samples_mgvi=1, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=1)),
    )
    deltas, *_ = draw_residuals(prob, jax.random.key(0), jnp.zeros(3), tau, 3, st, rt)
    assert np.all(np.isfinite(np.asarray(deltas)))


@pytest.mark.parametrize("n_samples", [1, 4])
def test_one_newton_step_gives_exact_local_mean(n_samples):
    p = LinGauss.make(d=6, n=9)
    opts = tight_options(n_samples=n_samples, newton_maxiter=1)
    prob, st, rt = setup(p.lik(), jnp.zeros(6), opts)
    rng = np.random.default_rng(2)
    for tau in (0.0, 0.3, 50.0):
        y = jnp.asarray(rng.normal(size=6) * (1 + tau))
        x0 = jnp.asarray(rng.normal(size=6))
        xb, _, _, stats = local_fit(prob, jax.random.key(1), x0, tau, y, st, rt)
        np.testing.assert_allclose(xb, p.local_mean(tau, np.asarray(y)), rtol=1e-8, atol=1e-10)
        assert int(stats.newton.iterations) == 1


def test_zero_fisher_newton_needs_no_cg_iterations():
    opts = tight_options(newton_maxiter=1)
    prob, st, rt = setup(zero_fisher_lik(3), jnp.zeros(3), opts)
    y = jnp.array([1.0, -2.0, 0.5])
    deltas = jnp.zeros((2, 3))
    x, _counts, solves = newton(prob, jnp.zeros(3), deltas, 1.5, y, st, rt)
    assert int(solves.iter_sum) == 0
    np.testing.assert_allclose(x, y / 2.5, rtol=1e-14)


def test_newton_matches_prototype():
    f = np.load(FIXTURE)
    p = Nonlinear2D(b=float(f["b"]), sd=float(f["sd"]), data=tuple(f["data"]))
    cg = mgsl.CGOptions(tol=1e-4, miniter=0, maxiter=50)
    opts = mgsl.MGVIOptions(
        n_samples_mgvi=4,
        n_iter_mgvi=1,
        newton=mgsl.NewtonOptions(
            maxiter=2, xtol=0.0, adaptive_cg=False, cg=cg, line_search="armijo"
        ),
        draw_cg=cg,
    )
    prob, st, rt = setup(p.lik(), jnp.zeros(2), opts)
    x, _counts, _ = newton(
        prob,
        jnp.asarray(f["newton_xbar0"]),
        jnp.asarray(f["newton_deltas"]),
        float(f["newton_tau"]),
        jnp.asarray(f["newton_z"]),
        st,
        rt,
    )
    np.testing.assert_allclose(x, f["newton_xbar2"], rtol=1e-6, atol=1e-8)


def test_fit_reaches_mode_from_far_start():
    p = Nonlinear2D()
    opts = tight_options(n_samples=2, n_iter=3, newton_maxiter=30)
    prob, st, rt = setup(p.lik(), jnp.zeros(2), opts)
    tau = 1e5
    target = np.array([0.4, -0.3])
    y = jnp.asarray(tau * target)
    xb, _, _, stats = local_fit(prob, jax.random.key(0), jnp.array([4.0, -5.0]), tau, y, st, rt)
    np.testing.assert_allclose(xb, target, atol=1e-3)
    assert int(stats.newton.line_search_failures) == 0


def test_nonfinite_objective_freezes_and_is_counted():
    p = LinGauss.make(d=3, n=4)
    good = p.lik()
    bad = mgsl.Likelihood(
        log_likelihood=lambda x: jnp.where(x[0] > 5.0, jnp.nan, good.log_likelihood(x)),
        fisher=good.fisher,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    prob, st, rt = setup(bad, jnp.zeros(3), tight_options(n_samples=1))
    x0 = jnp.array([6.0, 0.0, 0.0])
    x, counts, _ = newton(prob, x0, jnp.zeros((1, 3)), 1.0, jnp.zeros(3), st, rt)
    np.testing.assert_array_equal(x, x0)
    assert int(counts.nonfinite) == 1


@pytest.mark.parametrize("line_search", ["armijo", "nifty"])
def test_cg_breakdown_still_decreases(line_search):
    """A NaN Fisher breaks every CG solve; the step must stay finite and still go downhill."""
    p = Nonlinear2D()
    good = p.lik()
    nan_fisher = mgsl.Likelihood(
        log_likelihood=good.log_likelihood,
        fisher=lambda x, v: v * jnp.nan,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    opts = tight_options(n_samples=1, newton_maxiter=3, line_search=line_search)
    prob, st, rt = setup(nan_fisher, jnp.zeros(2), opts)
    x0 = jnp.array([2.0, -2.0])
    u = lambda x: 0.5 * x @ x - good.log_likelihood(x)
    x, _counts, solves = newton(prob, x0, jnp.zeros((1, 2)), 0.0, jnp.zeros(2), st, rt)
    assert int(solves.breakdown) >= 1
    assert np.all(np.isfinite(np.asarray(x))) and float(u(x)) < float(u(x0))


def test_armijo_replaces_ascent_direction_by_steepest_descent():
    p = Nonlinear2D(sd=1e-3)
    good = p.lik()
    indefinite = mgsl.Likelihood(
        log_likelihood=good.log_likelihood,
        fisher=lambda x, v: -1e3 * v,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    opts = tight_options(n_samples=1, newton_maxiter=1, line_search="armijo")
    prob, st, rt = setup(indefinite, jnp.zeros(2), opts)
    x0 = jnp.array([0.6, 0.9])
    u = lambda x: 0.5 * x @ x - good.log_likelihood(x)
    x, counts, _ = newton(prob, x0, jnp.zeros((1, 2)), 0.0, jnp.zeros(2), st, rt)
    assert int(counts.sd_fallbacks) == 1 and float(u(x)) < float(u(x0))


def test_nifty_search_restarts_along_gradient():
    p = Nonlinear2D(b=500.0, sd=1e-2)
    lik = p.lik()
    prob, st, rt = setup(lik, jnp.zeros(2), tight_options(n_samples=1, newton_maxiter=1))
    x0 = jnp.array([0.05, 0.0])
    u = lambda x: 0.5 * x @ x - lik.log_likelihood(x)
    x, counts, _ = newton(prob, x0, jnp.zeros((1, 2)), 0.0, jnp.zeros(2), st, rt)
    assert int(counts.sd_fallbacks) == 1 and int(counts.line_search_failures) == 0
    assert float(u(x)) < float(u(x0))


def test_newton_stopping_rules():
    p = Nonlinear2D()

    def iters(**kw):
        cg = mgsl.CGOptions(tol=1e-12, miniter=0, maxiter=100)
        newton_opts = mgsl.NewtonOptions(adaptive_cg=False, cg=cg, **kw)
        opts = mgsl.MGVIOptions(n_samples_mgvi=1, n_iter_mgvi=1, newton=newton_opts)
        prob, st, rt = setup(p.lik(), jnp.zeros(2), opts)
        _, c, _ = newton(
            prob, jnp.array([3.0, -3.0]), jnp.zeros((1, 2)), 0.0, jnp.zeros(2), st, rt
        )
        return int(c.iterations)

    assert iters(maxiter=3, xtol=0.0) == 3
    assert iters(maxiter=50, absdelta=1e9) == 1
    loose, strict = iters(maxiter=50, xtol=1e-2), iters(maxiter=50, xtol=0.0)
    assert 1 < loose < strict <= 50


def test_mgvi_fit_linear_gaussian():
    p = LinGauss.make(d=5, n=8)
    fit = mgsl.mgvi_fit(jax.random.key(0), p.lik(), jnp.zeros(5), tight_options(n_iter=2))
    np.testing.assert_allclose(fit.mean, p.mean, rtol=1e-9)
    assert np.asarray(fit.residuals).shape == (2, 5)
    assert not fit.failures.any()
    pp = LinGaussPytree.make()
    fitp = mgsl.mgvi_fit(jax.random.key(0), pp.lik(), pp.zeros(), tight_options(n_iter=2))
    np.testing.assert_allclose(LinGaussPytree.flat(fitp.mean), pp.base.mean, rtol=1e-9)
    assert fitp.residuals["a"].shape == (2, 2, 3)


def test_preconditioned_fit_is_unchanged():
    p = LinGauss.make(d=6, n=9, scale=10.0)
    exact = np.linalg.inv(p.precision)  # tau = 0 inverse metric

    def precond(x, tau):
        Minv = jnp.asarray(np.linalg.inv(p.precision + 0.0 * np.eye(6)))
        return lambda r: Minv @ r

    opts = tight_options(preconditioner=precond)
    fit = mgsl.mgvi_fit(jax.random.key(0), p.lik(), jnp.zeros(6), replace(opts, n_iter_mgvi=1))
    np.testing.assert_allclose(fit.mean, p.mean, rtol=1e-9)
    assert exact.shape == (6, 6)


@pytest.mark.nifty
@pytest.mark.parametrize(
    "b, sd, start, zero_deltas",
    [
        (3.0, 0.2, (3.0, -3.0), False),
        (3.0, 0.2, (-4.0, 6.0), False),
        (500.0, 1e-2, (0.05, 0.0), True),  # needs the restart along the gradient
    ],
)
def test_nifty_line_search_matches_nifty_newton_cg(b, sd, start, zero_deltas):
    """Same objective and curvature, tight solves: our Newton iterates equal NIFTy.re's."""
    jft = pytest.importorskip("nifty.re")
    p = Nonlinear2D(b=b, sd=sd)
    lik = p.lik()
    deltas = jnp.zeros((2, 2)) if zero_deltas else jnp.array([[0.3, -0.2], [0.1, 0.4]])
    tau, y = (0.0, jnp.zeros(2)) if zero_deltas else (0.5, jnp.array([0.2, -0.1]))

    def u(x):
        return 0.5 * (1 + tau) * x @ x - y @ x - lik.log_likelihood(x)

    def kl(x):
        xs = jnp.concatenate([x + deltas, x - deltas])
        return jnp.mean(jax.vmap(u)(xs))

    def hessp(x, v):
        xs = jnp.concatenate([x + deltas, x - deltas])
        return (1 + tau) * v + jnp.mean(jax.vmap(lik.fisher, (0, None))(xs, v), 0)

    x0 = jnp.asarray(start)
    for k in (1, 2, 4):
        cg = {"absdelta": None, "resnorm": None, "tol": 1e-14, "miniter": 0, "maxiter": 200}
        ref = jft.newton_cg(
            fun_and_grad=jax.value_and_grad(kl),
            x0=x0,
            hessp=hessp,
            maxiter=k,
            xtol=0.0,
            cg_kwargs=cg,
        )
        cgo = mgsl.CGOptions(tol=1e-14, miniter=0, maxiter=200)
        opts = mgsl.MGVIOptions(
            n_samples_mgvi=2,
            n_iter_mgvi=1,
            newton=mgsl.NewtonOptions(maxiter=k, xtol=0.0, adaptive_cg=False, cg=cgo),
        )
        prob, st, rt = setup(lik, jnp.zeros(2), opts)
        x, _counts, _ = newton(prob, x0, deltas, tau, y, st, rt)
        np.testing.assert_allclose(x, ref, rtol=1e-8, atol=1e-10)


def test_newton_cg_near_convergence_with_default_options():
    """At an essentially converged centre the step is tiny; the inner CG must still solve for it
    (not break down on an unreachable tolerance) with the default adaptive settings."""
    p = LinGauss.make(d=6, n=9)
    opts = mgsl.MGVIOptions(n_samples_mgvi=2, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=3))
    prob, st, rt = setup(p.lik(), jnp.zeros(6), opts)
    tau, y = 3.0, jnp.asarray(np.linspace(-1, 1, 6))
    deltas = jnp.asarray(np.random.default_rng(0).normal(size=(2, 6)) * 0.1)
    x, _, solves = newton(
        prob, jnp.asarray(p.local_mean(tau, np.asarray(y))), deltas, tau, y, st, rt
    )
    assert int(solves.breakdown) == 0 and int(solves.capped) == 0
    # the true residual of the unpreconditioned system: a solver drifting on rounding noise can
    # end with a garbage step without tripping the capped or breakdown flags
    assert float(solves.max_res) < 1e-8
    np.testing.assert_allclose(x, p.local_mean(tau, np.asarray(y)), rtol=1e-10, atol=1e-12)
