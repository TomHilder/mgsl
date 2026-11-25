"""Resuming runs: exact chunking, finalize, saving and loading states."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import LinGauss, LinGaussPytree, Nonlinear2D, tight_options

import mgsl


def chunked(lik, mu, taus, opts, cuts, key=0, **kw):
    state = mgsl.init(jax.random.key(key), lik, mu, n_paths=10, tau_start=taus[0], options=opts)
    reps = []
    bounds = [0, *cuts, len(taus) - 1]
    for a, b in itertools.pairwise(bounds):
        state, rep = mgsl.advance(lik, state, taus[a : b + 1], opts, **kw)
        reps.append(rep)
    samples, _ = mgsl.finalize(lik, state, opts)
    return samples, state, reps


@pytest.mark.parametrize("problem", ["lin", "nonlinear"])
def test_chunked_equals_unchunked(problem):
    if problem == "lin":
        p = LinGauss.make(d=5, n=7)
        lik, mu = p.lik(), jnp.asarray(p.mean)
    else:
        lik, mu = Nonlinear2D().lik(), jnp.array([0.35, 0.8])
    opts = tight_options()
    sched = mgsl.Schedule(10, 0.5, 2e3, 10)
    ref = mgsl.sample(jax.random.key(0), lik, mu, sched, opts)
    for cuts in ([4], [2, 7]):
        samples, state, _ = chunked(lik, mu, sched.taus, opts, cuts)
        np.testing.assert_allclose(samples, ref.samples, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(state.y, ref.state.y, rtol=1e-12)
        assert state.step == ref.state.step == 9


def test_advance_checks_grid_start():
    p = LinGauss.make(d=3, n=4)
    opts = tight_options()
    state = mgsl.init(
        jax.random.key(0), p.lik(), jnp.zeros(3), n_paths=2, tau_start=1.0, options=opts
    )
    with pytest.raises(ValueError, match=r"mgsl\.grid"):
        mgsl.advance(p.lik(), state, mgsl.grid(1.01, 10.0, 3), opts)
    with pytest.raises(ValueError, match="increasing"):
        mgsl.advance(p.lik(), state, np.array([1.0, 0.5]), opts)
    # options may change between chunks
    state, _ = mgsl.advance(p.lik(), state, mgsl.grid(1.0, 10.0, 3), opts)
    other = tight_options(n_samples=3, n_iter=2)
    state, rep = mgsl.advance(p.lik(), state, mgsl.grid(state.tau, 100.0, 3), other)
    assert state.tau == 100.0 and not rep.failures.any()


def test_finalize_does_not_change_state():
    p = LinGauss.make(d=4, n=6)
    lik, opts = p.lik(), tight_options()
    res = mgsl.sample(
        jax.random.key(1), lik, jnp.asarray(p.mean), mgsl.Schedule(6, 1, 100, 4), opts
    )
    s0 = res.state
    a, _ = mgsl.finalize(lik, s0, opts)
    b, _ = mgsl.finalize(lik, s0, opts)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, res.samples)
    s1, _ = mgsl.advance(lik, s0, mgsl.grid(s0.tau, 1e3, 3), opts)
    c, rep = mgsl.finalize(lik, s1, opts)
    assert np.asarray(c).shape == (6, 4) and rep.diagnostics.ess_fraction.shape == (6,)


def test_save_load_roundtrip(tmp_path):
    p = LinGaussPytree.make()
    lik, opts = p.lik(), tight_options()
    fit = mgsl.mgvi_fit(jax.random.key(0), lik, p.zeros(), opts, n_rounds=2)
    res = mgsl.sample(jax.random.key(2), lik, fit.mean, mgsl.Schedule(5, 1.0, 50.0, 4), opts)
    path = tmp_path / "state.npz"
    mgsl.save_state(path, res.state)
    loaded = mgsl.load_state(path, like=p.zeros())
    for f in ("y", "xbar", "key_data"):
        np.testing.assert_array_equal(getattr(loaded, f), getattr(res.state, f))
    assert (loaded.tau, loaded.step, loaded.key_impl) == (
        res.state.tau,
        res.state.step,
        res.state.key_impl,
    )
    assert jax.tree_util.tree_structure(loaded.struct) == jax.tree_util.tree_structure(
        res.state.struct
    )
    grid = mgsl.grid(loaded.tau, 500.0, 3)
    a, _ = mgsl.advance(lik, res.state, grid, opts)
    b, _ = mgsl.advance(lik, loaded, grid, opts)
    np.testing.assert_array_equal(a.y, b.y)
    with pytest.raises(ValueError, match="structure"):
        mgsl.load_state(path, like=jnp.zeros(10))
    with pytest.raises(ValueError, match="leaf"):
        mgsl.load_state(path, like={"a": jnp.zeros((3, 2)), "b": jnp.zeros(4)})
    with pytest.raises(ValueError, match="leaf"):
        mgsl.load_state(path, like={"a": jnp.zeros((2, 3), jnp.float32), "b": jnp.zeros(4)})


def test_load_rejects_other_format_version(tmp_path):
    p = LinGauss.make(d=3, n=4)
    opts = tight_options()
    state = mgsl.init(
        jax.random.key(0), p.lik(), jnp.zeros(3), n_paths=2, tau_start=1.0, options=opts
    )
    path = tmp_path / "s.npz"
    mgsl.save_state(path, state)
    with np.load(path) as f:
        content = dict(f)
    content["format_version"] = np.asarray(99)
    np.savez(path, **content)
    with pytest.raises(ValueError, match="version"):
        mgsl.load_state(path, like=jnp.zeros(3))


def test_concatenate_diagnostics():
    p = LinGauss.make(d=3, n=4)
    lik, opts = p.lik(), tight_options()
    s = mgsl.init(
        jax.random.key(0), lik, jnp.asarray(p.mean), n_paths=3, tau_start=1.0, options=opts
    )
    s1, r1 = mgsl.advance(lik, s, mgsl.grid(1.0, 10.0, 4), opts)
    _s2, r2 = mgsl.advance(lik, s1, mgsl.grid(10.0, 100.0, 3), opts)
    both = mgsl.MGSLDiagnostics.concatenate(r1.diagnostics, r2.diagnostics)
    assert both.ess_fraction.shape == (3, 5) and both.tau_end == 100.0
    np.testing.assert_allclose(
        both.taus, np.r_[mgsl.grid(1, 10, 4)[:-1], mgsl.grid(10, 100, 3)[:-1]]
    )
    with pytest.raises(ValueError, match="continuation"):
        mgsl.MGSLDiagnostics.concatenate(r2.diagnostics, r1.diagnostics)
    s_other = mgsl.init(
        jax.random.key(0), lik, jnp.asarray(p.mean), n_paths=2, tau_start=10.0, options=opts
    )
    _, r3 = mgsl.advance(lik, s_other, mgsl.grid(10.0, 100.0, 3), opts)
    with pytest.raises(ValueError, match="paths"):
        mgsl.MGSLDiagnostics.concatenate(r1.diagnostics, r3.diagnostics)


def test_resume_without_diagnostics():
    p = LinGauss.make(d=3, n=4)
    lik, opts = p.lik(), tight_options()
    res = mgsl.sample(
        jax.random.key(0),
        lik,
        jnp.asarray(p.mean),
        mgsl.Schedule(3, 1, 10, 3),
        opts,
        calculate_diagnostics=False,
    )
    state, rep = mgsl.advance(lik, res.state, mgsl.grid(10.0, 100.0, 3), opts)
    assert state.tau == 100.0 and rep.diagnostics is not None
