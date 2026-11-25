"""Sampler API and mechanics: keys, batching, devices, diagnostics, compilation, dtypes."""

import subprocess
import sys
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import LinGauss, tight_options

import mgsl

TESTS = Path(__file__).parent


@pytest.fixture(scope="module")
def lin():
    return LinGauss.make(d=6, n=8, seed=1)


def small(lin, n_paths=12, n_tau=5, key=0, **kw):
    sched = mgsl.Schedule(n_paths, 1.0, 1e3, n_tau)
    return mgsl.sample(
        jax.random.key(key), lin.lik(), jnp.asarray(lin.mean), sched, tight_options(), **kw
    )


def test_reproducible_and_paths_differ(lin):
    a, b, c = small(lin), small(lin), small(lin, key=1)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert not np.allclose(a.samples, c.samples)
    x = np.asarray(a.samples)
    assert len({tuple(r) for r in np.round(x, 12)}) == x.shape[0]


def test_legacy_keys_accepted(lin):
    a = small(lin)
    sched = mgsl.Schedule(12, 1.0, 1e3, 5)
    b = mgsl.sample(
        jax.random.key_data(jax.random.key(0)),
        lin.lik(),
        jnp.asarray(lin.mean),
        sched,
        tight_options(),
    )
    np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize("batch_size", [1, 5, 12, 50])
def test_batching_does_not_change_samples(lin, batch_size):
    ref = small(lin)
    out = small(lin, batch_size=batch_size)
    np.testing.assert_allclose(out.samples, ref.samples, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(out.state.y, ref.state.y, rtol=1e-8, atol=1e-10)


def test_diagnostics_off_gives_same_samples(lin):
    ref = small(lin)
    out = small(lin, calculate_diagnostics=False)
    np.testing.assert_array_equal(out.samples, ref.samples)
    assert out.diagnostics is None and out.failures is not None


def test_diagnostics_shapes(lin):
    res = small(lin, n_paths=7, n_tau=6)
    d = res.diagnostics
    np.testing.assert_allclose(d.taus, mgsl.grid(1.0, 1e3, 6)[:-1])
    assert d.tau_end == 1e3
    assert d.ess_fraction.shape == (7, 5) and d.newton_iter.shape == (7, 5)
    for s in (d.cg_newton, d.cg_draws):
        for f in s:
            assert np.shape(f) == (7, 5)
    assert np.all((d.ess_fraction > 0) & (d.ess_fraction <= 1 + 1e-9))
    assert np.all((d.cg_draws.cap_fraction >= 0) & (d.cg_draws.cap_fraction <= 1))
    assert d.terminal.ess_fraction.shape == (7,)
    assert res.schedule.n_paths == 7 and res.state.tau == 1e3 and res.state.step == 5


def test_capped_draws_warn_and_invalidate_ess():
    p = LinGauss.make(d=8, n=8, seed=2, scale=1e3)  # Fisher eigenvalues up to ~1e7
    cg = mgsl.CGOptions(tol=1e-10, miniter=0, maxiter=2)
    opts = mgsl.MGVIOptions(
        n_samples_mgvi=2, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=2, cg=cg), draw_cg=cg
    )
    sched = mgsl.Schedule(4, 1.0, 10.0, 3)
    with pytest.warns(RuntimeWarning, match="draw_cg_capped"):
        res = mgsl.sample(jax.random.key(0), p.lik(), jnp.zeros(8), sched, opts)
    assert res.failures.totals()["draw_cg_capped"] > 0
    assert np.all(np.isnan(res.diagnostics.ess_fraction))
    assert np.max(res.diagnostics.cg_draws.max_rel_residual) > 1e-3


def test_no_warning_when_clean(lin):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small(lin)


def test_no_retrace_for_new_data_or_settings(lin):
    traces = []
    A = jnp.asarray(lin.A)

    def forward(x):
        traces.append(1)
        return A @ x

    sched = mgsl.Schedule(4, 1.0, 1e3, 4)
    run = lambda data, sched, opts: mgsl.sample(
        jax.random.key(0),
        mgsl.gaussian(forward, data, noise_std=lin.sd),
        jnp.zeros(6),
        sched,
        opts,
    )
    run(lin.data, sched, tight_options())
    n = len(traces)
    run(lin.data + 1.0, sched, tight_options())  # new data, same shapes
    run(lin.data, mgsl.Schedule(4, 2.0, 5e3, 4), tight_options())  # new taus
    loose = mgsl.MGVIOptions(
        n_samples_mgvi=2,
        n_iter_mgvi=1,
        newton=mgsl.NewtonOptions(
            maxiter=7,
            xtol=1e-3,
            adaptive_cg=False,
            cg=mgsl.CGOptions(tol=1e-6, miniter=0, maxiter=300),
        ),
        draw_cg=mgsl.CGOptions(tol=1e-9, miniter=0, maxiter=400),
    )
    run(lin.data, sched, loose)  # new tolerances and iteration limits
    assert len(traces) == n


def test_type_errors(lin):
    sched = mgsl.Schedule(4, 1.0, 1e3, 4)
    lik, opts = lin.lik(), tight_options()
    res = small(lin, n_paths=4)
    fit = mgsl.mgvi_fit(jax.random.key(0), lik, jnp.zeros(6), replace(opts, n_iter_mgvi=1))
    with pytest.raises(TypeError, match="advance"):
        mgsl.sample(jax.random.key(0), lik, res.state, sched, opts)
    with pytest.raises(TypeError, match=r"\.mean"):
        mgsl.sample(jax.random.key(0), lik, fit, sched, opts)
    with pytest.raises(TypeError, match="complex"):
        mgsl.sample(jax.random.key(0), lik, jnp.zeros(6, complex), sched, opts)
    with pytest.raises(TypeError, match="Likelihood"):
        mgsl.sample(jax.random.key(0), lik.log_likelihood, jnp.zeros(6), sched, opts)


def run_subprocess(code, env_extra=None):
    import os

    env = {**os.environ, **(env_extra or {})}
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=TESTS,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_float32_run():
    out = run_subprocess(
        """
        import warnings
        import jax, jax.numpy as jnp, numpy as np
        import mgsl
        assert not jax.config.jax_enable_x64
        from problems import LinGauss
        p = LinGauss.make(d=6, n=8)
        lik = mgsl.gaussian(lambda x: jnp.asarray(p.A, jnp.float32) @ x,
                            p.data.astype(np.float32), noise_std=p.sd.astype(np.float32))
        opts = mgsl.MGVIOptions(n_samples_mgvi=2, n_iter_mgvi=1,
                                newton=mgsl.NewtonOptions(maxiter=3))
        res = mgsl.sample(jax.random.key(0), lik, jnp.asarray(p.mean, jnp.float32),
                          mgsl.Schedule(64, 1.0, 1e4, 8), opts)
        assert res.samples.dtype == jnp.float32, res.samples.dtype
        assert not jax.config.jax_enable_x64
        ess = res.diagnostics.ess_fraction
        print(np.nanmin(ess), np.isnan(ess).mean())
        """
    )
    ess_min, nan_frac = map(float, out.split())
    assert nan_frac < 0.1 and ess_min > 0.95


def test_devices_shard_identically():
    run_subprocess(
        """
        import jax, jax.numpy as jnp, numpy as np, pytest
        jax.config.update("jax_enable_x64", True)
        import mgsl
        from problems import LinGauss, tight_options
        assert len(jax.devices()) == 4
        p = LinGauss.make(d=6, n=8)
        sched = mgsl.Schedule(10, 1.0, 1e3, 4)
        go = lambda **kw: mgsl.sample(jax.random.key(0), p.lik(), jnp.asarray(p.mean), sched,
                                      tight_options(), **kw)
        ref = go()
        shard = go(devices=jax.devices(), batch_size=4)
        np.testing.assert_allclose(shard.samples, ref.samples, rtol=1e-8, atol=1e-10)
        with pytest.raises(ValueError, match="multiple"):
            go(devices=jax.devices(), batch_size=6)
        # no devices requested: nothing sharded, output on one device
        s = mgsl.init(jax.random.key(0), p.lik(), jnp.asarray(p.mean), n_paths=4,
                      tau_start=1.0, options=tight_options())
        assert isinstance(s.y, np.ndarray)
        """,
        {"XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
    )
