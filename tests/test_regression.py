"""Regression checks: the original MGSL implementation's samples, and a large latent."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from problems import Nonlinear2D
from scipy import stats

import mgsl

FIXTURE = Path(__file__).parent / "fixtures" / "prototype_nonlinear2d.npz"


def prototype_options():
    cg = mgsl.CGOptions(tol=1e-4, miniter=0, maxiter=50)
    return mgsl.MGVIOptions(
        n_samples_mgvi=4,
        n_iter_mgvi=1,
        newton=mgsl.NewtonOptions(
            maxiter=2, xtol=0.0, adaptive_cg=False, cg=cg, line_search="armijo"
        ),
        draw_cg=cg,
    )


def test_matches_prototype_samples():
    f = np.load(FIXTURE)
    p = Nonlinear2D(b=float(f["b"]), sd=float(f["sd"]), data=tuple(f["data"]))
    taus = f["sample_taus"]
    ref = f["sample_x"]
    sched = mgsl.Schedule(ref.shape[0], taus[0], taus[-1], len(taus))
    np.testing.assert_allclose(sched.taus, taus, rtol=1e-12)
    res = mgsl.sample(
        jax.random.key(7), p.lik(), jnp.asarray(f["sample_mu"]), sched, prototype_options()
    )
    x = np.asarray(res.samples)
    for i in range(2):
        assert stats.ks_2samp(x[:, i], ref[:, i]).pvalue > 1e-3
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        np.testing.assert_allclose(np.quantile(x[:, i], q), np.quantile(ref[:, i], q), atol=0.06)
    # radial direction around the curved ridge too
    r_new, r_ref = (np.hypot(*(a - a.mean(0)).T) for a in (x, ref))
    assert stats.ks_2samp(r_new, r_ref).pvalue > 1e-3


def test_large_latent_smoke():
    d = 10_000
    rng = np.random.default_rng(0)
    scale = jnp.asarray(rng.uniform(0.5, 50.0, d))

    def forward(x):  # diagonal in Fourier space; no dense d x d matrix anywhere
        return jnp.fft.irfft(jnp.fft.rfft(x) * jnp.fft.rfft(scale) / d, n=d) + x

    data = np.asarray(forward(jnp.asarray(rng.normal(size=d)))) + rng.normal(size=d)
    lik = mgsl.gaussian(forward, data, noise_std=1.0)
    opts = mgsl.MGVIOptions(n_samples_mgvi=2, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=2))
    fit = mgsl.mgvi_fit(jax.random.key(0), lik, jnp.zeros(d), opts, n_rounds=2)
    res = mgsl.sample(
        jax.random.key(1), lik, fit.mean, mgsl.Schedule(8, 1.0, 1e3, 4), opts, batch_size=4
    )
    assert np.asarray(res.samples).shape == (8, d)
    assert np.all(np.isfinite(np.asarray(res.samples)))
    assert not res.failures.any(), res.failures.totals()
