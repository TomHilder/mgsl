"""Linear-Gaussian control: with converged fits MGSL is exact at any grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import LinGauss, LinGaussPytree, assert_standard_normal, tight_options, whiten

import mgsl

N_PATHS = 4000
TAU_FINAL = 1e4


def run(problem, mu, n_tau, tau_start, n_samples=2, n_iter=1, key=0, **kw):
    lik = problem.lik()
    sched = mgsl.Schedule(N_PATHS, tau_start, TAU_FINAL, n_tau)
    opts = tight_options(n_samples=n_samples, n_iter=n_iter, **kw)
    return mgsl.sample(jax.random.key(key), lik, mu, sched, opts)


@pytest.fixture(scope="module")
def lin():
    return LinGauss.make(d=10, n=14, seed=0)


@pytest.mark.parametrize(
    "n_tau, tau_start, n_samples, n_iter",
    [(2, 1.0, 2, 1), (4, 1e-3, 1, 1), (32, 1.0, 4, 2), (4, 1.0, 1, 2), (32, 1e-3, 2, 1)],
)
def test_exact_samples(lin, n_tau, tau_start, n_samples, n_iter):
    res = run(lin, jnp.asarray(lin.mean), n_tau, tau_start, n_samples, n_iter)
    P, m, C = lin.precision, lin.mean, lin.cov
    tK = res.state.tau
    y = res.state.y
    # (1) the accumulated observation has its exact law N(tau m, tau^2 C + tau I)
    assert_standard_normal(whiten(y, tK * m, tK**2 * C + tK * np.eye(lin.d)))
    # (2) the terminal draw given y is the exact localised posterior
    cond_mean = np.linalg.solve(P + tK * np.eye(lin.d), (P @ m)[:, None] + y.T).T
    x = np.asarray(res.samples)
    assert_standard_normal(whiten(x - cond_mean, 0.0, np.linalg.inv(P + tK * np.eye(lin.d))))
    # (3) end to end
    assert_standard_normal(whiten(x, m, C))
    # local ESS is exactly 1 for a Gaussian with converged fits
    np.testing.assert_allclose(res.diagnostics.ess_fraction, 1.0, atol=1e-6)
    np.testing.assert_allclose(res.diagnostics.terminal.ess_fraction, 1.0, atol=1e-6)
    assert not res.failures.any()


def test_exact_samples_pytree():
    p = LinGaussPytree.make()
    fit = mgsl.mgvi_fit(jax.random.key(0), p.lik(), p.zeros(), tight_options(n_iter=2))
    res = run(p, fit.mean, 8, 1.0)
    assert res.samples["a"].shape == (N_PATHS, 2, 3) and res.samples["b"].shape == (N_PATHS, 4)
    x = LinGaussPytree.flat_batch(res.samples)
    assert_standard_normal(whiten(x, p.base.mean, p.base.cov))


def test_wrong_centre_bias_is_exact(lin):
    """Same key, centre m vs m + D: every sample shifts by tau0 (P + tau0)^-1 D exactly."""
    rng = np.random.default_rng(5)
    D = rng.normal(size=lin.d)
    tau0 = 1.0
    a = run(lin, jnp.asarray(lin.mean), 8, tau0, key=3)
    b = run(lin, jnp.asarray(lin.mean + D), 8, tau0, key=3)
    shift = tau0 * np.linalg.solve(lin.precision + tau0 * np.eye(lin.d), D)
    diff = np.asarray(b.samples) - np.asarray(a.samples)
    np.testing.assert_allclose(diff, np.broadcast_to(shift, diff.shape), atol=1e-8)


def test_exact_with_preconditioner(lin):
    P = lin.precision

    def precond(x, tau):
        Minv = jnp.linalg.inv(jnp.asarray(P) + tau * jnp.eye(P.shape[0]))
        return lambda r: Minv @ r

    res = run(lin, jnp.asarray(lin.mean), 8, 1.0, preconditioner=precond)
    assert_standard_normal(whiten(np.asarray(res.samples), lin.mean, lin.cov))
    d = res.diagnostics
    assert np.max(d.cg_draws.mean_iter) <= 2
    assert np.max(d.cg_draws.max_rel_residual) < 1e-8  # of the unpreconditioned system
