import jax.numpy as jnp
import numpy as np
import pytest

from mgsl.cg import cg


def spd(d=12, cond=1e3, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q @ np.diag(np.geomspace(1, cond, d)) @ Q.T, rng.normal(size=d)


def run(
    A, b, x0=None, tol=1e-10, atol=0.0, miniter=0, maxiter=500, ord=2, precond=None, refresh=20
):
    A, b = jnp.asarray(A), jnp.asarray(b)
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    resnorm = max(tol * float(np.linalg.norm(b, ord)), atol)
    return cg(lambda v: A @ v, b, x0, resnorm, miniter, maxiter, ord, precond, refresh)


def test_converges_to_solution():
    A, b = spd()
    x, info = run(A, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-7)
    assert not info.capped and not info.breakdown
    true = np.linalg.norm(b - A @ np.asarray(x)) / np.linalg.norm(b)
    np.testing.assert_allclose(float(info.rel_residual), true, rtol=1e-3, atol=1e-12)
    assert true < 1e-9


def test_maxiter_cap_reported():
    A, b = spd(cond=1e6)
    _x, info = run(A, b, maxiter=3)
    assert int(info.iterations) == 3 and bool(info.capped)


def test_miniter_and_atol():
    A, b = spd()
    _, info = run(A, b, tol=0.9, miniter=7)
    assert int(info.iterations) == 7
    _, loose = run(A, b, tol=0.0, atol=1e9)
    assert int(loose.iterations) == 0


@pytest.mark.parametrize("zero_rhs", [False, True])
def test_finite_at_degenerate_inputs(zero_rhs):
    A, b = spd()
    if zero_rhs:
        b = np.zeros_like(b)
        x0 = np.zeros_like(b)
    else:
        x0 = np.linalg.solve(A, b)
    x, info = run(A, b, x0=x0, miniter=6)
    assert np.all(np.isfinite(np.asarray(x)))
    np.testing.assert_allclose(x, x0, atol=1e-12)
    assert not info.breakdown


def test_breakdown_on_indefinite_matrix():
    A = np.diag([1.0, -1.0, 2.0])
    b = np.array([1.0, 1.0, 1.0])
    x, info = run(A, b)
    assert bool(info.breakdown)
    assert np.all(np.isfinite(np.asarray(x)))


def test_nonfinite_matvec_is_breakdown():
    b = jnp.ones(3)
    x, info = cg(lambda v: v * jnp.nan, b, jnp.zeros(3), 1e-8, 0, 50)
    assert bool(info.breakdown) and np.all(np.isfinite(np.asarray(x)))


def test_preconditioner():
    A, b = spd(cond=1e5)
    Ainv = jnp.asarray(np.linalg.inv(A))
    x, info = run(A, b, precond=lambda r: Ainv @ r)
    assert int(info.iterations) <= 2
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-8)
    dinv = jnp.asarray(1 / np.diag(A))
    x2, _ = run(A, b, precond=lambda r: dinv * r)
    np.testing.assert_allclose(x2, np.linalg.solve(A, b), rtol=1e-6)


def test_one_norm_stopping():
    A, b = spd()
    x, info = run(A, b, tol=1e-6, ord=1)
    r = np.asarray(b) - A @ np.asarray(x)
    assert np.abs(r).sum() <= 1e-6 * np.abs(b).sum() * 1.0001 or info.capped


@pytest.mark.parametrize("refresh", [1, 20, 10**6])
def test_convergence_is_judged_on_the_true_residual(refresh):
    """Ill-conditioned system and a tolerance near rounding level: the recurrence residual
    drifts from the true one. Whatever the refresh interval, the reported residual is the true
    one, and a solve reported as converged really is."""
    A, b = spd(d=40, cond=1e10, seed=3)
    tol = 1e-13
    x, info = run(A, b, tol=tol, maxiter=2000, refresh=refresh)
    true = np.linalg.norm(b - A @ np.asarray(x)) / np.linalg.norm(b)
    # at this conditioning A x itself is only accurate to ~1e-6 relative
    np.testing.assert_allclose(float(info.rel_residual), true, rtol=0.05, atol=1e-6)
    if not (bool(info.capped) or bool(info.breakdown)):
        assert true <= tol * 1.0001


def test_refresh_lets_badly_conditioned_solves_converge():
    """Without refreshing, the recurrence residual drifts: CG stalls (its objective stops
    decreasing) well short of the tolerance. Refreshing every 20 iterations keeps converging."""
    A, b = spd(d=60, cond=1e6, seed=4)
    res = {}
    for r in (20, 10**6):
        x, _ = run(A, b, tol=1e-12, maxiter=2000, refresh=r)
        res[r] = np.linalg.norm(b - A @ np.asarray(x)) / np.linalg.norm(b)
    assert res[20] < 1e-8 and res[10**6] > 1e-6
