"""Preconditioned conjugate gradients with iteration statistics."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax


class CGInfo(NamedTuple):
    """Outcome of one solve (scalars)."""

    iterations: jnp.ndarray  # int
    capped: jnp.ndarray  # bool: stopped at maxiter without meeting the tolerance
    breakdown: jnp.ndarray  # bool: stopped early without meeting the tolerance
    rel_residual: jnp.ndarray  # |b - A x|_2 / |b|_2 of the returned x (true residual)


def norm(v, ord: int):
    return jnp.sum(jnp.abs(v)) if ord == 1 else jnp.sqrt(v @ v)


def cg(
    matvec: Callable,
    b: jnp.ndarray,
    x0: jnp.ndarray,
    resnorm: jnp.ndarray,
    miniter: jnp.ndarray,
    maxiter: jnp.ndarray,
    norm_ord: int = 2,
    precond: Callable | None = None,
    refresh: jnp.ndarray | int = 20,
) -> tuple[jnp.ndarray, CGInfo]:
    """Solve ``matvec(x) = b`` from ``x0``.

    CG updates its residual by a recurrence, which drifts away from the true residual
    ``b - A x`` in floating point. Every ``refresh`` iterations, and whenever the recurrence
    says the solve has converged, the residual is replaced by the true one, so convergence is
    always judged on the true residual: if it has not converged, iterations continue.

    Stops once the residual norm is below ``resnorm`` after at least ``miniter`` iterations (or
    earlier, once it is at rounding level), at ``maxiter`` iterations, on an exactly zero
    residual, when the CG objective ``x.A.x / 2 - b.x`` stops decreasing (the iteration has
    reached rounding noise; the previous iterate is kept), or on a breakdown (``p^T A p <= 0``
    for a non-zero direction, or non-finite values). The iterate stays finite in all of these
    cases. Whether a solve counts as converged is decided on its true residual.
    """
    dtype = b.dtype
    apply_m = (lambda r: r) if precond is None else precond
    r0 = b - matvec(x0)
    z0 = apply_m(r0)
    rz0 = r0 @ z0
    bad0 = ~jnp.isfinite(rz0)
    # miniter is not honoured below rounding level: iterating on a residual that is pure rounding
    # error makes the iterate diverge
    floor = 10.0 * jnp.finfo(dtype).eps * norm(b, norm_ord)
    refresh = jnp.maximum(jnp.asarray(refresh), 1)

    def running(k, r, rz, stop):
        rn = norm(r, norm_ord)
        unconverged = ((k < miniter) & (rn > floor)) | (rn > resnorm)
        return (k < maxiter) & unconverged & (rz != 0) & ~stop

    def energy(x, r):
        # 0.5 x.A.x - b.x, using A x = b - r; it decreases at every exact CG iteration
        return -0.5 * (x @ (b + r))

    def body(c):
        k, x, r, p, rz, _, j = c
        ap = matvec(p)
        pap = p @ ap
        ok = jnp.isfinite(pap) & (pap > 0)
        alpha = jnp.where(ok, rz / jnp.where(ok, pap, 1.0), 0.0)
        x_new = x + alpha * p
        r_new = r - alpha * ap
        # an increase of the CG objective (beyond rounding) means the iteration has reached
        # rounding noise: keep the previous iterate and stop (as NIFTy.re's CG does)
        e_old, e_new = energy(x, r), energy(x_new, r_new)
        rising = e_new > e_old + 100.0 * jnp.finfo(dtype).eps * jnp.abs(e_old)
        ok = ok & ~rising
        x, r = jnp.where(ok, x_new, x), jnp.where(ok, r_new, r)
        z = apply_m(r)
        rz_new = r @ z
        ok = ok & jnp.isfinite(rz_new)
        beta = jnp.where(ok & (rz != 0), rz_new / jnp.where(rz != 0, rz, 1.0), 0.0)
        p = z + beta * p
        return k + 1, x, r, p, jnp.where(ok, rz_new, 0.0), ~ok, j + 1

    def block(c):
        """Up to ``refresh`` CG iterations, then replace the residual by the true one (keeping
        the search direction, as NIFTy.re does)."""
        k, x, r, p, rz, stop = c
        k, x, r, p, rz, stop, _ = lax.while_loop(
            lambda s: running(s[0], s[2], s[4], s[5]) & (s[6] < refresh),
            body,
            (k, x, r, p, rz, stop, jnp.asarray(0)),
        )
        r = b - matvec(x)
        z = apply_m(r)
        rz_true = r @ z
        ok = jnp.isfinite(rz_true)
        return k, x, r, p, jnp.where(ok, rz_true, 0.0), stop | ~ok

    init = (jnp.asarray(0), x0, r0, z0, jnp.where(bad0, 0.0, rz0).astype(dtype), bad0)
    k, x, r, _, _, stop = lax.while_loop(lambda c: running(c[0], c[2], c[4], c[5]), block, init)
    converged = norm(r, norm_ord) <= jnp.maximum(resnorm, 0.0)
    capped = (k >= maxiter) & ~converged & ~stop
    # any other exit without convergence (non-positive curvature, non-finite values, or a
    # vanishing preconditioned residual r.M^-1 r while r itself is large) is a breakdown
    bn, rn = jnp.sqrt(b @ b), jnp.sqrt(r @ r)
    rel = jnp.where(bn > 0, rn / jnp.where(bn > 0, bn, 1.0), rn)
    info = CGInfo(k, capped, ~converged & ~capped, jnp.where(jnp.isfinite(rel), rel, jnp.inf))
    return x, info
