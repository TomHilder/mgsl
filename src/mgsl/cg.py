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
) -> tuple[jnp.ndarray, CGInfo]:
    """Solve ``matvec(x) = b`` from ``x0``.

    Stops once the norm of the (recurrence) residual is below ``resnorm`` after at least
    ``miniter`` iterations (or earlier, once it is at rounding level), at ``maxiter`` iterations, on an exactly zero residual, or on a
    breakdown (``p^T A p <= 0`` for a non-zero direction, or non-finite values). The iterate
    stays finite in all of these cases.
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

    def cond(c):
        k, _, r, _, rz, stop = c
        rn = norm(r, norm_ord)
        unconverged = ((k < miniter) & (rn > floor)) | (rn > resnorm)
        return (k < maxiter) & unconverged & (rz != 0) & ~stop

    def body(c):
        k, x, r, p, rz, _ = c
        ap = matvec(p)
        pap = p @ ap
        ok = jnp.isfinite(pap) & (pap > 0)
        alpha = jnp.where(ok, rz / jnp.where(ok, pap, 1.0), 0.0)
        x = x + alpha * p
        r = r - alpha * ap
        z = apply_m(r)
        rz_new = r @ z
        ok = ok & jnp.isfinite(rz_new)
        beta = jnp.where(ok & (rz != 0), rz_new / jnp.where(rz != 0, rz, 1.0), 0.0)
        p = z + beta * p
        return k + 1, x, r, p, jnp.where(ok, rz_new, 0.0), ~ok

    init = (jnp.asarray(0), x0, r0, z0, jnp.where(bad0, 0.0, rz0).astype(dtype), bad0)
    k, x, r, _, _, stop = lax.while_loop(cond, body, init)
    converged = norm(r, norm_ord) <= resnorm
    capped = (k >= maxiter) & ~converged & ~stop
    # any other exit without convergence (non-positive curvature, non-finite values, or a
    # vanishing preconditioned residual r.M^-1 r while r itself is large) is a breakdown
    info = CGInfo(iterations=k, capped=capped, breakdown=~converged & ~capped)
    return x, info


def relative_residual(matvec: Callable, b: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """``|b - A x|_2 / |b|_2`` of the unpreconditioned system (0 when ``b = 0`` and ``x`` solves it)."""
    r = b - matvec(x)
    bn = jnp.sqrt(b @ b)
    rn = jnp.sqrt(r @ r)
    return jnp.where(bn > 0, rn / jnp.where(bn > 0, bn, 1.0), rn)
