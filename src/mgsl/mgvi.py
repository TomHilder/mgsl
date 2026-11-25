"""Residual draws and MGVI fits of the localised posterior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import _tree
from .cg import cg, norm, relative_residual
from .diagnostics import Failures, FitStats, NewtonCounts, Solves
from .likelihood import FlatLikelihood, Likelihood, flatten_likelihood
from .options import MGVIOptions

__all__ = ["MGVIResult", "mgvi_fit"]

ARMIJO = 1e-4
MIN_STEP = 1e-12


class Static(NamedTuple):
    """Compile-time settings of a fit."""

    n_samples: int
    n_iter: int
    adaptive_cg: bool
    line_search: str
    has_absdelta: bool
    diagnostics: bool


class Runtime(NamedTuple):
    """Run-time (traced) settings of a fit."""

    draw_tol: Any
    draw_atol: Any
    draw_miniter: Any
    draw_maxiter: Any
    newton_maxiter: Any
    xtol: Any
    absdelta: Any
    ncg_tol: Any
    ncg_atol: Any
    ncg_miniter: Any
    ncg_maxiter: Any


def split_options(options: MGVIOptions, d: int, dtype, diagnostics: bool):
    dmin, dmax = options.draw_cg.resolve(d)
    nmin, nmax = options.newton.cg.resolve(d)
    f = lambda v: jnp.asarray(v, dtype)
    i = lambda v: jnp.asarray(v, jnp.int32)
    static = Static(
        n_samples=options.n_samples_mgvi,
        n_iter=options.n_iter_mgvi,
        adaptive_cg=options.newton.adaptive_cg,
        line_search=options.newton.line_search,
        has_absdelta=options.newton.absdelta is not None,
        diagnostics=diagnostics,
    )
    runtime = Runtime(
        draw_tol=f(options.draw_cg.tol),
        draw_atol=f(options.draw_cg.atol),
        draw_miniter=i(dmin),
        draw_maxiter=i(dmax),
        newton_maxiter=i(options.newton.maxiter),
        xtol=f(options.newton.xtol * d),
        absdelta=f(0.0 if options.newton.absdelta is None else options.newton.absdelta),
        ncg_tol=f(options.newton.cg.tol),
        ncg_atol=f(options.newton.cg.atol),
        ncg_miniter=i(nmin),
        ncg_maxiter=i(nmax),
    )
    return static, runtime


class Problem(NamedTuple):
    """Everything a fit needs about the model (built inside traced code)."""

    fl: FlatLikelihood
    precond: Callable | None  # (x, tau) -> (v -> M^-1 v), flat
    d: int
    operator_given: bool  # the likelihood provides fisher_operator


def make_problem(lik: Likelihood, preconditioner, hstruct) -> Problem:
    ravel, unravel, d, _ = _tree.make_unravel(hstruct.struct)
    return Problem(
        flatten_likelihood(lik, unravel, ravel),
        preconditioner,
        d,
        lik.fisher_operator is not None,
    )


# --------------------------------------------------------------------------------------------
# Residual draws


def draw_residuals(prob: Problem, key, xbar, tau, n, st: Static, rt: Runtime):
    """``n`` residuals ``delta ~ N(0, Lambda^-1)`` at ``xbar``; returns deltas, zetas, stats."""
    fl = prob.fl
    fmv, fsqrt = fl.operator(xbar)
    shift = 1.0 + tau

    def matvec(v):
        return shift * v + fmv(v)

    precond = prob.precond(xbar, tau) if prob.precond is not None else None

    def one(k):
        k1, k2 = jax.random.split(k)
        e1 = jax.random.normal(k1, xbar.shape, xbar.dtype)
        zeta = jnp.sqrt(shift) * e1 + fsqrt(fl.draw_e(k2))
        resnorm = jnp.maximum(rt.draw_tol * jnp.sqrt(zeta @ zeta), rt.draw_atol)
        delta, info = cg(
            matvec,
            zeta,
            e1 / jnp.sqrt(shift),
            resnorm,
            rt.draw_miniter,
            rt.draw_maxiter,
            2,
            precond,
        )
        res = (
            relative_residual(matvec, zeta, delta)
            if st.diagnostics
            else jnp.asarray(0.0, xbar.dtype)
        )
        return delta, zeta, info, res

    deltas, zetas, infos, res = jax.vmap(one)(jax.random.split(key, n))
    stats = Solves(
        n=jnp.asarray(n, jnp.int32),
        iter_sum=infos.iterations.sum().astype(jnp.int32),
        iter_max=infos.iterations.max().astype(jnp.int32),
        capped=infos.capped.sum().astype(jnp.int32),
        breakdown=infos.breakdown.sum().astype(jnp.int32),
        max_res=res.max(),
    )
    valid = ~(infos.capped.any() | infos.breakdown.any())
    return deltas, zetas, stats, valid


# --------------------------------------------------------------------------------------------
# Newton-CG on the sample-averaged objective


def _objective(fl: FlatLikelihood, tau, y):
    def u(x):
        return 0.5 * (1.0 + tau) * (x @ x) - y @ x - fl.log_likelihood(x)

    return u


def newton(prob: Problem, xbar, deltas, tau, y, st: Static, rt: Runtime):
    """Newton-CG on the mean of the localised negative log posterior over ``xbar +- deltas``."""
    fl = prob.fl
    d = prob.d
    u = _objective(fl, tau, y)
    shift = 1.0 + tau
    eps = jnp.finfo(xbar.dtype).eps

    def kl(x):
        xs = jnp.concatenate([x + deltas, x - deltas])
        return jnp.mean(jax.vmap(u)(xs))

    def curvature(x):
        xs = jnp.concatenate([x + deltas, x - deltas])
        if prob.operator_given:
            # one linearisation per point and solve (unrolled: the number of points is static)
            ops = [fl.operator(xs[i])[0] for i in range(xs.shape[0])]

            def mv(v):
                return shift * v + sum(op(v) for op in ops) / len(ops)

        else:

            def mv(v):
                return shift * v + jnp.mean(jax.vmap(fl.fisher, (0, None))(xs, v), 0)

        return mv

    def cond(c):
        i, _, _, done, _, _ = c
        return (i < rt.newton_maxiter) & ~done

    def body(c):
        i, x, val, _, counts, solves = c
        g = jax.grad(kl)(x)
        mv = curvature(x)
        if st.adaptive_cg:
            g1 = norm(g, 1)
            resnorm, ord_ = jnp.minimum(0.5, jnp.sqrt(g1)) * g1, 1
        else:
            resnorm, ord_ = jnp.maximum(rt.ncg_tol * norm(g, 2), rt.ncg_atol), 2
        precond = prob.precond(x, tau) if prob.precond is not None else None
        p, info = cg(mv, -g, y / shift - x, resnorm, rt.ncg_miniter, rt.ncg_maxiter, ord_, precond)
        res = relative_residual(mv, -g, p) if st.diagnostics else jnp.asarray(0.0, x.dtype)
        solves = solves.add(info, res)

        gn = jnp.sqrt(g @ g)
        sd = -g * jnp.minimum(1.0, jnp.sqrt(d) / jnp.where(gn > 0, gn, 1.0))
        tiny = 100.0 * eps * jnp.maximum(jnp.abs(val), 1.0)
        search = _armijo if st.line_search == "armijo" else _nifty_search
        x_new, f_new, accept, fallback, significant, absdelta_ok = search(
            kl, mv, x, val, g, p, sd, tiny
        )
        decrease = val - f_new
        step = jnp.where(accept, norm(x_new - x, 1), 0.0)

        stop = ~accept | (step <= rt.xtol)
        if st.has_absdelta:
            stop = stop | ((decrease >= 0) & (decrease < rt.absdelta) & absdelta_ok)
        finite = jnp.isfinite(val)
        counts = NewtonCounts(
            counts.iterations + 1,
            counts.sd_fallbacks + (fallback & significant & finite).astype(jnp.int32),
            counts.line_search_failures + (~accept & significant & finite).astype(jnp.int32),
            counts.nonfinite,
        )
        return i + 1, x_new, f_new, stop, counts, solves

    val0 = kl(xbar)
    counts0 = NewtonCounts.zero()._replace(nonfinite=(~jnp.isfinite(val0)).astype(jnp.int32))
    init = (jnp.asarray(0), xbar, val0, jnp.asarray(False), counts0, Solves.zero(xbar.dtype))
    _, x, _, _, counts, solves = lax.while_loop(cond, body, init)
    return x, counts, solves


def _armijo(kl, mv, x, val, g, p, sd, tiny):
    """Backtracking with the Armijo condition (constant 1e-4) from step 1, halving down to
    MIN_STEP; a non-descent or non-finite CG direction is replaced by the capped steepest-descent
    step ``sd`` first. A step is kept only if the objective decreased."""
    slope = g @ p
    bad = ~(jnp.isfinite(slope) & (slope < 0))
    p = jnp.where(bad, sd, p)
    slope = jnp.where(bad, g @ sd, slope)
    significant = jnp.isfinite(slope) & (-slope > tiny)

    def cond(c):
        a, f = c
        return ~(f <= val + ARMIJO * a * slope) & (a > MIN_STEP)

    def body(c):
        a = 0.5 * c[0]
        return a, kl(x + a * p)

    a, f = lax.while_loop(cond, body, (jnp.asarray(1.0, x.dtype), kl(x + p)))
    accept = jnp.isfinite(f) & (f < val)
    x_new = jnp.where(accept, x + a * p, x)
    f_new = jnp.where(accept, f, val)
    return x_new, f_new, accept, bad, significant, jnp.asarray(True)


NIFTY_TRIES = 9  # step lengths tried by NIFTy.re's Newton-CG line search
NIFTY_RESET = 5  # after this many failed halvings (counting from 0) it switches to a gradient step


def _nifty_search(kl, mv, x, val, g, p, sd, tiny):
    """NIFTy.re's Newton-CG line search: try steps 1, 1/2, 1/4, ... along the Newton direction
    and accept the first that does not increase the objective; after six failures restart from
    step 1 along the gradient, scaled by ``g.g / g.H.g`` (the capped steepest-descent step if
    that curvature is not positive); give up (no step) after nine tries."""

    def cond(c):
        it, _, _, f, _ = c
        return ~(f <= val) & (it < NIFTY_TRIES - 1)

    def body(c):
        it, a, q, _, reset = c

        def to_gradient(_):
            curv = g @ mv(g)
            ok = jnp.isfinite(curv) & (curv > 0)
            return jnp.where(ok, -(g @ g) / jnp.where(ok, curv, 1.0) * g, sd)

        do_reset = it == NIFTY_RESET
        q = lax.cond(do_reset, to_gradient, lambda _: q, None)
        a = jnp.where(do_reset, 1.0, 0.5 * a)
        return it + 1, a, q, kl(x + a * q), reset | do_reset

    init = (jnp.asarray(0), jnp.asarray(1.0, x.dtype), p, kl(x + p), jnp.asarray(False))
    it, a, q, f, reset = lax.while_loop(cond, body, init)
    accept = jnp.isfinite(f) & (f <= val)
    x_new = jnp.where(accept, x + a * q, x)
    f_new = jnp.where(accept, f, val)
    pred = g @ sd  # predicted decrease of a capped gradient step: is there anything to gain?
    significant = jnp.isfinite(pred) & (-pred > tiny)
    # NIFTy.re applies absdelta only when the line search needed fewer than two halvings
    return x_new, f_new, accept, reset, significant, it < 2


# --------------------------------------------------------------------------------------------
# Local fit and ESS


def local_ess(fl: FlatLikelihood, xbar, deltas, zetas, tau, y, valid):
    """Effective sample size fraction of the mirrored residuals as importance samples of the
    localised posterior; NaN where the draws' solves did not converge."""
    shift = 1.0 + tau
    lin = (y - shift * xbar) @ deltas.T  # (n,)
    quad = -0.5 * shift * jnp.sum(deltas * deltas, -1) + 0.5 * jnp.sum(deltas * zetas, -1)
    l0 = fl.log_likelihood(xbar)
    lp = jax.vmap(fl.log_likelihood)(xbar + deltas) - l0
    lm = jax.vmap(fl.log_likelihood)(xbar - deltas) - l0
    logw = jnp.concatenate([lin + quad + lp, -lin + quad + lm])
    w = jax.nn.softmax(logw)
    ess = 1.0 / jnp.sum(w * w) / logw.shape[0]
    return jnp.where(valid & jnp.isfinite(ess), ess, jnp.nan)


def local_fit(prob: Problem, key, xbar, tau, y, st: Static, rt: Runtime):
    """MGVI fit of the localised posterior at ``(tau, y)`` from ``xbar``.

    Returns the centre, ``n_samples`` fresh residuals drawn at it (and their ``zeta``), and
    the fit's statistics.
    """
    k_rounds, k_final = jax.random.split(key)

    def round_(carry, k):
        x, counts, nsolves, dsolves = carry
        deltas, _, dstats, _ = draw_residuals(prob, k, x, tau, st.n_samples, st, rt)
        x, c, s = newton(prob, x, deltas, tau, y, st, rt)
        counts = counts + c
        return (x, counts, nsolves.merge(s), dsolves.merge(dstats)), None

    init = (xbar, NewtonCounts.zero(), Solves.zero(xbar.dtype), Solves.zero(xbar.dtype))
    (xbar, counts, nsolves, dsolves), _ = lax.scan(
        round_, init, jax.random.split(k_rounds, st.n_iter)
    )
    deltas, zetas, fstats, valid = draw_residuals(prob, k_final, xbar, tau, st.n_samples, st, rt)
    dsolves = dsolves.merge(fstats)
    ess = (
        local_ess(prob.fl, xbar, deltas, zetas, tau, y, valid)
        if st.diagnostics
        else jnp.asarray(jnp.nan, xbar.dtype)
    )
    stats = FitStats(dsolves, nsolves, counts, ess)
    return xbar, deltas, zetas, stats


# --------------------------------------------------------------------------------------------
# Global fit


class MGVIResult(NamedTuple):
    """Result of :func:`mgvi_fit`.

    Attributes
    ----------
    mean : pytree
        Centre of the fitted Gaussian, structured like the latent.
    residuals : pytree
        ``n_samples_mgvi`` residuals drawn at the centre, leading axis ``n_samples_mgvi``; the
        samples of the fitted Gaussian are ``mean +- residuals``.
    failures : Failures
        Solver failure counts of the fit (arrays of shape ``(1,)``).
    """

    mean: Any
    residuals: Any
    failures: Failures


def _fit_impl(lik, precond, key, x0, st, rt, hstruct):
    prob = make_problem(lik, precond, hstruct)
    zero = jnp.zeros_like(x0)
    tau = jnp.asarray(0.0, x0.dtype)
    xbar, deltas, _, stats = local_fit(prob, key, x0, tau, zero, st, rt)
    return xbar, deltas, stats


_fit_jit = jax.jit(_fit_impl, static_argnames=("st", "hstruct"))


def mgvi_fit(key, lik: Likelihood, x0, options: MGVIOptions) -> MGVIResult:
    """MGVI fit of the posterior (no localisation), e.g. to find the centre paths start from.

    Runs ``options.n_iter_mgvi`` rounds; each draws ``options.n_samples_mgvi`` mirrored
    residuals at the current centre and runs Newton-CG on the sample-averaged negative log
    posterior. A fit of the full posterior from a cold start usually needs more rounds than the
    warm-started fits inside a run, e.g. ``dataclasses.replace(options, n_iter_mgvi=10)``.
    Check convergence, e.g. by comparing the centre after more rounds.

    Parameters
    ----------
    key : jax.Array
    lik : Likelihood
    x0 : pytree
        Starting centre (e.g. zeros, the prior mean).
    options : MGVIOptions

    Returns
    -------
    MGVIResult
    """
    struct = _tree.latent_struct(x0)
    ravel, unravel, d, dtype = _tree.make_unravel(struct)
    st, rt = split_options(options, d, dtype, diagnostics=False)
    pre = _Precond(options.preconditioner) if options.preconditioner is not None else None
    xbar, deltas, stats = _fit_jit(lik, pre, _key(key), ravel(x0), st, rt, HashableStruct(struct))
    failures = Failures.from_stats(jax.tree_util.tree_map(lambda a: a[None], stats))
    failures.warn("mgvi_fit")
    return MGVIResult(
        mean=unravel(xbar),
        residuals=jax.vmap(unravel)(deltas),
        failures=failures,
    )


# --------------------------------------------------------------------------------------------
# helpers shared with the sampler


@jax.tree_util.register_pytree_node_class
class _Precond:
    """Pytree wrapper of a user preconditioner (array leaves dynamic)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, tau):
        return self.fn(x, tau)

    def tree_flatten(self):
        return _tree.partition((self.fn,))

    @classmethod
    def tree_unflatten(cls, aux, children):
        (fn,) = _tree.combine(aux, children)
        return cls(fn)


class HashableStruct:
    """Latent shape/dtype structure usable as a static jit argument."""

    def __init__(self, struct):
        leaves, treedef = jax.tree_util.tree_flatten(struct)
        self.struct = struct
        self.treedef = treedef
        self.leaves = tuple((tuple(s.shape), np.dtype(s.dtype).str) for s in leaves)

    def __hash__(self):
        return hash((self.treedef, self.leaves))

    def __eq__(self, other):
        return (
            isinstance(other, HashableStruct)
            and other.treedef == self.treedef
            and other.leaves == self.leaves
        )


def _key(key):
    """A typed PRNG key from a typed or legacy raw key."""
    key = jnp.asarray(key) if not isinstance(key, jax.Array) else key
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key
    return jax.random.wrap_key_data(key)
