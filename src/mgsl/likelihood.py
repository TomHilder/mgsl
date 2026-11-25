"""The likelihood interface, a Gaussian constructor and a consistency check."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from . import _tree

__all__ = ["Likelihood", "check_likelihood", "gaussian"]


def _as_shape_struct(s: Any) -> jax.ShapeDtypeStruct:
    if isinstance(s, jax.ShapeDtypeStruct):
        out = s
    elif hasattr(s, "shape") and hasattr(s, "dtype"):
        out = jax.ShapeDtypeStruct(tuple(s.shape), s.dtype)
    else:
        raise TypeError(f"data_shape leaves must have .shape and .dtype, got {s!r}")
    if np.issubdtype(out.dtype, np.complexfloating):
        raise NotImplementedError("complex data are not supported yet")
    if not jnp.issubdtype(out.dtype, jnp.floating):
        raise TypeError(
            f"data_shape leaves must have a floating dtype, got {out.dtype}: data_shape describes "
            "the standard-normal vectors e taken by fisher_sqrt, not the data themselves"
        )
    return out


def _is_shape_leaf(s: Any) -> bool:
    return hasattr(s, "shape") and hasattr(s, "dtype")


@jax.tree_util.register_pytree_node_class
class Likelihood:
    """A likelihood on a latent with a standard-normal prior.

    The latent ``x`` can be any pytree of real floating-point arrays. The prior on ``x`` is
    always ``N(0, I)``: a model with another prior is written in standardised (whitened)
    coordinates. For example, a parameter ``z ~ N(m, L L^T)`` becomes ``z = m + L x``, and a
    positive parameter with a log-normal prior becomes ``exp(mu + s x)``; the forward model
    then starts from ``x``.

    The object is a pytree: arrays it holds (for example data captured by a
    ``jax.tree_util.Partial``) are passed to compiled code at run time, and everything else
    (functions, shapes) is static. Functions that close over arrays directly get those arrays
    compiled in as constants instead.

    Parameters
    ----------
    log_likelihood : callable
        ``x -> log p(d | x)``, a scalar. Constants may be dropped.
    fisher : callable
        ``(x, v) -> J^T F J v``: the Fisher metric ``F`` of the noise model (e.g.
        ``diag(1 / sigma^2)`` for Gaussian noise) pulled back to the latent by the Jacobian
        ``J`` of the forward model, applied to a tangent ``v`` with the structure of ``x``.
        Must not include the prior.
    fisher_sqrt : callable
        ``(x, z) -> J^T F^{1/2} z`` for ``z`` with the structure of ``data_shape``, such that
        ``fisher_sqrt(x, z)`` with ``z ~ N(0, I)`` has covariance ``J^T F J``.
    data_shape : pytree of jax.ShapeDtypeStruct
        Structure, shapes and dtypes of the ``z`` taken by ``fisher_sqrt`` (usually those of
        the data).
    fisher_operator : callable, optional
        ``x -> (fmv, fsqrt)`` with ``fmv(v) = J^T F J v`` and ``fsqrt(z) = J^T F^{1/2} z``. If
        given, it is called once per solve and the returned functions are applied inside the
        solve, instead of ``fisher(x, .)`` and ``fisher_sqrt(x, .)``. Useful when linearising
        the model once is cheaper than re-evaluating it at every application; costs the memory
        of the stored linearisation. ``fmv`` and ``fsqrt`` must be pytrees (e.g.
        ``jax.tree_util.Partial``, or built from the function returned by ``jax.vjp``) so that
        operators at several points can be built and applied with ``jax.vmap``.

    Notes
    -----
    Functions (including bound methods) are compared by identity when JAX decides whether to
    reuse compiled code, so a ``Likelihood`` rebuilt after its data changed is recompiled. Data
    held inside a ``jax.tree_util.Partial`` are passed at run time and never recompile; data an
    unchanged function object reads from mutable state are not seen by already compiled code.
    """

    def __init__(
        self,
        log_likelihood: Callable,
        fisher: Callable,
        fisher_sqrt: Callable,
        data_shape: Any,
        fisher_operator: Callable | None = None,
    ):
        for name, fn in (
            ("log_likelihood", log_likelihood),
            ("fisher", fisher),
            ("fisher_sqrt", fisher_sqrt),
        ):
            if not callable(fn):
                raise TypeError(f"{name} must be callable")
        if fisher_operator is not None and not callable(fisher_operator):
            raise TypeError("fisher_operator must be callable")
        if data_shape is None:
            raise TypeError("data_shape is required")
        self.log_likelihood = log_likelihood
        self.fisher = fisher
        self.fisher_sqrt = fisher_sqrt
        self.data_shape = jax.tree_util.tree_map(
            _as_shape_struct, data_shape, is_leaf=_is_shape_leaf
        )
        self.fisher_operator = fisher_operator

    def _fields(self):
        return (
            self.log_likelihood,
            self.fisher,
            self.fisher_sqrt,
            self.data_shape,
            self.fisher_operator,
        )

    def tree_flatten(self):
        return _tree.partition(self._fields())

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.log_likelihood,
            obj.fisher,
            obj.fisher_sqrt,
            obj.data_shape,
            obj.fisher_operator,
        ) = _tree.combine(aux, children)
        return obj

    def __repr__(self):
        return f"Likelihood(data_shape={self.data_shape})"


# --------------------------------------------------------------------------------------------
# Gaussian likelihood


def _check_output(out, data):
    if getattr(out, "shape", None) != data.shape:
        raise ValueError(
            f"forward(x) has shape {getattr(out, 'shape', None)}, but the data have shape "
            f"{data.shape}; they must match exactly"
        )


def _gauss_log_likelihood(forward, data, weight, x):
    f = forward(x)
    _check_output(f, data)
    r = data - f.astype(data.dtype)
    return -0.5 * jnp.sum(weight * r * r)


def _gauss_fisher(forward, weight, x, v):
    # as NIFTy.re: linearise, then transpose the JVP instead of computing a separate VJP
    out, fwd = jax.linearize(forward, x)
    _check_output(out, weight)
    bwd = jax.linear_transpose(fwd, x)
    return bwd((weight * fwd(v)).astype(out.dtype))[0]


def _gauss_fisher_sqrt(forward, sqrt_weight, x, e):
    out, bwd = jax.vjp(forward, x)
    _check_output(out, sqrt_weight)
    return bwd((sqrt_weight * e).astype(out.dtype))[0]


def _op_fmv(vjp, weight, out, v):
    (jv,) = jax.linear_transpose(lambda c: vjp(c)[0], out)(v)
    return vjp((weight * jv).astype(out.dtype))[0]


def _op_fsqrt(vjp, sqrt_weight, out, e):
    return vjp((sqrt_weight * e).astype(out.dtype))[0]


def _gauss_fisher_operator(forward, weight, sqrt_weight, x):
    out, vjp = jax.vjp(forward, x)  # vjp is a pytree holding the linearisation
    _check_output(out, weight)
    return Partial(_op_fmv, vjp, weight, out), Partial(_op_fsqrt, vjp, sqrt_weight, out)


def gaussian(
    forward: Callable,
    data,
    *,
    noise_std,
    mask=None,
    linearize_once: bool = False,
) -> Likelihood:
    """Gaussian likelihood ``d = forward(x) + n``, ``n ~ N(0, diag(noise_std^2))``.

    Parameters
    ----------
    forward : callable
        The forward model, latent ``x`` (standard-normal prior) to an array of exactly the
        data's shape. Must be differentiable and vmappable. Its output is cast to the data's
        dtype. Compute it in the latent's precision: a float32 forward model with a float64
        latent limits the fits to float32 accuracy, which shows up as line-search failures.
    data : array
        The data. Masked entries may hold any value, including NaN.
    noise_std : float or array
        Noise standard deviation: a scalar or an array of exactly ``data.shape``. Positive and
        finite. (NIFTy.re's ``Gaussian`` takes the inverse standard deviation instead.)
    mask : array of bool, optional
        ``True`` where a data point is used, shape ``data.shape``. Masked points get zero
        weight.
    linearize_once : bool
        If True, also provide ``fisher_operator``, so each solve linearises the forward model
        once instead of at every application of ``J^T F J``.

    Returns
    -------
    Likelihood
        With the data and weights as run-time arrays.
    """
    if not callable(forward):
        raise TypeError("forward must be callable")
    data = np.asarray(data)
    if not np.issubdtype(data.dtype, np.floating):
        if np.issubdtype(data.dtype, np.complexfloating):
            raise NotImplementedError("complex data are not supported yet")
        data = data.astype(np.result_type(float))
    std = np.asarray(noise_std, dtype=data.dtype)
    if std.shape not in ((), data.shape):
        raise ValueError(
            f"noise_std must be a scalar or have the data's shape {data.shape}, "
            f"got shape {std.shape}"
        )
    if mask is None:
        mask = np.ones(data.shape, bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != data.shape:
            raise ValueError(f"mask must be a boolean array of shape {data.shape}")
    std_b = np.broadcast_to(std, data.shape)
    if not np.all(np.isfinite(std_b[mask]) & (std_b[mask] > 0)):
        raise ValueError("noise_std must be positive and finite at every unmasked point")
    if not np.all(np.isfinite(data[mask])):
        raise ValueError("data must be finite at every unmasked point (use mask=)")
    safe_std = np.where(mask, std_b, 1.0)
    weight = np.where(mask, 1.0 / safe_std**2, 0.0).astype(data.dtype)
    sqrt_weight = np.where(mask, 1.0 / safe_std, 0.0).astype(data.dtype)
    clean = np.where(mask, data, 0.0).astype(data.dtype)
    data_j, weight_j, sqrt_j = jnp.asarray(clean), jnp.asarray(weight), jnp.asarray(sqrt_weight)
    operator = (
        Partial(_gauss_fisher_operator, forward, weight_j, sqrt_j) if linearize_once else None
    )
    return Likelihood(
        log_likelihood=Partial(_gauss_log_likelihood, forward, data_j, weight_j),
        fisher=Partial(_gauss_fisher, forward, weight_j),
        fisher_sqrt=Partial(_gauss_fisher_sqrt, forward, sqrt_j),
        data_shape=jax.ShapeDtypeStruct(data.shape, data_j.dtype),
        fisher_operator=operator,
    )


# --------------------------------------------------------------------------------------------
# Flat (vectorised-latent) view used by the sampler


class FlatLikelihood(NamedTuple):
    log_likelihood: Callable  # x -> scalar
    fisher: Callable  # (x, v) -> vector
    fisher_sqrt: Callable  # (x, e) -> vector
    operator: Callable  # x -> (fmv, fsqrt)
    draw_e: Callable  # key -> e (pytree like data_shape)


def flatten_likelihood(lik: Likelihood, unravel: Callable, ravel: Callable) -> FlatLikelihood:
    def ll(x):
        return lik.log_likelihood(unravel(x))

    def fisher(x, v):
        return ravel(lik.fisher(unravel(x), unravel(v)))

    def fisher_sqrt(x, e):
        return ravel(lik.fisher_sqrt(unravel(x), e))

    if lik.fisher_operator is None:

        def plain_operator(x):
            return (lambda v: fisher(x, v)), (lambda e: fisher_sqrt(x, e))

        operator = plain_operator

    else:
        fisher_operator = lik.fisher_operator
        flat_in = functools.partial(_flat_tangent, ravel, unravel)
        flat_out = functools.partial(_flat_draw, ravel)

        def given_operator(x):
            ops = fisher_operator(unravel(x))
            if not (isinstance(ops, tuple) and len(ops) == 2):
                raise TypeError("fisher_operator(x) must return a tuple (fmv, fsqrt)")
            return Partial(flat_in, ops[0]), Partial(flat_out, ops[1])

        operator = given_operator

    leaves, treedef = jax.tree_util.tree_flatten(lik.data_shape)

    def draw_e(key):
        keys = jax.random.split(key, len(leaves))
        return jax.tree_util.tree_unflatten(
            treedef, [jax.random.normal(k, s.shape, s.dtype) for k, s in zip(keys, leaves)]
        )

    return FlatLikelihood(ll, fisher, fisher_sqrt, operator, draw_e)


def _flat_tangent(ravel, unravel, fmv, v):
    return ravel(fmv(unravel(v)))


def _flat_draw(ravel, fsqrt, e):
    return ravel(fsqrt(e))


# --------------------------------------------------------------------------------------------
# Consistency check


def check_likelihood(lik: Likelihood, x, key, n_dirs: int = 8, rtol: float | None = None):
    """Check that a likelihood's functions are consistent with each other at ``x``.

    Along ``n_dirs`` random directions ``v`` it compares

    - ``a = v . fisher(x, v)``,
    - ``b = |fisher_sqrt(x, .)^T v|^2`` (via ``jax.linear_transpose``), which must equal ``a``,
    - ``c = -v . H v`` with ``H`` the Hessian of ``log_likelihood``, which for a well-specified
      model near the data is close to ``a`` and always has the same sign.

    Parameters
    ----------
    lik : Likelihood
    x : pytree
        A latent value. Run it at a few points, e.g. zeros (the prior mean) and a fit's centre:
        the Hessian comparison is most informative near the posterior.
    key : jax.Array
        Random key for the directions.
    n_dirs : int
        Number of random directions.
    rtol : float, optional
        Tolerance for ``a == b``. Default 1e-6 in float64, 1e-3 in float32.

    Returns
    -------
    dict
        ``"fisher"``, ``"fisher_sqrt"`` and ``"hessian"``: the quadratic forms ``a``, ``b`` and
        ``c`` per direction; ``"hessian_ratio"``: ``sum(c) / sum(a)``.

    Raises
    ------
    ValueError
        If a function returns the wrong structure or shape, ``J^T F J`` is negative
        along some direction, ``fisher``, ``fisher_sqrt`` and ``fisher_operator`` (if given)
        disagree, or the Hessian of the log-likelihood has the opposite sign (e.g. a negative
        log-likelihood was passed).

    Warns
    -----
    UserWarning
        If the Hessian-to-Fisher ratio is outside [2/3, 3/2], which suggests a missing weight
        or factor (or a point far from the data).
    """
    struct = _tree.latent_struct(x)
    ravel, unravel, d, dtype = _tree.make_unravel(struct)
    x_flat = ravel(x)
    if rtol is None:
        rtol = 1e-6 if dtype == np.float64 else 1e-3

    def check_structure(name, out):
        out_struct = jax.tree_util.tree_structure(out)
        if out_struct != jax.tree_util.tree_structure(struct):
            raise ValueError(f"{name} returned structure {out_struct}, expected that of x")
        for o, s in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(struct)):
            if np.shape(o) != s.shape:
                raise ValueError(
                    f"{name} returned a leaf of shape {np.shape(o)}, expected {s.shape}"
                )

    ll = lik.log_likelihood(x)
    if np.shape(ll) != ():
        raise ValueError(f"log_likelihood must return a scalar, got shape {np.shape(ll)}")
    v0 = unravel(jnp.ones(d, dtype))
    check_structure("fisher", lik.fisher(x, v0))
    e_leaves, e_def = jax.tree_util.tree_flatten(lik.data_shape)
    e0 = jax.tree_util.tree_unflatten(e_def, [jnp.zeros(s.shape, s.dtype) for s in e_leaves])
    check_structure("fisher_sqrt", lik.fisher_sqrt(x, e0))

    fl = flatten_likelihood(lik, unravel, ravel)
    fsqrt_t = jax.linear_transpose(lambda e: fl.fisher_sqrt(x_flat, e), e0)
    grad = jax.grad(fl.log_likelihood)
    vs = jax.random.normal(key, (n_dirs, d), dtype)

    def forms(v):
        a = v @ fl.fisher(x_flat, v)
        (lt_v,) = fsqrt_t(v)
        b = sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(lt_v))
        _, hv = jax.jvp(grad, (x_flat,), (v,))
        return a, b, -(v @ hv)

    a, b, c = (np.asarray(t) for t in jax.vmap(forms)(vs))
    if lik.fisher_operator is not None:
        fmv, fsqrt = fl.operator(x_flat)
        a_op = np.asarray(jax.vmap(lambda v: v @ fmv(v))(vs))
        fsqrt_op_t = jax.linear_transpose(fsqrt, e0)
        b_op = np.asarray(
            jax.vmap(
                lambda v: sum(jnp.sum(t**2) for t in jax.tree_util.tree_leaves(fsqrt_op_t(v)))
            )(vs)
        )
        for name, val in (("fmv", a_op), ("fsqrt", b_op)):
            if np.any(np.abs(val - a) > rtol * np.maximum(np.abs(val), np.abs(a)) + 1e-30):
                raise ValueError(
                    f"fisher_operator's {name} disagrees with fisher: the quadratic forms "
                    f"differ (max relative difference "
                    f"{np.max(np.abs(val - a) / np.maximum(np.abs(a), 1e-300)):.3g})"
                )
    scale = np.maximum(np.abs(a), np.abs(b)).max() + np.finfo(dtype).tiny
    if np.any(a < -rtol * scale):
        raise ValueError("fisher is not positive semi-definite (v . fisher(x, v) < 0)")
    if np.any(np.abs(a - b) > rtol * np.maximum(np.abs(a), np.abs(b)) + rtol * scale * 1e-3):
        raise ValueError(
            "fisher and fisher_sqrt disagree: v . fisher(x, v) != |fisher_sqrt(x, .)^T v|^2 "
            f"(max relative difference {np.max(np.abs(a - b) / np.maximum(a, b)):.3g})"
        )
    ratio = float(c.sum() / a.sum()) if a.sum() > 0 else float("nan")
    if a.sum() > 0 and ratio < -0.5:
        raise ValueError(
            "the Hessian of log_likelihood has the opposite sign to J^T F J "
            f"(ratio {ratio:.3g}): log_likelihood is probably a negative log-likelihood "
            "(e.g. NIFTy.re's energy)"
        )
    if a.sum() > 0 and not 2 / 3 <= ratio <= 1.5:
        warnings.warn(
            f"the Hessian of log_likelihood is {ratio:.3g} times J^T F J along "
            "random directions (about 1 is expected near the data for a well-specified model). "
            "Check for missing noise weights in fisher, a missing factor 1/2 in "
            "log_likelihood, or evaluate at a point closer to the posterior.",
            UserWarning,
            stacklevel=2,
        )
    return {"fisher": a, "fisher_sqrt": b, "hessian": c, "hessian_ratio": ratio}
