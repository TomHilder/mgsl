"""Pytree helpers: objects whose array leaves are traced and whose other leaves are static."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def is_array(x: Any) -> bool:
    """Leaves passed to compiled code at run time: arrays and Python floats."""
    return isinstance(x, (jax.Array, np.ndarray, np.generic, float)) and not isinstance(x, bool)


class _ById:
    """Static leaf compared by identity.

    Used for callables and unhashable objects: two bound methods of the same object compare
    equal even after the object's attributes changed, so comparing them by value would reuse
    code compiled for the old attributes."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _ById) and other.obj is self.obj


def _static(x: Any) -> Any:
    if callable(x):
        return _ById(x)
    try:
        hash(x)
    except TypeError:
        return _ById(x)
    return x


def _unwrap(x: Any) -> Any:
    return x.obj if isinstance(x, _ById) else x


def partition(fields: tuple) -> tuple[list, tuple]:
    """Split a tuple of fields into array leaves (children) and hashable static data."""
    leaves, treedef = jax.tree_util.tree_flatten(fields)
    mask = tuple(is_array(leaf) for leaf in leaves)
    children = [leaf for leaf, m in zip(leaves, mask) if m]
    static = tuple(_static(leaf) for leaf, m in zip(leaves, mask) if not m)
    return children, (treedef, mask, static)


def combine(aux: tuple, children) -> tuple:
    treedef, mask, static = aux
    children, static = iter(children), iter(static)
    leaves = [next(children) if m else _unwrap(next(static)) for m in mask]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def latent_struct(x: Any) -> Any:
    """Shape-and-dtype pytree of a latent.

    Dtypes are canonicalised (float64 becomes float32 when x64 is disabled). Complex,
    non-floating and mixed floating dtypes are rejected."""

    def leaf(a):
        if isinstance(a, jax.ShapeDtypeStruct):
            shape, dtype = a.shape, a.dtype
        else:
            if not hasattr(a, "shape") or not hasattr(a, "dtype"):
                a = np.asarray(a)
            shape, dtype = np.shape(a), a.dtype
        if jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError("complex latents are not supported")
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(f"latent leaves must have a real floating dtype, got {dtype}")
        return jax.ShapeDtypeStruct(shape, jax.dtypes.canonicalize_dtype(dtype))

    struct = jax.tree_util.tree_map(leaf, x)
    dtypes = {np.dtype(s.dtype) for s in jax.tree_util.tree_leaves(struct)}
    if len(dtypes) > 1:
        raise TypeError(
            f"latent leaves must share one floating dtype, got {sorted(str(d) for d in dtypes)}"
        )
    return struct


def make_unravel(struct: Any):
    """``(ravel, unravel, d, dtype)`` for latents with the given shape-and-dtype pytree."""
    zeros = jax.tree_util.tree_map(lambda s: np.zeros(s.shape, s.dtype), struct)
    flat, unravel = ravel_pytree(zeros)
    dtype = flat.dtype

    def ravel(tree):
        return ravel_pytree(tree)[0].astype(dtype)

    return ravel, unravel, flat.size, dtype


def unravel_batch(flat: np.ndarray, struct: Any) -> Any:
    """Host-side unravel of ``flat`` (n, d) into the latent's structure with leading axis n
    (the inverse of ``ravel_pytree`` per row, without moving the array to a device)."""
    leaves, treedef = jax.tree_util.tree_flatten(struct)
    n = flat.shape[0]
    out, start = [], 0
    for s in leaves:
        size = int(np.prod(s.shape, dtype=np.int64))
        out.append(flat[:, start : start + size].reshape((n, *s.shape)).astype(s.dtype))
        start += size
    return jax.tree_util.tree_unflatten(treedef, out)
