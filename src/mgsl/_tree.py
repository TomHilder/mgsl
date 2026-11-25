"""Pytree helpers: objects whose array leaves are traced and whose other leaves are static."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax.flatten_util import ravel_pytree


def is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic))


class _ById:
    """Hashable wrapper for an unhashable static leaf (compared by identity)."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _ById) and other.obj is self.obj


def _hashable(x: Any) -> Any:
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
    static = tuple(_hashable(leaf) for leaf, m in zip(leaves, mask) if not m)
    return children, (treedef, mask, static)


def combine(aux: tuple, children) -> tuple:
    treedef, mask, static = aux
    children, static = iter(children), iter(static)
    leaves = [next(children) if m else _unwrap(next(static)) for m in mask]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def latent_struct(x: Any) -> Any:
    """Shape-and-dtype pytree of a latent, rejecting complex and non-array leaves."""

    def leaf(a):
        if isinstance(a, jax.ShapeDtypeStruct):
            return a
        if not hasattr(a, "shape") or not hasattr(a, "dtype"):
            a = np.asarray(a)
        if np.issubdtype(a.dtype, np.complexfloating):
            raise TypeError("complex latents are not supported")
        if not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"latent leaves must be floating point, got dtype {a.dtype}")
        return jax.ShapeDtypeStruct(np.shape(a), a.dtype)

    return jax.tree_util.tree_map(leaf, x)


def make_unravel(struct: Any):
    """``(ravel, unravel, d, dtype)`` for latents with the given shape-and-dtype pytree."""
    zeros = jax.tree_util.tree_map(lambda s: np.zeros(s.shape, s.dtype), struct)
    flat, unravel = ravel_pytree(zeros)
    dtype = flat.dtype

    def ravel(tree):
        return ravel_pytree(tree)[0].astype(dtype)

    return ravel, unravel, flat.size, dtype
