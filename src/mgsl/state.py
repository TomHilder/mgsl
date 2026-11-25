"""Sampler state, and saving and loading it."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import _tree

__all__ = ["SLState", "load_state", "save_state"]

FORMAT_VERSION = 1


@dataclass(frozen=True)
class SLState:
    """State of a set of localisation paths at one localisation time.

    It holds everything needed to continue the paths: :func:`mgsl.advance` continues them and
    :func:`mgsl.finalize` draws samples from them. Random numbers are derived from the run's
    key, the path index and the global step, so continuing a state gives exactly the samples
    an uninterrupted run would have given.

    Attributes
    ----------
    y : numpy.ndarray
        ``(n_paths, d)``: the accumulated localisation observation of each path (flat latent).
    xbar : numpy.ndarray
        ``(n_paths, d)``: the centre of each path's most recent local fit, the warm start of
        the next.
    tau : float
        Localisation time.
    step : int
        Number of steps taken since the start.
    key_data : numpy.ndarray
        Raw data of the run's random key.
    key_impl : str
        Name of the key's PRNG implementation.
    struct : pytree of jax.ShapeDtypeStruct
        Structure of the latent.
    """

    y: np.ndarray
    xbar: np.ndarray
    tau: float
    step: int
    key_data: np.ndarray
    key_impl: str
    struct: Any

    @property
    def n_paths(self) -> int:
        return int(self.y.shape[0])

    @property
    def d(self) -> int:
        return int(self.y.shape[1])

    @property
    def key(self) -> jax.Array:
        return jax.random.wrap_key_data(jnp.asarray(self.key_data), impl=self.key_impl)

    def __repr__(self):
        return f"SLState(n_paths={self.n_paths}, d={self.d}, tau={self.tau:.6g}, step={self.step})"


def _struct_meta(struct) -> tuple[str, list, list]:
    leaves, treedef = jax.tree_util.tree_flatten(struct)
    return str(treedef), [list(s.shape) for s in leaves], [np.dtype(s.dtype).str for s in leaves]


def save_state(path: str | os.PathLike, state: SLState) -> None:
    """Save a state to an ``.npz`` file.

    The file stores the arrays, the localisation time and step, the key, and the latent's
    structure, shapes and dtypes, which :func:`load_state` checks.
    """
    treedef, shapes, dtypes = _struct_meta(state.struct)
    with open(path, "wb") as f:
        np.savez(
            f,
            format_version=np.asarray(FORMAT_VERSION),
            y=np.asarray(state.y),
            xbar=np.asarray(state.xbar),
            tau=np.asarray(state.tau, np.float64),
            step=np.asarray(state.step, np.int64),
            key_data=np.asarray(state.key_data),
            key_impl=np.asarray(state.key_impl),
            meta=np.asarray(json.dumps({"treedef": treedef, "shapes": shapes, "dtypes": dtypes})),
        )


def load_state(path: str | os.PathLike, like: Any) -> SLState:
    """Load a state saved by :func:`save_state`.

    Parameters
    ----------
    path : str or path-like
    like : pytree
        A latent (arrays or ``jax.ShapeDtypeStruct``) with the structure the state was made
        with; it restores the pytree structure.

    Raises
    ------
    ValueError
        If the file's format version, structure, shapes or dtypes do not match.
    """
    with np.load(path, allow_pickle=False) as f:
        version = int(f["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"state file format version {version}, this mgsl reads version {FORMAT_VERSION}"
            )
        meta = json.loads(str(f["meta"]))
        struct = _tree.latent_struct(like)
        treedef, shapes, dtypes = _struct_meta(struct)
        if meta["treedef"] != treedef:
            raise ValueError(
                f"latent structure mismatch: file has {meta['treedef']}, like= has {treedef}"
            )
        if meta["shapes"] != shapes or meta["dtypes"] != dtypes:
            raise ValueError(
                f"latent leaf mismatch: file has shapes {meta['shapes']} and dtypes "
                f"{meta['dtypes']}, like= has {shapes} and {dtypes}"
            )
        return SLState(
            y=f["y"],
            xbar=f["xbar"],
            tau=float(f["tau"]),
            step=int(f["step"]),
            key_data=f["key_data"],
            key_impl=str(f["key_impl"]),
            struct=struct,
        )
