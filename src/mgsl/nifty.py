"""Adaptor for NIFTy.re (optional: imports ``nifty.re`` only when used).

Usage::

    import operator

    import nifty.re as jft
    from jax.tree_util import Partial

    import mgsl.nifty

    noise_std_inv = Partial(operator.mul, 1 / sd)     # NIFTy.re wants an operator here
    lh = jft.Gaussian(data, noise_std_inv=noise_std_inv).amend(model)  # model: a jft.Model
    lik = mgsl.nifty.from_likelihood(lh)
    result = mgsl.sample(key, lik, mu, schedule, options)
    samples = mgsl.nifty.to_samples(result)   # a nifty.re Samples
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from .likelihood import Likelihood

__all__ = ["from_likelihood", "to_samples"]


def _jft():
    try:
        import nifty.re as jft  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "mgsl.nifty needs NIFTy.re: install the 'nifty' package (provides nifty.re)"
        ) from e
    return jft


def _neg_energy(lh, kw, x):
    return -lh.energy(x, **kw)


def _metric(lh, kw, x, v):
    return lh.metric(x, v, **kw)


def _left_sqrt_metric(lh, kw, x, e):
    return lh.left_sqrt_metric(x, e, **kw)


def from_likelihood(lh, **primals_kw) -> Likelihood:
    """An :class:`mgsl.Likelihood` from a NIFTy.re likelihood on the standardised latent.

    ``lh`` must be a likelihood whose primals are the latent with a standard-normal prior,
    usually ``likelihood.amend(model)`` for a NIFTy.re ``Model``. NIFTy.re's ``energy`` is the
    negative log-likelihood, its ``metric`` the Fisher metric pulled back to the latent (``J^T F J``) and its
    ``left_sqrt_metric`` the Fisher square root; they map onto ``log_likelihood``, ``fisher``
    and ``fisher_sqrt``.

    Parameters
    ----------
    lh : nifty.re.Likelihood
        Not a ``StandardHamiltonian`` (which already includes the prior).
    **primals_kw
        Extra keyword arguments passed to every call of ``lh``'s methods.

    Notes
    -----
    ``optimize_kl``'s ``point_estimates`` and ``constants`` have no equivalent: fix such
    parameters inside the forward model instead.
    """
    jft = _jft()
    if "Hamiltonian" in type(lh).__name__:
        raise TypeError(
            f"got a {type(lh).__name__}, which includes the prior: pass its .likelihood"
        )
    if not isinstance(lh, jft.Likelihood):
        raise TypeError(f"expected a nifty.re Likelihood, got {type(lh).__name__}")
    shape = lh.lsm_tangents_shape
    if shape is None:
        raise ValueError(
            "the likelihood has no lsm_tangents_shape, so its Fisher square root cannot be "
            "sampled; construct it with lsm_tangents_shape"
        )
    if not isinstance(lh, (jft.likelihood.LikelihoodWithModel, jft.likelihood.LikelihoodSum)):
        warnings.warn(
            "the NIFTy.re likelihood is not amended with a model: mgsl assumes its primals are "
            "the latent with a standard-normal prior",
            stacklevel=2,
        )
    data_shape = jax.tree_util.tree_map(
        lambda s: jax.ShapeDtypeStruct(tuple(s.shape), s.dtype),
        shape,
        is_leaf=lambda s: hasattr(s, "shape") and hasattr(s, "dtype"),
    )
    kw = _FrozenKw(primals_kw)
    return Likelihood(
        log_likelihood=Partial(_neg_energy, lh, kw),
        fisher=Partial(_metric, lh, kw),
        fisher_sqrt=Partial(_left_sqrt_metric, lh, kw),
        data_shape=data_shape,
    )


@jax.tree_util.register_pytree_node_class
class _FrozenKw(dict):
    """Keyword arguments as a pytree (array values are run-time leaves)."""

    def tree_flatten(self):
        keys = tuple(sorted(self))
        return [self[k] for k in keys], keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))

    __hash__ = None  # type: ignore[assignment]


def to_samples(result) -> Any:
    """The samples of an :class:`mgsl.Result` as ``nifty.re.Samples``.

    ``pos`` is the sample mean and the samples are stored relative to it, so ``Samples.at``,
    ``.pos`` and NIFTy.re's tools work as for ``optimize_kl`` output.
    """
    jft = _jft()
    samples = result.samples
    mean = jax.tree_util.tree_map(lambda a: jnp.mean(a, axis=0), samples)
    rel = jax.tree_util.tree_map(lambda a, m: a - m[None], samples, mean)
    return jft.Samples(pos=mean, samples=rel)
