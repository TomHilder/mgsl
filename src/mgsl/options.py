"""Run settings: the localisation grid and the options of the local MGVI fits."""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CGOptions", "MGVIOptions", "NewtonOptions", "Schedule", "grid"]


def _as_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        if not (isinstance(value, (np.ndarray, np.generic)) and np.ndim(value) == 0):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if not np.issubdtype(np.asarray(value).dtype, np.integer):
            raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _as_float(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def grid(tau_from: float, tau_to: float, n: int) -> np.ndarray:
    """Geometric grid of ``n`` points from ``tau_from`` to ``tau_to``.

    The endpoints are exactly ``tau_from`` and ``tau_to`` (as float64), so a continuation built
    with ``grid(state.tau, ...)`` starts exactly where the state is.

    Parameters
    ----------
    tau_from, tau_to : float
        First and last point, ``0 < tau_from < tau_to``.
    n : int
        Number of points, at least 2.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n,)``.
    """
    tau_from = _as_float("tau_from", tau_from)
    tau_to = _as_float("tau_to", tau_to)
    n = _as_int("n", n, 2)
    if tau_from <= 0:
        raise ValueError(f"tau_from must be positive, got {tau_from}")
    if tau_to <= tau_from:
        raise ValueError(f"tau_to ({tau_to}) must be larger than tau_from ({tau_from})")
    taus = np.geomspace(tau_from, tau_to, n)
    taus[0], taus[-1] = tau_from, tau_to
    return taus


@dataclass(frozen=True)
class Schedule:
    """The localisation grid of a run. Every field is required.

    Parameters
    ----------
    n_paths : int
        Number of independent paths, i.e. of posterior samples.
    tau_start : float
        Localisation time at which the paths start (> 0).
    tau_final : float
        Localisation time of the last step and of the terminal fit.
    n_tau : int
        Number of grid points (geometric from ``tau_start`` to ``tau_final``), at least 2.
        A run makes ``n_tau - 1`` steps.
    """

    n_paths: int
    tau_start: float
    tau_final: float
    n_tau: int

    def __post_init__(self):
        object.__setattr__(self, "n_paths", _as_int("n_paths", self.n_paths, 1))
        object.__setattr__(self, "n_tau", _as_int("n_tau", self.n_tau, 2))
        object.__setattr__(self, "tau_start", _as_float("tau_start", self.tau_start))
        object.__setattr__(self, "tau_final", _as_float("tau_final", self.tau_final))
        if self.tau_start <= 0:
            raise ValueError(f"tau_start must be positive, got {self.tau_start}")
        if self.tau_final <= self.tau_start:
            raise ValueError(
                f"tau_final ({self.tau_final}) must be larger than tau_start ({self.tau_start})"
            )

    @property
    def taus(self) -> np.ndarray:
        """The grid, float64 array of shape ``(n_tau,)``."""
        return grid(self.tau_start, self.tau_final, self.n_tau)


@dataclass(frozen=True)
class CGOptions:
    """Stopping rules of a conjugate-gradient solve, with NIFTy.re's names and defaults.

    The solve stops once the residual norm is below ``max(tol * |b|, atol)``, after at least
    ``miniter`` and at most ``maxiter`` iterations.

    Parameters
    ----------
    tol : float
        Tolerance relative to the norm of the right-hand side.
    atol : float
        Absolute tolerance on the residual norm.
    miniter : int, optional
        Minimum number of iterations. Default ``min(6, maxiter)``.
    maxiter : int, optional
        Maximum number of iterations. Default ``min(200, 20 d)`` for a system of size ``d``.
    """

    tol: float = 1e-5
    atol: float = 0.0
    miniter: int | None = None
    maxiter: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "tol", _as_float("tol", self.tol))
        object.__setattr__(self, "atol", _as_float("atol", self.atol))
        if self.tol < 0 or self.atol < 0:
            raise ValueError("tol and atol must be non-negative")
        if self.miniter is not None:
            object.__setattr__(self, "miniter", _as_int("miniter", self.miniter, 0))
        if self.maxiter is not None:
            object.__setattr__(self, "maxiter", _as_int("maxiter", self.maxiter, 1))

    def resolve(self, size: int) -> tuple[int, int]:
        """``(miniter, maxiter)`` for a system of the given size, with NIFTy.re's fallbacks."""
        fallback = 20 * size
        miniter = (
            min(6, self.maxiter if self.maxiter is not None else fallback)
            if self.miniter is None
            else self.miniter
        )
        maxiter = max(min(200, fallback), miniter) if self.maxiter is None else self.maxiter
        return miniter, maxiter


@dataclass(frozen=True)
class NewtonOptions:
    """Stopping rules of the Newton-CG minimisation in each local fit.

    Names and meanings follow NIFTy.re's Newton-CG: the minimisation stops after ``maxiter``
    steps, or once the step taken has 1-norm ``<= xtol * d``, or once the objective decreased by
    less than ``absdelta``. The line search is a backtracking (Armijo) search with a
    steepest-descent fallback, which differs from NIFTy.re's.

    Parameters
    ----------
    maxiter : int
        Maximum number of Newton steps per round. Required.
    xtol : float
        Step-size tolerance, scaled by the dimension ``d``.
    absdelta : float, optional
        Stop when the objective decreased by less than this. Off by default.
    cg : CGOptions
        Iteration limits of the inner CG solves (and their tolerances if ``adaptive_cg`` is
        False).
    adaptive_cg : bool
        If True (default, as NIFTy.re), the inner CG stops at a residual 1-norm of
        ``min(0.5, sqrt(|g|_1)) |g|_1`` for gradient ``g``, ignoring ``cg.tol`` and ``cg.atol``.
    """

    maxiter: int
    xtol: float = 1e-5
    absdelta: float | None = None
    cg: CGOptions = field(default_factory=CGOptions)
    adaptive_cg: bool = True

    def __post_init__(self):
        object.__setattr__(self, "maxiter", _as_int("maxiter", self.maxiter, 1))
        object.__setattr__(self, "xtol", _as_float("xtol", self.xtol))
        if self.xtol < 0:
            raise ValueError("xtol must be non-negative")
        if self.absdelta is not None:
            object.__setattr__(self, "absdelta", _as_float("absdelta", self.absdelta))
        if not isinstance(self.cg, CGOptions):
            raise TypeError("cg must be a CGOptions")
        object.__setattr__(self, "adaptive_cg", bool(self.adaptive_cg))


@dataclass(frozen=True)
class MGVIOptions:
    """Settings of the local MGVI fits made at every step.

    Parameters
    ----------
    n_samples_mgvi : int
        Number of residual draws per fit. Each is mirrored, so the fit averages over
        ``2 * n_samples_mgvi`` points (NIFTy.re's ``n_samples`` with ``mirror_samples=True``).
        Required.
    n_iter_mgvi : int
        Number of rounds (draw residuals, then Newton) per fit. Required.
    newton : NewtonOptions
        Newton-CG settings. Required (its ``maxiter`` has no default).
    draw_cg : CGOptions
        Stopping rules of the CG solves that draw residuals.
    preconditioner : callable, optional
        ``preconditioner(x, tau)`` returning a function ``v -> M^{-1} v`` that approximates the
        inverse of the metric ``(1 + tau) I + I(x)`` (flat vectors). Used in every CG solve of a
        fit, built at the fit's centre. It may be a pytree (e.g. ``jax.tree_util.Partial``)
        whose array leaves are passed at run time rather than compiled in.
    """

    n_samples_mgvi: int
    n_iter_mgvi: int
    newton: NewtonOptions
    draw_cg: CGOptions = field(default_factory=CGOptions)
    preconditioner: Callable | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "n_samples_mgvi", _as_int("n_samples_mgvi", self.n_samples_mgvi, 1)
        )
        object.__setattr__(self, "n_iter_mgvi", _as_int("n_iter_mgvi", self.n_iter_mgvi, 1))
        if not isinstance(self.newton, NewtonOptions):
            raise TypeError("newton must be a NewtonOptions")
        if not isinstance(self.draw_cg, CGOptions):
            raise TypeError("draw_cg must be a CGOptions")
        if self.preconditioner is not None and not callable(self.preconditioner):
            raise TypeError("preconditioner must be callable")
