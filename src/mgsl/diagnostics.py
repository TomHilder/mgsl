"""Solver failure counters and run diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

__all__ = ["CGStats", "Failures", "FitDiagnostics", "MGSLDiagnostics"]


# --------------------------------------------------------------------------------------------
# Accumulators used inside compiled code


class Solves(NamedTuple):
    """Accumulated statistics of a group of CG solves (scalars, traced)."""

    n: Any
    iter_sum: Any
    iter_max: Any
    capped: Any
    breakdown: Any
    max_res: Any

    @staticmethod
    def zero(dtype):
        z = jnp.asarray(0, jnp.int32)
        return Solves(z, z, z, z, z, jnp.asarray(0.0, dtype))

    def add(self, info, res):
        it = info.iterations.astype(jnp.int32)
        return Solves(
            self.n + 1,
            self.iter_sum + it,
            jnp.maximum(self.iter_max, it),
            self.capped + info.capped.astype(jnp.int32),
            self.breakdown + info.breakdown.astype(jnp.int32),
            jnp.maximum(self.max_res, res),
        )

    def merge(self, other):
        return Solves(
            self.n + other.n,
            self.iter_sum + other.iter_sum,
            jnp.maximum(self.iter_max, other.iter_max),
            self.capped + other.capped,
            self.breakdown + other.breakdown,
            jnp.maximum(self.max_res, other.max_res),
        )


class NewtonCounts(NamedTuple):
    iterations: Any
    sd_fallbacks: Any
    line_search_failures: Any
    nonfinite: Any

    @staticmethod
    def zero():
        z = jnp.asarray(0, jnp.int32)
        return NewtonCounts(z, z, z, z)

    def __add__(self, other):
        return NewtonCounts(*(a + b for a, b in zip(self, other)))


class FitStats(NamedTuple):
    """Everything recorded about one local fit (scalars, traced)."""

    draws: Solves
    newton_cg: Solves
    newton: NewtonCounts
    ess: Any


# --------------------------------------------------------------------------------------------
# Public results


class Failures(NamedTuple):
    """Solver failure counts per path, always recorded.

    Each field is an integer array of shape ``(n_paths,)``, summed over all fits of the call.
    Non-zero counts mean some results rest on unconverged or failed numerics.

    Attributes
    ----------
    draw_cg_capped : array
        Residual-draw CG solves that stopped at ``maxiter``: those residuals are too narrow in
        the directions CG had not reached, and the local ESS of that fit is not computed.
    newton_cg_capped : array
        Newton CG solves that stopped at ``maxiter`` (inexact Newton steps; usually harmless
        if the Newton iteration still converges).
    cg_breakdowns : array
        CG solves stopped by non-positive curvature or non-finite values: the Fisher
        information is not positive semi-definite or the model returned NaN/inf.
    sd_fallbacks : array
        Newton steps where the CG direction was not a descent direction and steepest descent
        was used instead.
    line_search_failures : array
        Newton steps where the line search found no decrease although one was predicted.
    nonfinite : array
        Fits whose objective was not finite at the starting centre: the path cannot move
        ("frozen"), typically NaN data or a log-likelihood undefined there.
    """

    draw_cg_capped: Any
    newton_cg_capped: Any
    cg_breakdowns: Any
    sd_fallbacks: Any
    line_search_failures: Any
    nonfinite: Any

    @staticmethod
    def from_stats(stats: FitStats) -> Failures:
        """From stats with leading axis paths (and any further axes, summed over)."""

        def total(a):
            a = np.asarray(a)
            return a.reshape(a.shape[0], -1).sum(1)

        return Failures(
            draw_cg_capped=total(stats.draws.capped),
            newton_cg_capped=total(stats.newton_cg.capped),
            cg_breakdowns=total(stats.draws.breakdown) + total(stats.newton_cg.breakdown),
            sd_fallbacks=total(stats.newton.sd_fallbacks),
            line_search_failures=total(stats.newton.line_search_failures),
            nonfinite=total(stats.newton.nonfinite),
        )

    def __add__(self, other: Failures) -> Failures:  # type: ignore[override]
        return Failures(*(np.asarray(a) + np.asarray(b) for a, b in zip(self, other)))

    def totals(self) -> dict[str, int]:
        """Counts summed over paths."""
        return {k: int(np.sum(v)) for k, v in self._asdict().items()}

    def any(self) -> bool:
        return any(v > 0 for v in self.totals().values())

    def warn(self, where: str) -> None:
        """Issue a warning listing the non-zero counts."""
        if not self.any():
            return
        n_paths = len(np.asarray(self.nonfinite))
        parts = []
        for k, v in self._asdict().items():
            v = np.asarray(v)
            if v.sum() > 0:
                parts.append(f"{k}={int(v.sum())} (in {int((v > 0).sum())}/{n_paths} paths)")
        warnings.warn(
            f"mgsl {where}: solver failures: "
            + ", ".join(parts)
            + ". See the docstring of mgsl.Failures (result.failures) for what each count "
            "means; raising the CG or Newton maxiter, or checking the likelihood with "
            "mgsl.check_likelihood, usually helps.",
            RuntimeWarning,
            stacklevel=3,
        )


class CGStats(NamedTuple):
    """Statistics of the CG solves of each fit (arrays shaped like the fits).

    Attributes
    ----------
    mean_iter, max_iter : array
        Mean and maximum iterations per solve.
    max_rel_residual : array
        Largest ``|b - A x| / |b|`` of the unpreconditioned system.
    cap_fraction : array
        Fraction of solves that stopped at ``maxiter``.
    """

    mean_iter: Any
    max_iter: Any
    max_rel_residual: Any
    cap_fraction: Any

    @staticmethod
    def from_solves(s: Solves) -> CGStats:
        n = np.maximum(np.asarray(s.n), 1)
        return CGStats(
            mean_iter=np.asarray(s.iter_sum) / n,
            max_iter=np.asarray(s.iter_max),
            max_rel_residual=np.asarray(s.max_res),
            cap_fraction=np.asarray(s.capped) / n,
        )


class FitDiagnostics(NamedTuple):
    """Diagnostics of a set of local fits (arrays with leading axis paths).

    Attributes
    ----------
    ess_fraction : array
        Effective sample size of the fit's mirrored residuals as importance samples of the
        localised posterior, divided by ``2 * n_samples_mgvi``. 1 means the local Gaussian is
        exact; NaN where a residual's CG solve did not converge.
    newton_iter : array
        Newton steps taken per fit (all rounds).
    cg_newton, cg_draws : CGStats
        Newton and residual-draw CG statistics.
    """

    ess_fraction: Any
    newton_iter: Any
    cg_newton: CGStats
    cg_draws: CGStats

    @staticmethod
    def from_stats(stats: FitStats) -> FitDiagnostics:
        return FitDiagnostics(
            ess_fraction=np.asarray(stats.ess),
            newton_iter=np.asarray(stats.newton.iterations),
            cg_newton=CGStats.from_solves(stats.newton_cg),
            cg_draws=CGStats.from_solves(stats.draws),
        )


@dataclass(frozen=True)
class MGSLDiagnostics:
    """Per-path diagnostics of a run (numpy arrays only).

    Attributes
    ----------
    taus : numpy.ndarray
        ``(n_steps,)``: the localisation time of each step's fit.
    tau_end : float
        Localisation time reached after the last step.
    ess_fraction : numpy.ndarray
        ``(n_paths, n_steps)``, see :class:`FitDiagnostics`. The local ESS fraction over tau,
        per path, is the main convergence diagnostic: it rises towards 1 once the local
        Gaussian approximation becomes accurate.
    newton_iter : numpy.ndarray
        ``(n_paths, n_steps)``.
    cg_newton, cg_draws : CGStats
        Fields of shape ``(n_paths, n_steps)``.
    terminal : FitDiagnostics or None
        The terminal fit at ``tau_end`` that produced the samples (fields of shape
        ``(n_paths,)``), if the run was finalised.
    """

    taus: np.ndarray
    tau_end: float
    ess_fraction: np.ndarray
    newton_iter: np.ndarray
    cg_newton: CGStats
    cg_draws: CGStats
    terminal: FitDiagnostics | None = None

    @staticmethod
    def from_steps(taus, tau_end, stats: FitStats) -> MGSLDiagnostics:
        fd = FitDiagnostics.from_stats(stats)
        return MGSLDiagnostics(
            taus=np.asarray(taus),
            tau_end=float(tau_end),
            ess_fraction=fd.ess_fraction,
            newton_iter=fd.newton_iter,
            cg_newton=fd.cg_newton,
            cg_draws=fd.cg_draws,
        )

    def with_terminal(self, terminal: FitDiagnostics | None) -> MGSLDiagnostics:
        return MGSLDiagnostics(
            self.taus,
            self.tau_end,
            self.ess_fraction,
            self.newton_iter,
            self.cg_newton,
            self.cg_draws,
            terminal,
        )

    @staticmethod
    def concatenate(a: MGSLDiagnostics, b: MGSLDiagnostics) -> MGSLDiagnostics:
        """Join a run and its continuation along the step axis (terminal taken from ``b``)."""
        if a.ess_fraction.shape[0] != b.ess_fraction.shape[0]:
            raise ValueError("cannot concatenate diagnostics with different numbers of paths")
        if len(b.taus) and not np.isclose(b.taus[0], a.tau_end, rtol=1e-12, atol=0):
            raise ValueError(
                f"b starts at tau={b.taus[0]}, but a ends at tau={a.tau_end}: not a continuation"
            )

        def cat(x, y):
            return np.concatenate([np.asarray(x), np.asarray(y)], axis=-1)

        return MGSLDiagnostics(
            taus=np.concatenate([a.taus, b.taus]),
            tau_end=b.tau_end,
            ess_fraction=cat(a.ess_fraction, b.ess_fraction),
            newton_iter=cat(a.newton_iter, b.newton_iter),
            cg_newton=CGStats(*(cat(x, y) for x, y in zip(a.cg_newton, b.cg_newton))),
            cg_draws=CGStats(*(cat(x, y) for x, y in zip(a.cg_draws, b.cg_draws))),
            terminal=b.terminal,
        )
