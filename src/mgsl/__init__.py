"""Metric Gaussian Stochastic Localisation (MGSL) in JAX.

Posterior sampling by stochastic localisation, with the drift at each step from a matrix-free
MGVI fit of the localised posterior.
"""

from .diagnostics import CGStats, Failures, FitDiagnostics, MGSLDiagnostics
from .likelihood import Likelihood, check_likelihood, gaussian
from .mgvi import MGVIResult, mgvi_fit
from .options import CGOptions, MGVIOptions, NewtonOptions, Schedule, grid
from .sampler import Report, Result, advance, finalize, init, sample
from .state import SLState, load_state, save_state

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "CGOptions",
    "CGStats",
    "Failures",
    "FitDiagnostics",
    "Likelihood",
    "MGSLDiagnostics",
    "MGVIOptions",
    "MGVIResult",
    "NewtonOptions",
    "Report",
    "Result",
    "SLState",
    "Schedule",
    "advance",
    "check_likelihood",
    "finalize",
    "gaussian",
    "grid",
    "init",
    "load_state",
    "mgvi_fit",
    "sample",
    "save_state",
]
