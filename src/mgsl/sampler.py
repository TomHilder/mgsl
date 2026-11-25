"""The MGSL sampler: start, advance and finalise localisation paths."""

from __future__ import annotations

import functools
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import _tree
from .diagnostics import Failures, FitDiagnostics, FitStats, MGSLDiagnostics
from .likelihood import Likelihood
from .mgvi import (
    HashableStruct,
    MGVIResult,
    _key,
    _Precond,
    draw_residuals,
    local_fit,
    make_problem,
    split_options,
)
from .options import MGVIOptions, Schedule, _as_float, _as_int
from .state import SLState

__all__ = ["Report", "Result", "advance", "finalize", "init", "sample"]

logger = logging.getLogger("mgsl")

_START, _FIT, _TRANSITION, _TERMINAL = 0, 1, 2, 3


def _path_key(key_data, impl, purpose, path, step):
    key = jax.random.wrap_key_data(key_data, impl=impl)
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(key, purpose), path), step)


def _map(fn, args, vectorize):
    """Apply ``fn`` to each path: batched with vmap, or one path after another."""
    return jax.vmap(fn)(args) if vectorize else jax.lax.map(fn, args)


# --------------------------------------------------------------------------------------------
# Kernels: one call handles one batch of paths for one stage (start, one step, terminal fit).
# Arguments: replicated ones first, then the per-path ones (path indices, then arrays).


def _init_kernel(lik, pre, rt, key_data, tau0, mu, paths, *, impl, st, hstruct, vectorize):
    prob = make_problem(lik, pre, hstruct)
    zero_tau = jnp.zeros((), mu.dtype)

    def one(p):
        ka, kb = jax.random.split(_path_key(key_data, impl, _START, p, 0))
        deltas, _, solves, _ = draw_residuals(prob, ka, mu, zero_tau, 1, st, rt)
        xi = jax.random.normal(kb, mu.shape, mu.dtype)
        return tau0 * (mu + deltas[0]) + jnp.sqrt(tau0) * xi, solves

    return _map(one, paths, vectorize)


def _step_kernel(
    lik, pre, rt, key_data, tau, dtau, step, paths, y, xbar, *, impl, st, hstruct, vectorize
):
    prob = make_problem(lik, pre, hstruct)

    def one(args):
        p, y, xbar = args
        kf = _path_key(key_data, impl, _FIT, p, step)
        kt = _path_key(key_data, impl, _TRANSITION, p, step)
        xbar, deltas, _, stats = local_fit(prob, kf, xbar, tau, y, st, rt)
        xi = jax.random.normal(kt, y.shape, y.dtype)
        # over the step the unknown is drawn from the local Gaussian: X = xbar + delta
        y = y + dtau * (xbar + deltas[0]) + jnp.sqrt(dtau) * xi
        return y, xbar, stats

    return _map(one, (paths, y, xbar), vectorize)


def _terminal_kernel(
    lik, pre, rt, key_data, tau, step, paths, y, xbar, *, impl, st, hstruct, vectorize
):
    prob = make_problem(lik, pre, hstruct)

    def one(args):
        p, y, xbar = args
        k = _path_key(key_data, impl, _TERMINAL, p, step)
        xbar, deltas, _, stats = local_fit(prob, k, xbar, tau, y, st, rt)
        return xbar + deltas[0], stats

    return _map(one, (paths, y, xbar), vectorize)


_N_REPLICATED = {_init_kernel: 6, _step_kernel: 7, _terminal_kernel: 6}
_STATIC = ("impl", "st", "hstruct", "vectorize")
_JITTED = {k: jax.jit(k, static_argnames=_STATIC) for k in _N_REPLICATED}


@functools.lru_cache(maxsize=16)
def _sharded(kernel, devices: tuple, **static):
    mesh = Mesh(np.asarray(devices), ("paths",))
    n_rep = _N_REPLICATED[kernel]
    n_args = {_init_kernel: 7, _step_kernel: 10, _terminal_kernel: 9}[kernel]
    in_specs = tuple(P() if i < n_rep else P("paths") for i in range(n_args))
    fn = jax.shard_map(
        functools.partial(kernel, **static),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P("paths"),
        check_vma=False,
    )
    return jax.jit(fn), NamedSharding(mesh, P("paths"))


class _Batches:
    """Runs a kernel over all paths, in batches of ``batch_size`` (the last one padded).

    With one batch and no devices, per-path arrays stay on the device between calls."""

    def __init__(self, n_paths, batch_size, devices, vectorize):
        if devices is not None and not isinstance(devices, Sequence):
            raise TypeError("devices must be None or a sequence of jax devices")
        self.devices = None if devices is None else tuple(devices)
        if self.devices is not None and len(self.devices) == 0:
            raise ValueError("devices must be None or a non-empty sequence of devices")
        n_dev = 1 if self.devices is None else len(self.devices)
        if batch_size is None:
            bs = -(-n_paths // n_dev) * n_dev  # all paths, padded to a multiple of the devices
        else:
            bs = _as_int("batch_size", batch_size, 1)
            if bs % n_dev:
                raise ValueError(
                    f"batch_size ({bs}) must be a multiple of the number of devices ({n_dev})"
                )
            bs = min(bs, -(-n_paths // n_dev) * n_dev)
        self.n_paths, self.bs, self.vectorize = n_paths, bs, bool(vectorize)
        self.n_batches = -(-n_paths // bs)
        self.resident = self.n_batches == 1 and self.devices is None and bs == n_paths

    def run(self, kernel, replicated, per_path, **static):
        """``per_path``: arrays with leading axis n_paths (host or device). Returns outputs with
        leading axis n_paths: device arrays if ``resident``, else host arrays."""
        static = dict(static, vectorize=self.vectorize)
        if self.resident:
            paths = jnp.arange(self.n_paths, dtype=jnp.int32)
            return _JITTED[kernel](*replicated, paths, *per_path, **static)
        if self.devices is None:
            fn, sharding = functools.partial(_JITTED[kernel], **static), None
        else:
            fn, sharding = _sharded(kernel, self.devices, **static)
        out = None
        for b in range(self.n_batches):
            idx = np.arange(b * self.bs, min((b + 1) * self.bs, self.n_paths))
            padded = np.concatenate([idx, np.full(self.bs - idx.size, idx[-1])]).astype(np.int32)
            args = [padded] + [np.asarray(a)[padded] for a in per_path]
            args = [jax.device_put(a, sharding) if sharding is not None else a for a in args]
            res = jax.device_get(fn(*replicated, *args))
            if out is None:
                out = jax.tree_util.tree_map(
                    lambda a: np.empty((self.n_paths, *a.shape[1:]), a.dtype), res
                )
            jax.tree_util.tree_map(
                lambda o, r, n=idx.size, i=idx: o.__setitem__(i, r[:n]), out, res
            )
        assert out is not None
        return out


# --------------------------------------------------------------------------------------------
# Validation helpers


def _check_lik(lik):
    if not isinstance(lik, Likelihood):
        raise TypeError(
            f"expected an mgsl.Likelihood, got {type(lik).__name__} (for NIFTy.re likelihoods "
            "use mgsl.nifty.from_likelihood)"
        )


def _check_lik_accepts(lik, struct):
    """Trace the log-likelihood on the latent's structure, for a clear error on a mismatch."""
    try:
        out = jax.eval_shape(lik.log_likelihood, struct)
    except Exception as e:
        raise ValueError(
            "the likelihood does not accept a latent of the run's structure "
            f"{jax.tree_util.tree_structure(struct)} (shapes "
            f"{[s.shape for s in jax.tree_util.tree_leaves(struct)]}): {e}"
        ) from e
    if getattr(out, "shape", None) != ():
        raise ValueError(f"log_likelihood must return a scalar, got {out}")


def _check_options(options):
    if not isinstance(options, MGVIOptions):
        raise TypeError(f"expected MGVIOptions, got {type(options).__name__}")


def _check_centre(mu):
    if isinstance(mu, SLState):
        raise TypeError("got an SLState as the centre: to continue a state, use mgsl.advance")
    if isinstance(mu, MGVIResult):
        raise TypeError("got an MGVIResult as the centre: pass its .mean")
    if type(mu).__name__ == "Samples" and type(mu).__module__.startswith("nifty"):
        raise TypeError("got NIFTy.re Samples as the centre: pass samples.pos")
    if isinstance(mu, Result):
        raise TypeError("got an mgsl Result as the centre")
    return _tree.latent_struct(mu)


def _check_state(state):
    if not isinstance(state, SLState):
        raise TypeError(f"expected an SLState, got {type(state).__name__}")


def _precond(options):
    return None if options.preconditioner is None else _Precond(options.preconditioner)


class _Progress:
    """One line per step on stderr (and on the "mgsl" logger)."""

    def __init__(self, enabled, n_steps, n_paths):
        self.enabled, self.n_steps, self.n_paths = enabled, n_steps, n_paths
        self.t = time.perf_counter()

    def line(self, label, tau, stats, failures_total):
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        msg = f"mgsl  {label}  tau=10^{np.log10(tau):.2f}  {dt:.2f} s"
        ess = np.asarray(stats.ess)
        if np.any(np.isfinite(ess)):
            msg += f"  ESS fraction median {np.nanmedian(ess):.2f}, min {np.nanmin(ess):.2f}"
        msg += f"  failures {failures_total}"
        logger.info(msg)
        if self.enabled:
            print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# Public API


class Report(NamedTuple):
    """What a call recorded besides its main output.

    Attributes
    ----------
    diagnostics : MGSLDiagnostics or FitDiagnostics or None
        :class:`MGSLDiagnostics` from :func:`advance`, :class:`FitDiagnostics` (the terminal
        fit) from :func:`finalize`; None if ``calculate_diagnostics=False``.
    failures : Failures
        Always present.
    """

    diagnostics: Any
    failures: Failures


def init(
    key,
    lik: Likelihood,
    mu,
    *,
    n_paths: int,
    tau_start: float,
    options: MGVIOptions,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    vectorize: bool = False,
) -> SLState:
    """Start ``n_paths`` localisation paths at ``tau_start``.

    Each path starts from ``y = tau_start (mu + delta) + sqrt(tau_start) z``, with ``z``
    standard normal and ``delta ~ N(0, Lambda^{-1})`` a draw from the Gaussian approximation at
    ``mu``, whose precision metric is ``Lambda = I + J^T F J`` at ``mu`` (see :func:`advance`). Each
    path's first local fit is warm-started at ``mu``. If ``N(mu, Lambda^{-1})`` were the exact
    posterior, ``y`` would follow the exact localisation process at ``tau_start``. Solver
    failures of these draws are reported by a warning.

    Parameters
    ----------
    key : jax.Array
        Random key of the run (typed or legacy). All later random numbers derive from it.
    lik : Likelihood
    mu : pytree
        Centre the paths start from, e.g. ``mgvi_fit(...).mean``.
    n_paths : int
    tau_start : float
        Localisation time to start at (> 0).
    options : MGVIOptions
        Only ``draw_cg`` (and the preconditioner) are used here.
    batch_size, devices, vectorize
        See :func:`sample`.

    Returns
    -------
    SLState
    """
    state, failures = _init(
        key, lik, mu, n_paths, tau_start, options, batch_size, devices, vectorize
    )
    failures.warn("init")
    return state


def _init(key, lik, mu, n_paths, tau_start, options, batch_size, devices, vectorize):
    _check_lik(lik)
    _check_options(options)
    struct = _check_centre(mu)
    _check_lik_accepts(lik, struct)
    n_paths = _as_int("n_paths", n_paths, 1)
    tau_start = _as_float("tau_start", tau_start)
    if tau_start <= 0:
        raise ValueError(f"tau_start must be positive, got {tau_start}")
    ravel, _, d, dtype = _tree.make_unravel(struct)
    st, rt = split_options(options, d, dtype, diagnostics=False)
    key = _key(key)
    impl = str(jax.random.key_impl(key))
    key_data = jax.random.key_data(key)
    mu_flat = jnp.asarray(ravel(mu))
    batches = _Batches(n_paths, batch_size, devices, vectorize)
    y, solves = batches.run(
        _init_kernel,
        (lik, _precond(options), rt, key_data, jnp.asarray(tau_start, dtype), mu_flat),
        (),
        impl=impl,
        st=st,
        hstruct=HashableStruct(struct),
    )
    zeros = np.zeros(n_paths, np.int32)
    failures = Failures(
        draw_cg_capped=np.asarray(solves.capped),
        newton_cg_capped=zeros,
        cg_breakdowns=np.asarray(solves.breakdown),
        sd_fallbacks=zeros,
        line_search_failures=zeros,
        nonfinite=zeros,
    )
    state = SLState(
        y=np.asarray(y),
        xbar=np.broadcast_to(np.asarray(mu_flat), (n_paths, d)).copy(),
        tau=tau_start,
        step=0,
        key_data=np.asarray(key_data),
        key_impl=impl,
        struct=struct,
    )
    return state, failures


def advance(
    lik: Likelihood,
    state: SLState,
    taus,
    options: MGVIOptions,
    *,
    calculate_diagnostics: bool = True,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    vectorize: bool = False,
    progress: bool = True,
    _warn: bool = True,
) -> tuple[SLState, Report]:
    """Advance every path of ``state`` along the grid ``taus``.

    At each grid point ``tau`` the path's localising distribution is fitted by MGVI, warm-started
    from the previous centre. The fit is a Gaussian ``N(xbar, Lambda^{-1})`` with precision
    ``Lambda = (1 + tau) I + J^T F J`` at ``xbar`` (``J`` the Jacobian of the forward model,
    ``F`` the Fisher metric of the noise model). The observation then moves to the next grid point,
    ``dtau`` later, as if the unknown were a draw from that Gaussian:
    ``y += dtau (xbar + delta) + sqrt(dtau) z``, with ``delta ~ N(0, Lambda^{-1})`` and
    ``z ~ N(0, I)`` fresh for every step and path.

    Each step is one compiled call (for all paths, or per batch), so the grid's length and
    values do not cause recompilation.

    Parameters
    ----------
    lik : Likelihood
    state : SLState
    taus : array
        Increasing grid; ``taus[0]`` must equal ``state.tau`` (relative tolerance 1e-12). Use
        ``mgsl.grid(state.tau, tau_final, n)`` for a geometric continuation.
    options : MGVIOptions
    calculate_diagnostics, batch_size, devices, vectorize, progress
        See :func:`sample`.

    Returns
    -------
    state : SLState
        At ``taus[-1]``.
    report : Report
        With :class:`MGSLDiagnostics` (or None) and :class:`Failures`.
    """
    _check_lik(lik)
    _check_state(state)
    _check_options(options)
    _check_lik_accepts(lik, state.struct)
    taus = np.asarray(taus, np.float64)
    if taus.ndim != 1 or taus.size < 2:
        raise ValueError("taus must be a 1D array of at least 2 points")
    if not np.all(np.isfinite(taus)) or np.any(np.diff(taus) <= 0):
        raise ValueError("taus must be finite and strictly increasing")
    if not np.isclose(taus[0], state.tau, rtol=1e-12, atol=0):
        raise ValueError(
            f"taus[0] ({taus[0]!r}) must equal state.tau ({state.tau!r}); build the grid with "
            "mgsl.grid(state.tau, tau_final, n)"
        )
    dtype = state.y.dtype
    st, rt = split_options(options, state.d, dtype, calculate_diagnostics)
    batches = _Batches(state.n_paths, batch_size, devices, vectorize)
    pre, key_data = _precond(options), jnp.asarray(state.key_data)
    static = {"impl": state.key_impl, "st": st, "hstruct": HashableStruct(state.struct)}
    y, xbar = state.y, state.xbar
    if batches.resident:
        y, xbar = jnp.asarray(y), jnp.asarray(xbar)
    n_steps = taus.size - 1
    bar = _Progress(progress, n_steps, state.n_paths)
    per_step, failures_total = [], 0
    for k in range(n_steps):
        step = state.step + k
        y, xbar, stats = batches.run(
            _step_kernel,
            (
                lik,
                pre,
                rt,
                key_data,
                jnp.asarray(taus[k], dtype),
                jnp.asarray(taus[k + 1] - taus[k], dtype),
                jnp.asarray(step, jnp.int32),
            ),
            (y, xbar),
            **static,
        )
        stats = jax.device_get(stats)
        per_step.append(stats)
        failures_total += sum(Failures.from_stats(_with_step_axis(stats)).totals().values())
        bar.line(f"step {k + 1}/{n_steps}", taus[k], stats, failures_total)
    stats = jax.tree_util.tree_map(lambda *a: np.stack(a, axis=1), *per_step)
    new = SLState(
        y=np.asarray(y),
        xbar=np.asarray(xbar),
        tau=float(taus[-1]),
        step=state.step + n_steps,
        key_data=state.key_data,
        key_impl=state.key_impl,
        struct=state.struct,
    )
    failures = Failures.from_stats(stats)
    if _warn:
        failures.warn("advance")
    diag = (
        MGSLDiagnostics.from_steps(taus[:-1], taus[-1], stats) if calculate_diagnostics else None
    )
    return new, Report(diag, failures)


def _with_step_axis(stats: FitStats) -> FitStats:
    return jax.tree_util.tree_map(lambda a: np.asarray(a)[:, None], stats)


def finalize(
    lik: Likelihood,
    state: SLState,
    options: MGVIOptions,
    *,
    calculate_diagnostics: bool = True,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    vectorize: bool = False,
    progress: bool = True,
    _warn: bool = True,
) -> tuple[Any, Report]:
    """Draw one sample per path from ``state``.

    Fits each path's localising distribution at ``state.tau`` and returns ``xbar + delta``, with
    ``xbar`` the fit's centre and ``delta ~ N(0, Lambda^{-1})`` a draw from it (see
    :func:`advance`). The state is not changed, so it can still be advanced further and
    finalised again. Samples of paths whose fit hit a non-finite log-likelihood are NaN.

    Parameters
    ----------
    lik : Likelihood
    state : SLState
    options : MGVIOptions
    calculate_diagnostics, batch_size, devices, vectorize, progress
        See :func:`sample`.

    Returns
    -------
    samples : pytree
        Structured like the latent, with leading axis ``n_paths``; NumPy arrays on the host.
    report : Report
        With the terminal :class:`FitDiagnostics` (or None) and :class:`Failures`.
    """
    _check_lik(lik)
    _check_state(state)
    _check_options(options)
    _check_lik_accepts(lik, state.struct)
    dtype = state.y.dtype
    st, rt = split_options(options, state.d, dtype, calculate_diagnostics)
    batches = _Batches(state.n_paths, batch_size, devices, vectorize)
    bar = _Progress(progress, 1, state.n_paths)
    flat, stats = batches.run(
        _terminal_kernel,
        (
            lik,
            _precond(options),
            rt,
            jnp.asarray(state.key_data),
            jnp.asarray(state.tau, dtype),
            jnp.asarray(state.step, jnp.int32),
        ),
        (state.y, state.xbar),
        impl=state.key_impl,
        st=st,
        hstruct=HashableStruct(state.struct),
    )
    flat, stats = np.asarray(flat), jax.device_get(stats)
    failures = Failures.from_stats(_with_step_axis(stats))
    bar.line("terminal fit", state.tau, stats, sum(failures.totals().values()))
    flat = _nan_frozen(flat, failures.nonfinite)
    samples = _tree.unravel_batch(flat, state.struct)
    if _warn:
        failures.warn("finalize")
    diag = FitDiagnostics.from_stats(stats) if calculate_diagnostics else None
    return samples, Report(diag, failures)


def _nan_frozen(flat, nonfinite):
    frozen = np.asarray(nonfinite) > 0
    if frozen.any():
        flat = flat.copy()
        flat[frozen] = np.nan
    return flat


@dataclass(frozen=True)
class Result:
    """Output of :func:`sample`.

    Attributes
    ----------
    samples : pytree
        Structured like the latent, with leading axis ``n_paths``; NumPy arrays on the host.
        NaN for paths that hit a non-finite log-likelihood (see ``failures.nonfinite``).
    state : SLState
        The paths' state at ``tau_final``; continue it with :func:`advance`.
    schedule : Schedule
    diagnostics : MGSLDiagnostics or None
        None if ``calculate_diagnostics=False``.
    failures : Failures
        Solver failure counts of the whole run (start, steps and terminal fit); always
        present.
    """

    samples: Any
    state: SLState
    schedule: Schedule
    diagnostics: MGSLDiagnostics | None
    failures: Failures


def sample(
    key,
    lik: Likelihood,
    mu,
    schedule: Schedule,
    options: MGVIOptions,
    *,
    calculate_diagnostics: bool = True,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    vectorize: bool = False,
    progress: bool = True,
) -> Result:
    """Draw ``schedule.n_paths`` posterior samples with MGSL.

    Chains :func:`init` (at ``schedule.tau_start`` from centre ``mu``), :func:`advance` (over
    ``schedule.taus``) and :func:`finalize`. The samples are approximate: each step replaces
    the localising distribution by a local Gaussian fit. For a linear model with Gaussian noise and
    converged solves they are exact.

    The run uses the dtype of ``mu`` (float64 needs ``jax_enable_x64``; float32 works, with
    looser attainable solver tolerances).

    Parameters
    ----------
    key : jax.Array
        Random key (typed or legacy).
    lik : Likelihood
    mu : pytree
        Centre the paths start from, e.g. ``mgvi_fit(...).mean``. Its structure, shapes and
        dtypes define the latent.
    schedule : Schedule
    options : MGVIOptions
    calculate_diagnostics : bool
        Record the local ESS and the CG and Newton statistics. Costs ``2 n_samples_mgvi + 1``
        log-likelihood evaluations and one Fisher application per solve, per fit. Failure
        counts are recorded either way.
    batch_size : int, optional
        Paths per compiled call; batches run one after another and the last one is padded.
        Default: all paths in one call, kept on the device between steps. Smaller batches
        bound device memory.
    devices : sequence of jax devices, optional
        Shard each batch over exactly these devices (``batch_size`` must be a multiple of their
        number; by default all paths, padded to a multiple). Default: no sharding, whatever
        devices exist.
    vectorize : bool
        False (default): within a call, paths run one after another, each with as many solver
        iterations as it needs. True: paths are batched with ``jax.vmap``, which is faster for
        cheap models or on accelerators, but every solver loop then runs as long as its slowest
        path.
    progress : bool
        Print one line per step to stderr (time, local ESS fraction, failure count). The same
        lines go to the ``"mgsl"`` logger at INFO level.

    Returns
    -------
    Result
    """
    if not isinstance(schedule, Schedule):
        raise TypeError(f"expected a Schedule, got {type(schedule).__name__}")
    kw = {"batch_size": batch_size, "devices": devices, "vectorize": vectorize}
    state, ifail = _init(key, lik, mu, schedule.n_paths, schedule.tau_start, options, **kw)
    state, rep = advance(
        lik,
        state,
        schedule.taus,
        options,
        calculate_diagnostics=calculate_diagnostics,
        progress=progress,
        _warn=False,
        **kw,
    )
    samples, frep = finalize(
        lik,
        state,
        options,
        calculate_diagnostics=calculate_diagnostics,
        progress=progress,
        _warn=False,
        **kw,
    )
    failures = ifail + rep.failures + frep.failures
    if np.any(np.asarray(rep.failures.nonfinite) > 0):  # frozen during the steps
        frozen = np.asarray(rep.failures.nonfinite) > 0
        samples = jax.tree_util.tree_map(
            lambda a: np.where(frozen.reshape((-1,) + (1,) * (a.ndim - 1)), np.nan, a), samples
        )
    failures.warn("sample")
    diag = rep.diagnostics.with_terminal(frep.diagnostics) if calculate_diagnostics else None
    return Result(samples, state, schedule, diag, failures)
