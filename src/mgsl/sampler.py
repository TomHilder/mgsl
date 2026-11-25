"""The MGSL sampler: start, advance and finalise localisation paths."""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from . import _tree
from .diagnostics import Failures, FitDiagnostics, MGSLDiagnostics
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


# --------------------------------------------------------------------------------------------
# Per-batch kernels (vmapped over paths)


def _init_kernel(lik, pre, rt, key_data, tau0, mu, paths, *, impl, st, hstruct):
    prob = make_problem(lik, pre, hstruct)
    zero_tau = jnp.zeros((), mu.dtype)

    def one(p):
        ka, kb = jax.random.split(_path_key(key_data, impl, _START, p, 0))
        deltas, _, solves, _ = draw_residuals(prob, ka, mu, zero_tau, 1, st, rt)
        xi = jax.random.normal(kb, mu.shape, mu.dtype)
        return tau0 * (mu + deltas[0]) + jnp.sqrt(tau0) * xi, solves

    return jax.vmap(one)(paths)


def _advance_kernel(lik, pre, rt, key_data, taus, step0, paths, y, xbar, *, impl, st, hstruct):
    prob = make_problem(lik, pre, hstruct)
    dtaus = jnp.diff(taus)
    steps = step0 + jnp.arange(dtaus.shape[0])

    def path(p, y, xbar):
        def body(carry, inp):
            y, xbar = carry
            tau, dtau, k = inp
            kf = _path_key(key_data, impl, _FIT, p, k)
            kt = _path_key(key_data, impl, _TRANSITION, p, k)
            xbar, deltas, _, stats = local_fit(prob, kf, xbar, tau, y, st, rt)
            xi = jax.random.normal(kt, y.shape, y.dtype)
            # predictive transition: X ~ N(xbar, Lambda^-1) over the step, drawn as xbar + delta
            y = y + dtau * (xbar + deltas[0]) + jnp.sqrt(dtau) * xi
            return (y, xbar), stats

        (y, xbar), stats = jax.lax.scan(body, (y, xbar), (taus[:-1], dtaus, steps))
        return y, xbar, stats

    return jax.vmap(path)(paths, y, xbar)


def _finalize_kernel(lik, pre, rt, key_data, tau, step, paths, y, xbar, *, impl, st, hstruct):
    prob = make_problem(lik, pre, hstruct)

    def one(p, y, xbar):
        k = _path_key(key_data, impl, _TERMINAL, p, step)
        xbar, deltas, _, stats = local_fit(prob, k, xbar, tau, y, st, rt)
        return xbar + deltas[0], stats

    return jax.vmap(one)(paths, y, xbar)


# argument positions that are sharded over paths (the rest are replicated)
_SHARDED = {
    _init_kernel: (False, False, False, False, False, False, True),
    _advance_kernel: (False, False, False, False, False, False, True, True, True),
    _finalize_kernel: (False, False, False, False, False, False, True, True, True),
}


@functools.lru_cache(maxsize=64)
def _compiled(kernel, devices: tuple | None, **static):
    fn = functools.partial(kernel, **static)
    if devices is None:
        return jax.jit(fn)
    mesh = Mesh(np.asarray(devices), ("paths",))
    in_specs = tuple(P("paths") if s else P() for s in _SHARDED[kernel])
    return jax.jit(
        jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=P("paths"), check_vma=False)
    )


def _run(kernel, n_paths, batch_size, devices, replicated, sharded, **static):
    """Run ``kernel`` over all paths in batches; ``sharded`` are per-path arrays (leading axis
    n_paths), path indices are prepended to them. Returns host (numpy) outputs."""
    dev = None if devices is None else tuple(devices)
    if dev is not None and len(dev) == 0:
        raise ValueError("devices must be None or a non-empty sequence of devices")
    bs = n_paths if batch_size is None else _as_int("batch_size", batch_size, 1)
    if dev is None:
        bs = min(bs, n_paths)
    elif bs % len(dev):
        raise ValueError(
            f"batch_size ({bs}) must be a multiple of the number of devices ({len(dev)})"
        )
    fn = _compiled(kernel, dev, **static)
    n_batches = -(-n_paths // bs)
    outs = []
    for b in range(n_batches):
        idx = np.arange(b * bs, min((b + 1) * bs, n_paths))
        padded = np.concatenate([idx, np.full(bs - idx.size, idx[-1])]).astype(np.int32)
        args = [jnp.asarray(padded)] + [jnp.asarray(np.asarray(a)[padded]) for a in sharded]
        out = jax.device_get(fn(*replicated, *args))
        n_real = idx.size
        outs.append(jax.tree_util.tree_map(lambda a, n=n_real: a[:n], out))
        if n_batches > 1:
            logger.info("mgsl: batch %d/%d done", b + 1, n_batches)
    return jax.tree_util.tree_map(lambda *a: np.concatenate(a, 0), *outs)


# --------------------------------------------------------------------------------------------
# Validation helpers


def _check_lik(lik):
    if not isinstance(lik, Likelihood):
        raise TypeError(
            f"expected an mgsl.Likelihood, got {type(lik).__name__} (for NIFTy.re likelihoods "
            "use mgsl.nifty.from_likelihood)"
        )


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


def _check_devices(devices):
    if devices is not None and not isinstance(devices, Sequence):
        raise TypeError("devices must be None or a sequence of jax devices")
    return devices


def _precond(options):
    return None if options.preconditioner is None else _Precond(options.preconditioner)


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
) -> SLState:
    """Start ``n_paths`` localisation paths at ``tau_start``.

    Each path starts from ``y = tau_start (mu + delta) + sqrt(tau_start) xi``, with ``delta`` a
    residual of the Gaussian with centre ``mu`` and the posterior metric at ``mu``, and ``xi``
    standard normal; its first local fit is warm-started at ``mu``. If ``mu`` and the metric
    describe the posterior exactly, this is the exact localisation process at ``tau_start``.

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
    batch_size, devices
        See :func:`sample`.

    Returns
    -------
    SLState
    """
    _check_lik(lik)
    _check_options(options)
    struct = _check_centre(mu)
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
    y, solves = _run(
        _init_kernel,
        n_paths,
        batch_size,
        _check_devices(devices),
        (lik, _precond(options), rt, key_data, jnp.asarray(tau_start, dtype), mu_flat),
        (),
        impl=impl,
        st=st,
        hstruct=HashableStruct(struct),
    )
    zeros = np.zeros(n_paths, np.int32)
    Failures(
        draw_cg_capped=np.asarray(solves.capped),
        newton_cg_capped=zeros,
        cg_breakdowns=np.asarray(solves.breakdown),
        sd_fallbacks=zeros,
        line_search_failures=zeros,
        nonfinite=zeros,
    ).warn("init")
    return SLState(
        y=np.asarray(y),
        xbar=np.broadcast_to(np.asarray(mu_flat), (n_paths, d)).copy(),
        tau=tau_start,
        step=0,
        key_data=np.asarray(key_data),
        key_impl=impl,
        struct=struct,
    )


def _check_state(state):
    if not isinstance(state, SLState):
        raise TypeError(f"expected an SLState, got {type(state).__name__}")


def advance(
    lik: Likelihood,
    state: SLState,
    taus,
    options: MGVIOptions,
    *,
    calculate_diagnostics: bool = True,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    _warn: bool = True,
) -> tuple[SLState, Report]:
    """Advance every path of ``state`` along the grid ``taus``.

    At each grid point the path's localised posterior is fitted by MGVI (warm-started from the
    previous centre) and the observation is advanced with the predictive transition
    ``y += dtau (xbar + delta) + sqrt(dtau) xi``.

    Parameters
    ----------
    lik : Likelihood
    state : SLState
    taus : array
        Increasing grid; ``taus[0]`` must equal ``state.tau`` (relative tolerance 1e-12). Use
        ``mgsl.grid(state.tau, tau_final, n)`` for a geometric continuation.
    options : MGVIOptions
    calculate_diagnostics, batch_size, devices
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
    y, xbar, stats = _run(
        _advance_kernel,
        state.n_paths,
        batch_size,
        _check_devices(devices),
        (
            lik,
            _precond(options),
            rt,
            jnp.asarray(state.key_data),
            jnp.asarray(taus, dtype),
            jnp.asarray(state.step, jnp.int32),
        ),
        (state.y, state.xbar),
        impl=state.key_impl,
        st=st,
        hstruct=HashableStruct(state.struct),
    )
    new = SLState(
        y=np.asarray(y),
        xbar=np.asarray(xbar),
        tau=float(taus[-1]),
        step=state.step + taus.size - 1,
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


def finalize(
    lik: Likelihood,
    state: SLState,
    options: MGVIOptions,
    *,
    calculate_diagnostics: bool = True,
    batch_size: int | None = None,
    devices: Sequence | None = None,
    _warn: bool = True,
) -> tuple[Any, Report]:
    """Draw one sample per path from ``state``.

    Fits each path's localised posterior at ``state.tau`` and returns ``xbar + delta`` with
    ``delta`` a fresh residual of that fit. The state is not changed, so it can still be
    advanced further and finalised again.

    Returns
    -------
    samples : pytree
        Structured like the latent, with leading axis ``n_paths``.
    report : Report
        With the terminal :class:`FitDiagnostics` (or None) and :class:`Failures`.
    """
    _check_lik(lik)
    _check_state(state)
    _check_options(options)
    dtype = state.y.dtype
    st, rt = split_options(options, state.d, dtype, calculate_diagnostics)
    flat, stats = _run(
        _finalize_kernel,
        state.n_paths,
        batch_size,
        _check_devices(devices),
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
    _, unravel, _, _ = _tree.make_unravel(state.struct)
    samples = jax.device_get(jax.vmap(unravel)(jnp.asarray(flat)))
    failures = Failures.from_stats(jax.tree_util.tree_map(lambda a: a[:, None], stats))
    if _warn:
        failures.warn("finalize")
    diag = FitDiagnostics.from_stats(stats) if calculate_diagnostics else None
    return samples, Report(diag, failures)


@dataclass(frozen=True)
class Result:
    """Output of :func:`sample`.

    Attributes
    ----------
    samples : pytree
        Structured like the latent, leading axis ``n_paths``.
    state : SLState
        The paths' state at ``tau_final`` (before the terminal fit); continue it with
        :func:`advance`.
    schedule : Schedule
    diagnostics : MGSLDiagnostics or None
        None if ``calculate_diagnostics=False``.
    failures : Failures
        Always present.
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
) -> Result:
    """Draw ``schedule.n_paths`` posterior samples with MGSL.

    Chains :func:`init` (at ``schedule.tau_start`` from centre ``mu``), :func:`advance` (over
    ``schedule.taus``) and :func:`finalize`.

    Parameters
    ----------
    key : jax.Array
        Random key (typed or legacy).
    lik : Likelihood
    mu : pytree
        Centre the paths start from, e.g. ``mgvi_fit(...).mean``.
    schedule : Schedule
    options : MGVIOptions
    calculate_diagnostics : bool
        Record the local ESS and the CG and Newton statistics. Costs ``2 n_samples_mgvi + 1``
        log-likelihood evaluations and one Fisher application per solve, per fit. Failure
        counts are recorded either way.
    batch_size : int, optional
        Paths per compiled batch; the batches run one after another and the last one is
        padded. Default: all paths in one batch. ``batch_size=1`` runs the paths one at a time,
        which is often fastest for expensive models on CPU.
    devices : sequence of jax devices, optional
        Shard each batch over exactly these devices (``batch_size`` must be a multiple of their
        number). Default: no sharding, whatever devices exist.

    Returns
    -------
    Result
    """
    if not isinstance(schedule, Schedule):
        raise TypeError(f"expected a Schedule, got {type(schedule).__name__}")
    state = init(
        key,
        lik,
        mu,
        n_paths=schedule.n_paths,
        tau_start=schedule.tau_start,
        options=options,
        batch_size=batch_size,
        devices=devices,
    )
    state, rep = advance(
        lik,
        state,
        schedule.taus,
        options,
        calculate_diagnostics=calculate_diagnostics,
        batch_size=batch_size,
        devices=devices,
        _warn=False,
    )
    samples, frep = finalize(
        lik,
        state,
        options,
        calculate_diagnostics=calculate_diagnostics,
        batch_size=batch_size,
        devices=devices,
        _warn=False,
    )
    failures = rep.failures + frep.failures
    failures.warn("sample")
    diag = rep.diagnostics.with_terminal(frep.diagnostics) if calculate_diagnostics else None
    return Result(samples, state, schedule, diag, failures)
