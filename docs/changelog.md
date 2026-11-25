# Changelog

All notable changes to **mgsl** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sample`, and the three steps it chains: `init`, `advance`, `finalize`. Paths start at
  `tau_start` from a given centre, advance by drawing the unknown from each step's local
  Gaussian fit, and end with a terminal local fit.
- `mgvi_fit`: MGVI fit of the posterior, e.g. for the starting centre.
- `Likelihood` (a pytree: arrays are run-time inputs), `gaussian` (with `mask=`) and
  `check_likelihood`.
- Settings: `Schedule` (all fields required), `MGVIOptions` (`n_samples_mgvi`, `n_iter_mgvi`
  required), `NewtonOptions`, `CGOptions`, `grid`. Names, defaults and stopping rules follow
  NIFTy.re's `optimize_kl`; `line_search="nifty"` (default) or `"armijo"`.
- Exact resuming: `SLState`, `save_state`, `load_state`; random numbers derive from the run's
  key, the path and the global step, so chunked runs equal unchunked ones.
- Always-on `Failures` counts with warnings; optional `MGSLDiagnostics` (local ESS fraction,
  CG and Newton statistics, terminal fit).
- Batches of paths run in a Python loop (`batch_size`), optionally sharded over given
  `devices`.
- Power-user options: a CG preconditioner, `Likelihood(fisher_operator=...)`.
- `mgsl.nifty`: `from_likelihood`, `to_samples` (NIFTy.re is optional).
