# Changelog

All notable changes to **mgsl** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-11-26

### Fixed

- README matches the user guide: current status warning and the "Why MGSL" list.
- Typos in the user guide.

## [0.1.1] - 2026-11-26

First release on PyPI.

### Added

- Releases are uploaded to PyPI (`pip install mgsl`).
- README: badges (tests, PyPI, Python versions, docs, licence).

### Fixed

- README: maths renders on GitHub; links work from PyPI.

## [0.1.0] - 2026-11-26

First public release.

### Changed

- Licence: GNU General Public License v3.0 or later (was MIT).
- Releases no longer run the tests (they run on pull requests and on `main`); the docs are
  deployed with each release.
- NIFTy.re, a development-only dependency, now comes from PyPI.

### Added

- README: installation over HTTPS, documentation link, related work, licence, and how to
  report problems.
- Package metadata: project URLs, classifiers, keywords; `py.typed` marker.
- Documentation published at <https://tomhilder.github.io/mgsl/>; the home page, user guide
  and examples rewritten, and the README follows the documentation's home page.

## [0.0.2] - 2026-11-26

### Changed

- The default line search is now `"armijo"` (`NewtonOptions(line_search=...)`); NIFTy.re's
  line search (`"nifty"`, the previous default) gave up more often on strongly curved
  posteriors.

### Added

- Example notebooks in `examples/` (a curved 2D posterior; a field seen through binary data,
  compared with MGVI) and an `examples` dependency group.
- Docs workflow: builds the docs on pull requests; deploys to GitHub Pages on manual runs.
- README: citation section.

## [0.0.1] - 2026-11-26

First release. MGSL is an in-progress research project: expect bugs, quirks and breaking
changes, and see the user guide for what is not yet known about the method.

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
- Each localisation step is one compiled call, run from a Python loop that prints progress
  per step; paths run one after another within a call (or batched with `vectorize=True`), in
  batches of `batch_size`, optionally sharded over given `devices`.
- Power-user options: a CG preconditioner, `Likelihood(fisher_operator=...)`.
- `mgsl.nifty`: `from_likelihood`, `to_samples` (NIFTy.re is optional).
- Conjugate gradients judge convergence on the true residual, recomputed every `refresh`
  iterations (`CGOptions(refresh=20)`) and at apparent convergence, and stop when their
  objective stops decreasing.
- Newton-CG with NIFTy.re's line search (default) or a backtracking Armijo search
  (`NewtonOptions(line_search="armijo")`).
- Version from git tags; tagging `vX.Y.Z` makes a GitHub Release (no PyPI upload yet).
