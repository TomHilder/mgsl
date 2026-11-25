# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**mgsl**: Metric Gaussian Stochastic Localisation in JAX. A small library for posterior sampling
by stochastic localisation, with the drift at each step from a matrix-free MGVI fit of the
localised posterior. Usable with NIFTy.re models (through an optional adaptor) and with other JAX
models such as spectracles, without depending on NIFTy.re.

Scope: only the method itself. Experimental variants (frozen metric, path weights or
determinants, mode drift, Euler transitions, importance-corrected drift) do not belong here; they
live in the research prototype `../../mgsl-old` (its package also imports as `mgsl`, so never mix
the two environments). Power-user hooks (e.g. a preconditioner) are allowed but off by default.

## Layout and worktrees

- `software/mgsl` stays on `main`; `software/mgsl-dev` is a git worktree for development on a
  feature branch (currently `initial-impl`). Each has its own `.venv`.
- `SPEC.md` in the worktree is the local design spec. Never commit or push it (it is excluded via
  `.git/info/exclude`).

## Package management (uv)

- `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`
- Docs: `uv run mkdocs serve` (Material for MkDocs + mkdocstrings, NumPy docstrings).
- Dev dependencies: `uv add --group dev <package>`.
- Runtime dependencies: JAX only. NIFTy.re must stay optional (imported only by the adaptor).

## Code style

- Line length 99 (ruff). Type hints throughout. NumPy-style docstrings that describe what the code
  does, not its history.
- Keep it small: no feature without a use, no options that only an experiment needs.

## Git

Commit and push as you go on the feature branch, with the usual `Co-Authored-By` line.
