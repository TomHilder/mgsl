"""The documentation's examples run as written, with default solver settings, cleanly."""

import re
import warnings
from pathlib import Path

import numpy as np

DOCS = Path(__file__).parents[1] / "docs"


def python_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)


def test_quick_start_runs_cleanly():
    code = python_blocks(DOCS / "index.md")[0]
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any solver-failure warning fails the test
        exec(compile(code, "docs/index.md", "exec"), namespace)  # noqa: S102 (our own docs)
    result = namespace["result"]
    assert np.asarray(result.samples).shape == (256, 3)
    assert not result.failures.any(), result.failures.totals()
    # inexact Newton allows loose solves, but never a residual larger than the right-hand side
    d = result.diagnostics
    assert np.nanmax(d.cg_newton.max_rel_residual) <= 1.0
    assert np.nanmax(d.terminal.cg_newton.max_rel_residual) <= 1.0


def test_readme_quick_start_matches_docs():
    readme = python_blocks(DOCS.parent / "README.md")[0]
    docs = python_blocks(DOCS / "index.md")[0]
    assert readme == docs, "README quick start differs from docs/index.md: keep them in sync"
