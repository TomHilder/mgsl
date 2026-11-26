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


def words(text):
    """Prose without markdown quote/indent markers or line breaks."""
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def section(text, start, end):
    return text[text.index(start) : text.index(end, text.index(start))]


def test_readme_status_and_why_match_guide():
    readme = (DOCS.parent / "README.md").read_text()
    guide = (DOCS / "guide.md").read_text()
    readme_status = section(readme, "MGSL is an in-progress", "comparisons with MGSL.")
    guide_status = section(guide, "MGSL is an in-progress", "comparisons with MGSL.")
    assert words(readme_status) == words(guide_status), "README status differs from guide"
    readme_why = section(readme, "## Why MGSL", "## Installation")
    guide_why = section(guide, "## Why MGSL", "Notation:")
    assert words(readme_why) == words(guide_why), "README 'Why MGSL' differs from guide"
