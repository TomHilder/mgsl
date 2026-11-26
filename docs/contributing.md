# Contributing

## Development setup

``` bash
git clone https://github.com/TomHilder/mgsl.git
cd mgsl
uv sync
```

## Code style

- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting, line length 99
- Type hints throughout
- NumPy-style docstrings

``` bash
uv run ruff check .
uv run ruff format .
```

## Tests

``` bash
uv run pytest            # add -m slow to include slow tests
```

## Documentation

``` bash
uv run mkdocs serve
```

## Releasing

The version comes from git tags (hatch-vcs): untagged commits get development versions such as
`0.0.2.dev3+g1234abc`.

1. Move the `[Unreleased]` entries in `docs/changelog.md` under a new `## [X.Y.Z] - date`
   heading and merge to `main`.
2. Tag the merge commit and push the tag:

   ``` bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The `release` workflow then builds the sdist and wheel, checks that the version matches the tag,
and creates a GitHub Release with the changelog section as its notes and the built files
attached; the `docs` workflow deploys the documentation. The tests run on every pull request and
on `main` (the `tests` workflow), not in the release. Nothing is uploaded to PyPI yet.
