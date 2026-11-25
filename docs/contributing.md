# Contributing

## Development setup

``` bash
git clone git@github.com:TomHilder/mgsl.git
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
uv run pytest
```

## Documentation

``` bash
uv run mkdocs serve
```
