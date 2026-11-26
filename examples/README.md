# Examples

- `01_simple.ipynb`: a curved 2D posterior from a nonlinear forward model with Gaussian noise,
  compared with the exact posterior on a grid.
- `02_binary_field.ipynb`: a 1D Gaussian-process field observed only through binary (probit)
  labels, with the likelihood written by hand.

To run them, `uv sync` (the `examples` dependency group, with matplotlib and ipykernel, is
installed by default), then open a notebook with the repository's `.venv` as its kernel, or:

```sh
uv run --with jupyterlab jupyter lab examples/
```

You can also just them somewhere that `mgsl` is installed, as well as the other packages imported.