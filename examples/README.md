# Examples

- `01_simple.ipynb`: a curved 2D posterior from a nonlinear forward model with Gaussian noise,
  compared with the exact posterior on a grid.
- `02_binary_field.ipynb`: a 1D Gaussian-process field observed only through binary (probit)
  labels, with the likelihood written by hand.

To run them:

```sh
uv sync --group examples
uv run --group examples --with jupyterlab jupyter lab examples/   # or open them in your editor
```
