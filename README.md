# mgsl

Metric Gaussian Stochastic Localisation (MGSL) in JAX: posterior sampling by stochastic
localisation, with the drift at each step from a matrix-free MGVI fit of the localised posterior.

Works with any JAX model whose latent has a standard-normal prior and whose likelihood provides
its Fisher information as matrix-vector products; NIFTy.re models plug in through the optional
`mgsl.nifty` adaptor.

```python
import mgsl

lik = mgsl.gaussian(forward, data, noise_std=sigma)
options = mgsl.MGVIOptions(n_samples_mgvi=4, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=5))
fit = mgsl.mgvi_fit(key1, lik, x0, options, n_rounds=10)
schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=2e4, n_tau=64)
result = mgsl.sample(key2, lik, fit.mean, schedule, options)
```

Documentation: `uv run mkdocs serve` (user guide and API reference in `docs/`).

## Development

```sh
uv sync
uv run pytest            # add -m slow for slow tests
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict
```
