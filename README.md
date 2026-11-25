# mgsl

Metric Gaussian Stochastic Localisation (MGSL) in JAX: posterior sampling by stochastic
localisation, with the drift at each step from a matrix-free MGVI fit of the localising distribution.

Works with any JAX model whose latent has a standard-normal prior and whose likelihood provides
its Fisher metric (pulled back to the latent) as matrix-vector products; NIFTy.re models plug
in through the optional `mgsl.nifty` adaptor.

> **Status: under heavy development.** MGSL is an in-progress research project, and this code
> changes with it. Expect bugs, quirks, breaking changes and mistakes. Many properties of the
> method are still unknown: there is no established way to choose the tuning parameters (the
> number of observation-time steps, and the number of MGVI samples and iterations per step),
> and essentially nothing is known yet about its error properties.

## Why MGSL

- **No variational family.** MGVI approximates the posterior by a Gaussian, and geoVI by a
  Gaussian after a local transformation. MGSL assumes no family: it builds a global
  approximation out of many local Gaussian fits, one per step along each localisation path. It
  is designed for posteriors that are non-Gaussian beyond what a local transformation can
  capture, such as the funnels formed by Gaussian-process hyperparameters when the data are
  informative.
- **Scales like MGVI.** Every fit is implicit and matrix-free: it never builds a dense
  covariance matrix, only Jacobian-vector products and conjugate-gradient solves with the Fisher
  metric, which keeps all correlations between parameters (full rank) at linear cost in the
  number of parameters.
- **Embarrassingly parallel.** Each sample comes from its own path, independent of all others.
- **Large steps.** Each step draws the unknown from the local Gaussian fit, accounting for its
  uncertainty over the step, so the update is exact for the Gaussian part and steps need not
  be small.

Known limitation: MGSL is approximate, and on multimodal posteriors it fails silently (paths
find the modes but do not weight them correctly).

```python
from dataclasses import replace

import mgsl

lik = mgsl.gaussian(forward, data, noise_std=sigma)
options = mgsl.MGVIOptions(n_samples_mgvi=4, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=5))
fit = mgsl.mgvi_fit(key1, lik, x0, replace(options, n_iter_mgvi=10))
schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=2e4, n_tau=64)
result = mgsl.sample(key2, lik, fit.mean, schedule, options)
```

Install with `uv add "mgsl @ git+ssh://git@github.com/TomHilder/mgsl.git"`. A runnable quick
start, the user guide and the API reference are in `docs/` (`uv run mkdocs serve`).

## Development

```sh
uv sync
uv run pytest            # add -m slow for slow tests
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict
```
