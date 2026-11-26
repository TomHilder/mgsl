# mgsl

Metric Gaussian Stochastic Localisation (MGSL) in JAX: posterior sampling by stochastic
localisation, with the drift at each step from a matrix-free MGVI fit of the localising
distribution.

Works with any JAX model whose latent has a standard-normal prior and whose likelihood provides
its Fisher metric (pulled back to the latent) as matrix-vector products; NIFTy.re models plug
in through the optional `mgsl.nifty` adaptor.

Documentation: <https://tomhilder.github.io/mgsl/>

> **Status: under heavy development.** MGSL is an in-progress research project, and this code
> changes with it. Expect bugs, quirks, breaking changes and mistakes. Many properties of the
> method are still unknown: there is no established way to choose the tuning parameters (the
> number of observation-time steps, and the number of MGVI samples and iterations per step),
> and essentially nothing is known yet about its error properties.
>
> Please report bugs and problems through
> [GitHub Issues](https://github.com/TomHilder/mgsl/issues), and get in touch before publishing
> comparisons with MGSL.

## Why MGSL

- **No global variational family.** MGVI approximates the posterior by a Gaussian, and geoVI
  by a Gaussian after a local transformation. MGSL assumes no global family: it builds a global
  approximation out of many local Gaussian fits, one per step along each localisation path. It
  is designed for posteriors that are non-Gaussian beyond what a local transformation can
  capture.
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

## Installation

```sh
pip install "git+https://github.com/TomHilder/mgsl.git"
# or
uv add "mgsl @ git+https://github.com/TomHilder/mgsl.git"
```

Runtime dependency: JAX. NIFTy.re is only needed for the `mgsl.nifty` adaptor.

## Usage

```python
from dataclasses import replace

import mgsl

lik = mgsl.gaussian(forward, data, noise_std=sigma)
options = mgsl.MGVIOptions(n_samples_mgvi=4, n_iter_mgvi=1, newton=mgsl.NewtonOptions(maxiter=5))
fit = mgsl.mgvi_fit(key1, lik, x0, replace(options, n_iter_mgvi=10))
schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=2e4, n_tau=64)
result = mgsl.sample(key2, lik, fit.mean, schedule, options)
```

A runnable quick start, the user guide and the API reference are in the
[documentation](https://tomhilder.github.io/mgsl/). Example notebooks are in `examples/`.

## Citation

MGSL paper coming soon.

MGSL builds on previous work in:

- **Stochastic localisation:** R. Eldan (2013), *Thin shell implies spectral gap up to polylog
  via a stochastic localization scheme*, Geometric and Functional Analysis; A. El Alaoui,
  A. Montanari and M. Sellke (2022), *Sampling from the Sherrington-Kirkpatrick Gibbs measure
  via algorithmic stochastic localization*, FOCS; A. Montanari (2023), *Sampling, diffusions,
  and stochastic localization*.
- **Posterior sampling with stochastic localisation (SLIPS):** L. Grenioux, M. Noble,
  M. Gabrié and A. O. Durmus (2024), *Stochastic localization via iterative posterior
  sampling*, ICML.
- **Implicit Gaussian variational inference:** J. Knollmüller and T. A. Enßlin (2019),
  *Metric Gaussian variational inference* (MGVI); P. Frank, R. Leike and T. A. Enßlin (2021),
  *Geometric variational inference*, Entropy (geoVI); G. Edenhofer et al. (2024),
  *Re-envisioning numerical information field theory (NIFTy.re): a library for Gaussian
  processes and variational inference*, Journal of Open Source Software.

## Licence

GNU General Public License v3.0 or later; see [LICENSE](LICENSE).

## Development

```sh
uv sync
uv run pytest            # add -m slow for slow tests
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict
```
