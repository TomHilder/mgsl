<div align="center">

[![Tests](https://img.shields.io/github/actions/workflow/status/TomHilder/mgsl/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/TomHilder/mgsl/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/mgsl?style=flat-square)](https://pypi.org/project/mgsl/)
[![Python](https://img.shields.io/pypi/pyversions/mgsl?style=flat-square)](https://pypi.org/project/mgsl/)
[![Docs](https://img.shields.io/github/actions/workflow/status/TomHilder/mgsl/docs.yml?branch=main&label=docs&style=flat-square)](https://tomhilder.github.io/mgsl/)
[![License](https://img.shields.io/github/license/TomHilder/mgsl?style=flat-square)](https://github.com/TomHilder/mgsl/blob/main/LICENSE)

</div>

# mgsl

`mgsl` is an implementation of the Metric Gaussian Stochastic Localisation (MGSL) algorithm,
written in `jax`. Broadly, MGSL can be understood as a variational inference (VI) method that produces
independent posterior samples. The approximation to the posterior includes all correlations, and
does _not_ assume any particular variational family. As such, MGSL is particularly well suited
to massive-scale inference problems where the posterior is non-Gaussian, but still unimodal.

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

The algorithm operates by running localisation "paths". Each of these paths is independent, and produces
a single posterior sample.

In a bit more detail, each path follows the observation process
$`\mathbf{Y}_\tau = \tau \mathbf{X} + \mathbf{W}_\tau`$ where $`\mathbf{X} \sim p(\mathbf{x} \mid \mathbf{d})`$
is a currently unknown posterior sample, $`\mathbf{W}_\tau`$ is a standard Brownian motion process.
The observation time $`\tau`$ grows until $`\mathbf{Y}_\tau / \tau`$ converges to $`\mathbf{X}`$.
Given $`\mathbf{Y}_\tau = \mathbf{y}`$, the unknown follows the *localising distribution*
$`\pi_\tau(\mathbf{x} \mid \mathbf{y}) \propto p(\mathbf{x} \mid \mathbf{d})\, \mathcal N(\mathbf{y} \mid \tau \mathbf{x}, \tau \mathbf{I})`$.
At every step `mgsl` approximates $`\pi_\tau`$ as a Gaussian, with parameters determined by a
matrix-free MGVI fit, and then increments $`\mathbf{y}`$ as if $`\mathbf{X}`$ were drawn from that
Gaussian.

The model must be setup in a particular way to use `mgsl`. Namely, the parameters must be a single
vector with independent standard normal priors on all entries. Typically this is done by using transformations
to obtain the priors of interest. The likelihood must also provide its Fisher metric in terms of matrix-vector
products. `mgsl` can work with any `jax` model, and also works out-of-the-box with [`nifty.re`](https://github.com/nifty-ppl/nifty) models through a provided adaptor.

## Installation

```bash
pip install mgsl
# or
uv add mgsl
```

or the latest version from GitHub:

```bash
pip install "git+https://github.com/TomHilder/mgsl.git"
# or
uv add "mgsl @ git+https://github.com/TomHilder/mgsl.git"
```

(or of course by ssh instead).

`jax` is a runtime dependency. `nifty.re` is required if you wish to use the adaptor.

## Quick start

A simple linear model with Gaussian noise:

```python
from dataclasses import replace

import jax
import jax.numpy as jnp

import mgsl

jax.config.update("jax_enable_x64", True)  # recommended: tighter solver tolerances

n_latent, n_data, sigma = 3, 20, 0.1
A = jax.random.normal(jax.random.key(0), (n_data, n_latent))
data = A @ jnp.ones(n_latent) + sigma * jax.random.normal(jax.random.key(1), (n_data,))


def forward(x):  # latent x ~ N(0, I) -> prediction of the data
    return A @ x


lik = mgsl.gaussian(forward, data, noise_std=sigma)

options = mgsl.MGVIOptions(
    n_samples_mgvi=1,  # residual draws per local mgvi fit (doubled in practice via mirror samples)
    n_iter_mgvi=1,  # mgvi rounds per localisation step: (draw residuals, Newton) per step
    newton=mgsl.NewtonOptions(maxiter=5),
)

# Initialise from a longer MGVI fit
fit = mgsl.mgvi_fit(jax.random.key(3), lik, jnp.zeros(n_latent), replace(options, n_iter_mgvi=10))

schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=1e5, n_tau=32)
result = mgsl.sample(jax.random.key(4), lik, fit.mean, schedule, options)

result.samples  # (256, 3) NumPy array: posterior samples
result.diagnostics.ess_fraction  # (256, 31): local ESS fraction per path and step
result.failures.totals()  # solver failure counts (always recorded)
```

There are no defaults for the schedule or for the number of samples and rounds of the local
fits: they depend on the problem. This is an active research project and we currently do not
have a confident enough understanding of the algorithm to make many recommendations here.
The [user guide](https://tomhilder.github.io/mgsl/guide/#choosing-the-settings) goes into some
detail in terms of diagnostics however, and also covers `nifty.re` models, resuming runs, batches
and devices, and power-user options. Example notebooks are in [`examples/`](https://github.com/TomHilder/mgsl/tree/main/examples).

## Citation

MGSL paper coming soon.

MGSL builds on previous work in:

- **Stochastic localisation:** R. Eldan (2013), *Thin shell implies spectral gap up to polylog
  via a stochastic localization scheme*, Geometric and Functional Analysis; A. El Alaoui,
  A. Montanari and M. Sellke (2022), *Sampling from the Sherrington-Kirkpatrick Gibbs measure
  via algorithmic stochastic localization*, FOCS; A. Montanari (2023), *Sampling, diffusions,
  and stochastic localization*.
- **Stochastic localization via iterative posterior sampling (SLIPS):** L. Grenioux,
  M. Noble, M. Gabrié and A. O. Durmus (2024), ICML.
- **Implicit Gaussian variational inference:** J. Knollmüller and T. A. Enßlin (2019),
  *Metric Gaussian variational inference* (MGVI); P. Frank, R. Leike and T. A. Enßlin (2021),
  *Geometric variational inference*, Entropy (geoVI); G. Edenhofer et al. (2024),
  *Re-envisioning numerical information field theory (NIFTy.re): a library for Gaussian
  processes and variational inference*, Journal of Open Source Software.

## Licence

GNU General Public License v3.0 or later; see [LICENSE](https://github.com/TomHilder/mgsl/blob/main/LICENSE).

## Development

```sh
uv sync
uv run pytest            # add -m slow for slow tests
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict
```
