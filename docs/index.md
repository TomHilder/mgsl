# mgsl

**mgsl** draws approximate posterior samples with Metric Gaussian Stochastic Localisation
(MGSL), in JAX. It returns `n_paths` independent samples, one per localisation path.

Each path follows an observation \(y_\tau = \tau X + B_\tau\) of the unknown \(X\), where
\(B_\tau\) is a standard Brownian motion and the signal-to-noise ratio \(\tau\) grows until
\(y_\tau / \tau\) pins \(X\) down. Given \(y_\tau\), the unknown has the *localised posterior*
\(\pi_\tau(x \mid y) \propto p(x \mid \text{data})\, e^{\langle y, x\rangle - \tau |x|^2/2}\)
(\(|x|\) the Euclidean norm). At every step mgsl approximates it by a Gaussian found with a
matrix-free MGVI fit, and advances \(y\) as if \(X\) were drawn from that Gaussian. For a linear
model with Gaussian noise this is exact; otherwise the samples carry the error of the local
Gaussian fits, which shrinks as \(\tau\) grows.

The latent must have a standard-normal prior (write the model in whitened coordinates; see
the [user guide](guide.md#describing-the-problem)), and the likelihood must provide its Fisher
information as matrix-vector products. mgsl works with any JAX model, and with NIFTy.re
models through an optional adaptor.

## Installation

```bash
uv add "mgsl @ git+ssh://git@github.com/TomHilder/mgsl.git"
```

Runtime dependency: JAX. NIFTy.re is only needed for `mgsl.nifty`.

## Quick start

A linear model with Gaussian noise, so the exact posterior is known:

```python
from dataclasses import replace

import jax
import jax.numpy as jnp

import mgsl

jax.config.update("jax_enable_x64", True)  # recommended: tighter solver tolerances

n_latent, n_data, sigma = 3, 20, 0.1
A = jax.random.normal(jax.random.key(0), (n_data, n_latent))
data = A @ jnp.ones(n_latent) + sigma * jax.random.normal(jax.random.key(1), (n_data,))


def forward(x):  # latent x ~ N(0, Id) -> prediction of the data
    return A @ x


lik = mgsl.gaussian(forward, data, noise_std=sigma)
mgsl.check_likelihood(lik, jnp.zeros(n_latent), jax.random.key(2))  # raises if inconsistent

options = mgsl.MGVIOptions(
    n_samples_mgvi=4,  # residual draws per local fit (each used with its mirror image)
    n_iter_mgvi=1,  # rounds of (draw residuals, Newton) per localisation step
    newton=mgsl.NewtonOptions(maxiter=5),
)

# a centre to start the paths from: an MGVI fit of the posterior, with more rounds
fit = mgsl.mgvi_fit(jax.random.key(3), lik, jnp.zeros(n_latent), replace(options, n_iter_mgvi=10))

schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=1e5, n_tau=32)
result = mgsl.sample(jax.random.key(4), lik, fit.mean, schedule, options)

result.samples  # (256, 3) NumPy array: posterior samples
result.diagnostics.ess_fraction  # (256, 31): local ESS fraction per path and step
result.failures.totals()  # solver failure counts (always recorded)
```

There are no defaults for the schedule or for the number of samples and rounds of the local
fits: they depend on the problem, so every run states them. The [user
guide](guide.md#choosing-the-settings) explains how to check them with the diagnostics, and
covers NIFTy.re models, resuming runs, batches and devices, and the power-user options.
