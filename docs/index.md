# mgsl

**mgsl** draws posterior samples with Metric Gaussian Stochastic Localisation (MGSL), in JAX.

Each sample comes from its own stochastic localisation path: an observation
\(y_\tau = \tau X + B_\tau\) of the unknown \(X\) whose signal-to-noise ratio \(\tau\) grows
until it pins \(X\) down. At every step the path's localised posterior
\(\pi_\tau(x \mid y) \propto p(x \mid d)\, e^{\langle y, x\rangle - \tau |x|^2/2}\) is
approximated by a matrix-free MGVI fit, which supplies the drift of the next step. Paths are
independent.

The latent must have a standard-normal prior (write models in whitened coordinates, as in
NIFTy.re), and the likelihood must provide its Fisher information as matrix-vector products.
mgsl works with NIFTy.re models through an optional adaptor, and with any other JAX model.

## Installation

```bash
uv add mgsl   # or: pip install mgsl
```

Runtime dependency: JAX. NIFTy.re is only needed for `mgsl.nifty`.

## Quick start

```python
import jax
import jax.numpy as jnp
import mgsl

jax.config.update("jax_enable_x64", True)  # recommended

def forward(x):                 # latent x ~ N(0, I) -> model of the data
    return A @ x

lik = mgsl.gaussian(forward, data, noise_std=sigma)
mgsl.check_likelihood(lik, jnp.zeros(d), jax.random.key(0))   # optional sanity check

options = mgsl.MGVIOptions(
    n_samples_mgvi=4,            # residual draws per local fit (mirrored)
    n_iter_mgvi=1,               # rounds per localisation step
    newton=mgsl.NewtonOptions(maxiter=5),
)

# a centre to start the paths from: an MGVI fit of the posterior
fit = mgsl.mgvi_fit(jax.random.key(1), lik, jnp.zeros(d), options, n_rounds=10)

schedule = mgsl.Schedule(n_paths=256, tau_start=1.0, tau_final=2e4, n_tau=64)
result = mgsl.sample(jax.random.key(2), lik, fit.mean, schedule, options)

result.samples        # (256, d): posterior samples
result.diagnostics    # local ESS, CG and Newton statistics per path and step
result.failures       # solver failure counts (always recorded)
```

There are no defaults for the schedule or the number of samples and iterations of the local
fits: they depend on the problem, so every run states them.

See the [user guide](guide.md) for NIFTy.re models, resuming runs, diagnostics, batching and
devices, and the power-user options.
