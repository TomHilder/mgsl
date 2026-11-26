# User guide

!!! warning "Under heavy development"
    MGSL is an in-progress research project, and this code changes with it. Expect bugs,
    quirks, breaking changes and mistakes. Many properties of the method are still unknown:
    there is no established way to choose the tuning parameters (the number of observation-time
    steps, and the number of MGVI samples and iterations per step), and essentially nothing is
    known yet about its error properties.

    Please report bugs and problems through
    [GitHub Issues](https://github.com/TomHilder/mgsl/issues), and get in touch before
    publishing comparisons with MGSL.

## Why MGSL

- **No global variational family.** MGVI approximates the posterior by a Gaussian, and geoVI
  by a Gaussian after a local transformation. MGSL assumes no global family: it builds a global
  approximation out of many local Gaussian fits, one per step along each localisation path. It
  is designed for posteriors that are non-Gaussian beyond what a local transformation can
  capture.
- **Scales like MGVI.** Every fit is implicit and matrix-free: it never builds a dense
  covariance matrix, only Jacobian-vector products and conjugate-gradient solves with the Fisher
  metric, so it keeps all correlations between parameters (full rank) at a cost linear in the
  number of parameters.
- **Embarrassingly parallel.** Each sample comes from its own path, independent of all others.
- **Large steps.** Each step draws the unknown from the local Gaussian fit, accounting for its
  uncertainty over the step, so the update is exact for the Gaussian part and steps need not
  be small.

Known limitations: the samples are approximate, and on multimodal posteriors MGSL fails
silently: paths find the modes but do not weight them correctly.


Notation: \(\mathbf{x}\) is the latent (dimension \(n\)), \(\mathbf{I}\) the
identity, \(\mathbf{J}\) the Jacobian of the forward model, \(\mathbf{F}\) the Fisher metric of the
noise model in data space, and \(\tau\) the observation time of the localisation process
\(\mathbf{Y}_\tau = \tau \mathbf{X} + \mathbf{W}_\tau\). The *localising distribution*
\(\pi_\tau(\mathbf{x} \mid \mathbf{y})\) is the posterior of \(\mathbf{x}\) given the observation \(\mathbf{y}\) at time \(\tau\).

## Describing the problem

### Whitened latent

mgsl assumes the prior on the latent is \(\mathcal N(\mathbf{0}, \mathbf{I})\). A model with other priors
is written in terms of standard-normal variables through a transformation
\(\boldsymbol\theta = \mathbf{f}(\mathbf{x})\) to the physical parameters: a parameter
\(\boldsymbol\theta \sim \mathcal N(\mathbf{m}, \mathbf{L}\mathbf{L}^\top)\) becomes
\(\boldsymbol\theta = \mathbf{m} + \mathbf{L}\mathbf{x}\), a log-normal amplitude becomes \(\exp(m + s\,x)\), a
Gaussian process becomes its whitened Fourier or Karhunen-Loève coefficients, and so on. The
forward model starts from \(\mathbf{x}\):

```python
def forward(x):
    theta = m + L @ x  # prior N(m, L L^T) on theta
    return model(theta)  # prediction of the data
```

### The likelihood

A [`Likelihood`](api/likelihood.md) has

- `log_likelihood(x)`: \(\log p(\mathbf{d} \mid \mathbf{x})\), up to a constant;
- `fisher(x, v)`: \(\mathbf{J}^\dagger \mathbf{F} \mathbf{J}\,\mathbf{v}\), the Fisher metric pulled back to the latent and applied
  to a tangent \(\mathbf{v}\) (e.g. \(\mathbf{F} = \mathrm{diag}(1/\sigma^2)\) for Gaussian
  noise); no prior term;
- `fisher_sqrt(x, z)`: \(\mathbf{J}^\dagger \mathbf{F}^{1/2} \mathbf{z}\), which has covariance
  \(\mathbf{J}^\dagger \mathbf{F} \mathbf{J}\) for \(\mathbf{z} \sim \mathcal N(\mathbf{0}, \mathbf{I})\);
- `data_shape`: a pytree of `jax.ShapeDtypeStruct` with the structure, shapes and dtypes of
  \(\mathbf{z}\) (usually those of the data).

For Gaussian noise, `mgsl.gaussian(forward, data, noise_std=..., mask=...)` builds all four.
`noise_std` is a scalar or an array of the data's shape. Masked data points (`mask` False) get
zero weight and may hold NaN. For other noise models, write the four entries by hand, e.g.
with `jax.jvp` and `jax.vjp` of the forward model.

When you write the functions by hand, check them with `mgsl.check_likelihood(lik, x, key)`.
Along a few random directions (drawn with `key`) at the latent value `x`, it compares `fisher`
with `fisher_sqrt` (which must agree exactly) and with the Hessian of `log_likelihood` (which
should be close and must have the same sign). It raises `ValueError` naming the problem: a
negative log-likelihood passed as `log_likelihood`, a data-space Fisher matrix instead of
\(\mathbf{J}^\dagger \mathbf{F} \mathbf{J}\), a \(\mathbf{J}^\dagger \mathbf{F} \mathbf{J}\) that
is not positive semi-definite, or wrong output shapes; it warns when the Hessian and
\(\mathbf{J}^\dagger \mathbf{F} \mathbf{J}\) differ by more than a factor 1.5, a sign of a missing
weight or factor. It returns the quadratic forms it compared. Run it at zeros and near the
posterior (e.g. at a fit's centre). Likelihoods from `mgsl.gaussian` and the NIFTy.re adaptor
are consistent by construction.

Arrays (and Python floats) held by a `Likelihood` through `jax.tree_util.Partial`, as
`gaussian` does, are passed to compiled code at run time, so new data of the same shape does
not recompile. Arrays captured by a plain closure (a `lambda` or nested function) are compiled
in as constants. Functions, including bound methods, are compared by identity, so a
`Likelihood` rebuilt after its data changed is recompiled rather than run with stale data.

### NIFTy.re models

```python
import operator

import nifty.re as jft
from jax.tree_util import Partial

import mgsl.nifty

noise_std_inv = Partial(operator.mul, 1 / sigma)  # NIFTy.re takes an operator here
lh = jft.Gaussian(data, noise_std_inv=noise_std_inv).amend(model)  # model: a jft.Model
lik = mgsl.nifty.from_likelihood(lh)
result = mgsl.sample(key, lik, mu, schedule, options)
samples = mgsl.nifty.to_samples(result)  # nifty.re Samples, pos = the sample mean
```

The NIFTy.re likelihood must be amended with the model, so that its primals are the latent with
a standard-normal prior. `optimize_kl`'s `point_estimates` and `constants` have no equivalent:
fix such parameters inside the model. Dict and `jft.Vector` latents keep their type.

## Running

`mgsl.sample` chains three functions, which can also be called directly:

```python
state = mgsl.init(key, lik, mu, n_paths=256, tau_start=1.0, options=options)
state, report = mgsl.advance(lik, state, mgsl.grid(1.0, 2e4, 64), options)
samples, final = mgsl.finalize(lik, state, options)
```

The local fits are MGVI Gaussians \(\mathcal N(\bar{\mathbf{x}}, \boldsymbol{\Lambda}^{-1})\) of the localising
distribution, with centre \(\bar{\mathbf{x}}\) and precision metric
\(\boldsymbol{\Lambda} = (1 + \tau)\,\mathbf{I} + \mathbf{J}^\dagger \mathbf{F} \mathbf{J}\) (evaluated at \(\bar{\mathbf{x}}\)). A *residual* is a draw
\(\boldsymbol{\delta} \sim \mathcal N(\mathbf{0}, \boldsymbol{\Lambda}^{-1})\), made by solving
\(\boldsymbol{\Lambda}\boldsymbol{\delta} = \sqrt{1 + \tau}\,\mathbf{z}_1 + \mathbf{J}^\dagger \mathbf{F}^{1/2} \mathbf{z}_2\) with
conjugate gradients (CG), for standard-normal \(\mathbf{z}_1, \mathbf{z}_2\).

- `init` starts each path from \(\mathbf{y} = \tau_0(\boldsymbol{\mu} + \boldsymbol{\delta}) + \sqrt{\tau_0}\,\mathbf{z}\), with
  \(\boldsymbol{\delta}\) a residual at the centre \(\boldsymbol{\mu}\) (at \(\tau = 0\)) and \(\mathbf{z}\) standard normal.
  Small \(\tau_0\) approaches plain stochastic localisation from \(\mathbf{y} = \mathbf{0}\); larger \(\tau_0\)
  skips early steps but relies more on the centre. Check that the centre is converged, e.g.
  by running `mgvi_fit` with more rounds and comparing.
- `advance` moves every path along a grid of observation times. At each grid point it fits the
  localising distribution (warm-started from the previous fit) and sets
  \(\mathbf{y} \leftarrow \mathbf{y} + \Delta\tau\,(\bar{\mathbf{x}} + \boldsymbol{\delta}) + \sqrt{\Delta\tau}\,\mathbf{z}\), with
  \(\Delta\tau\) the distance to the next grid point. Each step is one compiled call, so
  grids of any length and values reuse the same compiled code, and progress is printed after
  every step (`progress=False` turns it off; the lines also go to the `"mgsl"` logger):

  ```
  mgsl  step 12/63  tau=10^2.08  3.41 s  ESS fraction median 0.97, min 0.81  failures 0
  ```
- `finalize` fits each path at its current \(\tau\) and returns \(\bar{\mathbf{x}} + \boldsymbol{\delta}\). It does not
  change the state. Paths whose log-likelihood became NaN or infinite (counted in
  `failures.nonfinite`) get NaN samples.

The run uses the dtype of the centre `mu`. Float64 (`jax.config.update("jax_enable_x64",
True)`) is recommended; float32 works with looser attainable CG tolerances.

### Resuming and checkpointing

The state holds everything needed to continue: the paths' observations and centres, the
observation time and step count, the run's key and the latent's structure. Random numbers are
derived from the key, the path and the global step, so a run split into several `advance`
calls gives the samples of one uninterrupted run over the same grid points with the same
options: bit for bit with the same `batch_size`, `devices` and `vectorize`, and to rounding
otherwise.

```python
mgsl.save_state("run.npz", result.state)
state = mgsl.load_state("run.npz", like=x0)  # x0: a latent (or ShapeDtypeStructs)
state, report = mgsl.advance(lik, state, mgsl.grid(state.tau, 1e5, 16), options)
samples, final = mgsl.finalize(lik, state, options)
```

`load_state` raises `ValueError` if `like` does not match the saved structure, shapes or
dtypes. For crash safety, advance in chunks and save the state after each one (`save_state`
writes atomically, so a crash during a save leaves the previous file intact):

```python
taus = mgsl.grid(state.tau, 2e4, 64)
for k in range(0, 63, 8):
    state, report = mgsl.advance(lik, state, taus[k : k + 9], options)
    mgsl.save_state("run.npz", state)
```

`MGSLDiagnostics.concatenate(a, b)` joins the chunks' diagnostics.

### Paths, batches and devices

- `vectorize`: within a compiled call, paths run one after another by default, each with as
  many solver iterations as it needs. `vectorize=True` batches them with `jax.vmap` instead,
  which is faster for cheap models and on accelerators, but every solver loop then runs as long
  as its slowest path.
- `batch_size`: paths per compiled call. By default all paths go in one call and stay on the
  device between steps; smaller batches bound device memory (they run one after another, and
  the last is padded, so there is one compilation).
- `devices`: shard each batch over exactly these devices. The batch size (default `n_paths`,
  padded) must be a multiple of `len(devices)`. Nothing is sharded unless asked.

Samples do not depend on these settings (beyond rounding).

## Settings

- [`Schedule`](api/options.md): `n_paths`, `tau_start`, `tau_final`, `n_tau` (a geometric grid
  of `n_tau` points from `tau_start` to `tau_final`).
- [`MGVIOptions`](api/options.md): `n_samples_mgvi` (residuals per fit, each also used
  mirrored, \(\bar{\mathbf{x}} \pm \boldsymbol{\delta}\)) and `n_iter_mgvi` (rounds per fit), both required; Newton
  and CG settings. Names, defaults and stopping rules follow NIFTy.re's `optimize_kl`, except
  the line search: `CGOptions(tol=1e-5, atol=0, miniter, maxiter, refresh=20)` for the residual
  draws and `NewtonOptions(maxiter, xtol=1e-5, absdelta, cg, adaptive_cg=True,
  line_search="armijo")` for the fits.
  - `line_search="armijo"` (default): backtracking until a sufficient decrease (Armijo
    condition) holds, with a steepest-descent fallback for bad Newton directions; it never
    gives up after a fixed number of tries.
  - `line_search="nifty"`: NIFTy.re's line search (accepts any step that does not increase
    the objective; restarts along the gradient after six halvings; gives up after nine). On
    strongly curved posteriors it gives up more often.

### Choosing the settings

mgsl has no recommended values; check a choice with the diagnostics:

- The local ESS fraction is expected to sit near its floor at small \(\tau\), where the
  localising distribution is far from Gaussian. What to look for is the observation time at
  which the paths' ESS fractions all rise to 1 together: `tau_final` should be beyond it, so
  that the terminal fit (`result.diagnostics.terminal.ess_fraction`) has an ESS fraction close
  to 1 for every path. If it does not, raise `tau_final`, e.g. by continuing the state with
  `advance`.
- Failure warnings (capped CG solves, line-search failures) mean the solves are not converged:
  raise the corresponding `maxiter`, or check the likelihood.
- `n_iter_mgvi=1` is often enough inside a run, because each fit starts from the previous
  path's centre; the cold-start `mgvi_fit` needs more rounds.

### Power-user options

- `MGVIOptions(preconditioner=...)`: `preconditioner(x, tau)` returns
  \(\mathbf{v} \mapsto \mathbf{M}^{-1} \mathbf{v}\), with \(\mathbf{M}\) approximating
  \(\boldsymbol{\Lambda} = (1 + \tau)\,\mathbf{I} + \mathbf{J}^\dagger \mathbf{F} \mathbf{J}\). `x` and `v` are 1-D arrays of length \(n\): the latent flattened as by `jax.flatten_util.ravel_pytree`. It is built
  before every group of solves (each round of residual draws and each Newton step), so
  building it should be cheap. The CG statistics report the residual of the unpreconditioned
  system.
- `Likelihood(fisher_operator=...)`, or `gaussian(..., linearize_once=True)`: linearise the
  forward model once per solve instead of at every Fisher application; costs the memory of the
  linearisation per path. The returned functions must be pytrees (e.g.
  `jax.tree_util.Partial`, or built from the function `jax.vjp` returns), so that operators at
  several points can be vmapped.

## Diagnostics

`result.failures` counts, per path, residual and Newton CG solves that stopped at `maxiter`,
CG breakdowns, line-search fallbacks and failures, and fits that started at a non-finite
objective ("frozen" paths). It is always recorded, and a warning is issued whenever a count is
non-zero. See [`Failures`](api/diagnostics.md) for what each count means.

`result.diagnostics` (with `calculate_diagnostics=True`, the default) holds, per path and
step:

- `taus`: the observation time of each step's fit;
- `ess_fraction`: the effective sample size (ESS) of the fit's \(m = 2\,\)`n_samples_mgvi`
  mirrored residuals as importance samples of the localising distribution, rescaled to
  \((\mathrm{ESS} - 1)/(m - 1)\): 1 means equal weights (the local Gaussian matches the
  localising distribution at these points), 0 means a single point carries all the weight.
  Plotted against `taus` per path, it sits near its floor early on; the convergence check is
  the observation time at which the paths all rise to 1 together, which `tau_final` should be
  beyond. NaN where a residual's CG solve did not converge. It only sees the directions the \(m\) residuals
  probe: with few residuals it can read close to 1 while the local Gaussian is still off, so
  judge convergence with a few more residuals than the run itself needs, or on a subset of
  paths;
- `newton_iter`: Newton steps per fit;
- `cg_newton`, `cg_draws`: [`CGStats`](api/diagnostics.md) of the Newton and residual-draw
  solves (mean and maximum iterations, largest relative residual, fraction stopped at
  `maxiter`);
- `terminal`: the same for the final fit that produced the samples.

```python
import matplotlib.pyplot as plt

d = result.diagnostics
plt.semilogx(d.taus, d.ess_fraction.T, color="k", alpha=0.2)
```

Turning diagnostics off saves `2 * n_samples_mgvi + 1` log-likelihood evaluations per fit.

Every CG solve judges convergence on its true residual \(\mathbf{b} - \mathbf{A}\mathbf{x}\),
recomputed every `refresh` iterations (`CGOptions(refresh=20)`, as in NIFTy.re) and whenever the
solve looks converged. CG's cheap running estimate of the residual drifts in floating point, and
in badly conditioned solves it can claim convergence for an iterate that is far off; the
recomputation prevents that. CG also stops, keeping its previous iterate, when its objective
\(\tfrac12 \mathbf{x}^\top \mathbf{A}\mathbf{x} - \mathbf{b}^\top \mathbf{x}\) stops
decreasing, which means it has reached rounding noise. Lowering `refresh` limits drift further but slows convergence when
done every few iterations.
