# User guide

Notation: \(x\) is the latent (dimension \(n\)), \(\mathrm{Id}\) the identity, \(I(x)\) the
Fisher information of the likelihood at \(x\), and \(\tau\) the localisation time.

## Describing the problem

### Whitened latent

mgsl assumes the prior on the latent is \(\mathcal N(0, \mathrm{Id})\). A model with another
prior is written in terms of standard-normal variables: a parameter \(z \sim \mathcal N(m,
LL^\top)\) becomes \(z = m + Lx\), a log-normal amplitude becomes \(\exp(\mu + s\,x)\), a
Gaussian process becomes its whitened Fourier or Karhunen-Loève coefficients, and so on. The
forward model starts from \(x\):

```python
def forward(x):
    z = m + L @ x  # prior N(m, L L^T) on z
    return model(z)  # prediction of the data
```

### The likelihood

A [`Likelihood`](api/likelihood.md) has

- `log_likelihood(x)`: \(\log p(\text{data} \mid x)\), up to a constant;
- `fisher(x, v)`: the Fisher information \(I(x) = J^\top F J\) applied to a tangent \(v\)
  (\(J\) the Jacobian of the forward model, \(F\) the Fisher metric of the noise model, e.g.
  \(\mathrm{diag}(1/\sigma^2)\) for Gaussian noise); no prior term;
- `fisher_sqrt(x, e)`: \(J^\top F^{1/2} e\), which has covariance \(I(x)\) for
  \(e \sim \mathcal N(0, \mathrm{Id})\);
- `data_shape`: a pytree of `jax.ShapeDtypeStruct` with the structure, shapes and dtypes of
  \(e\) (usually those of the data).

For Gaussian noise, `mgsl.gaussian(forward, data, noise_std=..., mask=...)` builds all four.
`noise_std` is a scalar or an array of the data's shape. Masked data points (`mask` False) get
zero weight and may hold NaN. For other noise models, write the four entries by hand, e.g.
with `jax.jvp` and `jax.vjp` of the forward model.

`mgsl.check_likelihood(lik, x, key)` checks that the functions agree with each other. It raises
`ValueError` naming the problem: a negative log-likelihood passed as `log_likelihood`, a
data-space Fisher matrix instead of \(J^\top F J\), a Fisher information that is not positive
semi-definite, or wrong output shapes. It returns the quadratic forms it compared. Run it at
zeros and near the posterior (e.g. at a fit's centre).

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

The local fits are Gaussians \(\mathcal N(\bar x, \Lambda^{-1})\), with centre \(\bar x\) and
precision \(\Lambda = (1 + \tau)\,\mathrm{Id} + I(\bar x)\) at localisation time \(\tau\). A
*residual* is a draw \(\delta \sim \mathcal N(0, \Lambda^{-1})\), made by solving
\(\Lambda\delta = \zeta\) with conjugate gradients (CG) for a \(\zeta \sim \mathcal N(0, \Lambda)\).

- `init` starts each path from \(y = \tau_0(\mu + \delta) + \sqrt{\tau_0}\,\xi\), with
  \(\delta\) a residual at the centre \(\mu\) (at \(\tau = 0\)) and \(\xi\) standard normal.
  Small \(\tau_0\) approaches plain stochastic localisation from \(y = 0\); larger \(\tau_0\)
  skips early steps but relies more on the centre. Check that the centre is converged, e.g.
  by running `mgvi_fit` with more rounds and comparing.
- `advance` moves every path along a grid of localisation times. At each grid point it fits the
  localised posterior (warm-started from the previous fit) and sets
  \(y \leftarrow y + \Delta\tau\,(\bar x + \delta) + \sqrt{\Delta\tau}\,\xi\), with
  \(\Delta\tau\) the distance to the next grid point. Each step is one compiled call, so
  grids of any length and values reuse the same compiled code, and progress is printed after
  every step (`progress=False` turns it off; the lines also go to the `"mgsl"` logger):

  ```
  mgsl  step 12/63  tau=120  3.41 s  ESS fraction median 0.97, min 0.81  failures 0
  ```
- `finalize` fits each path at its current \(\tau\) and returns \(\bar x + \delta\). It does not
  change the state. Paths whose log-likelihood became NaN or infinite (counted in
  `failures.nonfinite`) get NaN samples.

The run uses the dtype of the centre `mu`. Float64 (`jax.config.update("jax_enable_x64",
True)`) is recommended; float32 works with looser attainable CG tolerances.

### Resuming and checkpointing

The state holds everything needed to continue: the paths' observations and centres, the
localisation time and step count, the run's key and the latent's structure. Random numbers are
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
  mirrored, \(\bar x \pm \delta\)) and `n_iter_mgvi` (rounds per fit), both required; Newton
  and CG settings. Names, defaults and stopping rules follow NIFTy.re's `optimize_kl`:
  `CGOptions(tol=1e-5, atol=0, miniter, maxiter)` for the residual draws and
  `NewtonOptions(maxiter, xtol=1e-5, absdelta, cg, adaptive_cg=True, line_search="nifty")` for
  the fits.
  - `line_search="nifty"`: NIFTy.re's line search (accepts any step that does not increase
    the objective; restarts along the gradient after six halvings; gives up after nine).
  - `line_search="armijo"`: backtracking until a sufficient decrease (Armijo condition) holds,
    with a steepest-descent fallback for bad Newton directions; stricter about progress and
    never gives up after a fixed number of tries.

### Choosing the settings

mgsl has no recommended values; check a choice with the diagnostics:

- The terminal fit (`result.diagnostics.terminal.ess_fraction`) should have an ESS fraction
  close to 1 for every path (with enough residuals to make that meaningful, see
  [Diagnostics](#diagnostics)), which needs `tau_final` well above the largest eigenvalue of the
  posterior precision \(\mathrm{Id} + I(x)\) of the latent. If it is not, raise `tau_final`,
  e.g. by continuing the state with `advance`.
- Where `ess_fraction` is low or jumps between steps, the steps are too coarse there: use more
  grid points (`n_tau`).
- Failure warnings (capped CG solves, line-search failures) mean the solves are not converged:
  raise the corresponding `maxiter`, or check the likelihood.
- `n_iter_mgvi=1` is often enough inside a run, because each fit starts from the previous
  path's centre; the cold-start `mgvi_fit` needs more rounds.

### Power-user options

- `MGVIOptions(preconditioner=...)`: `preconditioner(x, tau)` returns \(v \mapsto M^{-1} v\),
  with \(M\) approximating \((1 + \tau)\,\mathrm{Id} + I(x)\). \(x\) and \(v\) are 1-D arrays
  of length \(n\): the latent flattened as by `jax.flatten_util.ravel_pytree`. It is built
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

- `taus`: the localisation time of each step's fit;
- `ess_fraction`: the effective sample size (ESS) of the fit's \(m = 2\,\)`n_samples_mgvi`
  mirrored residuals as importance samples of the localised posterior, rescaled to
  \((\mathrm{ESS} - 1)/(m - 1)\): 1 means equal weights (the local Gaussian matches the
  localised posterior at these points), 0 means a single point carries all the weight.
  Plotted against `taus`, per path, it is the main convergence diagnostic. NaN where a
  residual's CG solve did not converge. It only sees the directions the \(m\) residuals
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

Turning diagnostics off saves `2 * n_samples_mgvi + 1` log-likelihood evaluations and one
Fisher application per solve, per fit.
