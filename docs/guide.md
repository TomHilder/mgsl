# User guide

## Describing the problem

mgsl needs a latent \(x\) with prior \(\mathcal N(0, I)\) (any pytree of real arrays) and a
[`Likelihood`](api/likelihood.md) with

- `log_likelihood(x)`: \(\log p(d \mid x)\), up to a constant;
- `fisher(x, v)`: the Fisher information \(I(x) = J^\top F J\) applied to a tangent \(v\)
  (\(J\) the Jacobian of the forward model, \(F\) the Fisher metric of the noise model; no
  prior term);
- `fisher_sqrt(x, e)`: \(J^\top F^{1/2} e\), so that it has covariance \(I(x)\) for
  \(e \sim \mathcal N(0, I)\);
- `data_shape`: the shape of \(e\).

For Gaussian noise, `mgsl.gaussian(forward, data, noise_std=..., mask=...)` builds all four.
Masked data points get zero weight (they may hold NaN).

`mgsl.check_likelihood(lik, x, key)` checks that the functions agree with each other: it catches
a negative log-likelihood passed as `log_likelihood`, a data-space Fisher matrix instead of
\(J^\top F J\), a Fisher that is not positive semi-definite, and wrong output shapes.

Arrays held by a `Likelihood` through `jax.tree_util.Partial` (as `gaussian` does) are passed
to compiled code at run time, so new data of the same shape does not recompile. Arrays captured
by a plain closure are compiled in as constants.

### NIFTy.re models

```python
import nifty.re as jft
import mgsl.nifty

lh = jft.Gaussian(data, noise_std_inv=lambda r: r / sigma).amend(model)
lik = mgsl.nifty.from_likelihood(lh)
result = mgsl.sample(key, lik, mu, schedule, options)
samples = mgsl.nifty.to_samples(result)   # nifty.re Samples (pos = sample mean)
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

- `init` starts each path from \(y = \tau_0(\mu + \delta) + \sqrt{\tau_0}\,\xi\), with
  \(\delta\) a residual of the Gaussian at the centre \(\mu\). Small \(\tau_0\) approaches
  plain stochastic localisation from \(y = 0\); larger \(\tau_0\) skips the early steps but
  relies more on the centre. Check that the centre (e.g. from `mgvi_fit`) is converged.
- `advance` moves every path along a grid of localisation times.
- `finalize` fits each path at its current \(\tau\) and draws one sample. It does not change the
  state.

### Resuming and checkpointing

The state holds everything needed to continue: the paths' observations and centres, the
localisation time and step count, the run's key and the latent's structure. Random numbers are
derived from the key, the path and the global step, so continuing a state gives exactly the
samples an uninterrupted run would have given.

```python
mgsl.save_state("run.npz", result.state)
state = mgsl.load_state("run.npz", like=x0)          # like=: the latent's structure
state, report = mgsl.advance(lik, state, mgsl.grid(state.tau, 1e5, 16), options)
samples, _ = mgsl.finalize(lik, state, options)
```

For crash safety, advance in chunks and save the state after each one.

### Batches and devices

- `batch_size`: paths per compiled batch; batches run one after another (the last is padded,
  so there is one compilation). `batch_size=1` runs paths one at a time, which is often fastest
  for expensive models on CPU. Default: all paths in one batch.
- `devices`: shard each batch over exactly these devices. Nothing is sharded unless asked.

Samples do not depend on `batch_size` or `devices`.

## Settings

- [`Schedule`](api/options.md): `n_paths`, `tau_start`, `tau_final`, `n_tau` (geometric grid).
- [`MGVIOptions`](api/options.md): `n_samples_mgvi` and `n_iter_mgvi` (required), Newton and
  CG settings. Names, defaults and stopping rules follow NIFTy.re's `optimize_kl`:
  `CGOptions(tol, atol, miniter, maxiter)` for the residual draws;
  `NewtonOptions(maxiter, xtol, absdelta, cg, adaptive_cg, line_search)` for the fits.
  `line_search="nifty"` (default) is NIFTy.re's line search; `"armijo"` is the backtracking
  Armijo search of the original MGSL implementation.

### Power-user options

- `MGVIOptions(preconditioner=...)`: `preconditioner(x, tau)` returns \(v \mapsto M^{-1} v\)
  approximating the inverse of \((1 + \tau) I + I(x)\) (flat vectors). It is built at each fit's
  centre and used in all its CG solves. The CG statistics report the residual of the
  unpreconditioned system.
- `Likelihood(fisher_operator=...)`, or `gaussian(..., linearize_once=True)`: linearise the
  forward model once per solve instead of at every Fisher application; costs the memory of the
  linearisation per path.

## Diagnostics

`result.failures` counts, per path, residual and Newton CG solves that stopped at `maxiter`,
CG breakdowns, line-search fallbacks and failures, and fits that started at a non-finite
objective ("frozen" paths). It is always recorded, and a warning is issued whenever a count is
non-zero.

`result.diagnostics` (with `calculate_diagnostics=True`, the default) holds per path and step:

- `ess_fraction`: the effective sample size of the fit's mirrored residuals as importance samples
  of the localised posterior, divided by their number. 1 means the local Gaussian approximation
  is exact. Plotted against `taus` per path, it is the main convergence diagnostic. NaN where a
  residual's CG solve did not converge.
- `newton_iter`, `cg_newton`, `cg_draws`: iterations, residuals and cap fractions.
- `terminal`: the same for the final fit that produced the samples.

```python
import matplotlib.pyplot as plt

d = result.diagnostics
plt.semilogx(d.taus, d.ess_fraction.T, color="k", alpha=0.2)
```

Turning diagnostics off saves \(2\,\)`n_samples_mgvi`\(+1\) log-likelihood evaluations and one
Fisher application per solve, per fit.
