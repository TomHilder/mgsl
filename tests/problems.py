"""Test problems with known answers, and statistical helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import mgsl


@dataclass
class LinGauss:
    """``d = A x + n``, prior N(0, I), heteroscedastic Gaussian noise; exact posterior."""

    A: np.ndarray
    sd: np.ndarray
    data: np.ndarray

    @staticmethod
    def make(d=10, n=14, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        A = scale * rng.normal(size=(n, d))
        sd = rng.uniform(0.3, 1.0, n)
        x = rng.normal(size=d)
        return LinGauss(A, sd, A @ x + sd * rng.normal(size=n))

    @property
    def d(self):
        return self.A.shape[1]

    @property
    def precision(self):
        return np.eye(self.d) + self.A.T @ np.diag(self.sd**-2) @ self.A

    @property
    def cov(self):
        return np.linalg.inv(self.precision)

    @property
    def mean(self):
        return self.cov @ self.A.T @ (self.data / self.sd**2)

    def lik(self, **kw):
        A = jnp.asarray(self.A)
        return mgsl.gaussian(lambda x: A @ x, self.data, noise_std=self.sd, **kw)

    def local_mean(self, tau, y):
        """Mean of the localised posterior at (tau, y)."""
        P = self.precision
        return np.linalg.solve(P + tau * np.eye(self.d), P @ self.mean + y)


@dataclass
class LinGaussPytree:
    """LinGauss with the latent as a dict of a (2, 3) and a (4,) array."""

    base: LinGauss

    @staticmethod
    def make(seed=0):
        return LinGaussPytree(LinGauss.make(d=10, n=14, seed=seed))

    @staticmethod
    def flat(x):
        return jnp.concatenate([jnp.ravel(x["a"]), x["b"]], axis=-1)

    @staticmethod
    def flat_batch(x):
        n = x["a"].shape[0]
        return np.concatenate([np.asarray(x["a"]).reshape(n, -1), np.asarray(x["b"])], axis=1)

    def zeros(self):
        return {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4)}

    def lik(self):
        A = jnp.asarray(self.base.A)
        return mgsl.gaussian(lambda x: A @ self.flat(x), self.base.data, noise_std=self.base.sd)

    def unflat(self, v):
        v = jnp.asarray(v)
        return {"a": v[:6].reshape(2, 3), "b": v[6:]}


@dataclass
class Nonlinear2D:
    """``s = (x1, x2 + b x1^2)``, Gaussian noise; exact density on a grid."""

    b: float = 0.8
    sd: float = 0.5
    data: tuple = (0.6, 1.2)

    def forward(self, x):
        return jnp.stack([x[0], x[1] + self.b * x[0] ** 2])

    def lik(self):
        return mgsl.gaussian(self.forward, np.asarray(self.data), noise_std=self.sd)

    def log_post(self, x):
        r = np.asarray(self.data) - np.asarray(self.forward(x))
        return -0.5 * np.sum(r**2) / self.sd**2 - 0.5 * np.sum(np.asarray(x) ** 2)


def tight_options(n_samples=2, n_iter=1, newton_maxiter=5, line_search="armijo", **kw):
    cg = mgsl.CGOptions(tol=1e-12, miniter=0, maxiter=500)
    return mgsl.MGVIOptions(
        n_samples_mgvi=n_samples,
        n_iter_mgvi=n_iter,
        newton=mgsl.NewtonOptions(
            maxiter=newton_maxiter, xtol=0.0, adaptive_cg=False, cg=cg, line_search=line_search
        ),
        draw_cg=cg,
        **kw,
    )


def whiten(x, mean, cov):
    """Samples (n, d) whitened with N(mean, cov)."""
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, (np.asarray(x) - mean).T).T


def assert_standard_normal(z, alpha=1e-6):
    """Aggregate tests that rows of z (n, d) are N(0, I): Mahalanobis mean and |z|^2 ~ chi2_d."""
    n, d = z.shape
    m = z.mean(0)
    chi2_mean = n * (m @ m)
    p_mean = stats.chi2.sf(chi2_mean, d)
    p_norm = stats.kstest(np.sum(z * z, 1), stats.chi2(d).cdf).pvalue
    cov = np.cov(z.T)
    # each entry against 5 sigma of its sampling error: sqrt(2/n) on the diagonal, sqrt(1/n) off
    sd = np.where(np.eye(d, dtype=bool), np.sqrt(2.0 / n), np.sqrt(1.0 / n))
    worst = np.max(np.abs(cov - np.eye(d)) / sd)
    assert p_mean > alpha, f"mean off: chi2={chi2_mean:.1f} (d={d}), p={p_mean:.2e}"
    assert p_norm > alpha, f"|z|^2 not chi2_{d}: KS p={p_norm:.2e}"
    assert worst < 5.0, f"covariance off: worst entry {worst:.2f} sigma from the identity"


def key(i=0):
    return jax.random.key(i)
