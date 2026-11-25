import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial
from problems import LinGauss, LinGaussPytree, Nonlinear2D

import mgsl
from mgsl import _tree
from mgsl.likelihood import flatten_likelihood


def dense_fisher(forward, x, sd):
    J = jax.jacfwd(forward)(x)
    return J.T @ np.diag(1 / np.asarray(sd) ** 2) @ J


@pytest.mark.parametrize("problem", ["lin", "nonlin"])
def test_gaussian_log_likelihood_fisher_and_sqrt(problem):
    if problem == "lin":
        p = LinGauss.make(d=5, n=7)
        A = jnp.asarray(p.A)
        forward, data, sd = (lambda x: A @ x), p.data, p.sd
    else:
        p = Nonlinear2D()
        forward, data, sd = p.forward, np.asarray(p.data), np.full(2, p.sd)
    lik = mgsl.gaussian(forward, data, noise_std=sd)
    d = 5 if problem == "lin" else 2
    rng = np.random.default_rng(1)
    x, x2, v = (jnp.asarray(rng.normal(size=d)) for _ in range(3))

    def ref_ll(x):
        r = data - forward(x)
        return -0.5 * jnp.sum(r**2 / sd**2)

    np.testing.assert_allclose(
        lik.log_likelihood(x) - lik.log_likelihood(x2), ref_ll(x) - ref_ll(x2), rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.grad(lik.log_likelihood)(x), jax.grad(ref_ll)(x), rtol=1e-10, atol=1e-12
    )
    F = dense_fisher(forward, x, sd)
    np.testing.assert_allclose(lik.fisher(x, v), F @ v, rtol=1e-10, atol=1e-12)
    # fisher_sqrt: L L^T == F
    n = len(data)
    L = jax.vmap(lambda e: lik.fisher_sqrt(x, e), out_axes=1)(jnp.eye(n))
    np.testing.assert_allclose(L @ L.T, F, rtol=1e-10, atol=1e-12)
    out = mgsl.check_likelihood(lik, x, jax.random.key(0))
    if problem == "lin":
        np.testing.assert_allclose(out["hessian_ratio"], 1.0, rtol=1e-10)


def test_fisher_operator_matches():
    p = Nonlinear2D()
    lik = mgsl.gaussian(p.forward, np.asarray(p.data), noise_std=p.sd, linearize_once=True)
    x, v, e = jnp.array([0.3, -0.7]), jnp.array([1.0, 2.0]), jnp.array([-0.4, 0.9])
    fmv, fsqrt = lik.fisher_operator(x)
    np.testing.assert_allclose(fmv(v), lik.fisher(x, v), rtol=1e-12)
    np.testing.assert_allclose(fsqrt(e), lik.fisher_sqrt(x, e), rtol=1e-12)


def test_pytree_latent_roundtrip():
    p = LinGaussPytree.make()
    lik = p.lik()
    x = p.unflat(np.arange(10.0) / 10)
    struct = _tree.latent_struct(x)
    ravel, unravel, d, dtype = _tree.make_unravel(struct)
    assert d == 10 and dtype == jnp.float64
    back = unravel(ravel(x))
    assert jax.tree_util.tree_structure(back) == jax.tree_util.tree_structure(x)
    for a, b in zip(jax.tree_util.tree_leaves(back), jax.tree_util.tree_leaves(x)):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    fl = flatten_likelihood(lik, unravel, ravel)
    F = dense_fisher(lambda v: jnp.asarray(p.base.A) @ v, ravel(x), p.base.sd)
    np.testing.assert_allclose(fl.fisher(ravel(x), jnp.ones(10)), F @ np.ones(10), rtol=1e-10)
    mgsl.check_likelihood(lik, x, jax.random.key(0))


def test_complex_latent_rejected():
    with pytest.raises(TypeError, match="complex"):
        _tree.latent_struct(jnp.zeros(3, jnp.complex128))


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"noise_std": 0.0}, "positive"),
        ({"noise_std": np.inf}, "positive"),
        ({"noise_std": np.ones((4, 1))}, "shape"),
        ({"noise_std": np.ones(3)}, "shape"),
    ],
)
def test_gaussian_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        mgsl.gaussian(lambda x: x, np.zeros(4), **kw)


def test_gaussian_noise_std_keyword_only_and_finite_data():
    with pytest.raises(TypeError):
        mgsl.gaussian(lambda x: x, np.zeros(4), 1.0)  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        mgsl.gaussian(lambda x: x, np.array([0.0, np.nan]), noise_std=1.0)


def test_gaussian_mask_equals_dropping_points():
    p = LinGauss.make(d=4, n=8)
    mask = np.array([True, False, True, True, False, True, True, True])
    data = p.data.copy()
    data[~mask] = np.nan
    A = jnp.asarray(p.A)
    masked = mgsl.gaussian(lambda x: A @ x, data, noise_std=p.sd, mask=mask)
    Am = jnp.asarray(p.A[mask])
    dropped = mgsl.gaussian(lambda x: Am @ x, p.data[mask], noise_std=p.sd[mask])
    x, v = jnp.arange(4.0), jnp.ones(4)
    np.testing.assert_allclose(masked.log_likelihood(x), dropped.log_likelihood(x), rtol=1e-12)
    np.testing.assert_allclose(masked.fisher(x, v), dropped.fisher(x, v), rtol=1e-12)


def test_check_likelihood_catches_mistakes():
    p = LinGauss.make(d=6, n=6)  # n == d: a data-space Fisher has the right shape
    good = p.lik()
    x = jnp.zeros(6)
    k = jax.random.key(0)
    flipped = mgsl.Likelihood(
        log_likelihood=lambda x: -good.log_likelihood(x),
        fisher=good.fisher,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    with pytest.raises(ValueError, match="opposite sign"):
        mgsl.check_likelihood(flipped, x, k)
    W = jnp.asarray(np.diag(p.sd**-2))
    data_space = mgsl.Likelihood(
        log_likelihood=good.log_likelihood,
        fisher=lambda x, v: W @ v,
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    with pytest.raises(ValueError, match="disagree"):
        mgsl.check_likelihood(data_space, x, k)
    indefinite = mgsl.Likelihood(
        log_likelihood=good.log_likelihood,
        fisher=lambda x, v: v * jnp.array([1.0, -1, 1, -1, 1, -1]),
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    with pytest.raises(ValueError, match="positive semi-definite|disagree"):
        mgsl.check_likelihood(indefinite, x, k)
    wrong_shape = mgsl.Likelihood(
        log_likelihood=good.log_likelihood,
        fisher=lambda x, v: jnp.concatenate([v, v]),
        fisher_sqrt=good.fisher_sqrt,
        data_shape=good.data_shape,
    )
    with pytest.raises(ValueError, match="shape"):
        mgsl.check_likelihood(wrong_shape, x, k)


def test_likelihood_pytree_keeps_arrays_dynamic():
    p = LinGauss.make(d=3, n=4)
    lik = p.lik()
    leaves = jax.tree_util.tree_leaves(lik)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert any(leaf.shape == (4,) for leaf in leaves)  # data and weights are leaves
    back = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(lik), leaves)
    np.testing.assert_allclose(back.log_likelihood(jnp.ones(3)), lik.log_likelihood(jnp.ones(3)))
    # a Partial-wrapped user function: its arrays are leaves too
    data = jnp.ones(2)
    custom = mgsl.Likelihood(
        log_likelihood=Partial(lambda d, x: -jnp.sum((d - x) ** 2), data),
        fisher=lambda x, v: 2 * v,
        fisher_sqrt=lambda x, e: jnp.sqrt(2.0) * e,
        data_shape=jax.ShapeDtypeStruct((2,), jnp.float64),
    )
    assert len(jax.tree_util.tree_leaves(custom)) == 1


def test_complex_data_rejected():
    with pytest.raises(NotImplementedError):
        mgsl.gaussian(lambda x: x, np.zeros(3, complex), noise_std=1.0)
