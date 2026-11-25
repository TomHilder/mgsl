"""NIFTy.re adaptor."""

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from problems import LinGauss, tight_options

import mgsl

jft = pytest.importorskip("nifty.re")
import mgsl.nifty

pytestmark = pytest.mark.nifty


def nifty_gaussian(p):
    inv = jnp.asarray(1 / p.sd)
    return jft.Gaussian(jnp.asarray(p.data), noise_std_inv=lambda r: inv * r)


def dict_model(A):
    A = jnp.asarray(A)
    domain = {"a": jft.ShapeWithDtype((2, 3)), "b": jft.ShapeWithDtype((4,))}

    def call(x):
        return A @ jnp.concatenate([jnp.ravel(x["a"]), x["b"]])

    return jft.Model(call, domain=domain)


def test_from_likelihood_matches_gaussian():
    p = LinGauss.make(d=10, n=14)
    model = dict_model(p.A)
    lh = nifty_gaussian(p).amend(model)
    ours = mgsl.nifty.from_likelihood(lh)
    ref = mgsl.gaussian(model, p.data, noise_std=p.sd)
    assert jax.tree_util.tree_leaves(ours.data_shape)[0].shape == (14,)
    rng = np.random.default_rng(0)

    def rand():
        return {"a": jnp.asarray(rng.normal(size=(2, 3))), "b": jnp.asarray(rng.normal(size=4))}

    x, x2, v = rand(), rand(), rand()
    e = jnp.asarray(rng.normal(size=14))
    np.testing.assert_allclose(
        ours.log_likelihood(x) - ours.log_likelihood(x2),
        ref.log_likelihood(x) - ref.log_likelihood(x2),
        rtol=1e-12,
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(ours.fisher(x, v)), jax.tree_util.tree_leaves(ref.fisher(x, v))
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10)
    for a, b in zip(
        jax.tree_util.tree_leaves(ours.fisher_sqrt(x, e)),
        jax.tree_util.tree_leaves(ref.fisher_sqrt(x, e)),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10)
    mgsl.check_likelihood(ours, x, jax.random.key(0))


def test_end_to_end_dict_and_vector_latents():
    p = LinGauss.make(d=10, n=14)
    model = dict_model(p.A)
    lh = nifty_gaussian(p).amend(model)
    lik = mgsl.nifty.from_likelihood(lh)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, model.init(jax.random.key(0)))
    opts = tight_options()
    fit = mgsl.mgvi_fit(jax.random.key(0), lik, zeros, replace(opts, n_iter_mgvi=2))
    sched = mgsl.Schedule(200, 1.0, 1e3, 4)
    res = mgsl.sample(jax.random.key(1), lik, fit.mean, sched, opts)
    assert set(res.samples) == {"a", "b"} and res.samples["a"].shape == (200, 2, 3)
    flat = np.concatenate([np.asarray(res.samples["a"]).reshape(200, -1), res.samples["b"]], 1)
    assert np.abs(flat.mean(0) - p.mean).max() < 5 * np.sqrt(np.diag(p.cov)).max() / np.sqrt(200)
    # jft.Vector latents keep their container type
    vres = mgsl.sample(jax.random.key(1), lik, jft.Vector(fit.mean), sched, opts)
    assert isinstance(vres.samples, jft.Vector)
    np.testing.assert_allclose(vres.samples.tree["b"], res.samples["b"], rtol=1e-12)


def test_to_samples():
    p = LinGauss.make(d=10, n=14)
    lh = nifty_gaussian(p).amend(dict_model(p.A))
    lik = mgsl.nifty.from_likelihood(lh)
    zeros = {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4)}
    res = mgsl.sample(
        jax.random.key(0), lik, zeros, mgsl.Schedule(16, 1.0, 100.0, 3), tight_options()
    )
    s = mgsl.nifty.to_samples(res)
    assert len(s) == 16
    np.testing.assert_allclose(s.pos["b"], np.mean(res.samples["b"], 0), rtol=1e-12)
    np.testing.assert_allclose(s.samples["b"], res.samples["b"], rtol=1e-12)
    np.testing.assert_allclose(s[3]["a"], res.samples["a"][3], rtol=1e-12)
    moved = s.at(jax.tree_util.tree_map(lambda a: a + 1.0, s.pos))
    np.testing.assert_allclose(moved.samples["b"], np.asarray(res.samples["b"]) + 1.0, rtol=1e-12)


def test_rejections_and_warnings():
    from nifty.re.optimize_kl import _StandardHamiltonian

    p = LinGauss.make(d=10, n=14)
    gauss = nifty_gaussian(p)
    lh = gauss.amend(dict_model(p.A))
    with pytest.raises(TypeError, match="likelihood"):
        mgsl.nifty.from_likelihood(_StandardHamiltonian(lh))
    with pytest.warns(UserWarning, match="not amended"):
        mgsl.nifty.from_likelihood(gauss)

    class Bare(jft.Likelihood):
        def energy(self, x):
            return jnp.sum(x**2)

    bare = Bare(domain=jft.ShapeWithDtype((3,)))
    with pytest.raises(ValueError, match="lsm_tangents_shape"):
        mgsl.nifty.from_likelihood(bare)
    with pytest.raises(TypeError):
        mgsl.nifty.from_likelihood(mgsl.gaussian(lambda x: x, np.zeros(3), noise_std=1.0))


def test_mgvi_fit_agrees_with_optimize_kl():
    p = LinGauss.make(d=10, n=14)
    lh = nifty_gaussian(p).amend(dict_model(p.A))
    zeros = {"a": jnp.zeros((2, 3)), "b": jnp.zeros(4)}
    samples, _ = jft.optimize_kl(
        lh,
        jft.Vector(zeros),
        n_total_iterations=2,
        n_samples=2,
        key=jax.random.key(0),
        draw_linear_kwargs={"cg_kwargs": {"tol": 1e-12, "maxiter": 200}},
        kl_kwargs={"minimize_kwargs": {"maxiter": 10, "xtol": 0.0}},
        sample_mode="linear_resample",
    )
    fit = mgsl.mgvi_fit(
        jax.random.key(0), mgsl.nifty.from_likelihood(lh), zeros, tight_options(n_iter=2)
    )
    for k in ("a", "b"):
        np.testing.assert_allclose(fit.mean[k], samples.pos.tree[k], rtol=1e-6, atol=1e-8)


def test_import_without_nifty():
    code = """
        import sys
        sys.modules["nifty"] = None
        sys.modules["nifty.re"] = None
        import mgsl, mgsl.nifty
        try:
            mgsl.nifty.from_likelihood(object())
        except ImportError as e:
            assert "nifty" in str(e)
            print("ok")
    """
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=Path(__file__).parent,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"
