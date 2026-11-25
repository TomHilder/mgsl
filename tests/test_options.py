import jax.numpy as jnp
import numpy as np
import pytest

import mgsl


def newton():
    return mgsl.NewtonOptions(maxiter=2)


def test_schedule_requires_all_fields():
    with pytest.raises(TypeError):
        mgsl.Schedule(n_paths=4, tau_start=1.0, tau_final=10.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kw",
    [
        {"n_paths": 0, "tau_start": 1.0, "tau_final": 10.0, "n_tau": 4},
        {"n_paths": 4, "tau_start": 0.0, "tau_final": 10.0, "n_tau": 4},
        {"n_paths": 4, "tau_start": -1.0, "tau_final": 10.0, "n_tau": 4},
        {"n_paths": 4, "tau_start": 10.0, "tau_final": 10.0, "n_tau": 4},
        {"n_paths": 4, "tau_start": 1.0, "tau_final": 10.0, "n_tau": 1},
        {"n_paths": 4.5, "tau_start": 1.0, "tau_final": 10.0, "n_tau": 4},
        {"n_paths": 4, "tau_start": 1.0, "tau_final": np.inf, "n_tau": 4},
    ],
)
def test_schedule_rejects(kw):
    with pytest.raises((ValueError, TypeError)):
        mgsl.Schedule(**kw)


def test_schedule_coerces_numpy_and_jax_scalars():
    s = mgsl.Schedule(
        n_paths=np.int64(8), tau_start=jnp.float32(1.0), tau_final=np.float64(1e4), n_tau=5
    )
    assert type(s.n_paths) is int and type(s.tau_final) is float
    hash(s)
    with pytest.raises(AttributeError):
        s.n_paths = 3  # type: ignore[misc]


def test_grid_geometric_with_exact_endpoints():
    tau0, tau1 = 0.1 * 3, 2e4 / 3
    taus = mgsl.grid(tau0, tau1, 9)
    assert taus[0] == tau0 and taus[-1] == tau1
    r = taus[1:] / taus[:-1]
    np.testing.assert_allclose(r, r[0], rtol=1e-12)
    np.testing.assert_array_equal(mgsl.Schedule(4, tau0, tau1, 9).taus, taus)


def test_mgvi_options_required_counts():
    with pytest.raises(TypeError):
        mgsl.MGVIOptions(n_samples_mgvi=2, newton=newton())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        mgsl.NewtonOptions()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        mgsl.MGVIOptions(n_samples_mgvi=0, n_iter_mgvi=1, newton=newton())


def test_cg_defaults_follow_nifty():
    cg = mgsl.CGOptions()
    assert (cg.tol, cg.atol) == (1e-5, 0.0)
    assert cg.resolve(1000) == (6, 200)  # min(200, 20 d)
    assert cg.resolve(3) == (6, 60)  # 20 d for small d
    assert mgsl.CGOptions(maxiter=4).resolve(1000) == (4, 4)
    assert mgsl.CGOptions(miniter=0).resolve(1000) == (0, 200)
