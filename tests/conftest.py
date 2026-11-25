import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def key():
    return jax.random.key(0)
