"""Tests for tpu_features.parallelism.

These tests run on the CPU backend (no TPU required) to verify:

* Device mesh creation degrades gracefully to whatever devices are available.
* replicate / unreplicate round-trip correctly.
* pmap collective wrappers (all_reduce_sum, all_reduce_mean) produce the
  expected values.
"""

import pytest

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")

from tpu_features.parallelism import (
    all_reduce_mean,
    all_reduce_sum,
    create_device_mesh,
    make_named_sharding,
    replicate,
    unreplicate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_device_count() -> bool:
    return jax.device_count() == 1


# ---------------------------------------------------------------------------
# create_device_mesh
# ---------------------------------------------------------------------------


def test_create_device_mesh_shape():
    n = jax.device_count()
    mesh = create_device_mesh((1, n), axis_names=("batch", "model"))
    assert mesh.shape == {"batch": 1, "model": n}


def test_create_device_mesh_default_axis_names():
    n = jax.device_count()
    mesh = create_device_mesh((1, n))
    # Default names are "axis_0", "axis_1", ...
    assert "axis_0" in mesh.shape
    assert "axis_1" in mesh.shape


def test_create_device_mesh_bad_shape_raises():
    n = jax.device_count()
    with pytest.raises((ValueError, Exception)):
        create_device_mesh((n + 1, 1))  # product > available devices


# ---------------------------------------------------------------------------
# make_named_sharding
# ---------------------------------------------------------------------------


def test_make_named_sharding_returns_named_sharding():
    from jax.sharding import NamedSharding, PartitionSpec as P

    n = jax.device_count()
    mesh = create_device_mesh((1, n), axis_names=("batch", "model"))
    sharding = make_named_sharding(mesh, P("batch", "model"))
    assert isinstance(sharding, NamedSharding)


# ---------------------------------------------------------------------------
# replicate / unreplicate
# ---------------------------------------------------------------------------


def test_replicate_adds_device_axis():
    x = jnp.ones((4,))
    rep = replicate(x)
    n = jax.local_device_count()
    assert rep.shape == (n, 4)


def test_unreplicate_removes_device_axis():
    x = jnp.ones((4,))
    rep = replicate(x)
    restored = unreplicate(rep)
    assert restored.shape == (4,)
    assert jnp.allclose(restored, x)


def test_replicate_unreplicate_roundtrip():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 4))
    assert jnp.allclose(unreplicate(replicate(x)), x)


# ---------------------------------------------------------------------------
# all_reduce_sum / all_reduce_mean (via pmap)
# ---------------------------------------------------------------------------


def test_all_reduce_sum():
    n = jax.local_device_count()

    def fn(x):
        return all_reduce_sum(x, axis_name="devices")

    pmapped = jax.pmap(fn, axis_name="devices")

    # Each device contributes 1.0; sum should be n.
    x = jnp.ones((n,))
    result = pmapped(x)
    assert jnp.allclose(result, jnp.full((n,), float(n)))


def test_all_reduce_mean():
    n = jax.local_device_count()

    def fn(x):
        return all_reduce_mean(x, axis_name="devices")

    pmapped = jax.pmap(fn, axis_name="devices")

    x = jnp.ones((n,))
    result = pmapped(x)
    assert jnp.allclose(result, jnp.ones((n,)))
