"""Tests for tpu_features.optimizations.

These tests use JAX's CPU backend (always available in CI) so they can run
without TPU hardware.  They validate the correctness of padding, unpadding,
and dtype-conversion helpers.
"""

import math

import pytest

try:
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")

from tpu_features.optimizations import (
    cast_bfloat16,
    cast_float32,
    next_multiple,
    pad_to_multiple,
    tpu_matmul,
    unpad,
)


# ---------------------------------------------------------------------------
# next_multiple
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,multiple,expected",
    [
        (128, 128, 128),
        (129, 128, 256),
        (1, 128, 128),
        (0, 128, 0),
        (256, 128, 256),
        (257, 128, 384),
        (64, 64, 64),
        (65, 64, 128),
    ],
)
def test_next_multiple(n, multiple, expected):
    assert next_multiple(n, multiple) == expected


def test_next_multiple_invalid():
    with pytest.raises(ValueError):
        next_multiple(10, 0)

    with pytest.raises(ValueError):
        next_multiple(10, -1)


# ---------------------------------------------------------------------------
# cast_bfloat16
# ---------------------------------------------------------------------------


def test_cast_bfloat16_dtype():
    x = jnp.array([1.0, 2.0])
    assert cast_bfloat16(x).dtype == jnp.bfloat16


def test_cast_bfloat16_already_bfloat16():
    x = jnp.array([1.0], dtype=jnp.bfloat16)
    assert cast_bfloat16(x).dtype == jnp.bfloat16


def test_cast_bfloat16_from_int():
    x = jnp.array([1, 2, 3], dtype=jnp.int32)
    result = cast_bfloat16(x)
    assert result.dtype == jnp.bfloat16


# ---------------------------------------------------------------------------
# cast_float32
# ---------------------------------------------------------------------------


def test_cast_float32_dtype():
    x = jnp.array([1.0], dtype=jnp.bfloat16)
    assert cast_float32(x).dtype == jnp.float32


# ---------------------------------------------------------------------------
# pad_to_multiple
# ---------------------------------------------------------------------------


def test_pad_to_multiple_no_padding_needed():
    x = jnp.ones((128, 256))
    result = pad_to_multiple(x, 128)
    assert result.shape == (128, 256)
    assert result is x  # no copy when no padding needed


def test_pad_to_multiple_pads_rows():
    x = jnp.ones((130, 128))
    result = pad_to_multiple(x, 128)
    assert result.shape == (256, 128)


def test_pad_to_multiple_pads_cols():
    x = jnp.ones((128, 65))
    result = pad_to_multiple(x, 128)
    assert result.shape == (128, 128)


def test_pad_to_multiple_both_axes():
    x = jnp.ones((1, 1))
    result = pad_to_multiple(x, 128)
    assert result.shape == (128, 128)


def test_pad_to_multiple_specific_axis():
    x = jnp.ones((130, 130))
    result = pad_to_multiple(x, 128, axes=(0,))
    assert result.shape == (256, 130)  # only axis 0 padded


def test_pad_to_multiple_zero_padding():
    x = jnp.ones((1, 2))
    padded = pad_to_multiple(x, 128)
    # The padded region should be zero.
    assert float(padded[0, -1]) == 0.0


def test_pad_to_multiple_custom_pad_value():
    x = jnp.ones((1, 1))
    padded = pad_to_multiple(x, 4, pad_value=-1.0)
    assert float(padded[-1, -1]) == -1.0


# ---------------------------------------------------------------------------
# unpad
# ---------------------------------------------------------------------------


def test_unpad_restores_shape():
    x = jnp.ones((130, 64))
    padded = pad_to_multiple(x, 128)
    restored = unpad(padded, (130, 64))
    assert restored.shape == (130, 64)


def test_unpad_values_preserved():
    x = jnp.arange(12).reshape(3, 4).astype(jnp.float32)
    padded = pad_to_multiple(x, 128)
    restored = unpad(padded, (3, 4))
    assert jnp.allclose(restored, x)


# ---------------------------------------------------------------------------
# tpu_matmul
# ---------------------------------------------------------------------------


def test_tpu_matmul_output_shape():
    a = jnp.ones((256, 512))
    b = jnp.ones((512, 256))
    result = tpu_matmul(a, b)
    assert result.shape == (256, 256)


def test_tpu_matmul_non_multiple_shapes():
    """Input shapes that are NOT multiples of 128 must still give the right output shape."""
    a = jnp.ones((100, 200))
    b = jnp.ones((200, 150))
    result = tpu_matmul(a, b)
    assert result.shape == (100, 150)


def test_tpu_matmul_correctness():
    """Verify the numeric result against jnp.matmul with a known input."""
    a = jnp.eye(128, dtype=jnp.float32)
    b = jnp.eye(128, dtype=jnp.float32)
    result = tpu_matmul(a, b)
    # Identity @ Identity = Identity (in bfloat16 there may be tiny rounding).
    assert jnp.allclose(result, jnp.eye(128, dtype=jnp.float32), atol=1e-2)
