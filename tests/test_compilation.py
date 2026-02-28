"""Tests for tpu_features.compilation.

Run with: pytest tests/test_compilation.py
"""

import pytest

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")

from tpu_features.compilation import (
    aot_compile,
    jit_with_static,
    scan_loop,
    value_and_grad_jit,
)


# ---------------------------------------------------------------------------
# jit_with_static
# ---------------------------------------------------------------------------


def test_jit_with_static_basic():
    @jit_with_static
    def add(x, y):
        return x + y

    result = add(jnp.array(1.0), jnp.array(2.0))
    assert float(result) == pytest.approx(3.0)


def test_jit_with_static_static_argnums():
    call_count = [0]

    @jit_with_static(static_argnums=(1,))
    def scale(x, factor):
        call_count[0] += 1
        return x * factor

    result1 = scale(jnp.array(2.0), 3)
    result2 = scale(jnp.array(4.0), 3)
    assert float(result1) == pytest.approx(6.0)
    assert float(result2) == pytest.approx(12.0)


def test_jit_with_static_can_be_used_as_plain_decorator():
    @jit_with_static
    def square(x):
        return x ** 2

    assert float(square(jnp.array(5.0))) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# aot_compile
# ---------------------------------------------------------------------------


def test_aot_compile_basic():
    def matmul(a, b):
        return a @ b

    a = jnp.ones((64, 64), dtype=jnp.bfloat16)
    b = jnp.ones((64, 64), dtype=jnp.bfloat16)
    compiled = aot_compile(matmul, a, b)
    result = compiled(a, b)
    assert result.shape == (64, 64)


def test_aot_compile_returns_correct_values():
    def add_one(x):
        return x + 1

    x = jnp.zeros((4,))
    compiled = aot_compile(add_one, x)
    result = compiled(x)
    assert jnp.allclose(result, jnp.ones((4,)))


# ---------------------------------------------------------------------------
# value_and_grad_jit
# ---------------------------------------------------------------------------


def test_value_and_grad_jit_basic():
    loss_and_grad = value_and_grad_jit(lambda x: (x ** 2).sum())
    x = jnp.array([1.0, 2.0, 3.0])
    val, grad = loss_and_grad(x)
    assert float(val) == pytest.approx(14.0)
    assert jnp.allclose(grad, jnp.array([2.0, 4.0, 6.0]))


def test_value_and_grad_jit_argnums():
    loss_and_grad = value_and_grad_jit(lambda a, b: (a * b).sum(), argnums=0)
    a = jnp.ones((3,))
    b = jnp.array([1.0, 2.0, 3.0])
    val, grad = loss_and_grad(a, b)
    assert float(val) == pytest.approx(6.0)
    assert jnp.allclose(grad, b)


# ---------------------------------------------------------------------------
# scan_loop
# ---------------------------------------------------------------------------


def test_scan_loop_cumsum():
    """Use scan to compute a running sum."""

    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    xs = jnp.array([1.0, 2.0, 3.0, 4.0])
    final, ys = scan_loop(step, 0.0, xs)

    assert float(final) == pytest.approx(10.0)
    assert jnp.allclose(ys, jnp.array([1.0, 3.0, 6.0, 10.0]))


def test_scan_loop_output_shape():
    def identity_step(carry, x):
        return carry, x

    xs = jnp.ones((10, 4))
    _, ys = scan_loop(identity_step, jnp.zeros((4,)), xs)
    assert ys.shape == (10, 4)
