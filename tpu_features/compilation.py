"""JAX compilation utilities for TPU.

Provides thin wrappers around :func:`jax.jit` and ahead-of-time (AOT)
compilation that encode TPU-friendly defaults (BFloat16, static shape
annotation, lowering diagnostics).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence


# ---------------------------------------------------------------------------
# Lazy JAX import so the module is importable without JAX installed
# (useful in documentation-build or CI environments without TPU hardware).
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this function. "
            "Install it with: pip install 'jax[tpu]' "
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )


# ---------------------------------------------------------------------------
# JIT wrapper
# ---------------------------------------------------------------------------


def jit_with_static(
    fn: Callable = None,
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    donate_argnums: Sequence[int] = (),
) -> Callable:
    """Wrap *fn* with :func:`jax.jit` and TPU-friendly defaults.

    The main additions on top of a bare ``jax.jit`` call are:

    * ``donate_argnums`` defaults to the empty tuple (no donation) but is
      exposed so callers can opt in to buffer donation, which avoids an extra
      HBM allocation by reusing the input buffer for the output.

    Can be used both as a plain decorator and as a parameterised decorator::

        @jit_with_static
        def add(x, y): ...

        @jit_with_static(static_argnums=(1,))
        def scale(x, factor): ...

    Parameters
    ----------
    fn:
        The function to compile (supplied when used as a plain decorator).
    static_argnums:
        Positional argument indices that should be treated as compile-time
        constants (traced once per unique value).
    static_argnames:
        Keyword argument names that should be treated as compile-time
        constants.
    donate_argnums:
        Positional argument indices whose buffers may be donated
        (overwritten) to the output.  Enables zero-copy output on TPU.

    Returns
    -------
    Callable
        A JIT-compiled version of *fn*, or a decorator factory when *fn* is
        not provided.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.compilation import jit_with_static
    >>> @jit_with_static
    ... def add(x, y):
    ...     return x + y
    >>> add(jnp.array(1.0), jnp.array(2.0))
    Array(3., dtype=float32)
    """
    _require_jax()

    def _wrap(f: Callable) -> Callable:
        return jax.jit(
            f,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
        )

    if fn is not None:
        # Used as @jit_with_static (no parentheses).
        return _wrap(fn)

    # Used as @jit_with_static(...) – return a decorator.
    return _wrap


# ---------------------------------------------------------------------------
# AOT compilation
# ---------------------------------------------------------------------------


def aot_compile(fn: Callable, *example_args: Any, **example_kwargs: Any):
    """Ahead-of-time compile *fn* using :func:`jax.jit(...).lower(...).compile`.

    AOT compilation is useful when you want to pay the XLA compilation cost
    up front (e.g. during model initialisation) rather than on the first
    call.  The returned compiled object can be called directly with matching
    inputs.

    Parameters
    ----------
    fn:
        Function to compile.
    *example_args:
        Example positional arguments that determine the input shapes and
        dtypes used for compilation.
    **example_kwargs:
        Example keyword arguments.

    Returns
    -------
    jax.stages.Compiled
        A compiled executable object.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.compilation import aot_compile
    >>> def matmul(a, b):
    ...     return a @ b
    >>> a = jnp.ones((128, 128), dtype=jnp.bfloat16)
    >>> b = jnp.ones((128, 128), dtype=jnp.bfloat16)
    >>> compiled = aot_compile(matmul, a, b)
    >>> result = compiled(a, b)
    >>> result.shape
    (128, 128)
    """
    _require_jax()
    lowered = jax.jit(fn).lower(*example_args, **example_kwargs)
    return lowered.compile()


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def value_and_grad_jit(fn: Callable, argnums: int | Sequence[int] = 0) -> Callable:
    """Return a JIT-compiled function that computes *fn* and its gradient.

    Equivalent to ``jax.jit(jax.value_and_grad(fn, argnums=argnums))`` but
    expressed as a single convenience wrapper.

    Parameters
    ----------
    fn:
        A scalar-valued function (output must be a JAX array with shape ``()``).
    argnums:
        Which positional arguments to differentiate with respect to.

    Returns
    -------
    Callable
        A JIT-compiled ``(value, grad)`` function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.compilation import value_and_grad_jit
    >>> loss_and_grad = value_and_grad_jit(lambda x: (x ** 2).sum())
    >>> val, g = loss_and_grad(jnp.array([1.0, 2.0]))
    >>> val
    Array(5., dtype=float32)
    """
    _require_jax()
    return jax.jit(jax.value_and_grad(fn, argnums=argnums))


# ---------------------------------------------------------------------------
# Scan helper
# ---------------------------------------------------------------------------


def scan_loop(fn: Callable, init: Any, xs: Any) -> tuple:
    """Run a ``lax.scan`` loop, the TPU-preferred sequential iteration pattern.

    Unlike a Python ``for`` loop, :func:`jax.lax.scan` does **not** unroll
    the loop body into the HLO graph.  This keeps compiled binary size small
    and avoids compilation time blow-up for long sequences.

    Parameters
    ----------
    fn:
        Carry-update function with signature ``(carry, x) -> (new_carry, y)``.
    init:
        Initial carry value.
    xs:
        Array of inputs to scan over (leading axis is the loop axis).

    Returns
    -------
    tuple
        ``(final_carry, stacked_ys)``
    """
    _require_jax()
    return jax.lax.scan(fn, init, xs)
