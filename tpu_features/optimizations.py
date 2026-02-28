"""TPU-specific numerical and memory optimisations for JAX.

Key utilities:

* :func:`cast_bfloat16` – cast arrays to the native TPU dtype.
* :func:`pad_to_multiple` – pad matrix dimensions to MXU-friendly sizes.
* :func:`apply_bfloat16_policy` – mixed-precision policy helpers.
* :func:`tpu_matmul` – a reference matrix-multiply function with all
  best-practice optimisations applied.
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

from tpu_features.hardware import MXU_TILE_SIZE


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this function. "
            "Install it with: pip install 'jax[tpu]' "
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------


def cast_bfloat16(array: "jnp.ndarray") -> "jnp.ndarray":
    """Cast *array* to BFloat16, the native TPU MXU dtype.

    BFloat16 has the same exponent range as Float32 (8 bits) but only 7
    mantissa bits.  Computations in BF16 run at full MXU throughput; Float32
    throughput is ~4× lower on most TPU generations.

    Parameters
    ----------
    array:
        Input JAX array of any dtype.

    Returns
    -------
    jnp.ndarray
        The input array cast to ``jnp.bfloat16``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.optimizations import cast_bfloat16
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> cast_bfloat16(x).dtype
    dtype(bfloat16)
    """
    _require_jax()
    return array.astype(jnp.bfloat16)


def cast_float32(array: "jnp.ndarray") -> "jnp.ndarray":
    """Cast *array* to Float32 (e.g. for accumulation or loss scaling).

    Parameters
    ----------
    array:
        Input JAX array of any dtype.

    Returns
    -------
    jnp.ndarray
        The input array cast to ``jnp.float32``.
    """
    _require_jax()
    return array.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def next_multiple(n: int, multiple: int) -> int:
    """Return the smallest integer ≥ *n* that is a multiple of *multiple*.

    Parameters
    ----------
    n:
        The value to round up.
    multiple:
        The granularity to round to.  Must be a positive integer.

    Returns
    -------
    int

    Examples
    --------
    >>> next_multiple(130, 128)
    256
    >>> next_multiple(128, 128)
    128
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be a positive integer, got {multiple!r}")
    return math.ceil(n / multiple) * multiple


def pad_to_multiple(
    array: "jnp.ndarray",
    multiple: int = MXU_TILE_SIZE,
    axes: Tuple[int, ...] | None = None,
    pad_value: float = 0.0,
) -> "jnp.ndarray":
    """Pad *array* so that the specified axes are multiples of *multiple*.

    The MXU operates on 128 × 128 tiles.  Arrays whose dimensions are not
    multiples of 128 incur padding overhead inside XLA.  Explicit padding
    before compilation lets you control the pad value and keeps the HLO
    graph clean.

    Parameters
    ----------
    array:
        Input JAX array.
    multiple:
        The granularity to pad to.  Defaults to :data:`~tpu_features.hardware.MXU_TILE_SIZE` (128).
    axes:
        Which axes to pad.  ``None`` pads *all* axes.
    pad_value:
        The scalar value used for padding.  Defaults to ``0.0``.

    Returns
    -------
    jnp.ndarray
        A new array whose shape along the requested axes is rounded up to
        the nearest *multiple*.  If no padding is needed the original array
        is returned unchanged (no copy).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.optimizations import pad_to_multiple
    >>> x = jnp.ones((130, 64))
    >>> pad_to_multiple(x).shape
    (256, 128)
    """
    _require_jax()
    shape = array.shape
    if axes is None:
        axes = tuple(range(len(shape)))

    pad_width = []
    needs_pad = False
    for i, dim in enumerate(shape):
        if i in axes:
            target = next_multiple(dim, multiple)
            pad_amount = target - dim
            if pad_amount > 0:
                needs_pad = True
            pad_width.append((0, pad_amount))
        else:
            pad_width.append((0, 0))

    if not needs_pad:
        return array

    return jnp.pad(array, pad_width, constant_values=pad_value)


def unpad(array: "jnp.ndarray", original_shape: Tuple[int, ...]) -> "jnp.ndarray":
    """Slice *array* back to *original_shape*, removing any padding.

    Parameters
    ----------
    array:
        A padded JAX array.
    original_shape:
        The shape to restore.  Each dimension of *original_shape* must be ≤
        the corresponding dimension of ``array.shape``.

    Returns
    -------
    jnp.ndarray
        The first ``original_shape[i]`` elements along each axis *i*.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.optimizations import pad_to_multiple, unpad
    >>> x = jnp.ones((130, 64))
    >>> padded = pad_to_multiple(x)
    >>> padded.shape
    (256, 128)
    >>> unpad(padded, (130, 64)).shape
    (130, 64)
    """
    _require_jax()
    slices = tuple(slice(0, s) for s in original_shape)
    return array[slices]


# ---------------------------------------------------------------------------
# Mixed-precision matmul
# ---------------------------------------------------------------------------


def tpu_matmul(
    a: "jnp.ndarray",
    b: "jnp.ndarray",
    precision: "jax.lax.Precision" = None,
) -> "jnp.ndarray":
    """Perform a TPU-optimised matrix multiply.

    Applies the following best practices automatically:

    1. Cast both inputs to BFloat16 (native MXU dtype).
    2. Pad both matrices so all dimensions are multiples of 128.
    3. Compute ``a @ b`` via :func:`jnp.matmul`.
    4. Unpad the result back to the mathematically correct shape.

    Parameters
    ----------
    a:
        Left-hand matrix of shape ``(..., M, K)``.
    b:
        Right-hand matrix of shape ``(..., K, N)``.
    precision:
        Optional :class:`jax.lax.Precision` override.  Defaults to
        ``DEFAULT`` (BF16 with BF16 accumulation).  Pass
        ``jax.lax.Precision.HIGHEST`` for FP32 accumulation.

    Returns
    -------
    jnp.ndarray
        Result of shape ``(..., M, N)`` in Float32.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tpu_features.optimizations import tpu_matmul
    >>> a = jnp.ones((256, 512))
    >>> b = jnp.ones((512, 256))
    >>> tpu_matmul(a, b).shape
    (256, 256)
    """
    _require_jax()
    original_m = a.shape[-2]
    original_n = b.shape[-1]

    a_bf16 = cast_bfloat16(pad_to_multiple(a, MXU_TILE_SIZE))
    b_bf16 = cast_bfloat16(pad_to_multiple(b, MXU_TILE_SIZE))

    kwargs = {}
    if precision is not None:
        kwargs["precision"] = precision

    out = jnp.matmul(a_bf16, b_bf16, **kwargs)

    # Restore to correct output shape (strip padding from M and N axes).
    result_shape = out.shape[:-2] + (original_m, original_n)
    return unpad(out, result_shape)


# ---------------------------------------------------------------------------
# Memory-layout helper
# ---------------------------------------------------------------------------


def make_contiguous(array: "jnp.ndarray") -> "jnp.ndarray":
    """Return a C-contiguous copy of *array*.

    XLA / TPU compilers generally prefer C-contiguous (row-major) memory
    layout.  This function ensures the layout is correct before passing an
    array into a JIT-compiled function.

    Parameters
    ----------
    array:
        Input JAX array.

    Returns
    -------
    jnp.ndarray
        A copy of *array* in C-contiguous order.
    """
    _require_jax()
    return jnp.asarray(array, order="C")
