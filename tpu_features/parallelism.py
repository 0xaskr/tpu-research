"""Parallelism utilities for multi-chip / multi-host TPU training with JAX.

Covers the two main JAX parallelism APIs:

* **pmap** – the original single-program multiple-data (SPMD) API.
* **Named sharding / GSPMD** – the modern approach based on
  :class:`jax.sharding.Mesh` and :class:`jax.sharding.NamedSharding`.
"""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this function. "
            "Install it with: pip install 'jax[tpu]' "
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )


# ---------------------------------------------------------------------------
# Device mesh helpers
# ---------------------------------------------------------------------------


def create_device_mesh(
    mesh_shape: Tuple[int, ...],
    axis_names: Tuple[str, ...] | None = None,
) -> "Mesh":
    """Create a :class:`jax.sharding.Mesh` from all available devices.

    Parameters
    ----------
    mesh_shape:
        The logical shape of the device mesh, e.g. ``(2, 4)`` for 2 rows
        and 4 columns.  The product must equal the number of available
        devices.
    axis_names:
        Names for each mesh axis.  Defaults to ``("axis_0", "axis_1", ...)``.

    Returns
    -------
    jax.sharding.Mesh
        A mesh object ready for use with :class:`~jax.sharding.NamedSharding`.

    Raises
    ------
    ValueError
        If the product of *mesh_shape* does not equal the number of
        available devices.

    Examples
    --------
    >>> mesh = create_device_mesh((2, 4))
    >>> mesh.shape
    {'axis_0': 2, 'axis_1': 4}
    """
    _require_jax()
    try:
        from jax.experimental import mesh_utils as _mu

        devices = _mu.create_device_mesh(mesh_shape)
    except Exception:
        # Fallback: reshape the flat device list directly.
        import numpy as np

        flat = jax.devices()
        if len(flat) != _product(mesh_shape):
            raise ValueError(
                f"mesh_shape product ({_product(mesh_shape)}) must equal the "
                f"number of available devices ({len(flat)})."
            )
        devices = np.array(flat).reshape(mesh_shape)

    if axis_names is None:
        axis_names = tuple(f"axis_{i}" for i in range(len(mesh_shape)))

    return Mesh(devices, axis_names=axis_names)


def _product(shape: Tuple[int, ...]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result


# ---------------------------------------------------------------------------
# NamedSharding factory
# ---------------------------------------------------------------------------


def make_named_sharding(
    mesh: "Mesh",
    partition_spec: "PartitionSpec",
) -> "NamedSharding":
    """Convenience wrapper around :class:`jax.sharding.NamedSharding`.

    Parameters
    ----------
    mesh:
        The device mesh to shard over.
    partition_spec:
        A :class:`~jax.sharding.PartitionSpec` describing which array axes
        map to which mesh axes.  Use ``None`` for axes that should be
        replicated across all devices on that mesh dimension.

    Returns
    -------
    jax.sharding.NamedSharding

    Examples
    --------
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = create_device_mesh((2, 4), ("batch", "model"))
    >>> sharding = make_named_sharding(mesh, P("batch", "model"))
    """
    _require_jax()
    return NamedSharding(mesh, partition_spec)


# ---------------------------------------------------------------------------
# pmap helpers
# ---------------------------------------------------------------------------


def pmap_mean(fn):
    """Wrap *fn* with :func:`jax.pmap` and average results across devices.

    The wrapped function performs an ``lax.pmean`` collective over all
    participating devices so the output is the device-averaged result.

    Parameters
    ----------
    fn:
        A function that returns a scalar JAX array (e.g. a loss value).

    Returns
    -------
    Callable
        A pmap'd version of *fn* whose output is averaged across devices.

    Notes
    -----
    Input arrays must have a leading batch dimension equal to the number of
    local devices (``jax.local_device_count()``).
    """
    _require_jax()

    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        return jax.lax.pmean(result, axis_name="devices")

    return jax.pmap(wrapped, axis_name="devices")


def replicate(array: "jnp.ndarray") -> "jnp.ndarray":
    """Replicate *array* across all local devices for use with :func:`jax.pmap`.

    Creates a leading device axis by stacking identical copies of *array*,
    one per local device.

    Parameters
    ----------
    array:
        A JAX array without a leading device dimension.

    Returns
    -------
    jnp.ndarray
        An array of shape ``(n_devices, *array.shape)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.ones((4,))
    >>> replicated = replicate(x)
    >>> replicated.shape  # (n_devices, 4)
    """
    _require_jax()
    n = jax.local_device_count()
    return jnp.stack([array] * n)


def unreplicate(array: "jnp.ndarray") -> "jnp.ndarray":
    """Extract a single-device copy from a replicated pmap output.

    Takes the first device's slice (index 0 along the leading axis).  All
    devices hold identical values after an ``lax.pmean`` / ``lax.psum``
    collective, so this is safe.

    Parameters
    ----------
    array:
        A pmap-replicated array of shape ``(n_devices, ...)``.

    Returns
    -------
    jnp.ndarray
        The first-device slice with shape ``array.shape[1:]``.
    """
    _require_jax()
    return array[0]


# ---------------------------------------------------------------------------
# Collective primitives documentation wrappers
# ---------------------------------------------------------------------------


def all_reduce_sum(x: "jnp.ndarray", axis_name: str = "devices") -> "jnp.ndarray":
    """Sum *x* across all devices in the named pmap axis.

    Must be called inside a :func:`jax.pmap`-compiled function.

    Parameters
    ----------
    x:
        Per-device array to reduce.
    axis_name:
        The pmap axis name used when the outer function was compiled.

    Returns
    -------
    jnp.ndarray
        The summed value, identical on every device.
    """
    _require_jax()
    return jax.lax.psum(x, axis_name=axis_name)


def all_reduce_mean(x: "jnp.ndarray", axis_name: str = "devices") -> "jnp.ndarray":
    """Average *x* across all devices in the named pmap axis.

    Must be called inside a :func:`jax.pmap`-compiled function.

    Parameters
    ----------
    x:
        Per-device array to average.
    axis_name:
        The pmap axis name used when the outer function was compiled.

    Returns
    -------
    jnp.ndarray
        The averaged value, identical on every device.
    """
    _require_jax()
    return jax.lax.pmean(x, axis_name=axis_name)
