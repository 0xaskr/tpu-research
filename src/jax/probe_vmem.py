import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VMEM_CEIL = 512 * 1024 * 1024  # 512 MB, 远超任何已知 TPU


def _ensure_tpu():
  devices = jax.devices()
  if not any(d.platform == "tpu" for d in devices):
    raise RuntimeError(
      "No TPU device found. "
      f"Available devices: {[d.platform for d in devices]}"
    )


def _fill_kernel(o_ref):
  o_ref[...] = jnp.zeros(o_ref.shape, dtype=o_ref.dtype)


def _try_alloc(
  size_bytes: int,
  dtype: jnp.dtype,
  vmem_limit_bytes: int | None = None,
) -> bool:
  elem_size = jnp.dtype(dtype).itemsize
  n_elements = size_bytes // elem_size
  n_elements = (n_elements // 128) * 128
  if n_elements == 0:
    return True
  kwargs = {}
  if vmem_limit_bytes is not None:
    kwargs["compiler_params"] = pltpu.CompilerParams(
      vmem_limit_bytes=vmem_limit_bytes,
    )
  try:
    result = pl.pallas_call(
      _fill_kernel,
      out_shape=jax.ShapeDtypeStruct((n_elements,), dtype),
      in_specs=[],
      out_specs=pl.BlockSpec(
        memory_space=pltpu.VMEM, block_shape=(n_elements,)
      ),
      grid=(1,),
      **kwargs,
    )()
    result.block_until_ready()
    return True
  except Exception:
    return False


def _bisect(dtype, tolerance_bytes, vmem_limit_bytes=None):
  lo = 0
  hi = 1024 * 1024

  while _try_alloc(hi, dtype, vmem_limit_bytes):
    lo = hi
    hi *= 2

  while hi - lo > tolerance_bytes:
    mid = (lo + hi) // 2
    if _try_alloc(mid, dtype, vmem_limit_bytes):
      lo = mid
    else:
      hi = mid

  return lo


def probe_vmem_physical_bytes(
  dtype=jnp.float32,
  tolerance_bytes: int = 1024 * 1024,
) -> int:
  """通过提升 scoped limit 后二分法探测硬件物理 VMEM 总量（字节）。

  设置 --xla_tpu_scoped_vmem_limit_kib 和 CompilerParams.vmem_limit_bytes
  将 scoped limit 提升到远超物理 VMEM 的值，使二分法直接命中硬件上限。

  Args:
    dtype: 测试分配使用的数据类型，默认 float32。
    tolerance_bytes: 二分精度（字节），默认 1MB。

  Returns:
    物理 VMEM 字节数。

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  _ensure_tpu()

  existing = os.environ.get("LIBTPU_INIT_ARGS", "")
  scoped_kib = _VMEM_CEIL // 1024
  flag = f"--xla_tpu_scoped_vmem_limit_kib={scoped_kib}"
  if flag not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = (
      f"{existing} {flag}".strip()
    )

  return _bisect(dtype, tolerance_bytes, vmem_limit_bytes=_VMEM_CEIL)


def probe_vmem_scoped_bytes(
  dtype=jnp.float32,
  tolerance_bytes: int = 1024 * 1024,
) -> int:
  """通过二分法探测单个 Pallas kernel 的 scoped VMEM limit（字节）。

  这是编译器默认允许单个 kernel 分配的最大 VMEM，通常小于物理总量。

  Args:
    dtype: 测试分配使用的数据类型，默认 float32。
    tolerance_bytes: 二分精度（字节），默认 1MB。

  Returns:
    scoped VMEM limit 字节数。

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  _ensure_tpu()
  return _bisect(dtype, tolerance_bytes)


def probe_vmem_bytes(
  dtype=jnp.float32,
  tolerance_bytes: int = 1024 * 1024,
) -> dict:
  """探测当前 TPU 的 VMEM 信息。

  返回物理 VMEM 总量和单 kernel scoped limit 两个值。

  Args:
    dtype: 测试分配使用的数据类型，默认 float32。
    tolerance_bytes: 二分精度（字节），默认 1MB。

  Returns:
    dict with keys:
      physical: 硬件物理 VMEM 总量（字节）
      scoped_limit: 单 kernel scoped VMEM limit（字节）
      device_kind: TPU 设备类型字符串

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  _ensure_tpu()

  device_kind = next(
    d.device_kind for d in jax.devices() if d.platform == "tpu"
  )

  return {
    "physical": probe_vmem_physical_bytes(dtype, tolerance_bytes),
    "scoped_limit": probe_vmem_scoped_bytes(dtype, tolerance_bytes),
    "device_kind": device_kind,
  }
