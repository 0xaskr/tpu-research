import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax._src.pallas.mosaic.tpu_info import get_tpu_info


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


def probe_vmem_physical_bytes() -> int:
  """返回单个 TensorCore 的物理 VMEM 容量（字节）。

  从 jax._src.pallas.mosaic.get_tpu_info() 直接读取，
  不发起 XLA 编译。

  Returns:
    物理 VMEM 字节数（per TensorCore）。

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  _ensure_tpu()
  return get_tpu_info().vmem_capacity_bytes


def probe_vmem_scoped_bytes(
  dtype=jnp.float32,
  tolerance_bytes: int = 1024 * 1024,
) -> int:
  """通过二分法探测单 kernel 的 scoped VMEM limit（字节）。

  这是编译器默认允许单个 kernel 分配的最大 VMEM，
  XLA 侧没有提供 Python 回读接口，因此只能动态探测。

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

  物理 VMEM 从 get_tpu_info() 读取，scoped limit 通过二分探测。

  Args:
    dtype: 测试分配使用的数据类型，默认 float32。
    tolerance_bytes: 二分精度（字节），默认 1MB。

  Returns:
    dict with keys:
      device_kind:     TPU 设备类型字符串
      chip_version:    芯片版本 (v5e, v5p, v6e, ...)
      generation:      代际（5, 6, 7, ...）
      num_cores:       TensorCore 数量
      physical:        单 core 物理 VMEM（字节）
      total_physical:  芯片总物理 VMEM（bytes）= physical * num_cores
      hbm_capacity:    HBM 容量（字节）
      scoped_limit:    单 kernel scoped VMEM limit（字节）

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  _ensure_tpu()

  info = get_tpu_info()
  device_kind = next(
    d.device_kind for d in jax.devices() if d.platform == "tpu"
  )

  return {
    "device_kind": device_kind,
    "chip_version": info.chip_version.value,
    "generation": info.generation,
    "num_cores": info.num_cores,
    "physical": info.vmem_capacity_bytes,
    "scoped_limit": _bisect(dtype, tolerance_bytes),
  }
