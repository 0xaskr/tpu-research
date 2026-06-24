import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fill_kernel(o_ref):
  o_ref[...] = jnp.zeros(o_ref.shape, dtype=o_ref.dtype)


def _try_alloc(size_bytes: int, dtype: jnp.dtype) -> bool:
  elem_size = jnp.dtype(dtype).itemsize
  n_elements = size_bytes // elem_size
  n_elements = (n_elements // 128) * 128
  if n_elements == 0:
    return True
  try:
    result = pl.pallas_call(
      _fill_kernel,
      out_shape=jax.ShapeDtypeStruct((n_elements,), dtype),
      in_specs=[],
      out_specs=pl.BlockSpec(
        memory_space=pltpu.VMEM, block_shape=(n_elements,)
      ),
      grid=(1,),
    )()
    result.block_until_ready()
    return True
  except Exception:
    return False


def probe_vmem_bytes(
  dtype=jnp.float32,
  tolerance_bytes: int = 1024 * 1024,
) -> int:
  """探测当前 TPU 可分配的最大 VMEM（字节）。

  使用无输入 fill kernel，只分配单个 VMEM 输出 buffer，
  通过指数倍增 + 二分搜索找到最大可分配边界。

  Args:
    dtype: 测试分配使用的数据类型，默认 float32。
    tolerance_bytes: 二分精度（字节），默认 1MB。

  Returns:
    最大可分配的 VMEM 字节数。

  Raises:
    RuntimeError: 当前环境没有 TPU 设备。
  """
  devices = jax.devices()
  has_tpu = any(d.platform == "tpu" for d in devices)
  if not has_tpu:
    raise RuntimeError(
      "No TPU device found. "
      f"Available devices: {[d.platform for d in devices]}"
    )

  lo = 0
  hi = 1024 * 1024  # 从 1MB 开始

  # 指数倍增：找到第一个失败的上界
  while _try_alloc(hi, dtype):
    lo = hi
    hi *= 2

  # 二分搜索：精确定位边界
  while hi - lo > tolerance_bytes:
    mid = (lo + hi) // 2
    if _try_alloc(mid, dtype):
      lo = mid
    else:
      hi = mid

  return lo
