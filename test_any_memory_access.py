import os
import sys
import traceback

os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def any_memory_kernel(x_ref, out_ref):
  # Try to read directly from ANY (HBM) memory space
  # This should fail if ANY memory space can only be accessed via async copy
  val = x_ref[...]
  out_ref[...] = val

def test_any_memory_direct_access():
  print("\n--- Testing Direct Access for ANY (HBM) Memory Space ---")
  size = 128
  x = jnp.arange(size, dtype=jnp.float32)

  try:
    # Use ANY memory space for input
    # Note: Using pltpu.ANY might not be directly exposed as a constant like VMEM,
    # but pl.BlockSpec(memory_space=pl.ANY, ...) is the typical way.
    # Let's check if pltpu.ANY exists, otherwise use pl.ANY
    mem_space = pl.ANY

    result = pl.pallas_call(
        any_memory_kernel,
        out_shape=jax.ShapeDtypeStruct((size,), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=mem_space, index_map=lambda i: (0,), block_shape=(size,))
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(size,)),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result shape: {result.shape}")
  except Exception as e:
    print("Failed! Error:")
    traceback.print_exc()
    print(e)


def any_memory_async_copy_kernel(x_ref, out_ref):
  # Correct way: Async copy from ANY to VMEM
  # But here we just want to verify the FAILURE of direct access.
  # So the kernel above is enough for the failure case.
  # If we want to show success, we would need to implement async copy (dma_start/wait).
  pass

if __name__ == "__main__":
  print("Starting ANY Memory Space Tests...")
  test_any_memory_direct_access()
