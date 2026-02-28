import os
import sys
import traceback

os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def reduction_kernel(x_ref, out_ref):
  out_ref[0] = jnp.sum(x_ref[...], axis=0)

if __name__ == "__main__":
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  def test_uint32_reduction():
    print("\n--- Testing Reduction over uint32 ---")
    size = 128
    x = jnp.ones((size,), dtype=jnp.uint32)

    try:
      _ = pl.pallas_call(
          reduction_kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.uint32),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(size,))
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(1,)),
          grid=(1,)
      )(x)
      print("Success! (Unexpected)")
    except Exception as e:
      # print(f"Failed as expected! Error: {e}")
      traceback.print_exc()

  def test_multi_axis_reduction():
      print("\n--- Testing Reduction over Multiple axes (b, m, n) -> (b,) ---")
      b, m, n = 32, 32, 64
      try:
        def kernel(x_ref, o_ref):
            # x_ref: (b, m, n)
            # reduce over last two axes (m, n)
            o_ref[...] = jnp.sum(x_ref[...], axis=(1, 2))

        x = jnp.ones((b, m, n), dtype=jnp.float32)

        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((b,), jnp.float32),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0, 0, 0), block_shape=(b, m, n))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(b,)),
            grid=(1,)
        )(x)
        result.block_until_ready()
        print(f"Success! Result shape: {result.shape}")
      except Exception as e:
        print("Failed!")
        traceback.print_exc()

  # test_uint32_reduction()
  test_multi_axis_reduction()
