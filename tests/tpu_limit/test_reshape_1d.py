import os
import sys
import traceback

# 设置环境变量以便 dumping IR
os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def reshape_1d_to_2d_kernel(x_ref, z_ref):
  # x_ref: (N,)
  # z_ref: (R, C)
  z_ref[...] = x_ref[...].reshape(z_ref.shape)

def pallas_reshape_1d_to_2d(x, m, n):
  size = x.shape[0]
  print(f"Testing 1D to 2D Reshape: ({size},) -> ({m}, {n})")

  return pl.pallas_call(
      reshape_1d_to_2d_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(size,)),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0, 0), block_shape=(m, n)),
      grid=(1,)  # Single block covering the whole array
  )(x)

def reshape_2d_to_1d_kernel(x_ref, z_ref):
  # x_ref: (R, C)
  # z_ref: (N,)
  z_ref[...] = x_ref[...].reshape(z_ref.shape)

def pallas_reshape_2d_to_1d(x):
  m, n = x.shape
  size = m * n
  print(f"Testing 2D to 1D Reshape: ({m}, {n}) -> ({size},)")

  return pl.pallas_call(
      reshape_2d_to_1d_kernel,
      out_shape=jax.ShapeDtypeStruct((size,), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0, 0), block_shape=(m, n)),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda k: (0,), block_shape=(size,)),
      grid=(1,) # Single block covering the whole array
  )(x)


def test_1d_to_2d():
  print("\n--- Test 1D to 2D ---")
  # Test Case 1: 128 -> (8, 16)
  size = 128
  m, n = 8, 16
  x = jnp.arange(size, dtype=jnp.float32)

  try:
    z = jax.jit(pallas_reshape_1d_to_2d, static_argnames=['m', 'n'])(x, m, n)
    z.block_until_ready()
    print(f"Success! Output shape: {z.shape}")
    expected = x.reshape((m, n))
    if jnp.allclose(z, expected):
        print("Verification Passed")
    else:
        print("Verification Failed")
  except Exception as e:
    print(f"Failed! Error:")
    traceback.print_exc()

def test_2d_to_1d():
  print("\n--- Test 2D to 1D ---")
  # Test Case 1: (8, 16) -> 128
  m, n = 8, 16
  x = jnp.arange(m*n, dtype=jnp.float32).reshape((m, n))

  try:
    z = jax.jit(pallas_reshape_2d_to_1d)(x)
    z.block_until_ready()
    print(f"Success! Output shape: {z.shape}")
    expected = x.reshape((m*n,))
    if jnp.allclose(z, expected):
        print("Verification Passed")
    else:
        print("Verification Failed")
  except Exception as e:
    print(f"Failed! Error:")
    traceback.print_exc()

if __name__ == "__main__":
  print("Starting Reshape Tests...")
  test_1d_to_2d()
  test_2d_to_1d()
