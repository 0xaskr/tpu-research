import os
import sys
import traceback

os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def scalar_kernel(x_ref, out_ref):
  # Try to load scalar from ref
  val = x_ref[...][0]
  # Try to store scalar to ref
  print("val.shape = ", val.shape)
  # out_ref[...] = jnp.full(out_ref.shape, val)
  out_ref[0] = val

def test_vmem_scalar():
  print("\n--- Testing Scalar in VMEM ---")
  x = jnp.array([1, 2], dtype=jnp.float32)

  # Try to map a scalar to VMEM
  try:
    result = pl.pallas_call(
        scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(2,))
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(2,)),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result: {result}")
  except Exception as e:
    print("Failed! Error:")
    traceback.print_exc()
    # print(e)

def test_vmem_1_element_array():
  print("\n--- Testing 1-element Array in VMEM ---")
  x = jnp.array([1.0], dtype=jnp.float32)

  # Try to map a (1,) array to VMEM
  try:
    result = pl.pallas_call(
        scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(1,))
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(1,)),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result: {result}")
  except Exception as e:
    print("Failed! Error:")
    # traceback.print_exc()
    print(e)

def test_smem_scalar():
  print("\n--- Testing Scalar in SMEM ---")
  x = jnp.array(1.0, dtype=jnp.float32)

  # Try to map a scalar to SMEM
  try:
    result = pl.pallas_call(
        scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (), block_shape=())
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (), block_shape=()),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result: {result}")
  except Exception as e:
    print("Failed! Error:")
    # traceback.print_exc()
    print(e)

def test_smem_1_element_array():
  print("\n--- Testing 1-element Array in SMEM ---")
  x = jnp.array([1.0], dtype=jnp.float32)

  # Try to map a (1,) array to SMEM
  try:
    result = pl.pallas_call(
        scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (0,), block_shape=(1,))
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (0,), block_shape=(1,)),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result: {result}")
  except Exception as e:
    print("Failed! Error:")
    # traceback.print_exc()
    print(e)

if __name__ == "__main__":
  print("Starting Scalar Memory Space Tests...")
  test_vmem_scalar()  # store scalar to VMEM should fail, but store scalar to SMEM should succeed
  # test_vmem_1_element_array()
  # test_smem_scalar()
  # test_smem_1_element_array()
