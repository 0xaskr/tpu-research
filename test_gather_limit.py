
import os
import sys
import traceback
os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

def gather_1d_kernel(x_ref, indices_ref, o_ref):
    # x: (N,), indices: (M,) -> out: (M,)
    # expected to fail because input rank != 2
    # Load into registers first
    x_val = x_ref[...]
    indices_val = indices_ref[...]
    o_ref[...] = jnp.take(x_val, indices_val, axis=0)

def test_gather_1d():
    print("\n--- Testing 1D Gather (Expect Failure) ---")
    N = 128
    x = jnp.arange(N, dtype=jnp.float32)
    indices = jnp.array([0, 2, 4, 6], dtype=jnp.int32)

    # Block definitions don't matter too much if lowering fails early,
    # but we need valid ones.
    # 1D blocks
    try:
        kernel = pl.pallas_call(
            gather_1d_kernel,
            out_shape=jax.ShapeDtypeStruct(indices.shape, x.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(N,)),
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(indices.shape[0],))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(indices.shape[0],)),
            grid=(1,)
        )
        z = kernel(x, indices)
        z.block_until_ready()
        print("Success (Unexpected)!")
    except Exception as e:
        print(f"Failed as expected: {e}")
        # traceback.print_exc()

def gather_2d_kernel_axis0(x_ref, indices_ref, o_ref):
    x_val = x_ref[...]
    indices_val = indices_ref[...]
    o_ref[...] = jnp.take_along_axis(x_val, indices_val, axis=0)

def test_gather_2d_pass():
    # Use native vector size (8 sublanes x 128 lanes) for f32
    print("\n--- Testing 2D Gather (take_along_axis) with Native Vector Shape (8, 128) ---")
    M, N = 8, 128
    x = jnp.arange(M * N, dtype=jnp.float32).reshape(M, N)
    indices = jnp.zeros((M, N), dtype=jnp.int32)

    try:
        kernel = pl.pallas_call(
            gather_2d_kernel_axis0,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(M, N)),
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(M, N))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(M, N)),
            grid=(1, 1)
        )
        z = kernel(x, indices)
        z.block_until_ready()
        print("Success!")
    except Exception as e:
        print(f"Failed (Unexpected): {e}")
        # traceback.print_exc()

def gather_3d_kernel(x_ref, indices_ref, o_ref):
    x_val = x_ref[...]
    indices_val = indices_ref[...]
    o_ref[...] = jnp.take_along_axis(x_val, indices_val, axis=0)

def test_gather_3d():
    print("\n--- Testing 3D Gather (Expect Failure) ---")
    B, M, N = 8, 32, 32
    x = jnp.zeros((B, M, N), dtype=jnp.float32)
    indices = jnp.zeros((B, M, N), dtype=jnp.int32)

    try:
        kernel = pl.pallas_call(
            gather_3d_kernel,
            out_shape=jax.ShapeDtypeStruct((B, M, N), x.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N)),
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N)),
            grid=(1, 1, 1)
        )
        z = kernel(x, indices)
        z.block_until_ready()
        print("Success (Unexpected)!")
    except Exception as e:
        print(f"Failed as expected: {e}")
        # traceback.print_exc()

if __name__ == "__main__":
    test_gather_1d()
    test_gather_2d_pass()
    test_gather_3d()

    B, M, N = 8, 32, 32
    x = jnp.zeros((B, M, N), dtype=jnp.float32)
    indices = jnp.zeros((B, M, N), dtype=jnp.int32)

    try:
        kernel = pl.pallas_call(
            gather_3d_kernel,
            out_shape=jax.ShapeDtypeStruct((B, M, N), x.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N)),
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j, k: (i, j, k), block_shape=(B, M, N)),
            grid=(1, 1, 1)
        )
        z = kernel(x, indices)
        z.block_until_ready()
        print("Success (Unexpected)!")
    except Exception as e:
        print(f"Failed as expected: {e}")
        # traceback.print_exc()

if __name__ == "__main__":
    test_gather_1d()
    test_gather_2d_pass()
    test_gather_3d()
