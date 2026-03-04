import os
import sys
import traceback
os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def simple_matmul_kernel(x_ref, y_ref, z_ref):
  # x_ref shape: (B, bm, k), y_ref shape: (B, k, bn)
  # jnp.matmul handles batched matrix multiplication correctly
  z_ref[...] = jnp.matmul(x_ref[...], y_ref[...])

def pallas_matmul(x, y, bm, bn):
  B = x.shape[0]
  m, k = x.shape[1], x.shape[2]
  _, n = y.shape[1], y.shape[2]
  print("pallas_call lhs block shape = ", (B, bm, k))
  print("pallas_call rhs block shape = ", (B, k, bn))
  print("pallas_call output block shape = ", (B, bm, bn))
  return pl.pallas_call(
      simple_matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((B, m, n), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (0, i, 0), block_shape=(B, bm, k)),
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (0, 0, j), block_shape=(B, k, bn))
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (0, i, j), block_shape=(B, bm, bn)),
      grid=(m // bm, n // bn)
  )(x, y)

def simple_add_1d_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] + y_ref[...]

def pallas_add_1d(x, y, block_size):
  size = x.shape[0]
  print(f"pallas_call 1D block shape = ({block_size},)")
  return pl.pallas_call(
      simple_add_1d_kernel,
      out_shape=jax.ShapeDtypeStruct((size,), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(block_size,)),
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(block_size,))
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(block_size,)),
      grid=(size // block_size,)
  )(x, y)

def simple_add_2d_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] + y_ref[...]

def pallas_add_2d(x, y, bm, bn):
  m, n = x.shape
  print(f"pallas_call 2D block shape = ({bm}, {bn})")
  return pl.pallas_call(
      simple_add_2d_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(bm, bn)),
          pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(bm, bn))
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i, j: (i, j), block_shape=(bm, bn)),
      grid=(m // bm, n // bn)
  )(x, y)

# 测试pallas call 对于最低维度的限制

def test_matmul_grid_2x2():
  B = 32

  # 测试不同的 shape，观察 TPU Pallas 对最低维度的限制
  shapes_to_test = [
      # (32, 32, 32),    # fail
      # (32, 32, 256),    # pass
      # (2, 32, 256),       # lhs fail, lhs [2, 32] -> block shape = [1, 32]
      (16, 32, 256),    # pass, lhs [16, 32] -> block shape = [8, 32]
      # (32, 32, 128),    # 正常的 32 对齐 -> rhs fail
      # (32, 8, 32),     # K 维度为 8
      # (32, 4, 32),     # K 维度为 4 (可能会触发限制)
      # (16, 32, 16),    # M, N 为 16
      # (8, 32, 8),      # M, N 为 8
      # (4, 32, 4),      # M, N 为 4
  ]

  for m, k, n in shapes_to_test:
    print(f"\n--- Testing Matmul Shape (B={B}, M={m}, K={k}, N={n}) ---")
    x = jnp.ones((B, m, k), dtype=jnp.float32)
    y = jnp.ones((B, k, n), dtype=jnp.float32)

    # 这里 grid 设为 (2, 2)
    bm, bn = m // 2, n // 2

    try:
      # 使用 jit 编译并运行
      z = jax.jit(pallas_matmul, static_argnames=['bm', 'bn'])(x, y, bm, bn)
      z.block_until_ready()
      print(f"Success! Output shape: {z.shape}")
    except Exception as e:
      print(f"Failed! Error:")
      traceback.print_exc()

def test_1d_block_alignment():
  print("\n==================================================")
  print("Testing 1D Block Alignment (f32 requires 128 multiple or equal to array size)")
  print("==================================================")

  test_cases = [
      # (1024 + 512, 1024),  # Pass: 128 is multiple of 128
      (64, 64),   # Fail: 64 is not multiple of 128, and not equal to array size 256
      # (64, 64),    # Pass: equal to array size
      # (384, 128),  # Pass: 128 is multiple of 128
      # (384, 192),  # Fail: 192 is not multiple of 128
  ]

  for size, block_size in test_cases:
    print(f"\n--- Testing 1D Add (Size={size}, Block Size={block_size}) ---")
    x = jnp.ones((size,), dtype=jnp.float32)
    y = jnp.ones((size,), dtype=jnp.float32)

    try:
      # 使用 jit 编译并运行
      z = jax.jit(pallas_add_1d, static_argnames=['block_size'])(x, y, block_size)
      z.block_until_ready()
      print(f"Success! Output shape: {z.shape}")
    except Exception as e:
      print(f"Failed! Error:")
      traceback.print_exc()

def test_add_2d_block_alignment():
  print("\n==================================================")
  print("Testing 2D Add Block Alignment (requires last two dims divisible by 8 and 128)")
  print("==================================================")

  test_cases = [
      # (Array Shape), (Block Shape)
      # ((256, 256), (128, 128)), # Pass: 128%8==0, 128%128==0
      # ((256, 256), (8, 128)),   # Pass: 8%8==0, 128%128==0
      ((256, 256), (4, 128)),   # Fail: 4 is not divisible by 8
      # ((256, 256), (8, 64)),    # Fail: 64 is not divisible by 128
      # ((64, 64), (64, 64)),     # Pass: block_shape == array_shape
      # ((256, 256), (16, 256)),  # Pass: 16%8==0, 256%128==0
  ]

  for (m, n), (bm, bn) in test_cases:
    print(f"\n--- Testing 2D Add (Shape=({m}, {n}), Block=({bm}, {bn})) ---")
    x = jnp.ones((m, n), dtype=jnp.float32)
    y = jnp.ones((m, n), dtype=jnp.float32)

    try:
      # 使用 jit 编译并运行
      z = jax.jit(pallas_add_2d, static_argnames=['bm', 'bn'])(x, y, bm, bn)
      z.block_until_ready()
      print(f"Success! Output shape: {z.shape}")
    except Exception as e:
      print(f"Failed! Error:")
      traceback.print_exc()

if __name__ == "__main__":
  print("Hello from tpu-research!")
  print("jax version:", jax.__version__)

  # test_matmul_grid_2x2()
  test_1d_block_alignment()
  # test_add_2d_block_alignment()
