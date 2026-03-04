
import os
import sys
import traceback
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Set environment variables for compilation dump
os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

def scalar_op_kernel(x_ref, o_ref):
  # Load scalar from x_ref
  val = x_ref[0]
  # Store scalar to o_ref
  o_ref[0] = val

def test_scalar_types():
  # Types to test
  # Common TPU types: f32, s32, bf16, s8, u8, s16, u16
  types_to_test = [
      ("float32", jnp.float32),
      ("int32", jnp.int32),
      ("bfloat16", jnp.bfloat16),
      ("int8", jnp.int8),
      ("int16", jnp.int16),
  ]

  for type_name, dtype in types_to_test:
      print(f"\n--- Testing Scalar Load/Store for {type_name} ---")

      # For scalars, we typically use single-element arrays
      # VMEM workaround: rank-1 array of size 1 (since rank-0 VMEM is rarely supported directly in Pallas/Mosaic)
      # SMEM: can support true scalars (rank-0) or rank-1 size 1.
      # Let's test SMEM specifically as the user asked about "scalar load/store".

      # However, Pallas kernels define block specs.
      # Let's try SMEM first, as that is the "scalar" memory.

      data = jnp.array([42], dtype=dtype)

      try:
          # Define a kernel using SMEM for input and output
          # Note: Pallas usually requires inputs to be in VMEM or SMEM.
          # Let's try to put them in SMEM.

          def pallas_wrapper(x):
              return pl.pallas_call(
                  scalar_op_kernel,
                  out_shape=jax.ShapeDtypeStruct((1,), dtype),
                  in_specs=[
                      pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (i,), block_shape=(1,))
                  ],
                  out_specs=pl.BlockSpec(memory_space=pltpu.SMEM, index_map=lambda i: (i,), block_shape=(1,)),
                  grid=(1,)
              )(x)

          jit_pallas = jax.jit(pallas_wrapper)
          result = jit_pallas(data)
          result.block_until_ready()
          print(f"Success map for {type_name} in SMEM!")

      except Exception as e:
          print(f"Failed for {type_name} in SMEM! Error:")
          # Print the error message, not the full trace for clarity (unless needed)
          traceback.print_exc()

      print(f"--- Testing 1D-Scalar (size 1) Load/Store for {type_name} in VMEM ---")
      try:
           def pallas_wrapper_vmem(x):
              return pl.pallas_call(
                  scalar_op_kernel,
                  out_shape=jax.ShapeDtypeStruct((1,), dtype),
                  in_specs=[
                      pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(1,))
                  ],
                  out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(1,)),
                  grid=(1,)
              )(x)

           jit_pallas_vmem = jax.jit(pallas_wrapper_vmem)
           result = jit_pallas_vmem(data)
           result.block_until_ready()
           print(f"Success for {type_name} in VMEM!")
      except Exception as e:
          print(f"Failed for {type_name} in VMEM! Error:")
          traceback.print_exc()

if __name__ == "__main__":
    print("Starting scalar type tests...")
    test_scalar_types()
