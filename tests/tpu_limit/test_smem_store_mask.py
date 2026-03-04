import os
import sys
import traceback

os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def smem_store_mask_kernel(x_ref, out_ref):
  # x_ref is (1,) in VMEM
  val = x_ref[0]

  # SMEM ref
  scratch_ref = pl.program_id(0) # Just a dummy val

  # Try to store to SMEM with a mask
  # Note: Pallas doesn't have explicit masked_store primitive easily accessible like vector.masked_store
  # But we can try to use jax.lax.select or similar within the kernel if it lowers to masked store,
  # OR rely on pl.when if it's supported for SMEM store.

  # However, pl.load/store don't take mask args directly in high-level pallas.
  # But we can try to use `pltpu.when` or similar control flow.
  # Let's try a simple conditional store which might lower to masked store or predication.

  # Actually, let's look at how Pallas exposes masking.
  # Usually it's via `mask` argument in `pl.load` or `pl.store` if they exist in `jax.experimental.pallas.primitives`?
  # The standard JAX Pallas doesn't expose `pl.store(ref, val, mask=...)` directly in the public API easily.

  # Let's try to simulate a scenario that would trigger a masked store lowering potentially.
  # Or simply check if `when` works.

  pass

def smem_masked_store_kernel(x_ref, out_ref):
  program_id = pl.program_id(0)
  # Trying to conditionally store to SMEM
  # This usually lowers to a predicated store or scf.if

  # We need an SMEM buffer. We can allocate one or pass one in.
  # Let's allocate a small SMEM scratchpad.
  smem_scratch = pl.program_local_allocation((1,), jnp.float32, memory_space=pltpu.SMEM)

  val = x_ref[0]

  # Mask: only store if program_id == 0 (which it is)
  mask = program_id == 0

  # There isn't a direct "masked_store" function.
  # But we can try:
  #   smem_scratch[0] = jnp.where(mask, val, smem_scratch[0]) ?? NO, this is read-modify-write

  # If we use `pl.when`, it might generate specific control flow.
  # with pl.when(mask):
  #   smem_scratch[0] = val

  # But if the user prompt says "store smem does not support mask", maybe they mean
  # the lowering rule for `store` complains if a mask is present in the JAXPR?
  # Currently Pallas `store` primitive *does* have a mask field in internal representation,
  # but triggering it from high level API is tricky.

  # As per the previous `lowering.py` review, let's look at `_store_lowering_rule`.
  # It takes `mask` and `transforms`.

  # If we cannot easily generate a masked store from source, we might skip this unless
  # we can use `pl.when` or `jax.lax.cond`.

  # Let's try to use `pl.when` to guard a store to SMEM.
  @pl.when(mask)
  def _():
      smem_scratch[0] = val

  # And write back to out_ref to prevent DCE
  out_ref[0] = smem_scratch[0]

def test_smem_store_mask():
  print("\n--- Testing SMEM Store with Mask (via pl.when) ---")
  x = jnp.array([42.0], dtype=jnp.float32)

  try:
    result = pl.pallas_call(
        smem_masked_store_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(1,))
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (0,), block_shape=(1,)),
        grid=(1,)
    )(x)
    result.block_until_ready()
    print(f"Success! Result: {result}")
  except Exception as e:
    print("Failed! Error:")
    traceback.print_exc()
    # print(e)

if __name__ == "__main__":
  print("Starting SMEM Store Mask Tests...")
  test_smem_store_mask()
