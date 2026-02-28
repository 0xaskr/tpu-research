
import os
import sys
import traceback

# Setup environment for Pallas Mosaic
os.environ["LIBTPU_INIT_ARGS"] = "--xla_mosaic_dump_to=./llo_ir/pallas_ir/ --xla_jf_dump_to=./llo_ir/jax_ir"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Attempt to import sc_primitives from internal path since it might not be fully exposed
try:
    from jax._src.pallas.mosaic import sc_primitives as scp
except ImportError:
    print("Could not import sc_primitives from jax._src.pallas.mosaic")
    sys.exit(1)

def sc_gather_kernel(x_ref, indices_ref, o_ref):
    # This kernel attempts to use load_gather on a non-supported dtype (bfloat16)
    # x_ref should be in VMEM
    # indices_ref should be in VMEM

    # Load indices first
    indices = indices_ref[...]

    # Attempt gather using jnp.take
    out = jnp.take(x_ref, indices, axis=0)

    o_ref[...] = out

def test_sc_gather_limit():
    print("\n--- Testing SparseCore Gather Limit (Expect TypeError for bfloat16) ---")

    # Parameters
    B = 128
    # Use bfloat16 to trigger the limitation
    dtype = jnp.bfloat16

    x = jnp.arange(B, dtype=dtype)
    indices = jnp.arange(B, dtype=jnp.int32)

    # We need a SparseCore-like environment or at least a Mosaic kernel that allows these primitives.
    # However, standard pallas_call might not support sc_primitives unless we are in the right mode.
    # sc_primitives seem to rely on sc_lowering.

    # Let's try running it. If it fails specifically with "ref.dtype=... must be int32 or float32", success.

    try:
        kernel = pl.pallas_call(
            sc_gather_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(B,)),
                pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(B,))
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM, index_map=lambda i: (i,), block_shape=(B,)),
            grid=(1,)
        )
        z = kernel(x, indices)
        z.block_until_ready()
        print("Unexpected Success! Kernel ran without error.")
    except TypeError as e:
        if "must be int32 or float32" in str(e):
            print(f"Success! Caught expected TypeError: {e}")
        else:
            print(f"Caught TypeError but message unexpected: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Caught unexpected Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_sc_gather_limit()
