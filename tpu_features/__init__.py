"""tpu_features – reference library for TPU hardware characteristics and JAX feature support."""

from tpu_features.hardware import TPUGeneration, get_tpu_spec
from tpu_features.compilation import jit_with_static, aot_compile
from tpu_features.parallelism import create_device_mesh, make_named_sharding
from tpu_features.optimizations import cast_bfloat16, pad_to_multiple

__all__ = [
    "TPUGeneration",
    "get_tpu_spec",
    "jit_with_static",
    "aot_compile",
    "create_device_mesh",
    "make_named_sharding",
    "cast_bfloat16",
    "pad_to_multiple",
]
