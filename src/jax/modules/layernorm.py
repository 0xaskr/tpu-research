import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    公式: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nnx.Param(jnp.ones(hidden_size, dtype=jnp.float32))
        else:
            self.weight = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_f32 = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        x_f32 = x_f32 * rms
        if self.weight is not None:
            x_f32 = x_f32 * jnp.asarray(self.weight, jnp.float32)
        return x_f32.astype(x.dtype)


# =============================================================================
# FusedRMSNormGated (nnx.Module, 替代 fla.modules.FusedRMSNormGated)
# =============================================================================

