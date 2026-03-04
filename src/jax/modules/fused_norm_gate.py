import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional

class FusedRMSNormGated(nnx.Module):
    """RMSNorm + SiLU gate fusion.

    output = RMSNorm(x) * silu(g)
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

    def __call__(self, x: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        x_f32 = x.astype(jnp.float32)
        g_f32 = g.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        x_f32 = x_f32 * rms
        if self.weight is not None:
            x_f32 = x_f32 * jnp.asarray(self.weight, jnp.float32)
        x_f32 = x_f32 * jax.nn.silu(g_f32)
        return x_f32.astype(x.dtype)


# =============================================================================
# ShortConvolution (nnx.Module, 替代 fla.modules.ShortConvolution)
# 基于 nnx.Conv 的因果深度可分离 1D 卷积
# =============================================================================

