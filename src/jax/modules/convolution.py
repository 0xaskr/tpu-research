import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional

class ShortConvolution(nnx.Module):
    """Causal depthwise 1D convolution.

    - groups = hidden_size (depthwise separable)
    - causal padding
    - optional SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(kernel_size,),
            feature_group_count=hidden_size,
            padding="CAUSAL",
            use_bias=bias,
            rngs=rngs,
        )

    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.activation in ('silu', 'swish'):
            return jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)
        return x

    def __call__(
        self,
        x: jnp.ndarray,
        cache: jnp.ndarray | None = None,
        output_final_state: bool = False,
        cu_seqlens: Any = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Forward pass.

        Args:
            x: [B, T, D]
            cache: previous conv state [N, kernel_size-1, D] or None.
            output_final_state: whether to return final conv states.
            cu_seqlens: [N+1] cumulative seq lengths (numpy array).

        Returns:
            y: [B, T, D]
            new_cache: [N, kernel_size-1, D] or None
        """
        W = self.kernel_size
        B, T, D = x.shape

        if cu_seqlens is not None:
            y = self._forward_varlen(x, cu_seqlens)
        else:
            if cache is not None:
                x_full = jnp.concatenate([cache, x], axis=1)
                y_full = self.conv(x_full)
                y = y_full[:, -T:, :]
            else:
                y = self.conv(x)

        y = self._apply_activation(y)

        # Compute final state
        new_cache = None
        if output_final_state and W > 1:
            if cu_seqlens is not None:
                N = len(cu_seqlens) - 1
                caches = []
                x_flat = x[0]  # [T_total, D]
                for i in range(N):
                    bos = int(cu_seqlens[i])
                    eos = int(cu_seqlens[i + 1])
                    seg = x_flat[bos:eos]  # [seg_len, D]
                    seg_len = eos - bos
                    if seg_len >= W - 1:
                        caches.append(seg[-(W - 1):])
                    else:
                        pad_len = (W - 1) - seg_len
                        caches.append(jnp.concatenate([
                            jnp.zeros((pad_len, D), dtype=x.dtype),
                            seg,
                        ], axis=0))
                new_cache = jnp.stack(caches, axis=0)  # [N, W-1, D]
            else:
                # x: [B, T, D] → take last (W-1) tokens
                if T >= W - 1:
                    new_cache = x[:, -(W - 1):, :]
                else:
                    pad_len = (W - 1) - T
                    new_cache = jnp.concatenate([
                        jnp.zeros((B, pad_len, D), dtype=x.dtype),
                        x,
                    ], axis=1)

        return y, new_cache

    def _forward_varlen(
        self,
        x: jnp.ndarray,
        cu_seqlens: Any,
    ) -> jnp.ndarray:
        """Varlen path: per-segment causal convolution."""
        B, T_total, D = x.shape
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        segments = []
        for i in range(N):
            bos = int(cu_seqlens[i])
            eos = int(cu_seqlens[i + 1])
            seg = x[:, bos:eos, :]  # [1, seg_len, D]
            seg_out = self.conv(seg)
            segments.append(seg_out)
        return jnp.concatenate(segments, axis=1)

    def step(
        self,
        x: jnp.ndarray,
        cache: jnp.ndarray | None,
        output_final_state: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Single token step decoding.

        x: [B, 1, D]
        cache: [B, kernel_size-1, D] or None
        """
        W = self.kernel_size
        B = x.shape[0]
        D = self.hidden_size

        if cache is None and output_final_state:
            cache = jnp.zeros((B, W - 1, D), dtype=x.dtype)

        if cache is not None:
            x_full = jnp.concatenate([cache, x], axis=1)  # [B, W, D]
            y = self.conv(x_full)[:, -1:, :]  # [B, 1, D]
            new_cache = x_full[:, -(W - 1):, :] if output_final_state else None
        else:
            y = self.conv(x)
            new_cache = None

        y = self._apply_activation(y)
        return y, new_cache


# =============================================================================
# GLA core operation: naive recurrent (纯 JAX 实现, 使用 jax.lax.scan)
# =============================================================================

