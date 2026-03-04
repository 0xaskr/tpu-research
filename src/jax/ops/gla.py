import jax
import jax.numpy as jnp
from typing import Any, Optional

def naive_recurrent_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    cu_seqlens: Any = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Naive recurrent GLA — 纯 JAX 实现.

    核心递归:
        h_t = h_{t-1} * exp(gk_t) + k_t^T v_t
        o_t = scale * q_t @ h_t  (沿 K 维度求和)

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        gk: [B, T, H, K] — 门控 (log-space)
        scale: 缩放因子, 默认 K^{-0.5}
        initial_state: [N, H, K, V]
        output_final_state: 是否输出最终状态
        cu_seqlens: 变长序列累积长度

    Returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] 或 None
    """
    if cu_seqlens is not None:
        return _naive_recurrent_gla_varlen(
            q, k, v, gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # float32 计算
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    gk = gk.astype(jnp.float32)

    # 转为 [B, H, T, K/V]
    q = jnp.transpose(q, (0, 2, 1, 3))    # [B, H, T, K]
    k = jnp.transpose(k, (0, 2, 1, 3))    # [B, H, T, K]
    v = jnp.transpose(v, (0, 2, 1, 3))    # [B, H, T, V]
    gk = jnp.transpose(gk, (0, 2, 1, 3))  # [B, H, T, K]

    # 初始化状态
    if initial_state is not None:
        h_init = initial_state.astype(jnp.float32)  # [B, H, K, V]
    else:
        h_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    # 使用 jax.lax.scan 实现递归
    def scan_fn(h, t_inputs):
        """单步递归.
        h: [B, H, K, V]
        t_inputs: (q_t, k_t, v_t, gk_t) — 当前时间步
        """
        q_t, k_t, v_t, gk_t = t_inputs
        # q_t: [B, H, K], k_t: [B, H, K], v_t: [B, H, V], gk_t: [B, H, K]
        q_t = q_t * scale
        gk_t_exp = jnp.exp(gk_t)  # [B, H, K]
        # 外积: k_t^T v_t → [B, H, K, V]
        kv_t = k_t[..., None] * v_t[..., None, :]
        # 状态更新
        h = h * gk_t_exp[..., None] + kv_t
        # 输出
        o_t = (q_t[..., None] * h).sum(axis=-2)  # [B, H, V]
        return h, o_t

    # scan 沿时间维 axis=2 → moveaxis 到 leading axis
    scan_inputs = (
        jnp.moveaxis(q, 2, 0),    # [T, B, H, K]
        jnp.moveaxis(k, 2, 0),    # [T, B, H, K]
        jnp.moveaxis(v, 2, 0),    # [T, B, H, V]
        jnp.moveaxis(gk, 2, 0),   # [T, B, H, K]
    )

    h_final, o_scan = jax.lax.scan(scan_fn, h_init, scan_inputs)
    # o_scan: [T, B, H, V], h_final: [B, H, K, V]

    # 转回 [B, T, H, V]
    o = jnp.transpose(o_scan, (1, 0, 2, 3))  # [B, T, H, V]

    final_state = h_final if output_final_state else None
    return o, final_state


def _naive_recurrent_gla_varlen(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    cu_seqlens: Any = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """变长序列版 naive recurrent GLA."""
    B, T_total, H, K = q.shape
    V = v.shape[-1]
    assert B == 1, "cu_seqlens requires B=1"

    if scale is None:
        scale = K ** -0.5

    N = len(cu_seqlens) - 1
    o_segments = []
    final_states = []

    for i in range(N):
        bos = int(cu_seqlens[i])
        eos = int(cu_seqlens[i + 1])

        q_seg = q[:, bos:eos, :, :]
        k_seg = k[:, bos:eos, :, :]
        v_seg = v[:, bos:eos, :, :]
        gk_seg = gk[:, bos:eos, :, :]

        h0 = None
        if initial_state is not None:
            h0 = initial_state[i:i + 1]  # [1, H, K, V]

        o_seg, h_seg = naive_recurrent_gla(
            q_seg, k_seg, v_seg, gk_seg,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )
        o_segments.append(o_seg)
        if output_final_state:
            final_states.append(h_seg[0])  # [H, K, V]

    o = jnp.concatenate(o_segments, axis=1)  # [1, T_total, H, V]
    final_state = jnp.stack(final_states, axis=0) if output_final_state else None
    return o, final_state


# =============================================================================
# Reshape / repeat utilities (替代 einops 在 JAX 侧的功能)
# =============================================================================

