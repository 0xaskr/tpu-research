import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked GLA — 纯 PyTorch CPU 实现.

    将序列分为大小为 chunk_size 的块并行计算, 块间通过隐状态传播.
    与 naive_recurrent_gla 数学等价, 但利用块内并行性.

    注意: 门控参数名为 ``g`` (与 FLA chunk_gla API 一致),
    而非 fused_recurrent_gla 的 ``gk``.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — 门控 (log-space, logsigmoid 之后)
        scale: 缩放因子, 默认 K^{-0.5}
        initial_state: [N, H, K, V]
        output_final_state: 是否输出最终状态
        cu_seqlens: [N+1] 变长序列累积长度
        chunk_size: 块大小, 默认 16

    Returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] 或 None
    """
    dtype = q.dtype
    # transpose: [B, T, H, K/V] → [B, H, T, K/V], float32 计算
    q, k, v, g = (x.transpose(1, 2).float() for x in (q, k, v, g))
    B, H, T, K = q.shape   # q: [B, H, T, K]
    V = v.shape[-1]         # v: [B, H, T, V]

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is not None:
        # 变长序列: 按 cu_seqlens 分段, 每段独立做 chunk
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = torch.zeros_like(v)  # [1, H, T_total, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            h0 = initial_state[i:i + 1] if initial_state is not None else None

            o_seg, h_seg = _chunk_gla_inner(
                q[:, :, bos:eos], k[:, :, bos:eos],
                v[:, :, bos:eos], g[:, :, bos:eos],
                scale, h0, chunk_size,
            )
            o[:, :, bos:eos] = o_seg
            if output_final_state:
                final_states.append(h_seg.squeeze(0))  # [H, K, V]

        final_state = torch.stack(final_states, dim=0) if output_final_state else None  # [N, H, K, V]
        return o.transpose(1, 2).to(dtype), final_state  # o: [B, T, H, V]
    else:
        o, h_final = _chunk_gla_inner(q, k, v, g, scale, initial_state, chunk_size)
        final_state = h_final if output_final_state else None  # [B, H, K, V] 或 None
        return o.transpose(1, 2).to(dtype), final_state  # o: [B, T, H, V]


# =============================================================================
# chunk_gla: 分块 GLA (纯 PyTorch, 与 naive_recurrent_gla 数学等价)
# =============================================================================

def _chunk_gla_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分块 GLA 内部实现. 输入已 transpose 为 [B, H, T, K/V] float32.

    将序列分为大小为 chunk_size 的块, 每块内部用并行注意力计算,
    块间用隐状态传播. 与逐步递归数学等价.

    算法:
        对于块 c (包含 C 个 token):
        G_c[t] = cumsum(g_chunk, dim=t)  (块内累积门控)
        inter-chunk: o_inter = scale * (q * exp(G)) @ h
        intra-chunk: A = (q * exp(G)) @ (k * exp(-G))^T  (因果掩码)
                     o_intra = scale * A @ v
        state update: h = h * exp(G_total) + sum_j k_j*exp(G_total-G_j) ⊗ v_j
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = (T + C - 1) // C
    T_padded = NT * C

    # 补齐到 chunk_size 的整数倍
    if T_padded > T:
        pad = T_padded - T
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))

    # 重塑为 [B, H, NT, C, K/V]
    q = q.view(B, H, NT, C, K)
    k = k.view(B, H, NT, C, K)
    v = v.view(B, H, NT, C, V)
    g = g.view(B, H, NT, C, K)

    # 块内累积门控
    g_cumsum = g.cumsum(dim=3)  # [B, H, NT, C, K]

    # 初始化隐状态
    h = q.new_zeros(B, H, K, V, dtype=torch.float32)  # [B, H, K, V]
    if initial_state is not None:
        h = h + initial_state.float()                  # [B, H, K, V]

    # 因果掩码 [C, C]
    causal_mask = torch.tril(
        torch.ones(C, C, device=q.device, dtype=torch.bool))

    outputs = []
    for c_idx in range(NT):
        q_c = q[:, :, c_idx]       # [B, H, C, K]
        k_c = k[:, :, c_idx]       # [B, H, C, K]
        v_c = v[:, :, c_idx]       # [B, H, C, V]
        gc = g_cumsum[:, :, c_idx]  # [B, H, C, K]

        # 门控 Q/K
        q_gated = q_c * gc.exp()      # [B, H, C, K]
        k_gated = k_c * (-gc).exp()   # [B, H, C, K]

        # 块间贡献: o_inter = scale * q_gated @ h
        o_inter = scale * torch.einsum('bhck,bhkv->bhcv', q_gated, h)  # [B, H, C, V]

        # 块内注意力: A = q_gated @ k_gated^T, 因果掩码后乘 V
        A = torch.einsum('bhik,bhjk->bhij', q_gated, k_gated)  # [B, H, C, C]
        A = A.masked_fill(~causal_mask, 0.0)
        o_intra = scale * torch.einsum('bhij,bhjv->bhiv', A, v_c)  # [B, H, C, V]

        outputs.append(o_inter + o_intra)  # [B, H, C, V]

        # 状态更新: h = h * exp(G_total) + Σ k_j·exp(G_total-G_j) ⊗ v_j
        g_total = gc[:, :, -1, :]  # [B, H, K]
        h = h * g_total.unsqueeze(-1).exp()                     # [B, H, K, V]
        k_state = k_c * (g_total.unsqueeze(2) - gc).exp()       # [B, H, C, K]
        h = h + torch.einsum('bhck,bhcv->bhkv', k_state, v_c)   # [B, H, K, V]

    # 合并输出, 裁剪 padding
    o = torch.stack(outputs, dim=2).reshape(B, H, T_padded, V)[:, :, :T, :]  # [B, H, T, V]
    return o, h  # o: [B, H, T, V], h: [B, H, K, V]



# =============================================================================
# fused_chunk_gla: 融合分块 GLA (委托给 chunk_gla)
# =============================================================================

def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused chunk GLA — 纯 PyTorch CPU 实现.

    与 chunk_gla 相同算法 (在 CPU 上融合/非融合无区别).
    原始 FLA 中此函数已废弃, 建议使用 chunk_gla.

    参数同 chunk_gla (参见 chunk_gla 文档).
    """
    return chunk_gla(
        q=q, k=k, v=v, g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

# =============================================================================
# GLA core operation: naive recurrent (替代 Triton 版 chunk_gla / fused_recurrent_gla)
# 核心 GLA 递归操作，纯 PyTorch 实现
# =============================================================================

def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent GLA — 纯 PyTorch CPU 实现.

    数学等价于 chunk_gla / fused_recurrent_gla / fused_chunk_gla,
    但使用朴素逐步递归代替 Triton 分块或融合内核。

    核心递归:
        h_t = h_{t-1} * exp(gk_t) + k_t^T v_t
        o_t = q_t * h_t  (然后沿 K 维度求和)

    Args:
        q: [B, T, H, K] — 查询
        k: [B, T, H, K] — 键
        v: [B, T, H, V] — 值
        gk: [B, T, H, K] — 门控 (log-space, 即 logsigmoid 之后的值)
        scale: 缩放因子, 默认 K^{-0.5}
        initial_state: [N, H, K, V] — 初始状态
        output_final_state: 是否输出最终状态
        cu_seqlens: [N+1] — 变长序列累积长度 (此时 B 必须为 1)

    Returns:
        o: [B, T, H, V] — 输出
        final_state: [N, H, K, V] 或 None
    """
    dtype = q.dtype
    # transpose: [B, T, H, K/V] → [B, H, T, K/V], float32 计算
    q, k, v, gk = (x.transpose(1, 2).float() for x in (q, k, v, gk))
    B, H, T_total, K = q.shape    # q: [B, H, T, K]
    V = v.shape[-1]               # v: [B, H, T, V]

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is not None:
        # 变长序列模式: B=1, 按 cu_seqlens 分段独立递归
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = torch.zeros_like(v)  # [1, H, T_total, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            seg_len = eos - bos

            # 提取本段数据 [1, H, seg_len, K/V]
            q_seg = q[:, :, bos:eos, :]     # [1, H, seg_len, K]
            k_seg = k[:, :, bos:eos, :]     # [1, H, seg_len, K]
            v_seg = v[:, :, bos:eos, :]     # [1, H, seg_len, V]
            gk_seg = gk[:, :, bos:eos, :]   # [1, H, seg_len, K]

            # 初始状态
            h = q.new_zeros(1, H, K, V, dtype=torch.float32)  # [1, H, K, V]
            if initial_state is not None:
                h = h + initial_state[i:i+1].float()           # [1, H, K, V]

            for t in range(seg_len):
                q_t = q_seg[:, :, t] * scale   # [1, H, K]
                k_t = k_seg[:, :, t]           # [1, H, K]
                v_t = v_seg[:, :, t]           # [1, H, V]
                gk_t = gk_seg[:, :, t].exp()   # [1, H, K]
                kv_t = k_t[..., None] * v_t[..., None, :]  # [1, H, K, V]
                h = h * gk_t[..., None] + kv_t             # [1, H, K, V]
                o[:, :, bos + t] = (q_t[..., None] * h).sum(-2)  # [1, H, V]

            if output_final_state:
                final_states.append(h.squeeze(0))  # [H, K, V]

        final_state = torch.stack(final_states, dim=0) if output_final_state else None  # [N, H, K, V]
        return o.transpose(1, 2).to(dtype), final_state  # o: [B, T, H, V]
    else:
        # 标准 batch 模式
        o = torch.zeros_like(v)  # [B, H, T, V]
        h = q.new_zeros(B, H, K, V, dtype=torch.float32)  # [B, H, K, V]
        if initial_state is not None:
            h = h + initial_state.float()                  # [B, H, K, V]

        for t in range(T_total):
            q_t = q[:, :, t] * scale     # [B, H, K]
            k_t = k[:, :, t]             # [B, H, K]
            v_t = v[:, :, t]             # [B, H, V]
            gk_t = gk[:, :, t].exp()     # [B, H, K]
            kv_t = k_t[..., None] * v_t[..., None, :]  # [B, H, K, V]
            h = h * gk_t[..., None] + kv_t              # [B, H, K, V]
            o[:, :, t] = (q_t[..., None] * h).sum(-2)   # [B, H, V]

        final_state = h if output_final_state else None   # [B, H, K, V] 或 None
        return o.transpose(1, 2).to(dtype), final_state   # o: [B, T, H, V]


def fused_recurrent_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    Pure PyTorch CPU implementation of fused_recurrent_gla_fwd.

    Mirrors the Triton kernel recurrence in fla/ops/common/fused_recurrent.py.

    Recurrence per time step t (or reversed when reverse=True):
        h_t = decay(h_{t-1}) + k_t ⊗ v_t
        o_t = q_t · h_t

    Decay gates (all in log domain, kernel applies exp internally):
        g        : [B, T, H]    scalar log-gate per (batch, time, head)
        g_gamma  : [H]          per-head constant log-gate (same every step)
        gk       : [B, T, H, K] key-wise log-gate, applied as exp(gk)[:, None]
        gv       : [B, T, H, V] value-wise log-gate, applied as exp(gv)[None, :]

    Args:
        q:             [B, T, H, K] queries
        k:             [B, T, H, K] keys
        v:             [B, T, H, V] values
        g:             [B, T, H]    scalar log-gate (optional)
        g_gamma:       [H]          per-head constant log-gate (optional)
        gk:            [B, T, H, K] key-wise log-gate (optional)
        gv:            [B, T, H, V] value-wise log-gate (optional)
        scale:         scalar, default K^-0.5
        initial_state: [N, H, K, V] initial hidden states
        output_final_state: whether to return final hidden state [N, H, K, V]
        reverse:       if True, iterate time steps from T-1 to 0
        cu_seqlens:    [N+1] cumulative sequence lengths (varlen mode, requires B=1)

    Returns:
        o:  [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    USE_G       = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK      = gk is not None
    USE_GV      = gv is not None

    # All accumulation in float32, matching the Triton kernel's tl.float32 accumulators
    q_f        = q.float().cpu()
    k_f        = k.float().cpu()
    v_f        = v.float().cpu()
    g_f        = g.float().cpu()        if USE_G       else None
    g_gamma_f  = g_gamma.float().cpu() if USE_G_GAMMA else None
    gk_f       = gk.float().cpu()      if USE_GK      else None
    gv_f       = gv.float().cpu()      if USE_GV      else None

    o = torch.zeros(B, T, H, V, dtype=torch.float32)

    ht_list = []

    def _run_seq(batch_idx, bos, seq_len):
        """Run recurrence for one sequence, return final hidden state [H, K, V]."""
        if initial_state is not None:
            h = initial_state[batch_idx].clone().float().cpu()  # [H, K, V]
        else:
            h = torch.zeros(H, K, V, dtype=torch.float32)

        time_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for i_t in time_range:
            t_idx = bos + i_t

            # b_idx is always 0 in varlen mode (B=1), otherwise the batch index
            b = 0 if cu_seqlens is not None else batch_idx

            q_t = q_f[b, t_idx] * scale  # [H, K]
            k_t = k_f[b, t_idx]           # [H, K]
            v_t = v_f[b, t_idx]           # [H, V]

            # Apply log-gates to h: [H, K, V]
            if USE_G:
                # g[b, t_idx] -> [H]; broadcast to [H, 1, 1]
                h = h * torch.exp(g_f[b, t_idx])[:, None, None]
            if USE_G_GAMMA:
                # g_gamma -> [H]; broadcast to [H, 1, 1]
                h = h * torch.exp(g_gamma_f)[:, None, None]
            if USE_GK:
                # gk[b, t_idx] -> [H, K]; reshape to [H, K, 1]
                h = h * torch.exp(gk_f[b, t_idx])[:, :, None]
            if USE_GV:
                # gv[b, t_idx] -> [H, V]; reshape to [H, 1, V]
                h = h * torch.exp(gv_f[b, t_idx])[:, None, :]

            # h += k ⊗ v  (outer product per head: [H, K] x [H, V] -> [H, K, V])
            h = h + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # o = sum_K(h * q[:, :, None])  -> [H, V]
            b_out = 0 if cu_seqlens is not None else batch_idx
            o[b_out, t_idx] = (h * q_t.unsqueeze(-1)).sum(1)

        return h

    if cu_seqlens is not None:
        # Varlen mode: B must be 1, sequences are packed into dim-1
        N = len(cu_seqlens) - 1
        for i_n in range(N):
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            h_final = _run_seq(i_n, bos, eos - bos)
            if output_final_state:
                ht_list.append(h_final)
    else:
        for i_n in range(B):
            h_final = _run_seq(i_n, 0, T)
            if output_final_state:
                ht_list.append(h_final)

    ht = torch.stack(ht_list, dim=0) if output_final_state else None
    return o, ht
