# =============================================================================
# Cross-implementation comparison: FLA (remote, CUDA) vs local CPU code
#
# FLA 远程库 (flash-linear-attention) 的 GLA 全模块 (ops / modules / layers)
# 在 CUDA 上运行, 与本地 src/torch/ 纯 CPU 实现的全面比对测试。
#
# 测试策略:
#   - Kernel 级: FLA naive/chunk/fused_recurrent (CUDA) vs local naive/chunk (CPU)
#   - Module 级: FLA RMSNorm/FusedRMSNormGated/ShortConvolution (CUDA) vs local (CPU)
#   - Layer 级:  FLA GatedLinearAttention (CUDA) vs local GatedLinearAttention (CPU)
#   - 独立正确性: local 模块/层的独立验证 + 梯度 + 稳定性
# =============================================================================
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- FLA remote library (CUDA) ---
from fla.ops.gla.naive import naive_recurrent_gla as fla_naive_recurrent_gla
from fla.ops.gla import chunk_gla as fla_chunk_gla
from fla.ops.gla import fused_recurrent_gla as fla_fused_recurrent_gla
from fla.layers.gla import GatedLinearAttention as FLA_GLA
from fla.modules import RMSNorm as FLA_RMSNorm
from fla.modules import FusedRMSNormGated as FLA_FNG
from fla.modules import ShortConvolution as FLA_ShortConv

# --- Local CPU implementations ---
from src.torch.ops.gla import (
    naive_recurrent_gla as local_naive_recurrent_gla,
    chunk_gla as local_chunk_gla,
    fused_chunk_gla as local_fused_chunk_gla,
)
from src.torch.layers.gla import GatedLinearAttention as LocalGLA
from src.torch.modules.convolution import ShortConvolution as LocalShortConv
from src.torch.modules.layernorm import RMSNorm as LocalRMSNorm
from src.torch.modules.fused_norm_gate import FusedRMSNormGated as LocalFNG
from src.torch.layers.utils import get_unpad_data, index_first_axis, pad_input

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Helpers
# =============================================================================

def compare(name: str, a: torch.Tensor, b: torch.Tensor,
            atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    a_np = a.detach().float().cpu().numpy()
    b_np = b.detach().float().cpu().numpy()
    diff = np.abs(a_np - b_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    ok = np.allclose(a_np, b_np, atol=atol, rtol=rtol)
    status = "✅" if ok else "❌"
    print(f"  {status} {name}: max={max_diff:.2e} mean={mean_diff:.2e} (atol={atol:.0e})")
    if not ok:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"      worst at {idx}: a={a_np[idx]:.8f} b={b_np[idx]:.8f}")
    return ok


def to_gpu(sd: dict) -> dict:
    return {k: v.to(DEVICE) for k, v in sd.items()}


def to_cpu(sd: dict) -> dict:
    return {k: v.cpu() for k, v in sd.items()}


# =============================================================================
# Category 1: FLA naive kernel (CPU) vs local naive kernel (CPU)
# =============================================================================

def test_kernel_basic() -> bool:
    """FLA naive vs local naive: basic shapes."""
    print("\n[Kernel] FLA vs local: basic (B=2, T=32, H=4, K=32, V=64)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_fla, _ = fla_naive_recurrent_gla(q, k, v, gk)
    o_local, _ = local_naive_recurrent_gla(q, k, v, gk)
    return compare("output", o_fla, o_local)


def test_kernel_large() -> bool:
    """FLA naive vs local naive: larger dims."""
    print("\n[Kernel] FLA vs local: large (B=1, T=128, H=2, K=64, V=128)")
    torch.manual_seed(7)
    B, T, H, K, V = 1, 128, 2, 64, 128
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_fla, _ = fla_naive_recurrent_gla(q, k, v, gk)
    o_local, _ = local_naive_recurrent_gla(q, k, v, gk)
    return compare("output", o_fla, o_local, atol=5e-5, rtol=5e-5)


def test_kernel_initial_state() -> bool:
    """FLA naive vs local naive: with initial state."""
    print("\n[Kernel] FLA vs local: initial + final state (B=2, T=64, H=4, K=32, V=64)")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_fla, s_fla = fla_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    o_local, s_local = local_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    ok = compare("output", o_fla, o_local)
    ok &= compare("final_state", s_fla, s_local)
    return ok


def test_kernel_various_shapes() -> bool:
    """FLA naive vs local naive: multiple shape combos."""
    print("\n[Kernel] FLA vs local: various shapes")
    torch.manual_seed(99)
    ok = True
    for B, T, H, K, V in [(1, 16, 1, 16, 16), (4, 64, 8, 32, 64), (1, 256, 2, 128, 128)]:
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        o_fla, _ = fla_naive_recurrent_gla(q, k, v, gk)
        o_local, _ = local_naive_recurrent_gla(q, k, v, gk)
        atol = 1e-4 if K > 64 or T > 128 else 1e-5
        ok &= compare(f"B={B} T={T} H={H} K={K} V={V}", o_fla, o_local, atol=atol, rtol=atol)
    return ok


def test_kernel_state_split() -> bool:
    """FLA naive vs local naive: split sequence consistency."""
    print("\n[Kernel] FLA vs local: state split (process in 2 halves)")
    torch.manual_seed(77)
    B, T, H, K, V = 1, 40, 2, 16, 32
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_full_fla, s_full_fla = fla_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    o_full_local, s_full_local = local_naive_recurrent_gla(q, k, v, gk, output_final_state=True)

    T1 = T // 2
    _, s1_fla = fla_naive_recurrent_gla(q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True)
    o2_fla, s2_fla = fla_naive_recurrent_gla(q[:, T1:], k[:, T1:], v[:, T1:], gk[:, T1:],
                                              initial_state=s1_fla, output_final_state=True)
    _, s1_local = local_naive_recurrent_gla(q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True)
    o2_local, s2_local = local_naive_recurrent_gla(q[:, T1:], k[:, T1:], v[:, T1:], gk[:, T1:],
                                                    initial_state=s1_local, output_final_state=True)

    ok = compare("full output (fla vs local)", o_full_fla, o_full_local)
    ok &= compare("full state (fla vs local)", s_full_fla, s_full_local)
    ok &= compare("split-2nd output (fla vs local)", o2_fla, o2_local)
    ok &= compare("split-2nd state (fla vs local)", s2_fla, s2_local)
    ok &= compare("fla: full vs split state", s_full_fla, s2_fla)
    ok &= compare("local: full vs split state", s_full_local, s2_local)
    return ok


# =============================================================================
# Category 2: FLA CUDA kernels vs local CPU kernels
# =============================================================================

def test_fla_chunk_cuda_vs_local() -> bool:
    """FLA chunk_gla (CUDA) vs local chunk_gla (CPU)."""
    print("\n[Kernel CUDA] FLA chunk_gla (GPU) vs local chunk_gla (CPU)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_fla, s_fla = fla_chunk_gla(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
                                  output_final_state=True)
    o_local, s_local = local_chunk_gla(q, k, v, gk, output_final_state=True)
    ok = compare("output", o_fla.cpu(), o_local, atol=2e-2, rtol=2e-2)
    ok &= compare("final_state", s_fla.cpu(), s_local, atol=2e-2, rtol=2e-2)
    return ok


def test_fla_fused_recurrent_cuda_vs_local() -> bool:
    """FLA fused_recurrent_gla (CUDA) vs local naive (CPU)."""
    print("\n[Kernel CUDA] FLA fused_recurrent_gla (GPU) vs local naive (CPU)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_fla, s_fla = fla_fused_recurrent_gla(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
                                            output_final_state=True)
    o_local, s_local = local_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    ok = compare("output", o_fla.cpu(), o_local, atol=1e-4, rtol=1e-4)
    ok &= compare("final_state", s_fla.cpu(), s_local, atol=1e-4, rtol=1e-4)
    return ok


def test_fla_chunk_cuda_vs_local_init_state() -> bool:
    """FLA chunk_gla (CUDA) vs local with initial state."""
    print("\n[Kernel CUDA] FLA chunk_gla vs local: initial state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_fla, s_fla = fla_chunk_gla(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
                                  initial_state=h0.to(DEVICE), output_final_state=True)
    o_local, s_local = local_chunk_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    ok = compare("output", o_fla.cpu(), o_local, atol=2e-2, rtol=2e-2)
    ok &= compare("final_state", s_fla.cpu(), s_local, atol=2e-2, rtol=2e-2)
    return ok


def test_fla_fused_recurrent_cuda_init_state() -> bool:
    """FLA fused_recurrent_gla (CUDA) vs local with initial state."""
    print("\n[Kernel CUDA] FLA fused_recurrent_gla vs local: initial state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_fla, s_fla = fla_fused_recurrent_gla(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
                                            initial_state=h0.to(DEVICE), output_final_state=True)
    o_local, s_local = local_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    ok = compare("output", o_fla.cpu(), o_local, atol=1e-4, rtol=1e-4)
    ok &= compare("final_state", s_fla.cpu(), s_local, atol=1e-4, rtol=1e-4)
    return ok


# =============================================================================
# Category 3: Local chunk_gla / fused_chunk_gla vs local naive (equivalence)
# =============================================================================

def test_chunk_vs_naive() -> bool:
    """Local chunk_gla vs local naive_recurrent_gla equivalence."""
    print("\n[chunk_gla] local chunk vs naive: basic (B=2, T=64, H=4, K=32, V=64)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_naive, s_naive = local_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    o_chunk, s_chunk = local_chunk_gla(q, k, v, gk, output_final_state=True)
    ok = compare("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
    ok &= compare("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)
    return ok


def test_chunk_vs_naive_init_state() -> bool:
    """Local chunk vs naive with initial state."""
    print("\n[chunk_gla] local chunk vs naive: initial+final state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_naive, s_naive = local_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    o_chunk, s_chunk = local_chunk_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    ok = compare("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
    ok &= compare("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)
    return ok


def test_chunk_vs_naive_varlen() -> bool:
    """Local chunk vs naive with cu_seqlens."""
    print("\n[chunk_gla] local chunk vs naive: varlen cu_seqlens")
    torch.manual_seed(7)
    H, K, V = 4, 32, 64
    T = 48
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    v = torch.randn(1, T, H, V)
    gk = F.logsigmoid(torch.randn(1, T, H, K))
    cu = torch.tensor([0, 16, 32, 48], dtype=torch.long)

    o_naive, _ = local_naive_recurrent_gla(q, k, v, gk, cu_seqlens=cu)
    o_chunk, _ = local_chunk_gla(q, k, v, gk, cu_seqlens=cu)
    return compare("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)


def test_fused_chunk_vs_naive() -> bool:
    """Local fused_chunk_gla vs naive equivalence."""
    print("\n[fused_chunk] local fused_chunk vs naive")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_naive, _ = local_naive_recurrent_gla(q, k, v, gk)
    o_fc, _ = local_fused_chunk_gla(q, k, v, gk)
    return compare("output", o_naive, o_fc, atol=5e-5, rtol=5e-5)


# =============================================================================
# Category 4: Local cu_seqlens consistency
# =============================================================================

def test_cu_seqlens_vs_separate() -> bool:
    """cu_seqlens packed == separate batch processing."""
    print("\n[cu_seqlens] packed vs separate batches")
    torch.manual_seed(123)
    H, K, V = 2, 16, 32
    s1_len, s2_len = 10, 14
    q1 = torch.randn(1, s1_len, H, K)
    k1 = torch.randn(1, s1_len, H, K)
    v1 = torch.randn(1, s1_len, H, V)
    g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
    q2 = torch.randn(1, s2_len, H, K)
    k2 = torch.randn(1, s2_len, H, K)
    v2 = torch.randn(1, s2_len, H, V)
    g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))

    o1, s1 = local_naive_recurrent_gla(q1, k1, v1, g1, output_final_state=True)
    o2, s2 = local_naive_recurrent_gla(q2, k2, v2, g2, output_final_state=True)

    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)
    o_cu, s_cu = local_naive_recurrent_gla(q_cat, k_cat, v_cat, g_cat, output_final_state=True, cu_seqlens=cu)

    ok = compare("seg1 output", o1, o_cu[:, :s1_len])
    ok &= compare("seg2 output", o2, o_cu[:, s1_len:])
    ok &= compare("seg1 state", s1.squeeze(0), s_cu[0])
    ok &= compare("seg2 state", s2.squeeze(0), s_cu[1])
    return ok


# =============================================================================
# Category 5: FLA modules (CUDA) vs local modules (CPU)
# =============================================================================

def test_rmsnorm_fla_vs_local() -> bool:
    """FLA RMSNorm (CUDA) vs local RMSNorm (CPU)."""
    print("\n[Module CUDA] FLA RMSNorm vs local RMSNorm")
    torch.manual_seed(0)
    dim = 64
    fla_norm = FLA_RMSNorm(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
    loc_norm = LocalRMSNorm(dim, elementwise_affine=True, eps=1e-5)
    loc_norm.load_state_dict(to_cpu(fla_norm.state_dict()))
    fla_norm.eval(); loc_norm.eval()

    x = torch.randn(2, 10, dim)
    with torch.no_grad():
        y_fla = fla_norm(x.to(DEVICE)).cpu()
        y_local = loc_norm(x)
    return compare("RMSNorm", y_fla, y_local, atol=1e-5, rtol=1e-5)


def test_fused_norm_gated_fla_vs_local() -> bool:
    """FLA FusedRMSNormGated (CUDA) vs local (CPU)."""
    print("\n[Module CUDA] FLA FusedRMSNormGated vs local")
    torch.manual_seed(0)
    dim = 64
    fla_fn = FLA_FNG(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
    loc_fn = LocalFNG(dim, elementwise_affine=True, eps=1e-5)
    loc_fn.load_state_dict(to_cpu(fla_fn.state_dict()))
    fla_fn.eval(); loc_fn.eval()

    x = torch.randn(2, 10, dim)
    g = torch.randn(2, 10, dim)
    with torch.no_grad():
        y_fla = fla_fn(x.to(DEVICE), g.to(DEVICE)).cpu()
        y_local = loc_fn(x, g)
    return compare("FusedRMSNormGated", y_fla, y_local, atol=1e-5, rtol=1e-5)


def test_short_conv_fla_vs_local() -> bool:
    """FLA ShortConvolution (CUDA) vs local (CPU)."""
    print("\n[Module CUDA] FLA ShortConvolution vs local")
    torch.manual_seed(42)
    fla_conv = FLA_ShortConv(hidden_size=32, kernel_size=4, bias=True, activation='silu').to(DEVICE)
    loc_conv = LocalShortConv(hidden_size=32, kernel_size=4, bias=True, activation='silu')
    loc_conv.load_state_dict(to_cpu(fla_conv.state_dict()))
    fla_conv.eval(); loc_conv.eval()

    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_fla, _ = fla_conv(x.to(DEVICE))
        y_local, _ = loc_conv(x)
    return compare("ShortConv output", y_fla.cpu(), y_local, atol=1e-5, rtol=1e-5)


def test_short_conv_fla_vs_local_cache() -> bool:
    """FLA ShortConvolution (CUDA) vs local (CPU): cache output."""
    print("\n[Module CUDA] FLA ShortConvolution vs local: cache")
    torch.manual_seed(42)
    fla_conv = FLA_ShortConv(hidden_size=16, kernel_size=4, bias=True, activation='silu').to(DEVICE)
    loc_conv = LocalShortConv(hidden_size=16, kernel_size=4, bias=True, activation='silu')
    loc_conv.load_state_dict(to_cpu(fla_conv.state_dict()))
    fla_conv.eval(); loc_conv.eval()

    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        y_fla, cache_fla = fla_conv(x.to(DEVICE), output_final_state=True)
        y_local, cache_local = loc_conv(x, output_final_state=True)
    ok = compare("output", y_fla.cpu(), y_local, atol=1e-5, rtol=1e-5)
    ok &= compare("cache", cache_fla.cpu(), cache_local, atol=1e-5, rtol=1e-5)
    return ok


# =============================================================================
# Category 6: Local modules standalone correctness
# =============================================================================

def test_rmsnorm_standalone() -> bool:
    """Local RMSNorm correctness."""
    print("\n[RMSNorm] standalone correctness")
    torch.manual_seed(0)
    dim = 64
    norm = LocalRMSNorm(dim, elementwise_affine=True, eps=1e-5)
    x = torch.randn(2, 10, dim)
    y = norm(x)
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_exp = (x_f * rms * norm.weight.float()).to(x.dtype)
    ok = compare("RMSNorm vs manual", y, y_exp, atol=1e-6, rtol=1e-6)
    norm2 = LocalRMSNorm(dim, elementwise_affine=False)
    y2 = norm2(x)
    assert y2.shape == x.shape, f"Shape mismatch: {y2.shape}"
    print("  ✅ no-affine shape OK")
    return ok


def test_fused_norm_gated_standalone() -> bool:
    """Local FusedRMSNormGated correctness."""
    print("\n[FusedRMSNormGated] standalone correctness")
    torch.manual_seed(0)
    dim = 64
    fnorm = LocalFNG(dim, elementwise_affine=True, eps=1e-5)
    x = torch.randn(2, 10, dim)
    g = torch.randn(2, 10, dim)
    y = fnorm(x, g)
    x_f, g_f = x.float(), g.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_exp = (x_f * rms * fnorm.weight.float() * F.silu(g_f)).to(x.dtype)
    return compare("FNG vs manual", y, y_exp, atol=1e-6, rtol=1e-6)


def test_short_conv_causal() -> bool:
    """Local ShortConvolution: causality property."""
    print("\n[ShortConv] causality verification")
    conv = LocalShortConv(hidden_size=8, kernel_size=3, bias=True, activation='silu')
    x1 = torch.randn(1, 16, 8)
    x2 = x1.clone()
    x2[:, 10:, :] = torch.randn(1, 6, 8)
    y1, _ = conv(x1)
    y2, _ = conv(x2)
    ok = torch.allclose(y1[:, :10], y2[:, :10], atol=1e-6)
    differs = not torch.allclose(y1[:, 10:], y2[:, 10:], atol=1e-6)
    if ok and differs:
        print("  ✅ causal: future changes don't affect past outputs")
    else:
        print("  ❌ causality violated")
    return ok and differs


def test_short_conv_cu_seqlens() -> bool:
    """Local ShortConvolution: cu_seqlens packed == separate."""
    print("\n[ShortConv] cu_seqlens vs separate")
    conv = LocalShortConv(hidden_size=8, kernel_size=3, bias=False, activation='silu')
    x_s1 = torch.randn(1, 6, 8)
    x_s2 = torch.randn(1, 10, 8)
    y_s1, _ = conv(x_s1)
    y_s2, _ = conv(x_s2)
    x_packed = torch.cat([x_s1, x_s2], dim=1)
    cu = torch.tensor([0, 6, 16], dtype=torch.long)
    y_packed, _ = conv(x_packed, cu_seqlens=cu)
    ok = compare("seg1", y_s1, y_packed[:, :6])
    ok &= compare("seg2", y_s2, y_packed[:, 6:])
    return ok


def test_short_conv_step() -> bool:
    """Local ShortConvolution: step (single-token decode with cache)."""
    print("\n[ShortConv] step decode with cache")
    conv = LocalShortConv(hidden_size=16, kernel_size=4, bias=True, activation='silu')
    x_pre = torch.randn(1, 8, 16)
    y_pre, cache = conv(x_pre, output_final_state=True)
    assert cache is not None and cache.shape == (1, 16, 4), f"Cache shape: {cache.shape}"
    x_dec = torch.randn(1, 1, 16)
    y_dec, cache_dec = conv.step(x_dec, cache, output_final_state=True)
    assert y_dec.shape == (1, 1, 16)
    assert cache_dec.shape == (1, 16, 4)
    print(f"  ✅ prefill cache: {cache.shape}, decode: {y_dec.shape}")
    return True


# =============================================================================
# Category 7: FLA GLA layer (CUDA) vs local GLA layer (CPU) — full forward
# =============================================================================

def test_fla_layer_vs_local_basic() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): basic forward."""
    print("\n[Layer CUDA] FLA vs local: basic (B=2, T=32, D=128, H=2)")
    torch.manual_seed(42)
    cfg = dict(hidden_size=128, num_heads=2, use_short_conv=False,
               use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


def test_fla_layer_vs_local_conv() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): with ShortConvolution."""
    print("\n[Layer CUDA] FLA vs local: with ShortConvolution")
    torch.manual_seed(42)
    cfg = dict(hidden_size=128, num_heads=2, use_short_conv=True, conv_size=4,
               use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


def test_fla_layer_vs_local_mqa() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): MQA config."""
    print("\n[Layer CUDA] FLA vs local: MQA (H=8, KV=2)")
    torch.manual_seed(42)
    cfg = dict(hidden_size=256, num_heads=8, num_kv_heads=2,
               use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(2, 64, 256)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


def test_fla_layer_vs_local_no_gate() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): no output gate."""
    print("\n[Layer CUDA] FLA vs local: no output gate")
    torch.manual_seed(42)
    cfg = dict(hidden_size=128, num_heads=2, use_output_gate=False,
               fuse_norm=False, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


def test_fla_layer_vs_local_expand() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): non-default expand_k/expand_v."""
    print("\n[Layer CUDA] FLA vs local: expand_k/expand_v combos")
    ok = True
    for ek, ev in [(1.0, 1.0), (0.25, 2.0)]:
        torch.manual_seed(42)
        cfg = dict(hidden_size=128, num_heads=4, expand_k=ek, expand_v=ev,
                   use_output_gate=True, fuse_norm=True, layer_idx=0)
        fla_m = FLA_GLA(**cfg).to(DEVICE)
        loc_m = LocalGLA(**cfg)
        loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
        fla_m.eval(); loc_m.eval()

        x = torch.randn(1, 32, 128)
        with torch.no_grad():
            o_fla = fla_m(x.to(DEVICE))[0].cpu()
            o_local = loc_m(x)[0]
        ok &= compare(f"ek={ek} ev={ev}", o_fla, o_local, atol=1e-4, rtol=1e-4)
    return ok


def test_fla_layer_vs_local_mask() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): attention_mask."""
    print("\n[Layer CUDA] FLA vs local: attention_mask")
    torch.manual_seed(42)
    cfg = dict(hidden_size=128, num_heads=2, use_output_gate=True,
               fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    B, T, D = 2, 32, 128
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -8:] = 0
    mask[1, -4:] = 0

    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE), attention_mask=mask.to(DEVICE))[0].cpu()
        o_local = loc_m(x, attention_mask=mask)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


def test_fla_layer_vs_local_long_seq() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): longer sequence (T=256)."""
    print("\n[Layer CUDA] FLA vs local: long seq (T=256)")
    torch.manual_seed(7)
    cfg = dict(hidden_size=128, num_heads=2, use_output_gate=True,
               fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(1, 256, 128)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=5e-4, rtol=5e-4)


def test_fla_layer_vs_local_clamp() -> bool:
    """FLA GLA layer (CUDA) vs local (CPU): clamp_min."""
    print("\n[Layer CUDA] FLA vs local: clamp_min=-1.0")
    torch.manual_seed(42)
    cfg = dict(hidden_size=128, num_heads=2, clamp_min=-1.0,
               use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg).to(DEVICE)
    loc_m = LocalGLA(**cfg)
    loc_m.load_state_dict(to_cpu(fla_m.state_dict()))
    fla_m.eval(); loc_m.eval()

    x = torch.randn(1, 32, 128)
    with torch.no_grad():
        o_fla = fla_m(x.to(DEVICE))[0].cpu()
        o_local = loc_m(x)[0]
    return compare("output", o_fla, o_local, atol=1e-4, rtol=1e-4)


# =============================================================================
# Category 8: FLA module architecture parity
# =============================================================================

def test_fla_module_state_dict_keys() -> bool:
    """FLA vs local: state_dict keys match across configs."""
    print("\n[FLA Arch] state_dict keys match across configs")
    configs = [
        dict(hidden_size=128, num_heads=2, use_short_conv=False,
             use_output_gate=True, fuse_norm=True, layer_idx=0),
        dict(hidden_size=128, num_heads=2, use_short_conv=True, conv_size=4,
             use_output_gate=True, fuse_norm=True, layer_idx=0),
        dict(hidden_size=128, num_heads=2, use_short_conv=False,
             use_output_gate=False, fuse_norm=False, layer_idx=0),
        dict(hidden_size=256, num_heads=8, num_kv_heads=2,
             use_output_gate=True, fuse_norm=True, layer_idx=0),
        dict(hidden_size=128, num_heads=4, expand_k=1.0, expand_v=2.0,
             use_output_gate=True, fuse_norm=True, layer_idx=0),
    ]
    ok = True
    for i, cfg in enumerate(configs):
        fla_m = FLA_GLA(**cfg)
        loc_m = LocalGLA(**cfg)
        fla_keys = sorted(fla_m.state_dict().keys())
        loc_keys = sorted(loc_m.state_dict().keys())
        match = fla_keys == loc_keys
        ok &= match
        label = ', '.join(f'{k}={v}' for k, v in cfg.items() if k != 'layer_idx')
        print(f"  {'✅' if match else '❌'} cfg{i}({label})")
        if not match:
            print(f"      FLA only: {set(fla_keys) - set(loc_keys)}")
            print(f"      Local only: {set(loc_keys) - set(fla_keys)}")
    return ok


def test_fla_module_weight_transfer() -> bool:
    """FLA <-> local: bidirectional weight transfer."""
    print("\n[FLA Arch] bidirectional weight transfer")
    torch.manual_seed(100)
    cfg = dict(hidden_size=256, num_heads=4, use_short_conv=True, conv_size=4,
               use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg)
    loc_m = LocalGLA(**cfg)

    loc_m.load_state_dict(fla_m.state_dict())
    for k in fla_m.state_dict():
        assert torch.equal(fla_m.state_dict()[k], loc_m.state_dict()[k])
    print("  ✅ FLA -> Local: all params identical")

    loc_m2 = LocalGLA(**cfg)
    fla_m.load_state_dict(loc_m2.state_dict())
    for k in loc_m2.state_dict():
        assert torch.equal(fla_m.state_dict()[k], loc_m2.state_dict()[k])
    print("  ✅ Local -> FLA: all params identical")
    return True


def test_fla_module_attributes() -> bool:
    """FLA vs local: key attributes/config match."""
    print("\n[FLA Arch] attribute parity")
    cfg = dict(hidden_size=256, num_heads=8, num_kv_heads=2,
               expand_k=0.5, expand_v=1.0, gate_logit_normalizer=16,
               clamp_min=-1.0, use_output_gate=True, fuse_norm=True, layer_idx=0)
    fla_m = FLA_GLA(**cfg)
    loc_m = LocalGLA(**cfg)
    attrs = [
        'hidden_size', 'num_heads', 'num_kv_heads', 'num_kv_groups',
        'key_dim', 'value_dim', 'head_k_dim', 'head_v_dim',
        'key_dim_per_group', 'value_dim_per_group',
        'expand_k', 'expand_v', 'gate_logit_normalizer',
        'clamp_min', 'use_output_gate', 'use_short_conv',
        'fuse_norm_and_gate',
    ]
    ok = True
    for attr in attrs:
        fla_val = getattr(fla_m, attr, 'MISSING')
        loc_val = getattr(loc_m, attr, 'MISSING')
        match = fla_val == loc_val
        ok &= match
        if not match:
            print(f"  ❌ {attr}: FLA={fla_val} vs Local={loc_val}")
    if ok:
        print(f"  ✅ all {len(attrs)} attributes match")
    return ok


def test_fla_module_param_count() -> bool:
    """FLA vs local: parameter count match."""
    print("\n[FLA Arch] parameter count match")
    configs = [
        dict(hidden_size=256, num_heads=4, layer_idx=0),
        dict(hidden_size=256, num_heads=4, use_short_conv=True, conv_size=4, layer_idx=0),
        dict(hidden_size=256, num_heads=8, num_kv_heads=2, layer_idx=0),
        dict(hidden_size=512, num_heads=8, expand_k=1.0, expand_v=2.0, layer_idx=0),
    ]
    ok = True
    for i, cfg in enumerate(configs):
        fla_m = FLA_GLA(**cfg)
        loc_m = LocalGLA(**cfg)
        fla_n = sum(p.numel() for p in fla_m.parameters())
        loc_n = sum(p.numel() for p in loc_m.parameters())
        match = fla_n == loc_n
        ok &= match
        print(f"  {'✅' if match else '❌'} cfg{i}: FLA={fla_n:,} Local={loc_n:,} params")
    return ok


# =============================================================================
# Category 9: Local layer standalone tests
# =============================================================================

def test_layer_basic() -> bool:
    """Local GLA layer: basic forward."""
    print("\n[Layer] basic forward (B=2, T=32, H=4, D=256)")
    torch.manual_seed(42)
    model = LocalGLA(mode='chunk', hidden_size=256, num_heads=4,
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    model.eval()
    x = torch.randn(2, 32, 256)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  ✅ output shape: {o.shape}")
    return True


def test_layer_feature_map() -> bool:
    """Local GLA layer: feature_map='relu'."""
    print("\n[Layer] feature_map=relu")
    model = LocalGLA(hidden_size=128, num_heads=2, feature_map='relu',
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    model.eval()
    x = torch.randn(1, 16, 128)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  ✅ output shape: {o.shape}")
    return True


def test_layer_normalizer() -> bool:
    """Different gate_logit_normalizer values produce different outputs."""
    print("\n[Layer] gate_logit_normalizer effect")
    torch.manual_seed(200)
    m16 = LocalGLA(hidden_size=128, num_heads=2, gate_logit_normalizer=16,
                   use_output_gate=True, fuse_norm=True, layer_idx=0)
    m1 = LocalGLA(hidden_size=128, num_heads=2, gate_logit_normalizer=1,
                  use_output_gate=True, fuse_norm=True, layer_idx=0)
    m1.load_state_dict(m16.state_dict())
    x = torch.randn(1, 16, 128)
    with torch.no_grad():
        o16, _, _ = m16(x)
        o1, _, _ = m1(x)
    ok = not torch.allclose(o16, o1, atol=1e-3)
    print(f"  {'✅' if ok else '❌'} different normalizers -> different outputs")
    return ok


def test_layer_seq1() -> bool:
    """Local GLA layer: single token (T=1)."""
    print("\n[Layer] T=1 single token")
    model = LocalGLA(mode='chunk', hidden_size=256, num_heads=4,
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    model.eval()
    x = torch.randn(2, 1, 256)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  ✅ output shape: {o.shape}")
    return True


# =============================================================================
# Category 10: Gradient tests
# =============================================================================

def test_grad_basic() -> bool:
    """Gradient backward pass."""
    print("\n[Grad] basic backward")
    model = LocalGLA(mode='chunk', hidden_size=256, num_heads=4,
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    n_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    ok = n_grads == n_params
    print(f"  {'✅' if ok else '❌'} {n_grads}/{n_params} params have gradients")
    return ok


def test_grad_conv() -> bool:
    """Gradient backward with ShortConvolution."""
    print("\n[Grad] backward with ShortConvolution")
    model = LocalGLA(mode='chunk', hidden_size=256, num_heads=4,
                     use_short_conv=True, conv_size=4,
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    ok = x.grad is not None
    print(f"  {'✅' if ok else '❌'} input grad computed")
    return ok


def test_grad_mqa() -> bool:
    """Gradient backward with MQA."""
    print("\n[Grad] backward with MQA")
    model = LocalGLA(mode='chunk', hidden_size=256, num_heads=8, num_kv_heads=2,
                     use_output_gate=True, fuse_norm=True, layer_idx=0)
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    ok = x.grad is not None
    print(f"  {'✅' if ok else '❌'} input grad computed")
    return ok


# =============================================================================
# Category 11: Numerical stability & determinism
# =============================================================================

def test_numerical_stability() -> bool:
    """No inf/nan with large/small inputs."""
    print("\n[Stability] large/small inputs")
    model = LocalGLA(hidden_size=128, num_heads=2, use_output_gate=True,
                     fuse_norm=True, layer_idx=0)
    model.eval()
    ok = True
    for scale, label in [(10.0, "large"), (0.001, "small")]:
        x = torch.randn(1, 16, 128) * scale
        with torch.no_grad():
            o, _, _ = model(x)
        finite = torch.isfinite(o).all().item()
        ok &= finite
        print(f"  {'✅' if finite else '❌'} {label} inputs: finite={finite}")
    return ok


def test_determinism() -> bool:
    """Same seed -> same output."""
    print("\n[Determinism] reproducibility")
    torch.manual_seed(42)
    m1 = LocalGLA(hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0)
    x1 = torch.randn(1, 16, 128)
    with torch.no_grad():
        o1, _, _ = m1(x1)
    torch.manual_seed(42)
    m2 = LocalGLA(hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0)
    x2 = torch.randn(1, 16, 128)
    with torch.no_grad():
        o2, _, _ = m2(x2)
    ok = torch.allclose(o1, o2, atol=1e-7)
    print(f"  {'✅' if ok else '❌'} deterministic outputs")
    return ok


# =============================================================================
# Category 12: Unpad/pad roundtrip
# =============================================================================

def test_unpad_pad_roundtrip() -> bool:
    """get_unpad_data / index_first_axis / pad_input roundtrip."""
    print("\n[Unpad/Pad] roundtrip")
    B, T, D = 3, 20, 32
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 15:] = 0
    mask[1, 18:] = 0
    mask[2, 10:] = 0
    indices, cu, max_len = get_unpad_data(mask)
    x = torch.randn(B, T, D)
    x_flat = rearrange(x, 'b s d -> (b s) d')
    x_packed = index_first_axis(x_flat, indices)
    x_restored = pad_input(x_packed, indices, B, T)

    ok = True
    for b in range(B):
        vl = mask[b].sum().item()
        ok &= torch.allclose(x[b, :vl], x_restored[b, :vl], atol=1e-7)
    ok &= (x_restored[0, 15:] == 0).all().item()
    ok &= (x_restored[2, 10:] == 0).all().item()
    print(f"  {'✅' if ok else '❌'} packed {indices.shape[0]} tokens, restored correctly")
    return ok


# =============================================================================
# Main
# =============================================================================

def main() -> bool:
    print("=" * 70)
    print(f"FLA (remote, CUDA={DEVICE}) vs Local CPU — GLA Full Tests")
    print("=" * 70)

    test_cases = [
        # Category 1: FLA naive kernel vs local naive kernel (CPU)
        ("FLA naive vs local: basic", test_kernel_basic),
        ("FLA naive vs local: large dims", test_kernel_large),
        ("FLA naive vs local: initial state", test_kernel_initial_state),
        ("FLA naive vs local: various shapes", test_kernel_various_shapes),
        ("FLA naive vs local: state split", test_kernel_state_split),
        # Category 2: FLA CUDA kernels vs local CPU kernels
        ("FLA chunk_gla CUDA vs local", test_fla_chunk_cuda_vs_local),
        ("FLA fused_recurrent CUDA vs local", test_fla_fused_recurrent_cuda_vs_local),
        ("FLA chunk_gla CUDA init state", test_fla_chunk_cuda_vs_local_init_state),
        ("FLA fused_recurrent CUDA init state", test_fla_fused_recurrent_cuda_init_state),
        # Category 3: Local chunk_gla equivalence
        ("chunk vs naive", test_chunk_vs_naive),
        ("chunk vs naive: init state", test_chunk_vs_naive_init_state),
        ("chunk vs naive: varlen", test_chunk_vs_naive_varlen),
        ("fused_chunk vs naive", test_fused_chunk_vs_naive),
        # Category 4: cu_seqlens
        ("cu_seqlens vs separate", test_cu_seqlens_vs_separate),
        # Category 5: FLA modules (CUDA) vs local modules (CPU)
        ("FLA RMSNorm CUDA vs local", test_rmsnorm_fla_vs_local),
        ("FLA FusedRMSNormGated CUDA vs local", test_fused_norm_gated_fla_vs_local),
        ("FLA ShortConv CUDA vs local", test_short_conv_fla_vs_local),
        ("FLA ShortConv CUDA cache", test_short_conv_fla_vs_local_cache),
        # Category 6: Local modules standalone
        ("RMSNorm standalone", test_rmsnorm_standalone),
        ("FusedRMSNormGated standalone", test_fused_norm_gated_standalone),
        ("ShortConv causal", test_short_conv_causal),
        ("ShortConv cu_seqlens", test_short_conv_cu_seqlens),
        ("ShortConv step", test_short_conv_step),
        # Category 7: FLA GLA layer (CUDA) vs local (CPU)
        ("FLA layer CUDA vs local: basic", test_fla_layer_vs_local_basic),
        ("FLA layer CUDA vs local: conv", test_fla_layer_vs_local_conv),
        ("FLA layer CUDA vs local: MQA", test_fla_layer_vs_local_mqa),
        ("FLA layer CUDA vs local: no gate", test_fla_layer_vs_local_no_gate),
        ("FLA layer CUDA vs local: expand", test_fla_layer_vs_local_expand),
        ("FLA layer CUDA vs local: mask", test_fla_layer_vs_local_mask),
        ("FLA layer CUDA vs local: long seq", test_fla_layer_vs_local_long_seq),
        ("FLA layer CUDA vs local: clamp", test_fla_layer_vs_local_clamp),
        # Category 8: FLA module architecture parity
        ("FLA arch: state_dict keys", test_fla_module_state_dict_keys),
        ("FLA arch: weight transfer", test_fla_module_weight_transfer),
        ("FLA arch: attributes", test_fla_module_attributes),
        ("FLA arch: param count", test_fla_module_param_count),
        # Category 9: Local layer standalone
        ("Layer basic", test_layer_basic),
        ("Layer feature_map", test_layer_feature_map),
        ("Layer normalizer", test_layer_normalizer),
        ("Layer T=1", test_layer_seq1),
        # Category 10: Gradients
        ("Grad basic", test_grad_basic),
        ("Grad conv", test_grad_conv),
        ("Grad MQA", test_grad_mqa),
        # Category 11: Stability
        ("Numerical stability", test_numerical_stability),
        ("Determinism", test_determinism),
        # Category 12: Utilities
        ("Unpad/pad roundtrip", test_unpad_pad_roundtrip),
    ]

    passed = 0
    total = len(test_cases)

    for i, (name, fn) in enumerate(test_cases):
        print(f"\n{'#' * 70}")
        print(f"Test {i + 1}/{total}: {name}")
        print(f"{'#' * 70}")
        try:
            if fn():
                passed += 1
            else:
                print(f"  >>> FAILED")
        except Exception:
            print(f"  ❌ Exception:")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Summary: {passed}/{total} passed")
    print(f"{'=' * 70}")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {total - passed} test(s) FAILED")
    return passed == total


if __name__ == '__main__':
    exit(0 if main() else 1)
