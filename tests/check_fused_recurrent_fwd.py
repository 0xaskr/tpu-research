import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import traceback

import torch
import torch.nn.functional as F
from fla.ops.common.fused_recurrent import fused_recurrent_fwd as fla_fused_recurrent_fwd_triton_gpu


def fused_recurrent_fwd_torch_cpu(
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
    Pure PyTorch CPU implementation of fused_recurrent_fwd.

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


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_triton_vs_torch(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    cu_seqlens_list: list | None = None,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    reverse: bool = False,
    use_g: bool = False,
    use_gk: bool = False,
    use_gv: bool = False,
    use_g_gamma: bool = False,
    scale: float | None = None,
    gate_mode: str = 'logsigmoid',
):
    """Compare FLA Triton GPU fused_recurrent_fwd with PyTorch CPU reference.

    gate_mode controls how gate tensors are initialised:
      'logsigmoid'  : F.logsigmoid(randn), values in (-inf, 0]  (default)
      'zero'        : all zeros → exp(0)=1, no decay
      'strong_decay': all -5.0  → exp(-5)≈0.007, near-complete reset per step
      'positive'    : all +0.3  → exp(0.3)≈1.35, growing state (use short T)
    """
    print(f"\n=============================================")
    print(f"Testing fused_recurrent_fwd")
    print(f"B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"cu_seqlens={cu_seqlens_list}")
    print(f"gates: g={use_g}, gk={use_gk}, gv={use_gv}, g_gamma={use_g_gamma}  gate_mode={gate_mode!r}")
    print(f"init_state={use_initial_state}, final_state={output_final_state}, reverse={reverse}, scale={scale}")
    print(f"=============================================")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    device_gpu = "cuda"
    dtype = torch.float32

    # ---------- cu_seqlens bookkeeping ----------
    if cu_seqlens_list is not None:
        assert B == 1, "varlen mode requires B=1"
        cu_seqlens = torch.LongTensor(cu_seqlens_list)
        N = len(cu_seqlens) - 1
        total_T = cu_seqlens[-1].item()
        assert total_T == T, f"T must equal cu_seqlens[-1]={total_T}"
    else:
        cu_seqlens = None
        N = B

    # ---------- generate inputs ----------
    torch.manual_seed(42)
    q_gpu  = torch.randn(B, T, H, K, device=device_gpu, dtype=dtype)
    k_gpu  = torch.randn(B, T, H, K, device=device_gpu, dtype=dtype)
    v_gpu  = torch.randn(B, T, H, V, device=device_gpu, dtype=dtype)

    # Gate generation dispatch
    def _make_gate(shape):
        if gate_mode == 'logsigmoid':
            return F.logsigmoid(torch.randn(shape, device=device_gpu, dtype=dtype))
        elif gate_mode == 'zero':
            return torch.zeros(shape, device=device_gpu, dtype=dtype)
        elif gate_mode == 'strong_decay':
            return torch.full(shape, -5.0, device=device_gpu, dtype=dtype)
        elif gate_mode == 'positive':
            return torch.full(shape, 0.3, device=device_gpu, dtype=dtype)
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode!r}")

    # Gates are in log-domain; the kernel applies exp() internally.
    g_gpu        = _make_gate((B, T, H))     if use_g       else None
    g_gamma_gpu  = _make_gate((H,))          if use_g_gamma else None
    gk_gpu       = _make_gate((B, T, H, K))  if use_gk      else None
    gv_gpu       = _make_gate((B, T, H, V))  if use_gv      else None

    initial_state_gpu = torch.randn(N, H, K, V, device=device_gpu, dtype=dtype) if use_initial_state else None

    scale = scale if scale is not None else K ** -0.5

    # ---------- run Triton GPU ----------
    print("Running FLA Triton GPU version...")
    gpu_o, gpu_ht = fla_fused_recurrent_fwd_triton_gpu(
        q=q_gpu, k=k_gpu, v=v_gpu,
        g=g_gpu, g_gamma=g_gamma_gpu, gk=gk_gpu, gv=gv_gpu,
        scale=scale,
        initial_state=initial_state_gpu,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens.to(device_gpu) if cu_seqlens is not None else None,
    )

    # ---------- move to CPU ----------
    q_cpu             = q_gpu.cpu()
    k_cpu             = k_gpu.cpu()
    v_cpu             = v_gpu.cpu()
    g_cpu             = g_gpu.cpu()             if g_gpu is not None             else None
    g_gamma_cpu       = g_gamma_gpu.cpu()       if g_gamma_gpu is not None       else None
    gk_cpu            = gk_gpu.cpu()            if gk_gpu is not None            else None
    gv_cpu            = gv_gpu.cpu()            if gv_gpu is not None            else None
    initial_state_cpu = initial_state_gpu.cpu() if initial_state_gpu is not None else None

    # ---------- run CPU ----------
    print("Running Torch CPU version...")
    cpu_o, cpu_ht = fused_recurrent_fwd_torch_cpu(
        q=q_cpu, k=k_cpu, v=v_cpu,
        g=g_cpu, g_gamma=g_gamma_cpu, gk=gk_cpu, gv=gv_cpu,
        scale=scale,
        initial_state=initial_state_cpu,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )

    # ---------- shape assertions ----------
    expected_o_shape  = torch.Size([B, T, H, V])
    expected_ht_shape = torch.Size([N, H, K, V])
    for label, tensor in [("GPU o", gpu_o), ("CPU o", cpu_o)]:
        if tensor.shape != expected_o_shape:
            print(f"❌ {label} shape {tuple(tensor.shape)} != {tuple(expected_o_shape)}")
            return False
    if not output_final_state:
        for label, tensor in [("GPU ht", gpu_ht), ("CPU ht", cpu_ht)]:
            if tensor is not None:
                print(f"❌ {label} should be None when output_final_state=False, got {tuple(tensor.shape)}")
                return False
    else:
        for label, tensor in [("GPU ht", gpu_ht), ("CPU ht", cpu_ht)]:
            if tensor is None or tensor.shape != expected_ht_shape:
                got = "None" if tensor is None else tuple(tensor.shape)
                print(f"❌ {label} shape {got} != {tuple(expected_ht_shape)}")
                return False

    # ---------- compare o ----------
    gpu_o_np = gpu_o.detach().cpu().float().numpy()
    cpu_o_np = cpu_o.detach().float().numpy()

    diff_o     = np.abs(gpu_o_np - cpu_o_np)
    max_diff_o = diff_o.max()
    mean_diff_o = diff_o.mean()
    rel_diff_o  = diff_o / (np.abs(gpu_o_np) + 1e-8)
    max_rel_diff_o = rel_diff_o.max()

    print(f"\nOutput o  [{list(gpu_o_np.shape)}]:")
    print(f"  Max abs diff:  {max_diff_o:.6e}")
    print(f"  Mean abs diff: {mean_diff_o:.6e}")
    print(f"  Max rel diff:  {max_rel_diff_o:.6e}")

    # ---------- adaptive tolerance ----------
    # Both implementations use float32. Differences come from BK-blocked
    # K-reduction on GPU (NK = ceil(K/64) partial sums) vs. sequential CPU.
    # Accumulated error grows slowly with K, V, T.
    atol, rtol = 1e-4, 1e-4
    if K > 64 or V > 64:
        atol, rtol = 5e-4, 5e-4
    if K > 256 or V > 256:
        atol, rtol = 2e-3, 2e-3
    if T > 256:
        atol = max(atol, 1e-3)

    passed = np.allclose(gpu_o_np, cpu_o_np, atol=atol, rtol=rtol)

    # ---------- compare ht ----------
    if output_final_state and gpu_ht is not None and cpu_ht is not None:
        gpu_ht_np = gpu_ht.detach().cpu().float().numpy()
        cpu_ht_np = cpu_ht.detach().float().numpy()

        diff_ht     = np.abs(gpu_ht_np - cpu_ht_np)
        max_diff_ht = diff_ht.max()
        rel_diff_ht = diff_ht / (np.abs(gpu_ht_np) + 1e-8)
        max_rel_diff_ht = rel_diff_ht.max()

        print(f"\nFinal state ht [{list(gpu_ht_np.shape)}]:")
        print(f"  Max abs diff:  {max_diff_ht:.6e}")
        print(f"  Max rel diff:  {max_rel_diff_ht:.6e}")

        passed_ht = np.allclose(gpu_ht_np, cpu_ht_np, atol=atol, rtol=rtol)
        if not passed_ht:
            max_idx = np.unravel_index(np.argmax(diff_ht), diff_ht.shape)
            print(f"  ht Max mismatch at {max_idx}: GPU={gpu_ht_np[max_idx]:.6f} CPU={cpu_ht_np[max_idx]:.6f}")
        passed = passed and passed_ht

    # ---------- report ----------
    if passed:
        print("\n✅ Results MATCH (within tolerance).")
    else:
        print("\n❌ MISMATCH DETECTED!")
        max_idx = np.unravel_index(np.argmax(diff_o), diff_o.shape)
        print(f"  o Max mismatch at {max_idx}")
        print(f"  GPU: {gpu_o_np[max_idx]:.6f}  CPU: {cpu_o_np[max_idx]:.6f}")
        print(f"  abs diff: {diff_o[max_idx]:.6e}  rel diff: {rel_diff_o[max_idx]:.6e}")
        print(f"  (atol={atol:.0e}, rtol={rtol:.0e})")

    return passed


# ---------------------------------------------------------------------------
# Supplemental correctness tests (CPU-only, or CPU+GPU equivalence)
# ---------------------------------------------------------------------------

def run_test_reverse_differs_fwd(
    B: int, T: int, H: int, K: int, V: int,
    use_g: bool = False, use_gk: bool = True,
    use_gv: bool = False, use_g_gamma: bool = False,
):
    """Smoke test: reverse=True must produce a different output than reverse=False (for T>1).

    If both return the same array the 'reverse' flag is silently ignored — a hard bug to
    spot through GPU-vs-CPU comparison alone.
    """
    assert T > 1, "T must be >1 for reverse to differ from forward"
    torch.manual_seed(7)
    scale = K ** -0.5
    q        = torch.randn(B, T, H, K)
    k        = torch.randn(B, T, H, K)
    v        = torch.randn(B, T, H, V)
    g        = F.logsigmoid(torch.randn(B, T, H))    if use_g       else None
    g_gamma  = F.logsigmoid(torch.randn(H))          if use_g_gamma else None
    gk       = F.logsigmoid(torch.randn(B, T, H, K)) if use_gk      else None
    gv       = F.logsigmoid(torch.randn(B, T, H, V)) if use_gv      else None

    o_fwd, _ = fused_recurrent_fwd_torch_cpu(q, k, v, g=g, g_gamma=g_gamma, gk=gk, gv=gv,
                                              scale=scale, reverse=False)
    o_rev, _ = fused_recurrent_fwd_torch_cpu(q, k, v, g=g, g_gamma=g_gamma, gk=gk, gv=gv,
                                              scale=scale, reverse=True)
    if torch.allclose(o_fwd, o_rev, atol=1e-6):
        print("  ❌ reverse=True == reverse=False (reverse flag has no effect!)")
        return False
    max_diff = (o_fwd - o_rev).abs().max().item()
    print(f"  ✅ reverse differs from forward (max diff={max_diff:.4f})")
    return True


def run_test_varlen_equiv_nonvarlen(
    T: int, H: int, K: int, V: int,
    use_g: bool = False, use_gk: bool = True, use_g_gamma: bool = False,
):
    """Single-sequence varlen (cu_seqlens=[0,T]) must be numerically identical to
    non-varlen with B=1.  Tests both the CPU reference and the GPU Triton kernel.
    """
    torch.manual_seed(13)
    scale   = K ** -0.5
    q       = torch.randn(1, T, H, K)
    k       = torch.randn(1, T, H, K)
    v       = torch.randn(1, T, H, V)
    g       = F.logsigmoid(torch.randn(1, T, H))    if use_g       else None
    g_gamma = F.logsigmoid(torch.randn(H))          if use_g_gamma else None
    gk      = F.logsigmoid(torch.randn(1, T, H, K)) if use_gk      else None
    h0      = torch.randn(1, H, K, V)
    cu      = torch.LongTensor([0, T])

    # CPU check
    o_nv, ht_nv = fused_recurrent_fwd_torch_cpu(
        q, k, v, g=g, g_gamma=g_gamma, gk=gk, scale=scale,
        initial_state=h0, output_final_state=True)
    o_vl, ht_vl = fused_recurrent_fwd_torch_cpu(
        q, k, v, g=g, g_gamma=g_gamma, gk=gk, scale=scale,
        initial_state=h0, output_final_state=True, cu_seqlens=cu)
    assert ht_nv is not None and ht_vl is not None

    ok_cpu = torch.allclose(o_nv, o_vl, atol=1e-7) and torch.allclose(ht_nv, ht_vl, atol=1e-7)
    if not ok_cpu:
        d_o  = (o_nv  - o_vl).abs().max().item()
        d_ht = (ht_nv - ht_vl).abs().max().item()
        print(f"  ❌ CPU: varlen != non-varlen  (o diff={d_o:.2e}  ht diff={d_ht:.2e})")
        return False

    # GPU check
    if not torch.cuda.is_available():
        print("  ✅ CPU OK (GPU unavailable, skipped)")
        return True

    dev = 'cuda'
    q_g, k_g, v_g = q.to(dev), k.to(dev), v.to(dev)
    g_g  = g.to(dev)       if g       is not None else None
    gg_g = g_gamma.to(dev) if g_gamma is not None else None
    gk_g = gk.to(dev)      if gk      is not None else None
    h0_g = h0.to(dev)

    o_nv_g, ht_nv_g = fla_fused_recurrent_fwd_triton_gpu(
        q_g, k_g, v_g, g=g_g, g_gamma=gg_g, gk=gk_g, scale=scale,
        initial_state=h0_g, output_final_state=True)
    o_vl_g, ht_vl_g = fla_fused_recurrent_fwd_triton_gpu(
        q_g, k_g, v_g, g=g_g, g_gamma=gg_g, gk=gk_g, scale=scale,
        initial_state=h0_g, output_final_state=True,
        cu_seqlens=cu.to(dev).long())  # type: ignore[arg-type]
    assert ht_nv_g is not None and ht_vl_g is not None

    ok_gpu = (torch.allclose(o_nv_g.cpu(), o_vl_g.cpu(), atol=1e-6) and
              torch.allclose(ht_nv_g.cpu(), ht_vl_g.cpu(), atol=1e-6))
    if not ok_gpu:
        d_o  = (o_nv_g - o_vl_g).abs().max().item()
        d_ht = (ht_nv_g - ht_vl_g).abs().max().item()
        print(f"  ❌ GPU: varlen != non-varlen  (o diff={d_o:.2e}  ht diff={d_ht:.2e})")
        return False

    print("  ✅ CPU + GPU: single-seq varlen == non-varlen")
    return True


def run_test_golden_reference(
    B: int, T: int, H: int, K: int, V: int,
    use_g: bool = False, use_gk: bool = True,
    use_gv: bool = False, use_g_gamma: bool = False,
    reverse: bool = False,
):
    """Compare the vectorised CPU implementation against a naive triple-loop reference.

    The triple loop is an independent, maximally-readable implementation that is
    trivially correct by inspection.  Agreement with the vectorised version
    validates algorithmic correctness independent of GPU/CPU comparison.
    """
    torch.manual_seed(0)
    scale   = K ** -0.5
    q       = torch.randn(B, T, H, K)
    k       = torch.randn(B, T, H, K)
    v       = torch.randn(B, T, H, V)
    g       = F.logsigmoid(torch.randn(B, T, H))    if use_g       else None
    g_gamma = F.logsigmoid(torch.randn(H))          if use_g_gamma else None
    gk      = F.logsigmoid(torch.randn(B, T, H, K)) if use_gk      else None
    gv      = F.logsigmoid(torch.randn(B, T, H, V)) if use_gv      else None

    # --- naive triple-loop reference ---
    o_ref = torch.zeros(B, T, H, V)
    time_iter = range(T - 1, -1, -1) if reverse else range(T)
    for b in range(B):
        for h_idx in range(H):
            h_state = torch.zeros(K, V)
            for t in time_iter:
                if g       is not None: h_state = h_state * torch.exp(g[b, t, h_idx])
                if g_gamma is not None: h_state = h_state * torch.exp(g_gamma[h_idx])
                if gk      is not None: h_state = h_state * torch.exp(gk[b, t, h_idx])[:, None]
                if gv      is not None: h_state = h_state * torch.exp(gv[b, t, h_idx])[None, :]
                h_state = h_state + k[b, t, h_idx, :, None] * v[b, t, h_idx, None, :]
                o_ref[b, t, h_idx] = (h_state * (q[b, t, h_idx] * scale)[:, None]).sum(0)

    # --- vectorised CPU impl ---
    o_cpu, _ = fused_recurrent_fwd_torch_cpu(
        q, k, v, g=g, g_gamma=g_gamma, gk=gk, gv=gv, scale=scale, reverse=reverse)

    diff = (o_ref - o_cpu).abs().max().item()
    if diff < 1e-5:
        print(f"  ✅ matches triple-loop reference (max diff={diff:.2e})")
        return True
    else:
        print(f"  ❌ mismatch vs triple-loop reference (max diff={diff:.2e})")
        idx = (o_ref - o_cpu).abs().argmax()
        idx = np.unravel_index(idx.item(), o_ref.shape)
        print(f"     at {idx}: ref={o_ref[idx]:.6f}  cpu={o_cpu[idx]:.6f}")
        return False


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def test_triton_vs_torch():
    """
    Comparison tests: FLA Triton GPU fused_recurrent_fwd vs PyTorch CPU reference.

    Test coverage (~60 cases):
    ==========================
    Group 1  (10): Basic GLA-style tests (gk gate only) — varied B/T/H/K/V
    Group 2  (11): Gate combination tests — g, gk, gv, g_gamma, all combos
    Group 3  (10): Dimension edge cases — extreme K/V/H/B values
    Group 4  ( 6): Hidden-state tests — initial state, final state, both
    Group 5  ( 5): Reverse-mode tests
    Group 6  ( 8): Sequence-length edge cases — T=1 through T=1024
    Group 7  (10): Varlen mode — 2-4 sequences, with/without state, reverse

    Gate tensors are generated in log domain via logsigmoid, giving stable
    decay factors in (0, 1] matching typical GLA usage.

    Tolerance:
      atol/rtol = 1e-4 (K,V<=64), 5e-4 (K,V<=256), 2e-3 (K,V>256)
      For T>256: atol >= 1e-3 to account for state accumulation.
    """
    test_cases = [
        # ========================================
        # Group 1: Basic GLA (gk only)
        # ========================================
        dict(name="Basic: Standard config B=2 T=64",
             B=2, T=64, H=4, K=32, V=32, use_gk=True),
        dict(name="Basic: H=1 single head",
             B=1, T=32, H=1, K=16, V=16, use_gk=True),
        dict(name="Basic: H=8",
             B=1, T=64, H=8, K=32, V=32, use_gk=True),
        dict(name="Basic: K=64 (single BK block)",
             B=2, T=128, H=4, K=64, V=64, use_gk=True),
        dict(name="Basic: K=128 (two BK blocks)",
             B=1, T=64, H=4, K=128, V=64, use_gk=True),
        dict(name="Basic: K=256 (four BK blocks)",
             B=1, T=32, H=4, K=256, V=64, use_gk=True),
        dict(name="Basic: V=128",
             B=1, T=64, H=4, K=32, V=128, use_gk=True),
        dict(name="Basic: V=256",
             B=1, T=32, H=4, K=32, V=256, use_gk=True),
        dict(name="Basic: B=8",
             B=8, T=32, H=4, K=32, V=32, use_gk=True),
        dict(name="Basic: Long T=512",
             B=1, T=512, H=4, K=64, V=64, use_gk=True),

        # ========================================
        # Group 2: Gate combinations
        # ========================================
        dict(name="Gate: use_g only (scalar head gate)",
             B=2, T=64, H=4, K=32, V=32, use_g=True),
        dict(name="Gate: use_gv only (value-wise gate)",
             B=1, T=64, H=4, K=32, V=32, use_gv=True),
        dict(name="Gate: use_gk only",
             B=2, T=64, H=4, K=32, V=32, use_gk=True),
        dict(name="Gate: use_gk + use_gv",
             B=1, T=64, H=4, K=32, V=32, use_gk=True, use_gv=True),
        dict(name="Gate: use_g + use_gk",
             B=2, T=64, H=4, K=32, V=32, use_g=True, use_gk=True),
        dict(name="Gate: use_g + use_gv",
             B=1, T=64, H=4, K=32, V=32, use_g=True, use_gv=True),
        dict(name="Gate: use_g + use_gk + use_gv (all except g_gamma)",
             B=2, T=64, H=4, K=32, V=32, use_g=True, use_gk=True, use_gv=True),
        dict(name="Gate: use_g_gamma only (per-head constant decay)",
             B=1, T=64, H=4, K=32, V=32, use_g_gamma=True),
        dict(name="Gate: use_g_gamma + use_gk",
             B=1, T=64, H=4, K=32, V=32, use_g_gamma=True, use_gk=True),
        dict(name="Gate: all gates (g + g_gamma + gk + gv)",
             B=1, T=64, H=4, K=32, V=32,
             use_g=True, use_g_gamma=True, use_gk=True, use_gv=True),
        dict(name="Gate: no gates (pure linear attention)",
             B=1, T=32, H=4, K=32, V=32),

        # ========================================
        # Group 3: Dimension edge cases
        # ========================================
        dict(name="Dim: H=16",
             B=1, T=64, H=16, K=32, V=32, use_gk=True),
        dict(name="Dim: H=32",
             B=1, T=32, H=32, K=16, V=16, use_gk=True),
        dict(name="Dim: K<V (K=16, V=64)",
             B=1, T=64, H=4, K=16, V=64, use_gk=True),
        dict(name="Dim: K>V (K=128, V=32)",
             B=1, T=64, H=4, K=128, V=32, use_gk=True),
        dict(name="Dim: K=512 (eight BK blocks)",
             B=1, T=16, H=2, K=512, V=64, use_gk=True),
        dict(name="Dim: V=512",
             B=1, T=16, H=2, K=64, V=512, use_gk=True),
        dict(name="Dim: K=512, V=512",
             B=1, T=16, H=2, K=512, V=512, use_gk=True),
        dict(name="Dim: B=16",
             B=16, T=32, H=4, K=32, V=32, use_gk=True),
        dict(name="Dim: B=32",
             B=32, T=16, H=2, K=16, V=16, use_gk=True),
        dict(name="Dim: Odd K=37 V=53",
             B=1, T=64, H=4, K=37, V=53, use_gk=True),

        # ========================================
        # Group 4: Hidden-state tests
        # ========================================
        dict(name="State: With initial state (gk)",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, use_initial_state=True),
        dict(name="State: Output final state (gk)",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, output_final_state=True),
        dict(name="State: Both initial and final (gk)",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, use_initial_state=True, output_final_state=True),
        dict(name="State: g+gk with both states",
             B=1, T=64, H=4, K=32, V=32,
             use_g=True, use_gk=True,
             use_initial_state=True, output_final_state=True),
        dict(name="State: gk+gv with both states",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, use_gv=True,
             use_initial_state=True, output_final_state=True),
        dict(name="State: Long T=256 with both states",
             B=1, T=256, H=4, K=64, V=64,
             use_gk=True, use_initial_state=True, output_final_state=True),

        # ========================================
        # Group 5: Reverse mode
        # ========================================
        dict(name="Reverse: gk only",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, reverse=True),
        dict(name="Reverse: g+gk+gv",
             B=1, T=64, H=4, K=32, V=32,
             use_g=True, use_gk=True, use_gv=True, reverse=True),
        dict(name="Reverse: With initial state",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, reverse=True, use_initial_state=True),
        dict(name="Reverse: Output final state",
             B=1, T=64, H=4, K=32, V=32,
             use_gk=True, reverse=True, output_final_state=True),
        dict(name="Reverse: Long T=256",
             B=1, T=256, H=4, K=64, V=64,
             use_gk=True, reverse=True),

        # ========================================
        # Group 6: Sequence-length edge cases
        # ========================================
        dict(name="SeqLen: T=1 (single step)",
             B=1, T=1, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=2",
             B=2, T=2, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=4",
             B=1, T=4, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=8",
             B=2, T=8, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=16",
             B=2, T=16, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=32",
             B=2, T=32, H=4, K=32, V=32, use_gk=True),
        dict(name="SeqLen: T=256",
             B=1, T=256, H=4, K=64, V=64, use_gk=True),
        dict(name="SeqLen: T=1024",
             B=1, T=1024, H=4, K=64, V=64, use_gk=True),

        # ========================================
        # Group 7: Varlen mode
        # In varlen mode B=1; T must equal cu_seqlens[-1].
        # ========================================
        dict(name="Varlen: Two equal seqs (64+64=128)",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128], use_gk=True),
        dict(name="Varlen: Two diff seqs (32+96=128)",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 32, 128], use_gk=True),
        dict(name="Varlen: Three equal seqs (64+64+64=192)",
             B=1, T=192, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128, 192], use_gk=True),
        dict(name="Varlen: Four seqs (256 total)",
             B=1, T=256, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128, 192, 256], use_gk=True),
        dict(name="Varlen: Mixed lengths (20+40+40=100)",
             B=1, T=100, H=4, K=32, V=32,
             cu_seqlens_list=[0, 20, 60, 100], use_gk=True),
        dict(name="Varlen: Short seqs (T=1 each)",
             B=1, T=2, H=4, K=32, V=32,
             cu_seqlens_list=[0, 1, 2], use_gk=True),
        dict(name="Varlen: With initial state",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_gk=True, use_initial_state=True),
        dict(name="Varlen: Output final state",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_gk=True, output_final_state=True),
        dict(name="Varlen: g+gk gates",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_g=True, use_gk=True),
        dict(name="Varlen: Reverse mode",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_gk=True, reverse=True),

        # ========================================
        # Group 8: g_gamma extended coverage
        # ========================================
        dict(name="g_gamma: + varlen (two seqs)",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128], use_g_gamma=True, use_gk=True),
        dict(name="g_gamma: all gates + varlen",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_g=True, use_g_gamma=True, use_gk=True, use_gv=True),
        dict(name="g_gamma: + initial_state",
             B=2, T=64, H=4, K=32, V=32,
             use_g_gamma=True, use_gk=True, use_initial_state=True),
        dict(name="g_gamma: + output_final_state",
             B=2, T=64, H=4, K=32, V=32,
             use_g_gamma=True, use_gk=True, output_final_state=True),
        dict(name="g_gamma: + both states",
             B=2, T=64, H=4, K=32, V=32,
             use_g_gamma=True, use_gk=True,
             use_initial_state=True, output_final_state=True),
        dict(name="g_gamma: + reverse",
             B=2, T=64, H=4, K=32, V=32,
             use_g_gamma=True, use_gk=True, reverse=True),

        # ========================================
        # Group 9: Varlen edge cases
        # ========================================
        dict(name="Varlen edge: H=1",
             B=1, T=128, H=1, K=32, V=32,
             cu_seqlens_list=[0, 64, 128], use_gk=True),
        dict(name="Varlen edge: gv gate",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128], use_gk=True, use_gv=True),
        dict(name="Varlen edge: all gates",
             B=1, T=128, H=4, K=32, V=32,
             cu_seqlens_list=[0, 64, 128],
             use_g=True, use_g_gamma=True, use_gk=True, use_gv=True),
        dict(name="Varlen edge: asymmetric (1+256=257)",
             B=1, T=257, H=4, K=32, V=32,
             cu_seqlens_list=[0, 1, 257], use_gk=True),
        dict(name="Varlen edge: asymmetric + both states",
             B=1, T=257, H=4, K=32, V=32,
             cu_seqlens_list=[0, 1, 257],
             use_gk=True, use_initial_state=True, output_final_state=True),

        # ========================================
        # Group 10: Gate value distributions
        # ========================================
        dict(name="GateMode: zero (no decay, exp(0)=1)",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, gate_mode='zero'),
        dict(name="GateMode: strong_decay (exp(-5)≈0.007)",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, gate_mode='strong_decay'),
        dict(name="GateMode: positive (exp(0.3)≈1.35, T=8)",
             B=2, T=8, H=4, K=32, V=32,
             use_gk=True, gate_mode='positive'),
        dict(name="GateMode: all gates zero (no decay anywhere)",
             B=2, T=64, H=4, K=32, V=32,
             use_g=True, use_gk=True, use_gv=True, gate_mode='zero'),

        # ========================================
        # Group 11: Custom scale values
        # ========================================
        dict(name="Scale: scale=1.0",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, scale=1.0),
        dict(name="Scale: scale=0.1",
             B=2, T=64, H=4, K=32, V=32,
             use_gk=True, scale=0.1),
        dict(name="Scale: scale=2.0",
             B=2, T=32, H=4, K=32, V=32,
             use_gk=True, scale=2.0),
    ]

    all_passed   = True
    passed_count = 0
    failed_count = 0
    total_count  = len(test_cases)

    for i, test_case in enumerate(test_cases):
        test_name = test_case.pop("name", f"Test {i+1}")

        print(f"\n{'='*70}")
        print(f"Test {i+1}/{total_count}: {test_name}")
        print(f"{'='*70}")

        try:
            passed = run_test_triton_vs_torch(**test_case)
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        except Exception:
            print("❌ Exception occurred:")
            traceback.print_exc()
            passed = False
            failed_count += 1

        all_passed = all_passed and passed

    print(f"\n{'='*70}")
    print(f"Test Summary:")
    print(f"  ✅ Passed: {passed_count}/{total_count}")
    print(f"  ❌ Failed: {failed_count}/{total_count}")
    print(f"{'='*70}")

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")

    # ========================================
    # Supplemental correctness tests
    # ========================================
    print(f"\n{'='*70}")
    print("Supplemental Tests")
    print(f"{'='*70}")

    supp_passed = 0
    supp_failed = 0

    # --- [A] smoke test: reverse differs from forward ---
    smoke_cases = [
        dict(B=2, T=64, H=4, K=32, V=32, use_gk=True),
        dict(B=1, T=32, H=4, K=32, V=32, use_g=True, use_gk=True, use_gv=True),
        dict(B=1, T=16, H=4, K=32, V=32, use_gk=True, use_g_gamma=True),
    ]
    print("\n[A] Reverse differs from forward (CPU smoke test):")
    for case in smoke_cases:
        g_flags = (f"g={case.get('use_g', False)} gk={case.get('use_gk', False)}"
                   f" gv={case.get('use_gv', False)} g_gamma={case.get('use_g_gamma', False)}")
        print(f"  B={case['B']} T={case['T']}  {g_flags}")
        ok = run_test_reverse_differs_fwd(**case)  # type: ignore[arg-type]
        if ok:
            supp_passed += 1
        else:
            supp_failed += 1
            all_passed = False

    # --- [B] single-seq varlen == non-varlen ---
    varlen_equiv_cases = [
        dict(T=64,  H=4, K=32, V=32, use_gk=True),
        dict(T=128, H=4, K=64, V=64, use_gk=True, use_g=True),
        dict(T=64,  H=4, K=32, V=32, use_gk=True, use_g_gamma=True),
        dict(T=32,  H=1, K=16, V=16, use_gk=True),
    ]
    print("\n[B] Single-seq varlen == non-varlen (CPU + GPU):")
    for case in varlen_equiv_cases:
        g_flags = (f"g={case.get('use_g', False)} gk={case.get('use_gk', False)}"
                   f" g_gamma={case.get('use_g_gamma', False)}")
        print(f"  T={case['T']} H={case['H']} K={case['K']} V={case['V']}  {g_flags}")
        ok = run_test_varlen_equiv_nonvarlen(**case)  # type: ignore[arg-type]
        if ok:
            supp_passed += 1
        else:
            supp_failed += 1
            all_passed = False

    # --- [C] golden reference: triple-loop vs vectorized CPU ---
    golden_cases = [
        dict(B=2, T=32, H=4, K=32, V=32, use_gk=True),
        dict(B=1, T=32, H=4, K=32, V=32, use_g=True, use_gk=True, use_gv=True),
        dict(B=2, T=32, H=4, K=32, V=32, use_gk=True, use_g_gamma=True),
        dict(B=1, T=32, H=4, K=32, V=32, use_gk=True, reverse=True),
    ]
    print("\n[C] Triple-loop golden reference vs vectorized CPU:")
    for case in golden_cases:
        g_flags = (f"g={case.get('use_g', False)} gk={case.get('use_gk', False)}"
                   f" gv={case.get('use_gv', False)} g_gamma={case.get('use_g_gamma', False)}"
                   f" rev={case.get('reverse', False)}")
        print(f"  B={case['B']} T={case['T']}  {g_flags}")
        ok = run_test_golden_reference(**case)  # type: ignore[arg-type]
        if ok:
            supp_passed += 1
        else:
            supp_failed += 1
            all_passed = False

    supp_total = supp_passed + supp_failed
    print(f"\n{'='*70}")
    print(f"Supplemental Summary:  ✅ {supp_passed}/{supp_total}  ❌ {supp_failed}/{supp_total}")
    print(f"{'='*70}")

    return all_passed


if __name__ == "__main__":
    success = True

    if torch.cuda.is_available():
        success = test_triton_vs_torch() and success

    if success:
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

    if (success):
        exit(0)
    else:
        exit(1)


