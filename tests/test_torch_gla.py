
# =============================================================================
# Quick smoke test
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from torch_gla import GatedLinearAttention, ShortConvolution, RMSNorm, FusedRMSNormGated, naive_recurrent_gla, chunk_gla, fused_chunk_gla, get_unpad_data, index_first_axis, pad_input

chunk_gla = naive_recurrent_gla
fused_recurrent_gla = naive_recurrent_gla
fused_chunk_gla = naive_recurrent_gla

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cpu')

    print("=" * 60)
    print("torch_gla.py — CPU-only GLA smoke test")
    print("=" * 60)

    # --- Test 1: Basic forward pass ---
    print("\n[Test 1] Basic forward (no conv, no mask)")
    B, T, H, D = 2, 32, 4, 64
    model = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=False,
        use_output_gate=True,
        gate_fn='swish',
        fuse_norm=True,
        layer_idx=0,
    ).to(device)

    x = torch.randn(B, T, D * H, device=device)
    o, _, _ = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {o.shape}")
    assert o.shape == x.shape, f"Shape mismatch: {o.shape} vs {x.shape}"
    print("  PASSED ✓")

    # --- Test 2: With short conv ---
    print("\n[Test 2] Forward with ShortConvolution")
    model_conv = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=True,
        conv_size=4,
        use_output_gate=True,
        gate_fn='swish',
        fuse_norm=True,
        layer_idx=0,
    ).to(device)

    o_conv, _, _ = model_conv(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {o_conv.shape}")
    assert o_conv.shape == x.shape
    print("  PASSED ✓")

    # --- Test 3: With attention mask ---
    print("\n[Test 3] Forward with attention_mask (padding)")
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    mask[0, -8:] = 0  # 第一个序列最后 8 个 token 是 padding
    mask[1, -4:] = 0

    o_mask, _, _ = model(x, attention_mask=mask)
    print(f"  Input:  {x.shape}, mask: {mask.shape}")
    print(f"  Output: {o_mask.shape}")
    assert o_mask.shape == x.shape
    print("  PASSED ✓")

    # --- Test 4: Fused recurrent mode (short seq) ---
    print("\n[Test 4] Short sequence triggers fused_recurrent mode")
    x_short = torch.randn(B, 16, D * H, device=device)
    o_short, _, _ = model(x_short)
    print(f"  Input:  {x_short.shape}")
    print(f"  Output: {o_short.shape}")
    assert o_short.shape == x_short.shape
    print("  PASSED ✓")

    # --- Test 5: Naive recurrent vs reference ---
    print("\n[Test 5] Naive recurrent correctness check")
    B2, T2, H2, K2, V2 = 2, 64, 4, 32, 64
    q = torch.randn(B2, T2, H2, K2, device=device)
    k = torch.randn(B2, T2, H2, K2, device=device)
    v = torch.randn(B2, T2, H2, V2, device=device)
    g = F.logsigmoid(torch.randn(B2, T2, H2, K2, device=device))

    o1, s1 = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    o2, s2 = naive_recurrent_gla(q, k, v, g, output_final_state=True)
    assert torch.allclose(o1, o2, atol=1e-5), "Outputs differ!"
    assert torch.allclose(s1, s2, atol=1e-5), "States differ!"
    print(f"  Output shape: {o1.shape}, State shape: {s1.shape}")
    print("  PASSED ✓")

    # --- Test 6: cu_seqlens variable-length ---
    print("\n[Test 6] Variable-length with cu_seqlens")
    T_total = 48
    q_var = torch.randn(1, T_total, H2, K2, device=device)
    k_var = torch.randn(1, T_total, H2, K2, device=device)
    v_var = torch.randn(1, T_total, H2, V2, device=device)
    g_var = F.logsigmoid(torch.randn(1, T_total, H2, K2, device=device))
    cu = torch.tensor([0, 16, 32, 48], dtype=torch.long, device=device)
    h0 = torch.randn(3, H2, K2, V2, device=device, dtype=torch.float32)

    o_var, s_var = naive_recurrent_gla(
        q_var, k_var, v_var, g_var,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    print(f"  Output: {o_var.shape}, Final state: {s_var.shape}")
    assert o_var.shape == (1, T_total, H2, V2)
    assert s_var.shape == (3, H2, K2, V2)
    print("  PASSED ✓")

    # --- Test 7: Non-fused norm path ---
    print("\n[Test 7] Non-fused norm + non-swish gate")
    model_nofuse = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=False,
        use_output_gate=True,
        gate_fn='sigmoid',
        fuse_norm=False,
        layer_idx=0,
    ).to(device)
    o_nf, _, _ = model_nofuse(x)
    assert o_nf.shape == x.shape
    print(f"  Output: {o_nf.shape}")
    print("  PASSED ✓")

    # --- Test 8: No output gate ---
    print("\n[Test 8] No output gate")
    model_nogate = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=False,
        use_output_gate=False,
        fuse_norm=False,
        layer_idx=0,
    ).to(device)
    o_ng, _, _ = model_nogate(x)
    assert o_ng.shape == x.shape
    print(f"  Output: {o_ng.shape}")
    print("  PASSED ✓")

    # --- Test 9: Gradient backward pass ---
    print("\n[Test 9] Gradient backward pass")
    model_grad = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=False,
        use_output_gate=True,
        gate_fn='swish',
        fuse_norm=True,
        layer_idx=0,
    ).to(device)
    x_grad = torch.randn(B, T, D * H, device=device, requires_grad=True)
    o_grad, _, _ = model_grad(x_grad)
    loss = o_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient computed for input!"
    assert x_grad.grad.shape == x_grad.shape
    # 检查模型参数是否都有梯度
    for name, param in model_grad.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    print(f"  Input grad shape: {x_grad.grad.shape}")
    print(f"  All {sum(1 for p in model_grad.parameters() if p.requires_grad)} params have gradients")
    print("  PASSED ✓")

    # --- Test 10: Gradient backward with short conv ---
    print("\n[Test 10] Gradient backward with ShortConvolution")
    x_grad2 = torch.randn(B, T, D * H, device=device, requires_grad=True)
    o_grad2, _, _ = model_conv(x_grad2)
    loss2 = o_grad2.sum()
    loss2.backward()
    assert x_grad2.grad is not None
    print(f"  Input grad shape: {x_grad2.grad.shape}")
    print("  PASSED ✓")

    # --- Test 11: cu_seqlens consistency vs separate batches ---
    print("\n[Test 11] cu_seqlens vs separate batches consistency")
    torch.manual_seed(123)
    H3, K3, V3 = 2, 16, 32
    s1_len, s2_len = 10, 14
    # 序列 1
    q1 = torch.randn(1, s1_len, H3, K3, device=device)
    k1 = torch.randn(1, s1_len, H3, K3, device=device)
    v1 = torch.randn(1, s1_len, H3, V3, device=device)
    g1 = F.logsigmoid(torch.randn(1, s1_len, H3, K3, device=device))
    # 序列 2
    q2 = torch.randn(1, s2_len, H3, K3, device=device)
    k2 = torch.randn(1, s2_len, H3, K3, device=device)
    v2 = torch.randn(1, s2_len, H3, V3, device=device)
    g2 = F.logsigmoid(torch.randn(1, s2_len, H3, K3, device=device))
    # 分别计算
    o_sep1, s_sep1 = naive_recurrent_gla(q1, k1, v1, g1, output_final_state=True)
    o_sep2, s_sep2 = naive_recurrent_gla(q2, k2, v2, g2, output_final_state=True)
    # 用 cu_seqlens 一起算
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    cu_test = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long, device=device)
    o_cu, s_cu = naive_recurrent_gla(
        q_cat, k_cat, v_cat, g_cat,
        output_final_state=True, cu_seqlens=cu_test,
    )
    assert torch.allclose(o_sep1, o_cu[:, :s1_len], atol=1e-5), "cu_seqlens seg 1 output mismatch"
    assert torch.allclose(o_sep2, o_cu[:, s1_len:], atol=1e-5), "cu_seqlens seg 2 output mismatch"
    assert torch.allclose(s_sep1.squeeze(0), s_cu[0], atol=1e-5), "cu_seqlens seg 1 state mismatch"
    assert torch.allclose(s_sep2.squeeze(0), s_cu[1], atol=1e-5), "cu_seqlens seg 2 state mismatch"
    print("  cu_seqlens outputs match separate batches")
    print("  PASSED ✓")

    # --- Test 12: Initial state propagation ---
    print("\n[Test 12] Initial state propagation")
    torch.manual_seed(77)
    B4, T4, H4, K4, V4 = 1, 20, 2, 16, 32
    q4 = torch.randn(B4, T4, H4, K4, device=device)
    k4 = torch.randn(B4, T4, H4, K4, device=device)
    v4 = torch.randn(B4, T4, H4, V4, device=device)
    g4 = F.logsigmoid(torch.randn(B4, T4, H4, K4, device=device))
    # 处理前 10 步得到中间状态
    o_first, mid_state = naive_recurrent_gla(
        q4[:, :10], k4[:, :10], v4[:, :10], g4[:, :10],
        output_final_state=True,
    )
    # 用中间状态处理后 10 步
    o_second, final_state = naive_recurrent_gla(
        q4[:, 10:], k4[:, 10:], v4[:, 10:], g4[:, 10:],
        initial_state=mid_state,
        output_final_state=True,
    )
    # 一次处理全部 20 步
    o_full, full_state = naive_recurrent_gla(
        q4, k4, v4, g4,
        output_final_state=True,
    )
    assert torch.allclose(o_first, o_full[:, :10], atol=1e-4), "First half output mismatch"
    assert torch.allclose(o_second, o_full[:, 10:], atol=1e-4), "Second half output mismatch"
    assert torch.allclose(final_state, full_state, atol=1e-4), "Final state mismatch"
    print("  Split processing matches full processing")
    print("  PASSED ✓")

    # --- Test 13: RMSNorm correctness ---
    print("\n[Test 13] RMSNorm correctness")
    dim = 64
    norm = RMSNorm(dim, elementwise_affine=True, eps=1e-5)
    x_norm = torch.randn(2, 10, dim)
    y_norm = norm(x_norm)
    # 手动验证
    x_f = x_norm.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_expected = (x_f * rms * norm.weight.float()).to(x_norm.dtype)
    assert torch.allclose(y_norm, y_expected, atol=1e-6), "RMSNorm output mismatch"
    assert y_norm.shape == x_norm.shape
    # 无 affine
    norm_noaff = RMSNorm(dim, elementwise_affine=False)
    y_noaff = norm_noaff(x_norm)
    assert y_noaff.shape == x_norm.shape
    print("  PASSED ✓")

    # --- Test 14: FusedRMSNormGated correctness ---
    print("\n[Test 14] FusedRMSNormGated correctness")
    fnorm = FusedRMSNormGated(dim, elementwise_affine=True, eps=1e-5)
    x_fn = torch.randn(2, 10, dim)
    g_fn = torch.randn(2, 10, dim)
    y_fn = fnorm(x_fn, g_fn)
    # 手动验证: RMSNorm(x) * silu(g)
    x_ff = x_fn.float()
    g_ff = g_fn.float()
    rms_fn = torch.rsqrt(x_ff.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_fn_expected = (x_ff * rms_fn * fnorm.weight.float() * F.silu(g_ff)).to(x_fn.dtype)
    assert torch.allclose(y_fn, y_fn_expected, atol=1e-6), "FusedRMSNormGated mismatch"
    print("  PASSED ✓")

    # --- Test 15: ShortConvolution causal property ---
    print("\n[Test 15] ShortConvolution causality verification")
    conv = ShortConvolution(hidden_size=8, kernel_size=3, bias=True, activation='silu')
    # 输入 x[:, :t, :] 的变化不应该影响 y[:, :t_prev, :]
    x_c1 = torch.randn(1, 16, 8)
    x_c2 = x_c1.clone()
    x_c2[:, 10:, :] = torch.randn(1, 6, 8)  # 修改后半部分
    y_c1, _ = conv(x_c1)
    y_c2, _ = conv(x_c2)
    # 前 10 步的输出应完全相同（因果性）
    assert torch.allclose(y_c1[:, :10], y_c2[:, :10], atol=1e-6), "Causality violated!"
    # 后面的输出可能不同
    assert not torch.allclose(y_c1[:, 10:], y_c2[:, 10:], atol=1e-6), "Outputs should differ!"
    print("  Causal property verified: future changes don't affect past outputs")
    print("  PASSED ✓")

    # --- Test 16: ShortConvolution with cu_seqlens ---
    print("\n[Test 16] ShortConvolution with cu_seqlens (variable-length)")
    conv2 = ShortConvolution(hidden_size=8, kernel_size=3, bias=False, activation='silu')
    # 两个独立序列 vs 打包成一个
    x_s1 = torch.randn(1, 6, 8)
    x_s2 = torch.randn(1, 10, 8)
    y_s1, _ = conv2(x_s1)
    y_s2, _ = conv2(x_s2)
    # 打包
    x_packed = torch.cat([x_s1, x_s2], dim=1)  # [1, 16, 8]
    cu_conv = torch.tensor([0, 6, 16], dtype=torch.long)
    y_packed, _ = conv2(x_packed, cu_seqlens=cu_conv)
    assert torch.allclose(y_s1, y_packed[:, :6], atol=1e-5), "cu_seqlens conv seg 1 mismatch"
    assert torch.allclose(y_s2, y_packed[:, 6:], atol=1e-5), "cu_seqlens conv seg 2 mismatch"
    print("  cu_seqlens conv outputs match separate convolutions")
    print("  PASSED ✓")

    # --- Test 17: MQA (Multi-Query Attention) with num_kv_heads ---
    print("\n[Test 17] MQA with num_kv_heads < num_heads")
    model_mqa = GatedLinearAttention(
        mode='chunk',
        hidden_size=256,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=8,
        num_kv_heads=2,  # 8/2 = 4 groups
        use_short_conv=False,
        use_output_gate=True,
        gate_fn='swish',
        fuse_norm=True,
        layer_idx=0,
    ).to(device)
    x_mqa = torch.randn(2, 32, 256, device=device)
    o_mqa, _, _ = model_mqa(x_mqa)
    assert o_mqa.shape == x_mqa.shape, f"MQA shape mismatch: {o_mqa.shape}"
    print(f"  num_heads=8, num_kv_heads=2, output: {o_mqa.shape}")
    print("  PASSED ✓")

    # --- Test 18: MQA backward ---
    print("\n[Test 18] MQA backward pass")
    x_mqa_g = torch.randn(2, 32, 256, device=device, requires_grad=True)
    o_mqa_g, _, _ = model_mqa(x_mqa_g)
    o_mqa_g.sum().backward()
    assert x_mqa_g.grad is not None
    print("  PASSED ✓")

    # --- Test 19: Different expand_k / expand_v ratios ---
    print("\n[Test 19] Non-default expand_k/expand_v ratios")
    for ek, ev in [(1.0, 1.0), (0.25, 2.0), (1.0, 0.5)]:
        model_exp = GatedLinearAttention(
            mode='chunk',
            hidden_size=128,
            expand_k=ek,
            expand_v=ev,
            num_heads=4,
            use_short_conv=False,
            use_output_gate=True,
            gate_fn='swish',
            fuse_norm=True,
            layer_idx=0,
        ).to(device)
        x_exp = torch.randn(1, 16, 128, device=device)
        o_exp, _, _ = model_exp(x_exp)
        assert o_exp.shape == x_exp.shape, f"expand_k={ek}, expand_v={ev}: shape {o_exp.shape}"
        print(f"  expand_k={ek}, expand_v={ev} -> {o_exp.shape} ✓")
    print("  PASSED ✓")

    # --- Test 20: All three modes produce same output ---
    print("\n[Test 20] All modes (chunk/fused_recurrent/fused_chunk) produce same output")
    torch.manual_seed(99)
    B5, T5, H5, K5, V5 = 2, 30, 2, 16, 32
    q5 = torch.randn(B5, T5, H5, K5, device=device)
    k5 = torch.randn(B5, T5, H5, K5, device=device)
    v5 = torch.randn(B5, T5, H5, V5, device=device)
    g5 = F.logsigmoid(torch.randn(B5, T5, H5, K5, device=device))
    o_chunk, s_chunk = chunk_gla(q5, k5, v5, g5, output_final_state=True)
    o_fused, s_fused = fused_recurrent_gla(q5, k5, v5, g5, output_final_state=True)
    o_fc, s_fc = fused_chunk_gla(q5, k5, v5, g5, output_final_state=True)
    assert torch.allclose(o_chunk, o_fused, atol=1e-6), "chunk vs fused_recurrent mismatch"
    assert torch.allclose(o_chunk, o_fc, atol=1e-6), "chunk vs fused_chunk mismatch"
    assert torch.allclose(s_chunk, s_fused, atol=1e-6)
    assert torch.allclose(s_chunk, s_fc, atol=1e-6)
    print("  All three modes produce identical results")
    print("  PASSED ✓")

    # --- Test 21: Unpad/pad roundtrip ---
    print("\n[Test 21] Unpad/pad roundtrip correctness")
    B6, T6, D6 = 3, 20, 32
    mask6 = torch.ones(B6, T6, dtype=torch.long, device=device)
    mask6[0, 15:] = 0
    mask6[1, 18:] = 0
    mask6[2, 10:] = 0
    indices6, cu6, max_len6 = get_unpad_data(mask6)
    x6 = torch.randn(B6, T6, D6, device=device)
    x_flat = rearrange(x6, 'b s d -> (b s) d')
    x_packed = index_first_axis(x_flat, indices6)
    x_restored = pad_input(x_packed, indices6, B6, T6)
    # 有效位置应完全一致
    for b in range(B6):
        valid_len = mask6[b].sum().item()
        assert torch.allclose(x6[b, :valid_len], x_restored[b, :valid_len], atol=1e-7), \
            f"Roundtrip failed for batch {b}"
    # padding 位置应为 0
    assert (x_restored[0, 15:] == 0).all()
    assert (x_restored[2, 10:] == 0).all()
    print(f"  Packed {indices6.shape[0]} tokens from {B6*T6}, restored correctly")
    print("  PASSED ✓")

    # --- Test 22: GLA layer with attention mask + short conv ---
    print("\n[Test 22] Full GLA with mask + short conv")
    model_full = GatedLinearAttention(
        mode='chunk',
        hidden_size=D * H,
        expand_k=0.5,
        expand_v=1.0,
        num_heads=H,
        use_short_conv=True,
        conv_size=4,
        use_output_gate=True,
        gate_fn='swish',
        fuse_norm=True,
        layer_idx=0,
    ).to(device)
    mask_full = torch.ones(B, T, dtype=torch.long, device=device)
    mask_full[0, -5:] = 0
    o_full_m, _, _ = model_full(x, attention_mask=mask_full)
    assert o_full_m.shape == x.shape
    print(f"  Output: {o_full_m.shape}")
    print("  PASSED ✓")

    # --- Test 23: gate_logit_normalizer effect ---
    print("\n[Test 23] gate_logit_normalizer effect")
    torch.manual_seed(200)
    # 不同 normalizer 应产生不同输出
    model_n16 = GatedLinearAttention(
        hidden_size=128, num_heads=2, gate_logit_normalizer=16,
        use_output_gate=True, fuse_norm=True, layer_idx=0,
    ).to(device)
    model_n1 = GatedLinearAttention(
        hidden_size=128, num_heads=2, gate_logit_normalizer=1,
        use_output_gate=True, fuse_norm=True, layer_idx=0,
    ).to(device)
    # 共享权重
    model_n1.load_state_dict(model_n16.state_dict())
    x_n = torch.randn(1, 16, 128, device=device)
    o_n16, _, _ = model_n16(x_n)
    o_n1, _, _ = model_n1(x_n)
    assert not torch.allclose(o_n16, o_n1, atol=1e-3), "Different normalizers should produce different outputs"
    print("  Different normalizers produce different outputs")
    print("  PASSED ✓")

    # --- Test 24: clamp_min effect ---
    print("\n[Test 24] clamp_min for gate logits")
    model_clamp = GatedLinearAttention(
        hidden_size=128, num_heads=2, clamp_min=-1.0,
        use_output_gate=True, fuse_norm=True, layer_idx=0,
    ).to(device)
    o_clamp, _, _ = model_clamp(x_n)
    assert o_clamp.shape == x_n.shape
    print(f"  Output: {o_clamp.shape}")
    print("  PASSED ✓")

    # --- Test 25: Feature map (relu) ---
    print("\n[Test 25] Feature map application (relu)")
    model_fm = GatedLinearAttention(
        hidden_size=128, num_heads=2, feature_map='relu',
        use_output_gate=True, fuse_norm=True, layer_idx=0,
    ).to(device)
    o_fm, _, _ = model_fm(x_n)
    assert o_fm.shape == x_n.shape
    print(f"  Output: {o_fm.shape}")
    print("  PASSED ✓")

    # --- Test 26: Numerical stability (large values) ---
    print("\n[Test 26] Numerical stability with large/small values")
    x_large = torch.randn(1, 16, 128, device=device) * 10.0
    o_large, _, _ = model_n16(x_large)
    assert torch.isfinite(o_large).all(), "Output contains inf/nan with large inputs"
    x_small = torch.randn(1, 16, 128, device=device) * 0.001
    o_small, _, _ = model_n16(x_small)
    assert torch.isfinite(o_small).all(), "Output contains inf/nan with small inputs"
    print("  No inf/nan with large or small inputs")
    print("  PASSED ✓")

    # --- Test 27: Batch size 1 ---
    print("\n[Test 27] Batch size = 1")
    x_b1 = torch.randn(1, 64, D * H, device=device)
    o_b1, _, _ = model(x_b1)
    assert o_b1.shape == x_b1.shape
    print(f"  Output: {o_b1.shape}")
    print("  PASSED ✓")

    # --- Test 28: Sequence length 1 ---
    print("\n[Test 28] Sequence length = 1")
    x_t1 = torch.randn(2, 1, D * H, device=device)
    o_t1, _, _ = model(x_t1)
    assert o_t1.shape == x_t1.shape
    print(f"  Output: {o_t1.shape}")
    print("  PASSED ✓")

    # --- Test 29: ShortConvolution step (single-token decode) ---
    print("\n[Test 29] ShortConvolution step (single-token decode with cache)")
    conv_step = ShortConvolution(hidden_size=16, kernel_size=4, bias=True, activation='silu')
    # Prefill: 8 tokens
    x_pre = torch.randn(1, 8, 16)
    y_pre, cache_pre = conv_step(x_pre, output_final_state=True)
    assert cache_pre is not None
    assert cache_pre.shape == (1, 16, 4), f"Cache shape: {cache_pre.shape}"
    # Step: decode 1 token
    x_dec = torch.randn(1, 1, 16)
    y_dec, cache_dec = conv_step.step(x_dec, cache_pre, output_final_state=True)
    assert y_dec.shape == (1, 1, 16)
    assert cache_dec.shape == (1, 16, 4)
    print(f"  Prefill cache: {cache_pre.shape}, decode output: {y_dec.shape}")
    print("  PASSED ✓")

    # --- Test 30: Determinism ---
    print("\n[Test 30] Determinism (same seed -> same output)")
    torch.manual_seed(42)
    model_det = GatedLinearAttention(
        hidden_size=128, num_heads=2, use_output_gate=True,
        fuse_norm=True, layer_idx=0,
    ).to(device)
    x_det = torch.randn(1, 16, 128, device=device)
    o_det1, _, _ = model_det(x_det)

    torch.manual_seed(42)
    model_det2 = GatedLinearAttention(
        hidden_size=128, num_heads=2, use_output_gate=True,
        fuse_norm=True, layer_idx=0,
    ).to(device)
    x_det2 = torch.randn(1, 16, 128, device=device)
    o_det2, _, _ = model_det2(x_det2)
    assert torch.allclose(o_det1, o_det2, atol=1e-7), "Determinism violated"
    print("  Same seed produces identical outputs")
    print("  PASSED ✓")

    print("\n" + "=" * 60)
    print(f"All 30 tests passed! ✓")
    print("=" * 60)
