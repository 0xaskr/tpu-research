import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# =============================================================================
# ShortConvolution (替代 fla.modules.ShortConvolution, 纯 PyTorch)
# 基于 nn.Conv1d 的因果深度可分离卷积
# =============================================================================

class ShortConvolution(nn.Conv1d):
    """Causal depthwise 1D convolution (纯 CPU 版本).

    - groups = hidden_size (深度可分离)
    - causal padding = kernel_size - 1 (左侧 padding)
    - 可选 SiLU 激活
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ('silu', 'swish'):
            return F.silu(x)
        return x

    def _causal_conv1d(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Apply causal conv1d.
        x: [B, T, D] -> [B, T, D]  (depthwise causal convolution + optional SiLU)
        """
        W = self.kernel_size[0]

        if cu_seqlens is not None:
            # 变长序列：按 cu_seqlens 分段独立做因果卷积
            B, T_total, D = x.shape
            assert B == 1, "cu_seqlens requires B=1"
            N = len(cu_seqlens) - 1
            segments = []
            for i in range(N):
                bos = cu_seqlens[i].item()
                eos = cu_seqlens[i + 1].item()
                seg = x[:, bos:eos, :]  # [1, seg_len, D]
                seg = rearrange(seg, 'b t d -> b d t')  # [1, D, seg_len]
                # 手动左 padding
                seg_padded = F.pad(seg, (W - 1, 0))
                seg_out = F.conv1d(seg_padded, self.weight, self.bias, groups=self.hidden_size)
                seg_out = rearrange(seg_out, 'b d t -> b t d')
                segments.append(seg_out)
            y = torch.cat(segments, dim=1)
        else:
            # 标准情况：整个序列一起做卷积
            x_t = rearrange(x, 'b t d -> b d t')
            # nn.Conv1d with padding=kernel_size-1 会在两边 pad
            # 我们需要因果卷积，所以手动处理
            x_padded = F.pad(x_t, (W - 1, 0))
            y = F.conv1d(x_padded, self.weight, self.bias, groups=self.hidden_size)
            y = rearrange(y, 'b d t -> b t d')

        return self._apply_activation(y)

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-step decoding with cache update.
        单步解码，使用缓存更新。

        x: [B, 1, D] or [1, N, D] with cu_seqlens
        cache: [N, D, W] where W is kernel_size
        """
        W = self.kernel_size[0]
        B = x.shape[0]
        D = self.hidden_size
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)

        # 取出当前 token
        x_step = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)  # [N, D] or [B, D]

        if cache is not None:
            # 滚动 cache 并将新 token 放入最后位置
            cache = cache.roll(shifts=-1, dims=-1)
            cache[:, :, -1] = x_step
            # 与卷积权重做点积
            w = rearrange(self.weight, 'd 1 w -> d w')
            y = (cache * w).sum(dim=-1)  # [N, D]
            if self.bias is not None:
                y = y + self.bias
        else:
            # 没有 cache 的情况，直接计算
            w = rearrange(self.weight, 'd 1 w -> d w')
            # 只用最后一个权重
            y = x_step * w[:, -1]
            if self.bias is not None:
                y = y + self.bias

        y = self._apply_activation(y)
        y = y.view(x.shape)
        return y, cache

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [B, T, D]  — 投影后的 q/k/v (D = key_dim 或 value_dim_per_group 等)
        # cache: [N, D, W] 或 None (W = kernel_size)
        # cu_seqlens: [N+1] 或 None
        # -> y: [B, T, D], final_state: [N, D, W] 或 None
        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        W = self.kernel_size[0]

        # Decoding mode: single token per sequence
        if B * T == N:
            return self.step(x, cache, output_final_state, cu_seqlens)

        # Prefill / training mode
        y = self._causal_conv1d(x, cu_seqlens)

        # 计算 final state (最后 W 个 token 的窗口)
        final_state = None
        if output_final_state:
            if cu_seqlens is not None:
                final_states = []
                for i in range(N):
                    bos = cu_seqlens[i].item()
                    eos = cu_seqlens[i + 1].item()
                    seg = x[0, bos:eos, :]  # [seg_len, D]
                    seg_t = rearrange(seg, 't d -> d t')
                    # 左 pad 保证至少 W 个 token
                    if seg_t.shape[-1] < W:
                        seg_t = F.pad(seg_t, (W - seg_t.shape[-1], 0))
                    final_states.append(seg_t[:, -W:])  # [D, W]
                final_state = torch.stack(final_states, dim=0)  # [N, D, W]
            else:
                x_t = rearrange(x, 'b t d -> b d t')
                if T < W:
                    x_t = F.pad(x_t, (W - T, 0))
                final_state = x_t[:, :, -W:]  # [B, D, W]

        # 如果有传入 cache，将 final_state 写入
        if cache is not None and final_state is not None:
            cache.copy_(final_state)
            final_state = cache

        return y, final_state