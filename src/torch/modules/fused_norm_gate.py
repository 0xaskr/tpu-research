import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# =============================================================================
# FusedRMSNormGated (替代 fla.modules.FusedRMSNormGated, 纯 PyTorch)
# =============================================================================

class FusedRMSNormGated(nn.Module):
    """RMSNorm + SiLU gate fusion (纯 CPU 版本).

    output = RMSNorm(x) * silu(g)
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, V]  — kernel 输出 (需要归一化)
        # g: [B, T, H, V]  — gate 值 (从 g_proj 投影 + rearrange 得到)
        # -> [B, T, H, V]  = RMSNorm(x) * SiLU(g)
        input_dtype = x.dtype
        x = x.float()
        g = g.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # [B, T, H, 1]
        x = x * rms                     # [B, T, H, V]
        if self.weight is not None:
            x = x * self.weight.float()  # [B, T, H, V]  element-wise
        x = x * F.silu(g)               # [B, T, H, V]
        return x.to(input_dtype)

