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
        input_dtype = x.dtype
        x = x.float()
        g = g.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        if self.weight is not None:
            x = x * self.weight.float()
        x = x * F.silu(g)
        return x.to(input_dtype)

