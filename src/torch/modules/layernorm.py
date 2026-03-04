import torch
import torch.nn as nn

# =============================================================================
# RMSNorm (替代 fla.modules.RMSNorm, 纯 PyTorch)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (纯 CPU 版本).

    公式: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype or torch.float32))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(input_dtype)
