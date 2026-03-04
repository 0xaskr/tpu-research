
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """从 attention mask 计算每个序列的长度."""
    return mask.sum(dim=-1, dtype=torch.int32)

def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype = torch.int32,
) -> torch.LongTensor:
    """从 attention mask 计算累积序列长度 (cu_seqlens)."""
    lens = prepare_lens_from_mask(mask)
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))



def index_put_first_axis(
    x: torch.Tensor, indices: torch.Tensor, first_axis_dim: int,
) -> torch.Tensor:
    """将紧凑张量放回展平的第一维度指定位置.

    x: [num_selected, ...], indices: [num_selected]
    returns: [first_axis_dim, ...]
    """
    assert indices.ndim == 1
    assert x.ndim >= 2
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    return y



def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """获取 unpadding 所需的索引数据.

    Args:
        attention_mask: [batch_size, seq_len], 1 表示有效，0 表示 padding.

    Returns:
        indices: 展平序列中有效 token 的索引.
        cu_seqlens: 累积序列长度 [batch_size + 1].
        max_seqlen_in_batch: 批次中最大序列长度.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch

def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """从展平的第一维度中选取指定索引的元素.

    x: [total_tokens, ...], indices: [num_selected]
    returns: [num_selected, ...]
    """
    assert x.ndim >= 2
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, 'b ... -> b (...)'), 0,
        repeat(indices, 'z -> z d', d=second_dim),
    ).reshape(-1, *other_shape)

def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """将紧凑张量重新填充为 [batch_size, seq_len, ...] 的稠密张量."""
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, '(b s) ... -> b s ...', b=batch_size)

