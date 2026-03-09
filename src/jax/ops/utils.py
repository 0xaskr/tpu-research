import jax
import jax.numpy as jnp

from src.utils import next_power_of_2, cdiv, align_up, pad_to_multiple

def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    cu_seqlens_cpu: jax.Array | None = None,
) -> jax.Array:
    """
    Generate chunk indices for variable length sequences.

    Example:
        >>> cu_seqlens = torch.tensor([0, 2, 5])
        >>> chunk_size = 2
        >>> prepare_chunk_indices(cu_seqlens, chunk_size)
        tensor([[0, 0],
                [1, 0],
                [1, 1]])
        Explanation:
        - Sequence 0 (len 2): 1 chunk (chunk 0)
        - Sequence 1 (len 3): 2 chunks (chunk 0, chunk 1)

    Returns:
        torch.LongTensor: A tensor of shape [Num_Total_Chunks, 2].
        Each row is (sequence_id, chunk_id).
    """
    if cu_seqlens_cpu is not None:
        # Calculate number of chunks for each sequence: ceil(seq_len / chunk_size)
        indices = jnp.concatenate([jnp.array(jnp.arange(n), device=cu_seqlens_cpu.device)
                            for n in cdiv(jnp.diff(cu_seqlens_cpu), chunk_size).tolist()])
        # Stack sequence_id and chunk_id
        # indices.eq(0) finds where chunk_id resets to 0 (start of new sequence)
        # cumsum counts these resets to get sequence_id
        return jnp.array(jnp.stack([(indices == 0).cumsum(0) - 1, indices], 1), dtype=cu_seqlens_cpu.dtype, device=cu_seqlens_cpu.device)

    indices = jnp.concatenate([jnp.arange(n) for n in cdiv(jnp.diff(cu_seqlens), chunk_size).tolist()])
    return jnp.array(jnp.stack([(indices == 0).cumsum(0) - 1, indices], 1), dtype = cu_seqlens.dtype, device=cu_seqlens.device)

def prepare_chunk_offsets(seqlens: jax.Array, chunk_size:int):
  return jnp.pad(cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32), (1, 0), constant_values=0).cumsum(-1)
