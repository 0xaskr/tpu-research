import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from src.utils import next_power_of_2, cdiv

__all__ = ['fused_recurrent_gla_fwd']

def fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None):
  B, T, H, K = q.shape
  V = v.shape[-1]
  N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
  scale = scale if scale is not None else K ** -0.5

  assert k.shape == (B, T, H, K), f"Expected k shape {(B, T, H, K)}, got {k.shape}"
  assert v.shape == (B, T, H, V), f"Expected v shape {(B, T, H, V)}, got {v.shape}"
  assert (gk is None) or (gk.shape == (B, T, H, K)), f"Expected gk shape {(B, T, H, K)}, got {gk.shape}"
  assert (gv is None) or (gv.shape == (B, T, H, V)), f"Expected gv shape {(B, T, H, V)}, got {gv.shape}"
  assert (initial_state is None) or (initial_state.shape == (N, H, K, V)), f"Expected initial_state shape {(N, H, K, V)}, got {initial_state.shape}"
  assert (cu_seqlens is None) or (B == 1), f"Batch size must be 1 when using cu_seqlens, got {B}"
  assert scale is not None, "ignore pylance warning about unused variable `scale`, which is actually used in the kernel call"
  assert reverse == False, "Reverse mode is not yet implemented in the JAX version"
  # TODO(0xaskr) support it.
  assert K % 8 == 0, f"K must be a multiple of 8 for the current implementation, got {K}"
  # TODO(0xaskr): support non-multiple of 8 K in the JAX version, which would require padding the input and output.
  assert V % 8 == 0, f"V must be a multiple of 8 for the current implementation, got {V}"
  # TODO(0xaskr): support non-multiple of 8 V in the JAX version, which would require padding the input and output.

  if cu_seqlens is not None:
    o, ht = _fused_recurrent_gla_fwd_varlen(
      q=q,
      k=k,
      v=v,
      gk=gk,
      gv=gv,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      reverse=reverse,
      cu_seqlens=cu_seqlens,
    )
  else:
    o, ht = _fused_recurrent_gla_fwd(
      q=q,
      k=k,
      v=v,
      gk=gk,
      gv=gv,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      reverse=reverse,
      use_gk=gk is not None,
      use_gv=gv is not None,
      use_init_state=initial_state is not None,
      use_final_state=output_final_state,
    )
  return o, ht

@jax.jit(static_argnames=["use_gk", "use_gv", "use_init_state", "use_final_state"])
def _fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    use_gk: bool = False,
    use_gv: bool = False,
    use_init_state: bool = False,
    use_final_state: bool = False):
  B, T ,H, K, V = *q.shape, v.shape[-1]
  N = B
  BK, BV = min(next_power_of_2(K), 64), min(next_power_of_2(V), 64)
  # BK, BV = 128, 128
  NK, NV = cdiv(K, BK), cdiv(V, BV)
  h0 = initial_state
  ht = jnp.zeros((N, H, K, V), dtype=jnp.float32) if use_final_state else None
  o = jnp.zeros([NK, B, H, V, T], dtype=jnp.float32)
  o_spec = jax.ShapeDtypeStruct(o.shape, o.dtype)
  ht_spec = jax.ShapeDtypeStruct(ht.shape, ht.dtype) if ht is not None else None

  # [B, T, H, K] -> [B, H, K, T]
  q_trans = q.transpose(0, 2, 3, 1)
  k_trans = k.transpose(0, 2, 3, 1)
  v_trans = v.transpose(0, 2, 3, 1)
  gk_trans = gk.transpose(0, 2, 3, 1) if use_gk else None
  gv_trans = gv.transpose(0, 2, 3, 1) if use_gv else None
  h0_trans = h0 if use_init_state else None

  def qk_index_map(idx_v, idx_k, idx_nh):
    return (idx_nh // H, idx_nh % H, idx_k * BK, 0)

  def v_index_map(idx_v, idx_k, idx_nh):
    return (idx_nh // H, idx_nh % H, idx_v * BV, 0)

  def h0_index_map(idx_v, idx_k, idx_nh):
    return (idx_nh // H, idx_nh % H, idx_k * BK, idx_v * BV)

  def o_index_map(idx_v, idx_k, idx_nh):
    return (idx_k, idx_nh // H, idx_nh % H, idx_v * BV, 0)

  q_blockspec = pl.BlockSpec([1, 1, BK, T], qk_index_map)
  k_blockspec = pl.BlockSpec([1, 1, BK, T], qk_index_map)
  v_blockspec = pl.BlockSpec([1, 1, BV, T], v_index_map)
  gk_blockspec = pl.BlockSpec([1, 1, BK, T], qk_index_map) if use_gk else None
  gv_blockspec = pl.BlockSpec([1, 1, BV, T], v_index_map) if use_gv else None
  h0_blockspec = pl.BlockSpec([1, 1, K, V], h0_index_map) if use_init_state else None

  o_blockspec = pl.BlockSpec([1, 1, 1, BV, T], o_index_map)
  ht_blockspec = pl.BlockSpec([1, 1, BK, BV], h0_index_map) if use_final_state else None
  call_func = functools.partial(
    _fused_recurrent_gla_fwd_kernel,
    T=T,
    B=B,
    H=H,
    K=K,
    V=V,
    BK=BK,
    BV=BV,
    USE_G=False,
    USE_GK=gk is not None,
    USE_GV=gv is not None,
    OUTPUT_FINAL_STATE=output_final_state,
    REVERSE=reverse,
    SCALE=scale
  )

  grid = (NV, NK, N*H)
  results = pl.pallas_call(
    call_func,
    out_shape=o_spec if ht is None else [o_spec, ht_spec],
    grid=grid,
    in_specs=[q_blockspec, k_blockspec, v_blockspec, gk_blockspec, gv_blockspec, h0_blockspec],
    out_specs=[o_blockspec, ht_blockspec] if ht is not None else [o_blockspec],
  )(q_trans, k_trans, v_trans, gk_trans, gv_trans, h0_trans)
  if use_final_state:
    o, ht = results
  else:
    o, ht = results, None
  o = o.sum(0)
  return o, ht

def _fused_recurrent_gla_fwd_kernel(
  # in
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  gk: jax.Array,
  gv: jax.Array,
  h0: jax.Array,
  # out
  o: jax.Array,
  ht: jax.Array,
  # static args
  SCALE: float,
  T:int,
  B:int,
  H:int,
  K:int,
  V:int,
  BK:int,
  BV:int,
  USE_G: bool,
  USE_GK: bool,
  USE_GV: bool,
  OUTPUT_FINAL_STATE: bool,
  REVERSE: bool,
):
  print("q.shape = ", q.shape)
  q = q.reshape(BK, T).transpose(1, 0)
  k = k.reshape(BK, T).transpose(1, 0)
  v = v.reshape(BV, T).transpose(1, 0)
  # o = o.reshape(BV, T).transpose(1, 0)
  if USE_GK:
    gk = gk.reshape(BK, T).transpose(1, 0)
  if USE_GV:
    gv = gv.reshape(BV, T).transpose(1, 0)
  if h0 is not None:
    h0 = h0.reshape(K, V).transpose(1, 0)
  if ht is not None:
    ht = ht.reshape(K, V).transpose(1, 0)

  idx_v, idx_k, idx_nh = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  idx_n, idx_h = idx_nh // H, idx_nh % H

  ALL_T = B * T
  bos, eos = idx_n * T, idx_n * T + T
  o_k = idx_k * BK + jnp.arange(0, BK)
  o_v = idx_v * BV + jnp.arange(0, BV)
  m_k = o_k < K
  m_v = o_v < V
  m_h = m_k[:, None] & m_v[None, :]
  b_h = jnp.zeros((BK, BV), dtype=jnp.float32)

  for idx_t in range(0, T):
    if USE_GK:
      pass
    if USE_GV:
      pass
    b_q = jnp.where(m_k, q[idx_t,0:BK], 0).astype(jnp.float32) * SCALE
    b_k = jnp.where(m_k, k[idx_t,0:BK], 0).astype(jnp.float32)
    b_v = jnp.where(m_v, v[idx_t,0:BV], 0).astype(jnp.float32)

    b_h += b_k[:, None] * b_v[None, :]
    b_o = b_h * b_q[:,None]
    b_o = jnp.sum(b_o, axis=0)
    o[1, 1, 1, :, idx_t] = jnp.where(m_h, b_o[:, None], 0).astype(o.dtype)

  if OUTPUT_FINAL_STATE:
    ht = ht.at[:BK, :BV].set(b_h.astype(ht.dtype))


def _fused_recurrent_gla_fwd_varlen(q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    cu_seqlens: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False):

  # Placeholder for the actual implementation, which would mirror the logic of the PyTorch version.
  # This function would be called by the main fused_recurrent_gla_fwd and would contain the core recurrence logic.
  pass
  return None, None


