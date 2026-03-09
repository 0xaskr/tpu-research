import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from numpy import pad

from src.utils import cdiv, align_up, pad_to_multiple
from src.jax.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

__all__ = ["chunk_gla", "chunk_gla_fwd"]


def chunk_fwd_h(
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None=None,
    g_gamma: jax.Array | None=None,
    gk: jax.Array | None=None,
    gv: jax.Array | None=None,
    h0 : jax.Array | None=None,
    output_final_state: bool = False,
    cu_seqlens : jax.Array | None = None,
    chunk_size : int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False):
  B, T, H, K, V = *k.shape, v.shape[-1]
  BT = chunk_size
  BS = BT if split_size is None else split_size
  BK = 64
  BV = 64
  assert BK in [32, 64]
  assert BV in [32, 64]
  assert BS % BT == 0, "split_size must be a multiple of chunk_size"
  if cu_seqlens is None:
    N, NS, split_offsets = B, cdiv(T, BS), None
  else:
    split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
    N, NS = len(split_offsets) - 1, split_offsets[-1].item()

  if cu_seqlens is not None:
    return _chunk_fwd_h_varlen(k, v, g, g_gamma, gk, gv, h0, output_final_state, cu_seqlens, chunk_size, split_size, states_in_fp32)
  else:
    return _chunk_fwd_h(k, v, g, g_gamma, gk, gv, h0, output_final_state, chunk_size, split_size, states_in_fp32)

  # h = jnp.zeros((B, NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32)
  # ht = jnp.zeros((N, H, K, V), dtype=jnp.float32) if output_final_state else None
  # grid = (cdiv(K, BK), cdiv(V, BV), N*H)


def _chunk_fwd_h():
  pass

def _chunk_fwd_h_kernel():
  pass


def _chunk_fwd_h_varlen():
  pass

def _chunk_fwd_h_varlen_kernel():
  pass

def chunk_gla_fwd(
    q:jax.Array,
    k:jax.Array,
    v:jax.Array,
    g:jax.Array,
    g_cumsum:jax.Array | None,
    scale:float,
    initial_state:jax.Array,
    output_final_state:bool,
    cu_seqlens:jax.Array | None=None,
    chunk_size:int=64
):
  if g_cumsum is None:
    g_cumsum = jnp.zeros_like(g)

  h, ht = chunk_fwd_h(
    k=k,
    v=v,
    g=None,
    gk=g_cumsum,
    gv=None,
    h0=initial_state,
    output_final_state=output_final_state,
    states_in_fp32=False,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
  )

  A = chunk_gla_fwd_intra_gk(
    q=q,
    k=k,
    g=g_cumsum,
    scale=scale,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
  )

  o = chunk_gla_fwd_o_gk(
    q=q,
    v=v,
    g=g_cumsum,
    A=A,
    h=h,
    scale=scale,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
  )
  return g_cumsum, A, h, ht, 0

def chunk_gla_fwd_o_gk(
    q: jax.Array,          # [B, T, H, K]
    v: jax.Array,          # [B, T, H, V]
    g: jax.Array,          # [B, T, H, K]
    A: jax.Array,          # ([B, T, H, BT] or similar structure depending on chunking)
    h: jax.Array,          # [B, NT, H, K, V]
    scale: float,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    use_exp2: bool = False
):

  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  N = B if cu_seqlens is None else cu_seqlens.shape[-1] - 1
  NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

  assert q.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert g.shape == (B, T, H, K)
  assert A.shape == (B, T, H, BT)
  assert h.shape == (B, NT, H, K, V)
  assert (cu_seqlens is None) or (cu_seqlens.shape == (N+1,))
  assert (cu_seqlens is None) or ((cu_seqlens is not None) and (chunk_indices is not None))
  assert (cu_seqlens is None) or ((cu_seqlens is not None) and B == 1)
  assert (cu_seqlens is None) or (cu_seqlens.dtype == jnp.int32)
  assert (chunk_indices is None) or (chunk_indices.dtype == jnp.int32)

  if cu_seqlens is None:
    assert T % chunk_size == 0, "For non-varlen input, T must be divisible by chunk_size"
    # TODO(0xaskr): support non-aligned cu_seqlens by padding inside the kernel.
    # but for now we require alignment for simplicity.
    return _chunk_gla_fwd_o_gk(q, v, g, A, h, scale, chunk_size, use_exp2)
  else:
    cpu_device = jax.devices("cpu")[0]
    cu_seqlens_cpu = jax.device_put(cu_seqlens, cpu_device)
    assert jnp.all(cu_seqlens_cpu % chunk_size == 0), "cu_seqlens offset must be aligned to chunk_size"
    # TODO(0xaskr): support non-aligned cu_seqlens by padding inside the kernel.
    # but for now we require alignment for simplicity.

    if chunk_indices is None:
      chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return _chunk_gla_fwd_o_gk_varlen(
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2
    )

def _chunk_gla_fwd_o_gk(q : jax.Array,
                       v : jax.Array,
                       g : jax.Array,
                       a : jax.Array,
                       h : jax.Array,
                       scale: float,
                       chunk_size: int = 64,
                       use_exp2: bool = False) -> jax.Array:
  """
  Non-varlen GLA chunk forward output: o = (q * exp(g)) @ h * scale + A @ v

  Uses the same "load full sequence per head + pl.ds" block spec design as the
  varlen kernel to ensure correctness on TPU.
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)
  BK, BV = 64, 128

  assert q.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert g.shape == (B, T, H, K)
  assert a.shape == (B, T, H, BT)
  assert h.shape == (B, NT, H, K, V)

  orig_T = T
  orig_V = V

  # Pad T to a multiple of BT to avoid OOB access in the last chunk.
  if T % BT != 0:
    pad_T = cdiv(T, BT) * BT - T
    q = jnp.pad(q, ((0, 0), (0, pad_T), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, pad_T), (0, 0), (0, 0)))
    g = jnp.pad(g, ((0, 0), (0, pad_T), (0, 0), (0, 0)))
    a = jnp.pad(a, ((0, 0), (0, pad_T), (0, 0), (0, 0)))
    T = T + pad_T
    NT = cdiv(T, BT)

  # Pad K and V to multiples of BK/BV.
  # TPU requires block[-1] divisible by 128 or equal to array's last dim.
  # With BV=128, V padded to a multiple of 128 satisfies both conditions.
  K_padded = align_up(K, BK)
  V_padded = align_up(V, BV)
  q = pad_to_multiple(q, BK, 3, 0)
  g = pad_to_multiple(g, BK, 3, 0)
  v = pad_to_multiple(v, BV, 3, 0)
  h = pad_to_multiple(h, [BK, BV], [3, 4], 0)

  if K_padded > K or V_padded > V:
    h = jnp.pad(h, ((0, 0), (0, 0), (0, 0), (0, K_padded - K), (0, V_padded - V)))
  K, V = K_padded, V_padded

  # Transpose: (B, T, H, K) -> (H, B, T, K) so each head's data is contiguous.
  # The block spec loads one head's full (B, T, K/V) into VMEM per grid point,
  # then the kernel reshapes to (B*T, K/V) and uses pl.ds for time/K/V slicing.
  q = jnp.transpose(q, (2, 0, 1, 3))   # (H, B, T, K)
  v = jnp.transpose(v, (2, 0, 1, 3))   # (H, B, T, V)
  g = jnp.transpose(g, (2, 0, 1, 3))   # (H, B, T, K)
  a = jnp.transpose(a, (2, 0, 1, 3))   # (H, B, T, BT)
  h = jnp.transpose(h, (2, 0, 1, 3, 4))  # (H, B, NT, K, V)

  # Block specs: each grid point (i_v, i_t, i_bh) loads the full (B, T, K/V)
  # for head i_bh % H. block[-2] = T (equal to array's dim -> ✓),
  # block[-1] = K (equal to array's K -> ✓) or = V=multiple of 128 (divisible by 128 -> ✓).
  q_blockspec = pl.BlockSpec([1, B, T, K], index_map=lambda v, nt, bh: (bh % H, 0, 0, 0))
  v_blockspec = pl.BlockSpec([1, B, T, V], index_map=lambda v, nt, bh: (bh % H, 0, 0, 0))
  g_blockspec = pl.BlockSpec([1, B, T, K], index_map=lambda v, nt, bh: (bh % H, 0, 0, 0))
  h_blockspec = pl.BlockSpec([1, B, NT, K, V], index_map=lambda v, nt, bh: (bh % H, 0, 0, 0, 0))
  a_blockspec = pl.BlockSpec([1, B, T, BT], index_map=lambda v, nt, bh: (bh % H, 0, 0, 0))

  # Output: (B*H, NT, NV, BT, BV) — each grid point writes a unique block.
  NV = cdiv(V, BV)
  o_shape = jax.ShapeDtypeStruct([B * H, NT, NV, BT, BV], v.dtype)
  o_blockspec = pl.BlockSpec([1, 1, 1, BT, BV], index_map=lambda v, nt, bh: (bh, nt, v, 0, 0))

  grid = (NV, NT, B * H)
  o = pl.pallas_call(
    functools.partial(
        _chunk_gla_fwd_o_gk_kernel,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        USE_EXP2=use_exp2,
    ),
    grid=grid,
    out_shape=[o_shape],
    in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, a_blockspec],
    out_specs=[o_blockspec],
  )(q, v, g, h, a)

  if isinstance(o, (tuple, list)):
    o = o[0]

  # Post-process: (B*H, NT, NV, BT, BV) -> (B, T_padded, H, V_padded)
  o = o.transpose(0, 1, 3, 2, 4)    # (B*H, NT, BT, NV, BV)
  o = o.reshape(B, H, NT, BT, V)    # split B*H -> (B,H), merge NV*BV -> V
  o = o.reshape(B, H, NT * BT, V)   # merge NT*BT -> T_padded
  o = o.transpose(0, 2, 1, 3)       # (B, T_padded, H, V_padded)

  # Trim to original dimensions.
  if T > orig_T:
    o = o[:, :orig_T, ...]
  if V > orig_V:
    o = o[..., :orig_V]
  return o

# ================= gla_fwd_o_gk start =================
def _chunk_gla_fwd_o_gk_kernel(q_ref, v_ref, g_ref, h_ref, a_ref,
                              o_ref,
                              scale, B, T, H, K, V, BT, BK, BV, NT, USE_EXP2):
  """
  Non-varlen GLA forward O+GK kernel.

  Block specs load the full sequence per head:
    q/g/a: (1, B, T, K/BT)  v: (1, B, T, V)  h: (1, B, NT, K, V)
  These are reshaped to (B*T, dim) inside the kernel and accessed via pl.ds.
  Grid: (NV=cdiv(V,BV), NT, B*H). Each grid point owns a unique (1,1,1,BT,BV) output block.
  """
  i_v, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  i_b = i_bh // H
  i_tg = i_b * NT + i_t
  bos = i_b * T  # start of batch i_b in the (B*T) flattened layout

  m_s = jnp.arange(0, BT)[:, None] >= jnp.arange(0, BT)[None, :]
  b_o = jnp.zeros([BT, BV], dtype=jnp.float32)

  # Reshape Refs (not loaded arrays) so pl.ds dynamic slicing works on them.
  # q_ref/g_ref/v_ref/a_ref have block shape (1, B, T, K/V/BT).
  # h_ref has block shape (1, B, NT, K, V).
  q_2d = q_ref.reshape(B * T, K)
  g_2d = g_ref.reshape(B * T, K)
  h_3d = h_ref.reshape(B * NT, K, V)

  for i_k in range(cdiv(K, BK)):
    b_q = q_2d[pl.ds(bos + i_t * BT, BT), pl.ds(i_k * BK, BK)]
    b_g = g_2d[pl.ds(bos + i_t * BT, BT), pl.ds(i_k * BK, BK)].astype(jnp.float32)
    b_h = h_3d[i_tg, pl.ds(i_k * BK, BK), pl.ds(i_v * BV, BV)]
    if USE_EXP2:
      b_qg = (b_q * jnp.exp2(b_g)).astype(b_q.dtype)
    else:
      b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_o += jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                   precision=jax.lax.Precision.HIGHEST,
                   preferred_element_type=jnp.float32)

  b_o *= scale
  v_2d = v_ref.reshape(B * T, V)
  a_2d = a_ref.reshape(B * T, BT)
  b_v = v_2d[pl.ds(bos + i_t * BT, BT), pl.ds(i_v * BV, BV)]
  b_a = a_2d[pl.ds(bos + i_t * BT, BT), 0:BT]
  b_a = jnp.where(m_s, b_a, 0.0).astype(b_a.dtype)
  b_o += jnp.dot(b_a, b_v,
                 precision=jax.lax.Precision.HIGHEST,
                 preferred_element_type=jnp.float32)
  # Each grid point writes to a unique block — no write conflicts across the grid.
  o_ref[...] = b_o.reshape(1, 1, 1, BT, BV).astype(o_ref.dtype)

def _chunk_gla_fwd_o_gk_varlen(
    q: jax.Array,          # [B, T, H, K]
    v: jax.Array,          # [B, T, H, V]
    g: jax.Array,          # [B, T, H, K]
    A: jax.Array,          # ([B, T, H, BT] or similar structure depending on chunking)
    h: jax.Array,          # [B, H, K, V]
    scale: float,
    cu_seqlens: jax.Array,
    chunk_size: int,
    chunk_indices: jax.Array,
    use_exp2: bool,
) -> jax.Array:

  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  BK = 64
  BV = 128
  N = cu_seqlens.shape[-1] - 1
  NT = len(chunk_indices)

  # assert (chunk_indices is None) or (chunk_indices.shape == (cdiv(T,BT),2))
  assert BK in [32, 64]
  assert BV in [64, 128]

  # Padding Logic for Varlen to prevent OOB access in Pallas Kernel
  # The kernel reads fixed blocks of size BT. If the last chunk of a sequence
  # starts near the end of T (e.g., index 224 for T=256 with BT=64),
  # it will try to read up to 288. We must pad q, v, g, A.
  # We pad by BT to be safe.
  # Only apply padding if IS_VARLEN is true, as standard mode assumes perfect tiling or handles it differently.

  # Pad K and V dimensions to multiples of BK/BV for kernel block access
  orig_V = V
  K_padded = align_up(K, BK)
  V_padded = align_up(V, BV)
  q = pad_to_multiple(q, BK, 3, 0)
  g = pad_to_multiple(g, BK, 3, 0)
  v = pad_to_multiple(v, BV, 3, 0)
  h = pad_to_multiple(h, [BK, BV], [3, 4], 0)
  K = K_padded
  V = V_padded

  q_block_t_shape = [1, B, T, K]
  v_block_t_shape = [1, B, T, V]
  g_block_t_shape = [1, B, T, K]
  h_block_t_shape = [1, B, NT, K, V]
  A_block_t_shape = [1, B, T, BT]

  q_t = q.transpose(2, 0, 1, 3)
  v_t = v.transpose(2, 0, 1, 3)
  g_t = g.transpose(2, 0, 1, 3)
  h_t = h.transpose(2, 0, 1, 3, 4)
  A_t = A.transpose(2, 0, 1, 3)

  # Output: (B*H, NT, NV, BT, BV) where NV = cdiv(V, BV)
  # Each grid point (v, nt, bh) writes to a unique block.
  # This avoids TPU double-buffering corruption when multiple grid iterations
  # share the same output block, and also avoids TPU crash from non-contiguous
  # V-strided block access in 4D output.
  NV = cdiv(V, BV)
  o_shape = jax.ShapeDtypeStruct([B * H, NT, NV, BT, BV], v.dtype)
  o_blockspec = pl.BlockSpec([1, 1, 1, BT, BV], index_map=lambda v, nt, bh: (bh, nt, v, 0, 0))

  q_blockspec = pl.BlockSpec(q_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  v_blockspec = pl.BlockSpec(v_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  g_blockspec = pl.BlockSpec(g_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  h_blockspec = pl.BlockSpec(h_block_t_shape, index_map = lambda v, nt, bh: (bh%H, 0, 0, 0, 0))
  A_blockspec = pl.BlockSpec(A_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  cs_blockspec = pl.BlockSpec([N+1], index_map=lambda v, nt, bh: (0,), memory_space=pltpu.MemorySpace.SMEM)
  ci_blockspec = pl.BlockSpec([NT, 2], index_map=lambda v, nt, bh: (0, 0,), memory_space=pltpu.MemorySpace.SMEM)

  if cu_seqlens is None:
      cu_seqlens_arg = jnp.zeros((N+1,), dtype=jnp.int32)
      chunk_indices_arg = jnp.zeros((NT, 2), dtype=jnp.int32)
  else:
      cu_seqlens_arg = cu_seqlens
      chunk_indices_arg = chunk_indices

  grid = (cdiv(V, BV), NT, B * H)
  o = pl.pallas_call(
    functools.partial(
        _chunk_gla_fwd_o_gk_varlen_kernel,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_EXP2=use_exp2,
        scale=scale
    ),
    grid=grid,
    out_shape=o_shape,
    out_specs=o_blockspec,
    in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, A_blockspec,cs_blockspec, ci_blockspec],
  )(q_t,v_t,g_t,h_t,A_t,cu_seqlens_arg,chunk_indices_arg)

  # Post-process: (B*H, NT, NV, BT, BV) -> (B, T, H, orig_V)
  # bh = i_b * H + i_h (h changes fastest), so reshape to (B, H, ...)
  # Merge NV and BV dims to recover V: transpose to (B*H, NT, BT, NV, BV) then reshape
  o = o.transpose(0, 1, 3, 2, 4)  # (B*H, NT, BT, NV, BV)
  o = o.reshape(B, H, NT, BT, V)  # NV*BV = V

  # Scatter chunks to correct T positions based on chunk_indices
  # t_positions[nt] = cu_seqlens[chunk_indices[nt, 0]] + chunk_indices[nt, 1] * BT
  t_positions = cu_seqlens_arg[chunk_indices_arg[:, 0]] + chunk_indices_arg[:, 1] * BT
  # Expand to (NT, BT): all T indices covered by each chunk
  all_t = t_positions[:, None] + jnp.arange(BT)[None, :]  # (NT, BT)
  all_t = all_t.reshape(-1)  # (NT*BT,)
  # Transpose to (B, NT*BT, H, V) to avoid NumPy advanced indexing issue
  o_flat = o.reshape(B, H, NT * BT, V).transpose(0, 2, 1, 3)  # (B, NT*BT, H, V)
  final_o = jnp.zeros((B, T, H, V), dtype=o.dtype)
  final_o = final_o.at[0, all_t].set(o_flat[0])
  o = final_o  # already (B, T, H, V)

  if V_padded > orig_V:
      o = o[..., :orig_V]

  return o

def _chunk_gla_fwd_o_gk_varlen_kernel(
    # in
    q,
    v,
    g,
    h,
    A,
    cu_seqlens,
    chunk_indices,
    # out
    o,
    # static args
    T,
    H,
    K,
    V,
    BT,
    BK,
    BV,
    USE_EXP2,
    scale
):
  B, T, K = q.shape[1:]
  V = v.shape[-1]
  NT = h.shape[2]
  TOTAL_T = B * T
  TOTAL_NT = B * NT

  q = q.reshape(TOTAL_T, K)
  v = v.reshape(TOTAL_T, V)
  g = g.reshape(TOTAL_T, K)
  h = h.reshape(TOTAL_NT, K, V)
  A = A.reshape(TOTAL_T, BT)

  # i_t = 当前chunk/当前batch的哪个chunk
  i_v, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  i_b, i_h = i_bh // H, i_bh % H
  real_T = 0
  real_NT = 0
  i_tg = i_t
  i_n = chunk_indices[i_t, 0].astype(jnp.int32)
  i_t = chunk_indices[i_t, 1].astype(jnp.int32)
  bos = cu_seqlens[i_n].astype(jnp.int32)
  eos = cu_seqlens[i_n + 1].astype(jnp.int32)
  real_T = eos - bos
  real_NT = cdiv(real_T, BT)

  m_s = jnp.arange(0, BT)[:, None] >= jnp.arange(0, BT)[None,:]
  b_o = jnp.zeros([BT, BV], dtype=jnp.float32)

  for i_k in range(cdiv(K, BK)):
    b_q = q[pl.ds(bos+i_t * BT, BT), pl.ds(i_k * BK, BK)]
    b_g = g[pl.ds(bos+i_t * BT, BT), pl.ds(i_k * BK, BK)].astype(jnp.float32)
    b_h = h[i_tg, pl.ds(i_k * BK, BK), pl.ds(i_v * BV, BV)]
    if (USE_EXP2):
      b_qg = (b_q * jnp.exp2(b_g)).astype(b_q.dtype)
    else:
      b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    if i_k >= 0:
      b_o += jnp.dot(b_qg, b_h.astype(b_qg.dtype), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

  b_o *= scale
  b_v = v[pl.ds(bos+i_t* BT, BT), pl.ds(i_v * BV, BV)]
  b_A = A[pl.ds(bos+i_t* BT, BT), 0:BT]
  # Apply causal mask
  b_A = jnp.where(m_s, b_A, 0.0).astype(b_A.dtype)
  b_o += jnp.dot(b_A, b_v, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
  # Write to unique output block — each grid point owns its own block
  o[...] = b_o.reshape(1, 1, 1, BT, BV).astype(o.dtype)



class ChunkGLAFunction:
  def forawrd():
    pass
  def backward():
    pass
