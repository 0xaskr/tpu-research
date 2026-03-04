import os
import sys
from pathlib import Path
from typing import final

import torch
import numpy as np
import traceback
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.torch.ops.gla import fused_recurrent_gla_fwd as pt_fused_recurrent_gla_fwd
from src.jax.ops.pallas_gla import fused_recurrent_gla_fwd as pl_fused_recurrent_gla_fwd
from tests.utils import compare_tensor

try:
  # from src.jax.ops.gla import pallas_chunk_gated_delta_rule_fwd_h
  TPU_AVAILABLE = True
except (ImportError, Exception) as e:
  TPU_AVAILABLE = False
  _tpu_import_error = str(e)

def run_tpu_test(B:int, T:int, H:int, V:int, K:int,
            use_gk:bool, use_gv:bool,
            scale:float | None, use_init_state:bool, use_final_state: bool, reverse:bool,
            cu_seqlens: np.ndarray | None, dtype=np.float32):
  assert TPU_AVAILABLE, "TPU not available, cannot run TPU test"

  status = True
  q_cpu = np.random.random([B, T, H, K]).astype(dtype)
  k_cpu = np.random.random([B, T, H, K]).astype(dtype)
  v_cpu = np.random.random([B, T, H, V]).astype(dtype)
  gk_cpu = np.random.random([B, T, H, K]).astype(dtype) if use_gk else None
  gv_cpu = np.random.random([B, T, H, V]).astype(dtype) if use_gv else None
  scale = scale if scale is not None else K ** -0.5

  N = cu_seqlens.shape[-1] - 1 if cu_seqlens is not None else B
  init_state_cpu = np.random.random([N, H, K, V]).astype(dtype) if use_init_state else None

  q_pt = torch.from_numpy(q_cpu)
  k_pt = torch.from_numpy(k_cpu)
  v_pt = torch.from_numpy(v_cpu)
  gk_pt = torch.from_numpy(gk_cpu) if use_gk else None
  gv_pt = torch.from_numpy(gv_cpu) if use_gv else None
  init_state_pt = torch.from_numpy(init_state_cpu) if init_state_cpu is not None else None

  # tpu
  tpu_device = jax.devices("tpu")[0]
  q_jax = jnp.array(q_cpu, device=tpu_device)
  k_jax = jnp.array(k_cpu, device=tpu_device)
  v_jax = jnp.array(v_cpu, device=tpu_device)
  gk_jax = jnp.array(gk_cpu, device=tpu_device) if use_gk else None
  gv_jax = jnp.array(gv_cpu, device=tpu_device) if use_gv else None
  init_state_jax = jnp.array(init_state_cpu, device=tpu_device) if init_state_cpu is not None else None

  o_pt, final_state_pt = pt_fused_recurrent_gla_fwd(
      q=q_pt,
      k=k_pt,
      v=v_pt,
      gk=gk_pt,
      gv=gv_pt,
      scale=scale,
      initial_state=init_state_pt,
      output_final_state=use_final_state,
      reverse=reverse,
      cu_seqlens=torch.from_numpy(cu_seqlens) if cu_seqlens is not None else cu_seqlens,
  )
  results = pl_fused_recurrent_gla_fwd(
      q=q_jax,
      k=k_jax,
      v=v_jax,
      gk=gk_jax,
      gv=gv_jax,
      scale=scale,
      initial_state=init_state_jax,
      output_final_state=use_final_state,
      reverse=reverse,
      cu_seqlens=jnp.array(cu_seqlens) if cu_seqlens is not None else None,
  )
  if use_final_state:
    o_jax, final_state_jax = results
  else:
    o_jax, final_state_jax = results[0], None
  status = compare_tensor("Output", o_pt, o_jax) and status
  if use_final_state:
    status = compare_tensor("Final State", final_state_pt, final_state_jax) and status

  return status

def tpu_test():
  all_passed = True
  test_cases = [
    dict(name="Test Case 1: Basic functionality with all features enabled",
         B=2, T=4, H=2, V=8, K=8, use_gk=True, use_gv=False, scale=None, use_init_state=True, use_final_state=True, reverse=False, cu_seqlens=None, dtype=np.float32),
  ]


  for i, case in enumerate(test_cases):
    case_copy = case.copy()
    test_name = case_copy.pop("name", f"Case {i+1}")

    try:
        passed = run_tpu_test(**case_copy)
    except Exception as e:
        print(f"❌ Exception occurred:")
        traceback.print_exc()
        passed = False
    all_passed = all_passed and passed

  if all_passed:
    print("✅ All TPU test cases passed!")
  else:
    print("❌ Some TPU test cases failed.")



if __name__ == "__main__":
  success = True

  if TPU_AVAILABLE:
    success = tpu_test() and success
  else:
    print(f"TPU not available: {_tpu_import_error}")
