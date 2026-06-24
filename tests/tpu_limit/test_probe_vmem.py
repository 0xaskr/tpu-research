import os
import sys

from src.jax.probe_vmem import probe_vmem_bytes


def test_probe_vmem():
  vmem = probe_vmem_bytes()
  vmem_mb = vmem / (1024 * 1024)
  print(f"\n=== VMEM Probe Results ===")
  print(f"Max allocatable VMEM: {vmem:,} bytes ({vmem_mb:.1f} MB)")
  assert vmem > 0, "VMEM probe returned 0 — allocation failed at 1MB"
  assert vmem <= 256 * 1024 * 1024, (
    f"VMEM probe returned {vmem_mb:.1f} MB — "
    "unexpectedly large, check probe logic"
  )


if __name__ == "__main__":
  test_probe_vmem()
