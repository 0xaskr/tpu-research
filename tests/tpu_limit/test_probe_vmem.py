import os

os.environ["LIBTPU_INIT_ARGS"] = (
  "--xla_mosaic_dump_to=./llo_ir/probe_vmem/pallas_ir/ "
  "--xla_jf_dump_to=./llo_ir/probe_vmem/jax_ir"
)

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
