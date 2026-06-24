from src.jax.probe_vmem import probe_vmem_bytes


def test_probe_vmem():
  result = probe_vmem_bytes()

  physical_mb = result["physical"] / (1024 * 1024)
  scoped_mb = result["scoped_limit"] / (1024 * 1024)

  print(f"\n=== VMEM Probe Results ===")
  print(f"Device: {result['device_kind']}")
  print(f"Physical VMEM: {result['physical']:,} bytes ({physical_mb:.0f} MB)")
  print(f"Scoped limit:  {result['scoped_limit']:,} bytes ({scoped_mb:.0f} MB)")

  assert result["physical"] > 0
  assert result["scoped_limit"] > 0
  assert result["scoped_limit"] <= result["physical"]


if __name__ == "__main__":
  test_probe_vmem()
