"""BFloat16 matrix multiply benchmark on TPU.

This script demonstrates the recommended approach for performing large matrix
multiplications on a TPU:

1. Cast inputs to BFloat16 (native MXU dtype, ~4× faster than FP32).
2. Pad dimensions to multiples of 128 (avoids MXU tile-padding overhead).
3. Wrap the computation in ``jax.jit`` so it runs entirely on-device.

Run this script on a TPU VM or Google Colab TPU runtime:

    python examples/matmul_bfloat16.py
"""

import time

import jax
import jax.numpy as jnp

from tpu_features.optimizations import cast_bfloat16, pad_to_multiple, tpu_matmul


def benchmark_matmul(m: int, k: int, n: int, warmup: int = 3, repeats: int = 10) -> float:
    """Return the median wall-clock time (seconds) for an M×K @ K×N matmul."""
    a = jnp.ones((m, k), dtype=jnp.float32)
    b = jnp.ones((k, n), dtype=jnp.float32)

    jitted = jax.jit(tpu_matmul)

    # Warm up – compiles the function and fills caches.
    for _ in range(warmup):
        _ = jitted(a, b).block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jitted(a, b).block_until_ready()
        times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2]  # median


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}\n")

    shapes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for m, k, n in shapes:
        elapsed = benchmark_matmul(m, k, n)
        flops = 2 * m * k * n  # multiply-add
        tflops = flops / elapsed / 1e12
        print(f"  [{m}×{k}] @ [{k}×{n}]  →  {elapsed * 1e3:.2f} ms  ({tflops:.2f} TFLOPS)")


if __name__ == "__main__":
    main()
