"""Multi-chip all-reduce with pmap.

Demonstrates how to use :func:`jax.pmap` together with
:func:`jax.lax.psum` to perform a data-parallel gradient all-reduce across
all local TPU chips (cores).

Each device computes a local gradient and the all-reduce sums them together
so every device ends up with the globally accumulated gradient, exactly as
happens during synchronous data-parallel training.

Run on a multi-chip TPU VM:

    python examples/pmap_allreduce.py
"""

import jax
import jax.numpy as jnp

from tpu_features.parallelism import replicate, unreplicate


AXIS = "devices"


@jax.pmap(axis_name=AXIS)
def compute_and_reduce(params, inputs, labels):
    """Compute per-device loss + gradient, then all-reduce the gradient."""
    def loss_fn(p):
        logits = inputs @ p
        return jnp.mean((logits - labels) ** 2)

    loss, grad = jax.value_and_grad(loss_fn)(params)
    # Sum gradients across all devices (equivalent to AllReduce in other frameworks).
    grad = jax.lax.psum(grad, axis_name=AXIS)
    return loss, grad


def main() -> None:
    n_devices = jax.local_device_count()
    print(f"JAX backend : {jax.default_backend()}")
    print(f"Local devices: {n_devices}\n")

    key = jax.random.PRNGKey(0)
    feature_dim = 128

    # Initialise parameters and replicate across devices.
    params = jax.random.normal(key, (feature_dim,))
    params_rep = replicate(params)

    # Create per-device batches (leading axis = device axis).
    batch_size_per_device = 32
    inputs = jax.random.normal(key, (n_devices, batch_size_per_device, feature_dim))
    labels = jax.random.normal(key, (n_devices, batch_size_per_device))

    loss, grad = compute_and_reduce(params_rep, inputs, labels)

    print(f"Per-device losses : {loss}")
    print(f"Gradient (device 0, first 4 elements): {unreplicate(grad)[:4]}")
    print(
        "\nAll-reduce successful – each device holds the summed gradient "
        "from all devices."
    )


if __name__ == "__main__":
    main()
