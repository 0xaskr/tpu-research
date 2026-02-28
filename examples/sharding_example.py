"""Named sharding on a device mesh (GSPMD).

Demonstrates the modern JAX sharding API introduced in JAX 0.4+.  Instead
of the legacy ``pmap`` approach, we:

1. Build a logical 2-D device mesh with named axes (``batch`` and ``model``).
2. Annotate parameter and activation tensors with :class:`~jax.sharding.NamedSharding`.
3. Use ``jax.jit`` – the XLA compiler inserts the necessary collectives
   automatically.

This pattern scales from 8-chip TPU v4 pods all the way to 6 000-chip
v5p superclusters without changing application code.

Run on a multi-chip TPU VM:

    python examples/sharding_example.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_features.parallelism import create_device_mesh, make_named_sharding


def main() -> None:
    print(f"JAX backend  : {jax.default_backend()}")
    n = jax.device_count()
    print(f"Total devices: {n}\n")

    # Build a mesh.  Adjust the shape to match your hardware.
    # On a single-chip system this degrades gracefully to (1, 1).
    if n >= 8:
        mesh_shape = (2, n // 2)
    elif n >= 4:
        mesh_shape = (2, 2)
    elif n >= 2:
        mesh_shape = (1, 2)
    else:
        mesh_shape = (1, 1)

    mesh = create_device_mesh(mesh_shape, axis_names=("batch", "model"))
    print(f"Mesh shape   : {mesh.shape}\n")

    # Shard a weight matrix across the model axis.
    # Each device holds a horizontal slice of W.
    W_sharding = make_named_sharding(mesh, P(None, "model"))

    # Shard a batch of inputs across the batch axis.
    x_sharding = make_named_sharding(mesh, P("batch", None))

    hidden_dim = 512
    embed_dim = 256
    batch_size = 64

    key = jax.random.PRNGKey(42)
    W = jax.device_put(jax.random.normal(key, (embed_dim, hidden_dim)), W_sharding)
    x = jax.device_put(jax.random.normal(key, (batch_size, embed_dim)), x_sharding)

    print(f"W sharding   : {W.sharding}")
    print(f"x sharding   : {x.sharding}\n")

    @jax.jit
    def forward(params, inputs):
        return jnp.dot(inputs, params)

    out = forward(W, x)
    print(f"Output shape : {out.shape}")
    print(f"Output sharding: {out.sharding}")
    print(
        "\nSharded forward pass complete – XLA inserted all necessary "
        "collectives automatically."
    )


if __name__ == "__main__":
    main()
