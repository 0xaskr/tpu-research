"""TPU hardware characteristics: specifications, constants, and helper utilities.

Each TPU generation has different peak compute, memory capacity, and
interconnect topology.  The constants and helpers in this module let
application code query these properties without hard-coding values
scattered throughout the codebase.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Native MXU tile size (rows and columns).  Shapes that are multiples of this
#: value map directly onto the hardware without any padding overhead.
MXU_TILE_SIZE: int = 128

#: BFloat16 is the native numeric format of the TPU MXU.  Computations in this
#: dtype achieve the full advertised FLOPS rating.
NATIVE_DTYPE: str = "bfloat16"

#: Preferred memory alignment for on-chip VMEM transfers (bytes).
VMEM_ALIGNMENT_BYTES: int = 128

#: Typical PCIe latency (µs) for a host↔device transfer initiation.
HOST_DEVICE_TRANSFER_LATENCY_US: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPUSpec:
    """Immutable specification record for a single TPU generation."""

    generation: str
    """Human-readable generation label, e.g. ``"v4"``."""

    bf16_tflops_per_chip: float
    """Peak BFloat16 TFLOPS per single chip (both cores)."""

    hbm_capacity_gib: float
    """Total HBM capacity per chip in gibibytes."""

    hbm_bandwidth_tbps: float
    """Aggregate HBM bandwidth per chip in TB/s."""

    ici_bandwidth_gbps: float
    """Inter-chip interconnect bandwidth per link in GB/s."""

    interconnect_topology: str
    """Physical network topology, e.g. ``"2D torus"`` or ``"3D torus"``."""

    mxu_tile_size: int = MXU_TILE_SIZE
    """MXU tile dimension; always 128 for current TPU generations."""

    @property
    def hbm_capacity_bytes(self) -> int:
        """Return HBM capacity in bytes."""
        return int(self.hbm_capacity_gib * (1024 ** 3))

    @property
    def arithmetic_intensity_threshold(self) -> float:
        """FLOP/byte ratio at which compute becomes the bottleneck (roofline).

        Operations with arithmetic intensity *above* this value are
        compute-bound; those below are memory-bandwidth-bound.
        """
        peak_flops = self.bf16_tflops_per_chip * 1e12
        peak_bandwidth = self.hbm_bandwidth_tbps * 1e12
        return peak_flops / peak_bandwidth


# ---------------------------------------------------------------------------
# Generation enum & registry
# ---------------------------------------------------------------------------


class TPUGeneration(str, enum.Enum):
    """Supported TPU hardware generations."""

    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5E = "v5e"
    V5P = "v5p"


_TPU_SPECS: dict[TPUGeneration, TPUSpec] = {
    TPUGeneration.V2: TPUSpec(
        generation="v2",
        bf16_tflops_per_chip=45.0,
        hbm_capacity_gib=8.0,
        hbm_bandwidth_tbps=0.6,
        ici_bandwidth_gbps=496.0,
        interconnect_topology="2D torus",
    ),
    TPUGeneration.V3: TPUSpec(
        generation="v3",
        bf16_tflops_per_chip=123.0,
        hbm_capacity_gib=16.0,
        hbm_bandwidth_tbps=0.9,
        ici_bandwidth_gbps=656.0,
        interconnect_topology="2D torus",
    ),
    TPUGeneration.V4: TPUSpec(
        generation="v4",
        bf16_tflops_per_chip=275.0,
        hbm_capacity_gib=32.0,
        hbm_bandwidth_tbps=1.2,
        ici_bandwidth_gbps=1200.0,
        interconnect_topology="3D torus (OCS)",
    ),
    TPUGeneration.V5E: TPUSpec(
        generation="v5e",
        bf16_tflops_per_chip=197.0,
        hbm_capacity_gib=16.0,
        hbm_bandwidth_tbps=0.82,
        ici_bandwidth_gbps=1600.0,
        interconnect_topology="2D torus",
    ),
    TPUGeneration.V5P: TPUSpec(
        generation="v5p",
        bf16_tflops_per_chip=459.0,
        hbm_capacity_gib=95.0,
        hbm_bandwidth_tbps=2.76,
        ici_bandwidth_gbps=4800.0,
        interconnect_topology="3D torus",
    ),
}


def get_tpu_spec(generation: TPUGeneration | str) -> TPUSpec:
    """Return the :class:`TPUSpec` for a given TPU generation.

    Parameters
    ----------
    generation:
        A :class:`TPUGeneration` enum value or a plain string such as
        ``"v4"``, ``"v5e"``, etc.

    Returns
    -------
    TPUSpec
        Immutable specification record for the requested generation.

    Raises
    ------
    ValueError
        If *generation* is not recognised.

    Examples
    --------
    >>> spec = get_tpu_spec("v4")
    >>> spec.bf16_tflops_per_chip
    275.0
    """
    if isinstance(generation, str):
        try:
            generation = TPUGeneration(generation.lower())
        except ValueError as exc:
            valid = [g.value for g in TPUGeneration]
            raise ValueError(
                f"Unknown TPU generation {generation!r}. Valid values: {valid}"
            ) from exc
    return _TPU_SPECS[generation]


def list_generations() -> Tuple[str, ...]:
    """Return all known TPU generation labels in chronological order.

    Examples
    --------
    >>> list_generations()
    ('v2', 'v3', 'v4', 'v5e', 'v5p')
    """
    return tuple(g.value for g in TPUGeneration)


def compare_specs(gen_a: TPUGeneration | str, gen_b: TPUGeneration | str) -> dict:
    """Return a side-by-side comparison dictionary of two TPU specs.

    Parameters
    ----------
    gen_a, gen_b:
        Generation identifiers accepted by :func:`get_tpu_spec`.

    Returns
    -------
    dict
        Mapping of field name → ``(value_a, value_b)`` tuples.
    """
    spec_a = get_tpu_spec(gen_a)
    spec_b = get_tpu_spec(gen_b)
    fields = [
        "bf16_tflops_per_chip",
        "hbm_capacity_gib",
        "hbm_bandwidth_tbps",
        "ici_bandwidth_gbps",
        "interconnect_topology",
    ]
    return {field: (getattr(spec_a, field), getattr(spec_b, field)) for field in fields}
