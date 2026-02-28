"""Tests for tpu_features.hardware – no JAX required."""

import pytest

from tpu_features.hardware import (
    MXU_TILE_SIZE,
    NATIVE_DTYPE,
    TPUGeneration,
    TPUSpec,
    compare_specs,
    get_tpu_spec,
    list_generations,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_mxu_tile_size():
    assert MXU_TILE_SIZE == 128


def test_native_dtype():
    assert NATIVE_DTYPE == "bfloat16"


# ---------------------------------------------------------------------------
# get_tpu_spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen,expected_tflops",
    [
        (TPUGeneration.V2, 45.0),
        (TPUGeneration.V3, 123.0),
        (TPUGeneration.V4, 275.0),
        (TPUGeneration.V5E, 197.0),
        (TPUGeneration.V5P, 459.0),
    ],
)
def test_get_tpu_spec_by_enum(gen, expected_tflops):
    spec = get_tpu_spec(gen)
    assert spec.bf16_tflops_per_chip == expected_tflops


@pytest.mark.parametrize("gen_str", ["v2", "v3", "v4", "v5e", "v5p"])
def test_get_tpu_spec_by_string(gen_str):
    spec = get_tpu_spec(gen_str)
    assert isinstance(spec, TPUSpec)
    assert spec.generation == gen_str


def test_get_tpu_spec_string_case_insensitive():
    spec = get_tpu_spec("V4")
    assert spec.bf16_tflops_per_chip == 275.0


def test_get_tpu_spec_invalid_raises():
    with pytest.raises(ValueError, match="Unknown TPU generation"):
        get_tpu_spec("v99")


def test_spec_is_immutable():
    spec = get_tpu_spec("v4")
    with pytest.raises((AttributeError, TypeError)):
        spec.bf16_tflops_per_chip = 999  # frozen dataclass


# ---------------------------------------------------------------------------
# TPUSpec derived properties
# ---------------------------------------------------------------------------


def test_hbm_capacity_bytes():
    spec = get_tpu_spec("v4")
    expected = int(32.0 * (1024 ** 3))
    assert spec.hbm_capacity_bytes == expected


def test_arithmetic_intensity_threshold_positive():
    for gen in TPUGeneration:
        spec = get_tpu_spec(gen)
        assert spec.arithmetic_intensity_threshold > 0


# ---------------------------------------------------------------------------
# list_generations
# ---------------------------------------------------------------------------


def test_list_generations_returns_all():
    gens = list_generations()
    assert set(gens) == {"v2", "v3", "v4", "v5e", "v5p"}


def test_list_generations_is_tuple():
    assert isinstance(list_generations(), tuple)


# ---------------------------------------------------------------------------
# compare_specs
# ---------------------------------------------------------------------------


def test_compare_specs_keys():
    result = compare_specs("v3", "v4")
    assert "bf16_tflops_per_chip" in result
    assert "hbm_capacity_gib" in result
    assert "interconnect_topology" in result


def test_compare_specs_values():
    result = compare_specs("v3", "v4")
    assert result["bf16_tflops_per_chip"] == (123.0, 275.0)


def test_compare_specs_same_generation():
    result = compare_specs("v4", "v4")
    for v_a, v_b in result.values():
        assert v_a == v_b
