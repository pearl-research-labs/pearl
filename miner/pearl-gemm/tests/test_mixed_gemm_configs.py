"""Correctness coverage for mixed-GEMM scheduling configurations.

``C''`` is the peel/unscale output and is checked to tolerance against
host FP8-upcast peel algebra. Repeat launches under a given config must
be bit-identical. Invalid configs are rejected on the host.
"""

from dataclasses import asdict, replace

import pytest
import torch

from pearl_gemm import (
    HitSignal,
    HitSignalConfig,
    MixedGemmConfig,
    default_mixed_gemm_config,
    mixed_gemm,
    validate_mixed_gemm_config,
)
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.autotune import MIXED_GEMM_SPACE
from pearl_gemm.protocol_constants import R

_ARCH = arch_of()

# The family's library default (``default_mixed_gemm_config``).
_DEFAULT_CONFIG = asdict(default_mixed_gemm_config(_ARCH))

# Per-model / non-default smoke points on the PR gate: single-CTA fallback,
# explicit tile_k, and every committed lottery width. SM100 adds clustered
# and 2-CTA points; SM120 has neither. Full MIXED_GEMM_SPACE stays @slow.
_SM100_PER_MODEL_CONFIGS = [
    {"tile_m": 256, "tile_n": 128, "cluster_m": 1, "cluster_n": 1},
    {"tile_m": 256, "tile_n": 128, "cluster_m": 2, "cluster_n": 1, "tile_k": 128},
    {
        "tile_m": 128,
        "tile_n": 192,
        "cluster_m": 2,
        "cluster_n": 1,
        "ltile_cols": 64,
    },
    {
        "tile_m": 256,
        "tile_n": 64,
        "cluster_m": 2,
        "cluster_n": 1,
        "ltile_cols": 64,
    },
    {
        "tile_m": 256,
        "tile_n": 128,
        "cluster_m": 2,
        "cluster_n": 2,
        "ltile_cols": 64,
    },
    {
        "tile_m": 256,
        "tile_n": 192,
        "cluster_m": 1,
        "cluster_n": 1,
        "ltile_cols": 192,
    },
    {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_m": 1,
        "cluster_n": 1,
        "ltile_cols": 256,
    },
    # 64-row CTA tiles: the decode (64, 64) point (16 messages per CTA, so
    # the sub-warp hashing gate runs) and a clustered 128-wide-lottery point.
    {
        "tile_m": 64,
        "tile_n": 64,
        "cluster_m": 1,
        "cluster_n": 1,
        "ltile_cols": 64,
    },
    {
        "tile_m": 64,
        "tile_n": 128,
        "cluster_m": 2,
        "cluster_n": 1,
    },
]

# SM90 has tile_n >= 128 and only the 4-row family.
_SM90_PER_MODEL_CONFIGS = [
    {"tile_m": 256, "tile_n": 128, "cluster_m": 2, "cluster_n": 1, "tile_k": 128},
    {"tile_m": 128, "tile_n": 128, "cluster_m": 1, "cluster_n": 1},
    {"tile_m": 128, "tile_n": 192, "cluster_m": 2, "cluster_n": 1, "ltile_cols": 64},
    {"tile_m": 256, "tile_n": 128, "cluster_m": 2, "cluster_n": 2, "ltile_cols": 64},
    {"tile_m": 128, "tile_n": 192, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 192},
    {"tile_m": 64, "tile_n": 128, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 64},
    {"tile_m": 64, "tile_n": 128, "cluster_m": 1, "cluster_n": 1},
    {"tile_m": 64, "tile_n": 128, "cluster_m": 2, "cluster_n": 1},
]

_SM120_PER_MODEL_CONFIGS = [
    {"tile_m": 128, "tile_n": 128, "cluster_m": 1, "cluster_n": 1, "tile_k": 128},
    {"tile_m": 128, "tile_n": 192, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 64},
    {"tile_m": 128, "tile_n": 64, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 64},
    {"tile_m": 128, "tile_n": 192, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 192},
    {"tile_m": 128, "tile_n": 256, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 256},
    {"tile_m": 64, "tile_n": 64, "cluster_m": 1, "cluster_n": 1, "ltile_cols": 64},
    {"tile_m": 64, "tile_n": 128, "cluster_m": 1, "cluster_n": 1},
]

_PER_MODEL_CONFIGS = {
    Arch.SM90: _SM90_PER_MODEL_CONFIGS,
    Arch.SM100: _SM100_PER_MODEL_CONFIGS,
    Arch.SM120: _SM120_PER_MODEL_CONFIGS,
}[_ARCH]

_SMOKE_CONFIGS = [_DEFAULT_CONFIG, *_PER_MODEL_CONFIGS]


def _is_legal(config_fields: dict) -> bool:
    try:
        validate_mixed_gemm_config(1536, 768, 512, MixedGemmConfig(**config_fields))
    except ValueError:
        return False
    return True


# The tuned shortlist is SM100's; points the local family cannot run (clusters,
# 256-row tiles on SM120) are skipped.
_GATE_SPACE = [
    pytest.param(
        fields,
        marks=()
        if _is_legal(fields)
        else pytest.mark.skip(reason=f"not a {_ARCH.name} tile shape"),
    )
    for fields in MIXED_GEMM_SPACE
]

_CONSISTENCY_ITERS = 10_000


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(13)
    m, n, k = 1536, 768, 512
    a_prime = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b_prime = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    a_peel = torch.randn(m, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1
    b_peel = torch.randn(n, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_a = (torch.rand(m, device="cuda") + 0.5).to(torch.bfloat16)
    inv_alpha_b = torch.reciprocal(torch.rand(n, device="cuda") + 0.5)
    return {
        "a_prime": a_prime,
        "b_prime": b_prime,
        "a_peel": a_peel,
        "b_peel": b_peel,
        "alpha_a": alpha_a,
        "inv_alpha_b": inv_alpha_b,
        "pow_key": torch.arange(32, dtype=torch.uint8, device="cuda"),
        "threshold": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "hit_signal": HitSignal(HitSignalConfig(max_m=m, max_k=k)),
        "a_codes": torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda"),
        "a_scales": torch.rand(m, k // 8, dtype=torch.bfloat16, device="cuda"),
        "commitment_hash_b": torch.zeros(32, dtype=torch.uint8, device="cuda"),
    }


def _run(inputs, config_fields):
    m, k = inputs["a_prime"].shape
    n = inputs["b_prime"].shape[0]
    config = MixedGemmConfig(**config_fields)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm_inputs = dict(inputs)
    mixed_gemm(
        **gemm_inputs,
        out=out,
        config=config,
    )
    torch.cuda.synchronize()
    return out


def _output_reference(inputs):
    output = inputs["a_prime"].float() @ inputs["b_prime"].float().T
    output = output + inputs["a_peel"].float() @ inputs["b_peel"].float().T
    output *= torch.reciprocal(inputs["alpha_a"].float()).reshape(-1, 1)
    output *= inputs["inv_alpha_b"].reshape(1, -1)
    return output


def _assert_correct(inputs, out):
    torch.testing.assert_close(out.float(), _output_reference(inputs), rtol=5e-3, atol=0.25)


@pytest.mark.parametrize("config_fields", _SMOKE_CONFIGS)
def test_representative_configs(inputs, config_fields):
    out = _run(inputs, config_fields)
    _assert_correct(inputs, out)


@pytest.mark.slow
@pytest.mark.parametrize("config_fields", _GATE_SPACE)
def test_autotune_space_matches_host_peel_algebra(inputs, config_fields):
    out = _run(inputs, config_fields)
    _assert_correct(inputs, out)


@pytest.mark.slow
@pytest.mark.parametrize("config_fields", _GATE_SPACE)
def test_consistency_across_tuning_space(inputs, config_fields):
    """Relaunches under every tuned config are bit-identical (mixed GEMM consistency)."""
    out_first = _run(inputs, config_fields)
    for _ in range(_CONSISTENCY_ITERS):
        out = _run(inputs, config_fields)
        assert torch.equal(out, out_first)


# Host-only validation, so every family's rejections run on any machine.
_FAMILIES = pytest.mark.parametrize(
    "arch", [Arch.SM90, Arch.SM100, Arch.SM120], ids=lambda arch: arch.name.lower()
)


@_FAMILIES
@pytest.mark.parametrize(
    "m,n,k,config,match",
    [
        (512, 256, 512, {"tile_m": 32}, "tile_m"),
        (512, 256, 512, {"tile_m": 192}, "tile_m"),
        (512, 256, 512, {"tile_n": 48}, "tile_n"),
        (512, 256, 512, {"tile_k": 32}, "tile_k"),
        (512, 256, 512, {"tile_k": 64}, "tile_k"),
        (512, 256, 512, {"ltile_cols": 96}, "ltile_cols"),
        (512, 256, 512, {"ltile_rows": 8}, "ltile_rows must be one of"),
        # 16-row tiles are only committed at 32 cols, and the 64-row kernel
        # tile cannot fold them (two threads split every accumulator row).
        (512, 256, 512, {"ltile_rows": 16, "ltile_cols": 128}, "ltile_cols must be one of"),
        (
            512,
            256,
            512,
            {"tile_m": 64, "tile_n": 64, "ltile_rows": 16, "ltile_cols": 32},
            "4-row lottery family",
        ),
        # Integral floats and bools compare equal to the allowed values but
        # break the kernel's integer bit-shift during variant compilation.
        (512, 256, 512, {"ltile_rows": 16.0}, "ltile_rows must be an int"),
        (512, 256, 512, {"ltile_cols": 128.0}, "ltile_cols must be an int"),
        (512, 256, 512, {"ltile_rows": True}, "ltile_rows must be an int"),
        (512, 256, 512, {"tile_m": 64.0}, "tile_m must be an int"),
        (512, 256, 512, {"tile_n": 128.0}, "tile_n must be an int"),
        (512, 256, 512, {"cluster_m": True}, "cluster_m must be an int"),
        (512, 256, 512, {"tile_k": 64.0}, "tile_k must be an int"),
        (512, 256, 512, {"cluster_m": 0}, "cluster"),
        # SM90/SM100 reject the non-power-of-2 cluster; SM120 rejects any cluster.
        (512, 256, 512, {"cluster_m": 3}, "powers of 2|no thread-block clusters"),
        (512, 256, 513, {}, "divisible by 64"),
        (513, 256, 512, {}, "lottery tiles"),
        (512, 130, 512, {}, "lottery tiles"),
    ],
)
def test_invalid_configurations_are_rejected(arch, m, n, k, config, match):
    """One field at a time on top of the family default, so the rejection names
    the field under test rather than another family's tile."""
    with pytest.raises(ValueError, match=match):
        validate_mixed_gemm_config(
            m, n, k, replace(default_mixed_gemm_config(arch), **config), arch=arch
        )


@pytest.mark.parametrize(
    "m,n,k,config,match",
    [
        # Over the promoted mainloop's accumulator register budget.
        (512, 384, 512, {"tile_m": 256, "tile_n": 192, "ltile_cols": 64}, "accumulator"),
        # One k-tile is one promotion window of the verifier's H100 arithmetic.
        (512, 256, 512, {"tile_k": 160}, "promotion window"),
        (512, 256, 512, {"tile_k": 256}, "promotion window"),
        # WGMMA's fused epilogue and register budget admit tile_n 128 and 192.
        (512, 256, 512, {"tile_n": 256}, "SM90 tile_n"),
        (512, 256, 512, {"tile_n": 64, "ltile_cols": 64}, "SM90 tile_n"),
    ],
)
def test_invalid_sm90_configurations_are_rejected(m, n, k, config, match):
    """The promoted SM90 mainloop's own limits, on top of the Hopper default."""
    with pytest.raises(ValueError, match=match):
        validate_mixed_gemm_config(
            m, n, k, replace(default_mixed_gemm_config(Arch.SM90), **config), arch=Arch.SM90
        )


def test_tall_tile_config_and_legacy_positional_configs_validate():
    """``MixedGemmConfig`` is public API: ``ltile_rows`` is appended after
    ``ltile_cols`` so existing positional constructions keep their meaning.
    Validation is per family: SM100 runs the legacy 2-CTA default and the tall
    tile, SM120 (no clusters, no whole-row fold) rejects both, and SM90 runs
    the legacy default but not the tall tile."""
    legacy = MixedGemmConfig(256, 128, None, 2, 1, 128)
    assert (legacy.ltile_cols, legacy.ltile_rows) == (128, 4)
    tall = MixedGemmConfig(ltile_rows=16, ltile_cols=32)
    validate_mixed_gemm_config(256, 6144, 12288, legacy, arch=Arch.SM100)
    # The tall tile is what makes k=16384 (GLM o_proj) verifiable.
    validate_mixed_gemm_config(256, 6144, 16384, tall, arch=Arch.SM100)
    # Gemma-3 31B q/k/v and o_proj use k=5376: 64-aligned, not 512-aligned.
    validate_mixed_gemm_config(256, 128, 5376, MixedGemmConfig(), arch=Arch.SM100)
    sm120_default = default_mixed_gemm_config(Arch.SM120)
    with pytest.raises(ValueError, match="SM120 tile_m"):
        validate_mixed_gemm_config(256, 6144, 12288, legacy, arch=Arch.SM120)
    with pytest.raises(ValueError, match="4-row lottery family"):
        validate_mixed_gemm_config(
            256,
            6144,
            16384,
            replace(sm120_default, ltile_rows=16, ltile_cols=32),
            arch=Arch.SM120,
        )
    validate_mixed_gemm_config(256, 128, 5376, sm120_default, arch=Arch.SM120)
    sm90_default = default_mixed_gemm_config(Arch.SM90)
    # Whole-tile SM90 CTAs: the legacy 2x1 cluster needs an even CTA-row count.
    validate_mixed_gemm_config(512, 6144, 12288, legacy, arch=Arch.SM90)
    with pytest.raises(ValueError, match="4-row lottery family"):
        validate_mixed_gemm_config(256, 6144, 16384, tall, arch=Arch.SM90)
    validate_mixed_gemm_config(256, 128, 5376, sm90_default, arch=Arch.SM90)
    # The promotion window is the only legal explicit tile_k on SM90.
    validate_mixed_gemm_config(512, 256, 512, replace(sm90_default, tile_k=128), arch=Arch.SM90)


def test_supports_lottery_family_agrees_with_the_validator():
    """The runtime's load-time admission predicate must match what the
    validator later accepts for each family, so a layer is never encoded for
    a lottery whose kernel the device lacks."""
    from pearl_gemm import supports_lottery_family

    for capability, arch in (((9, 0), Arch.SM90), ((10, 0), Arch.SM100), ((12, 0), Arch.SM120)):
        base = default_mixed_gemm_config(arch)
        for rows, cols in ((4, 64), (16, 32)):
            config = replace(base, ltile_rows=rows, ltile_cols=cols)
            try:
                validate_mixed_gemm_config(256, 6144, 12288, config, arch=arch)
            except ValueError:
                assert not supports_lottery_family(rows, capability)
            else:
                assert supports_lottery_family(rows, capability)
    assert supports_lottery_family(4)
    assert supports_lottery_family(16) is (_ARCH is Arch.SM100)
    assert supports_lottery_family(4.0, (12, 0)) is False
    assert supports_lottery_family(True, (10, 0)) is False


@pytest.mark.parametrize("tag_name", ["layer_id"])
@pytest.mark.parametrize(
    "tag_value",
    [
        # Out of the u32 record word's range.
        -1,
        2**32,
        2**40,
        # Not an int. bool would pass a bare range check and stamp 0/1; the
        # rest would raise TypeError from the comparison if unguarded.
        True,
        False,
        1.0,
        "1",
        None,
    ],
)
def test_malformed_record_tag_is_rejected(tag_name, tag_value):
    """The record tag rejects truncation, bool coercion, and leaked TypeError."""
    from pearl_gemm.mixed_gemm._host import _validate_u32_tag

    with pytest.raises(ValueError, match=tag_name):
        _validate_u32_tag(tag_name, tag_value)


@pytest.mark.parametrize("tag_name", ["layer_id"])
@pytest.mark.parametrize("tag_value", [0, 1, 2**32 - 1])
def test_valid_record_tag_is_accepted(tag_name, tag_value):
    """The boundaries of the accepted domain."""
    from pearl_gemm.mixed_gemm._host import _validate_u32_tag

    _validate_u32_tag(tag_name, tag_value)
