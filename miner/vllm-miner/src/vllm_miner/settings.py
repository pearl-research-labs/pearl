"""GPU runtime knobs; see ``vllm-miner/README.md`` (Runtime configuration)."""

import re
from collections.abc import Iterable
from functools import cache
from typing import Annotated, cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="pearl_")

    # Exclusion list, vLLM ignore-list style: exact layer prefixes or "re:"
    # regexes re.match-ed against the unique runtime prefix. Empty: mine all.
    ignored_layers: Annotated[tuple[str, ...], NoDecode] = ()
    # Forwards below this token count (decode) take the FP8 fallback GEMM.
    min_mining_tokens: int = Field(default=1024, ge=4)
    # Ascending activation-row buckets; m pads to the smallest fit (bounds JIT variants).
    m_buckets: Annotated[tuple[int, ...], NoDecode] = (2048, 8192)
    # Compile pipeline variants off the serving path before mining engages.
    warmup_compile: bool = True
    # Pending event-gated checks of the process-wide hit signal.
    winner_check_inflight_limit: int = Field(default=8, ge=1)
    # Every credited launch, including no-gateway launches without a retained
    # winner, owns one bounded event-completion slot. This bound also sizes the
    # mining memory vLLM must keep off the KV cache (see memory.py): the peak
    # reservation scales roughly linearly with it, so a high value can starve KV
    # on large models. 8 keeps ample overlap while leaving KV headroom; raise it
    # only when profiling shows spare memory.
    completion_inflight_limit: int = Field(default=8, ge=1, le=256)
    # Device-wide serving protection after a mining allocation OOM. Exactly one
    # probe is admitted after this cooldown; another OOM restarts it.
    oom_cooldown_s: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    # Identity rows fed per delegate ``apply`` when decoding a quantized weight
    # back to BF16 (upcast). Bounds the transient workspace instead of a full
    # ``k x k`` eye; the free-memory budget check is the real OOM gate, so this
    # rarely needs tuning -- lower it only for a pathological shape that the
    # budget still admits but whose per-chunk transient is too large.
    recon_chunk_rows: int = Field(default=4096, ge=1)

    @field_validator("ignored_layers", mode="before")
    @classmethod
    def _parse_ignored_layers(cls, value: object) -> object:
        if isinstance(value, str):
            value = tuple(part.strip() for part in value.split(",") if part.strip())
        entries = tuple(str(v) for v in cast(Iterable[object], value))
        for entry in entries:
            if entry.startswith("re:"):
                try:
                    re.compile(entry[3:])
                except re.error as exc:
                    raise ValueError(f"invalid ignored-layers regex {entry!r}: {exc}") from exc
        return entries

    @field_validator("m_buckets", mode="before")
    @classmethod
    def _parse_m_buckets(cls, value: object) -> object:
        if isinstance(value, str):
            value = tuple(int(part) for part in value.split(",") if part.strip())
        buckets = tuple(sorted(int(v) for v in cast(Iterable[int | str], value)))
        if not buckets:
            raise ValueError("m_buckets must not be empty")
        # 64 keeps every committed lottery tile's row alignment (m % 16) and is
        # the smallest bucket the 64-row decode kernel tile fills. Sub-256
        # buckets let small decode batches pad less; on 4-row commitments those
        # buckets use the 64-row tile unless an exact autotune record exists.
        if any(b <= 0 or b % 64 for b in buckets):
            raise ValueError(f"m_buckets must be positive multiples of 64, got {buckets}")
        # The persistent hit signal reserves an ``(max_m, max_k)`` A-codes plane
        # whose element count is a uint32 (pearl_gemm HitSignalConfig). max_k is
        # the largest mineable k across committed tiles, so an oversized largest
        # bucket would overflow that product -- reject it here with the offending
        # value rather than failing opaquely at signal allocation.
        from .mining_config import max_mineable_k  # local: keep settings import-light

        max_m = ((1 << 32) - 1) // max_mineable_k()
        if buckets[-1] > max_m:
            raise ValueError(
                f"largest m_bucket {buckets[-1]} times max mineable k {max_mineable_k()} "
                f"overflows the hit signal's uint32 capacity; keep buckets <= {max_m}"
            )
        return buckets


_settings: RuntimeSettings | None = None


def runtime_settings() -> RuntimeSettings:
    global _settings
    if _settings is None:
        _settings = RuntimeSettings()
    return _settings


@cache
def _compiled(entry: str) -> re.Pattern[str]:
    return re.compile(entry[3:])


def is_layer_ignored(prefix: str, ignored_layers: tuple[str, ...]) -> bool:
    """vLLM ignore-list semantics: exact prefix, or ``re:`` regex via re.match."""
    for entry in ignored_layers:
        if entry.startswith("re:"):
            if _compiled(entry).match(prefix):
                return True
        elif entry == prefix:
            return True
    return False
