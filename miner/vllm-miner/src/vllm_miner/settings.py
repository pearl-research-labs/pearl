"""Process runtime settings (PEARL_* and MINER_*)."""

import re
from collections.abc import Iterable
from functools import cache
from typing import Annotated, Literal, cast

from miner_base.settings import MinerSettings
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse a Linux cpulist ('0-3,8,10-11') into ordered CPU indices."""
    cpus: list[int] = []
    for part in filter(None, (part.strip() for part in value.split(","))):
        matched = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if matched is None:
            raise ValueError(f"malformed cpulist entry {part!r} in {value!r}")
        first = int(matched[1])
        last = int(matched[2] or first)
        if last < first:
            raise ValueError(f"reversed cpulist range {part!r} in {value!r}")
        cpus.extend(range(first, last + 1))
    return tuple(cpus)


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="pearl_")

    ignored_layers: Annotated[tuple[str, ...], NoDecode] = ()
    min_mining_tokens: int = Field(default=1024, ge=4)
    m_buckets: Annotated[tuple[int, ...], NoDecode] = (2048, 8192)
    warmup_compile: bool = True
    winner_check_inflight_limit: int = Field(default=8, ge=1)
    completion_inflight_limit: int = Field(default=8, ge=1, le=256)
    oom_cooldown_s: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    recon_chunk_rows: int = Field(default=4096, ge=1)
    cpu_b_prepare_threads: int = Field(default=8, ge=1)
    cpu_b_prepare_cpus: Annotated[tuple[int, ...] | Literal["auto"], NoDecode] = ()
    cpu_b_replace_lead: int = Field(default=1, ge=1, le=3)

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

    @field_validator("cpu_b_prepare_cpus", mode="before")
    @classmethod
    def _parse_cpu_list(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if value.strip().lower() == "auto":
            return "auto"
        return parse_cpu_list(value)

    @field_validator("m_buckets", mode="before")
    @classmethod
    def _parse_m_buckets(cls, value: object) -> object:
        if isinstance(value, str):
            value = tuple(int(part) for part in value.split(",") if part.strip())
        buckets = tuple(sorted(int(v) for v in cast(Iterable[int | str], value)))
        if not buckets:
            raise ValueError("m_buckets must not be empty")
        if any(b <= 0 or b % 64 for b in buckets):
            raise ValueError(f"m_buckets must be positive multiples of 64, got {buckets}")
        # HitSignal reserves (max_m, max_k) indexed by uint32.
        from .mining_config import max_mineable_k

        max_m = ((1 << 32) - 1) // max_mineable_k()
        if buckets[-1] > max_m:
            raise ValueError(
                f"largest m_bucket {buckets[-1]} times max mineable k {max_mineable_k()} "
                f"overflows the hit signal's uint32 capacity; keep buckets <= {max_m}"
            )
        return buckets


_settings: RuntimeSettings | None = None
_miner_settings: MinerSettings | None = None


def runtime_settings() -> RuntimeSettings:
    global _settings
    if _settings is None:
        _settings = RuntimeSettings()
    return _settings


def miner_settings() -> MinerSettings:
    global _miner_settings
    if _miner_settings is None:
        _miner_settings = MinerSettings()
    return _miner_settings


def set_miner_settings(settings: MinerSettings) -> None:
    global _miner_settings
    _miner_settings = settings


@cache
def _compiled(entry: str) -> re.Pattern[str]:
    return re.compile(entry[3:])


def is_layer_ignored(prefix: str, ignored_layers: tuple[str, ...]) -> bool:
    for entry in ignored_layers:
        if entry.startswith("re:"):
            if _compiled(entry).match(prefix):
                return True
        elif entry == prefix:
            return True
    return False
