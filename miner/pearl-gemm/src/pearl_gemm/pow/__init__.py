"""Persistent per-process PoW hit signal (see ``_hit_signal``)."""

from ._hit_signal import (
    HIT_PAYLOAD_K_ALIGN,
    HIT_RECORD_MAGIC_WORDS,
    RECORD_WORDS,
    Hit,
    HitRecordLayout,
    HitSignal,
    HitSignalConfig,
    HitSignalPoisonedError,
)

__all__ = [
    "HIT_PAYLOAD_K_ALIGN",
    "HIT_RECORD_MAGIC_WORDS",
    "RECORD_WORDS",
    "Hit",
    "HitRecordLayout",
    "HitSignal",
    "HitSignalConfig",
    "HitSignalPoisonedError",
]
