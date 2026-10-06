"""Plaintext-PoUW scheme classification from a job's certificate version.

The single, torch-free classifier shared by every dispatch point: arch/layer
admission, the per-forward launch branch (fused FP8 ``_launch_stages`` vs. the
standalone FP16 ``search_block``), and submission (``PlainProofV4`` vs.
``Fp16PlainProof``). Kept out of :mod:`miner_base.block_submission` (which pulls
in torch) so the classifier -- and the FP16 submission path that reuses it --
imports under the proof interpreter alone.

Only ``pearl_mining`` (the cert-version constants) is required; the ``job`` is
duck-typed on its ``cert_version`` attribute, so no gateway/torch import is
needed here.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from pearl_mining import CERT_VERSION_PLAIN_FP8, CERT_VERSION_PLAIN_FP16

if TYPE_CHECKING:
    from pearl_gateway.comm.dataclasses import MiningJob

__all__ = [
    "Scheme",
    "is_plain_fp8_job",
    "is_plain_fp16_job",
    "is_submittable_plain_job",
    "scheme_of",
]


class Scheme(enum.Enum):
    """The plaintext-PoUW scheme a job's certificate version selects.

    ``FP8`` is cert-v4 (Hopper/Blackwell FP8-MMA, the fused mixed-GEMM lottery);
    ``FP16`` is cert-v5 (A100/sm_80, the standalone full-matrix FP16 search).
    ``OTHER`` is any version this miner does not plaintext-mine (ZK v1/v2/v3),
    which the runtime may execute and credit but never inspects, builds, or
    submits a proof for.
    """

    FP8 = "fp8"
    FP16 = "fp16"
    OTHER = "other"


def scheme_of(job: "MiningJob") -> Scheme:
    """The :class:`Scheme` a job advertises through its certificate version."""
    version = int(job.cert_version)
    if version == CERT_VERSION_PLAIN_FP8:
        return Scheme.FP8
    if version == CERT_VERSION_PLAIN_FP16:
        return Scheme.FP16
    return Scheme.OTHER


def is_plain_fp8_job(job: "MiningJob") -> bool:
    """Whether the issuing endpoint advertised certificate-v4 FP8 through ``job``."""
    return int(job.cert_version) == CERT_VERSION_PLAIN_FP8


def is_plain_fp16_job(job: "MiningJob") -> bool:
    """Whether the issuing endpoint advertised certificate-v5 FP16 (A100)."""
    return int(job.cert_version) == CERT_VERSION_PLAIN_FP16


def is_submittable_plain_job(job: "MiningJob") -> bool:
    """Whether this job runs a plaintext scheme this miner can submit a proof for
    (FP8/v4 or FP16/v5). The shared submission-admission gate keys on this so the
    AsyncLoopManager plumbing is scheme-agnostic while proof construction branches
    per scheme."""
    return scheme_of(job) in (Scheme.FP8, Scheme.FP16)
