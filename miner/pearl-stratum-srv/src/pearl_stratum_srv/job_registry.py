"""Maps stratum job_id ↔ BlockTemplate.

When the poller picks up a new template from pearld we mint a fresh job_id
(HEIGHT-SEQ hex, matching alphapool's `0000d446-3061` format) and broadcast
`mining.notify` to every connected miner. When a miner later submits a share,
we look up its job_id here to recover the template needed for
`SubmissionService.submit_plain_proof(plain_proof, template)`.

Bounded FIFO. Older jobs evict; submits against them return error[21]
("stale share") without closing the socket.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pearl_gateway.comm.dataclasses import BlockTemplate


@dataclass(slots=True)
class JobEntry:
    job_id: str
    template: "BlockTemplate"
    minted_at: float

    @property
    def height(self) -> int:
        return self.template.height

    @property
    def prev_hash_hex(self) -> str:
        """Hex of previous block hash, big-endian as the wire shows it."""
        return self.template.header.previous_block_hash.hex()

    @property
    def incomplete_header_hex(self) -> str:
        return self.template.header.serialize_without_proof_commitment().hex()

    @property
    def ntime_hex(self) -> str:
        return f"{self.template.header.timestamp:08x}"

    @property
    def nbits_hex(self) -> str:
        return f"{self.template.header.target_bits:08x}"


class JobRegistry:
    """Bounded job-id → JobEntry cache. Single-threaded asyncio access only."""

    def __init__(self, max_size: int = 16):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._jobs: OrderedDict[str, JobEntry] = OrderedDict()
        self._seq = 0

    def mint(self, template: "BlockTemplate") -> JobEntry:
        """Create and store a JobEntry for a fresh template. Evicts oldest if full."""
        self._seq = (self._seq + 1) & 0xFFFF
        job_id = f"{template.height & 0xFFFFFFFF:08x}-{self._seq:04x}"
        entry = JobEntry(job_id=job_id, template=template, minted_at=time.time())
        self._jobs[job_id] = entry
        while len(self._jobs) > self._max_size:
            self._jobs.popitem(last=False)
        return entry

    def get(self, job_id: str) -> JobEntry | None:
        return self._jobs.get(job_id)

    def latest(self) -> JobEntry | None:
        if not self._jobs:
            return None
        # OrderedDict preserves insertion order; latest minted is last.
        return next(reversed(self._jobs.values()))

    def __len__(self) -> int:
        return len(self._jobs)
