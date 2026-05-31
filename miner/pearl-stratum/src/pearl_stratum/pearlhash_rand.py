"""glibc `rand()` reimplemented in pure Python (TYPE_3, default state).

pearl-miner v2 calls `srand(0)` exactly once at TCP connection start and emits
`htonl(rand_N)` as the first 4 bytes of every C->S frame. We need the same
sequence to construct outgoing frames the pool will accept.

Reference (sequence verified bit-exact in `36_pearlhash_shim.md` §1.3):
    rand[0]  = 0x6b8b4567 = 1804289383   ← login req counter
    rand[1]  = 0x327b23c6 =  846930886   ← 1st keepalive counter
    ...
    rand[15] = 0x7545e146 = 1967513926

Algorithm details (from glibc/stdlib/random_r.c TYPE_3):
- 31-element int32 state vector.
- srand(0) is special-cased to srand(1) internally.
- Seeded via a Park-Miller-style LCG: r[i] = (16807 * r[i-1]) mod (2^31 - 1)
  for i in 1..30; r[0] = seed.
- After seeding, runs 310 warmup iterations of the additive feedback rule
  `r[front] = r[front] + r[rear]; front++; rear++` (mod 31) before the
  first observable output.
- Each output is `(r[front] >> 1) & 0x7FFFFFFF`.

The "310 warmup" count is the empirically verified value for the modern glibc
TYPE_3 initializer (older docs cite 313 or 344; those are wrong for current
glibc).
"""

from __future__ import annotations

from collections.abc import Iterator

_STATE_LEN = 31
_LCG_MULT = 16807
_LCG_MOD = 2147483647  # 2^31 - 1
_WARMUP_ITERATIONS = 310


class GlibcRand:
    """Stateful glibc-compatible rand() generator.

    Construct fresh per Pearlhash TCP connection (matches `srand(0)` on connect).
    Each call to `next_u31()` returns the same 31-bit value glibc's rand() would.
    """

    def __init__(self, seed: int = 0) -> None:
        # glibc replaces seed=0 with seed=1 internally.
        effective_seed = seed if seed != 0 else 1

        state = [0] * _STATE_LEN
        state[0] = effective_seed
        for i in range(1, _STATE_LEN):
            # Park-Miller seeding LCG; mod 2^31-1 keeps values positive.
            state[i] = (_LCG_MULT * state[i - 1]) % _LCG_MOD

        self._state = state
        self._front = 3
        self._rear = 0

        # Warmup: advance the additive-feedback machine before first output.
        for _ in range(_WARMUP_ITERATIONS):
            self._step()

    def _step(self) -> int:
        new = (self._state[self._front] + self._state[self._rear]) & 0xFFFFFFFF
        self._state[self._front] = new
        self._front = (self._front + 1) % _STATE_LEN
        self._rear = (self._rear + 1) % _STATE_LEN
        return new

    def next_u31(self) -> int:
        """Return the next rand() output (0..2^31-1)."""
        raw = self._step()
        return (raw >> 1) & 0x7FFFFFFF


def glibc_rand_sequence(n: int, seed: int = 0) -> list[int]:
    """Convenience helper: return the first `n` outputs of `srand(seed); rand()...`."""
    g = GlibcRand(seed)
    return [g.next_u31() for _ in range(n)]
