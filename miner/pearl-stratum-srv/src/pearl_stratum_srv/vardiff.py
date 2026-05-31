"""Per-worker variable difficulty (vardiff).

Each connection starts at `initial_diff`. Every `retarget_interval_secs` we
look at how many shares the worker submitted in that window and nudge their
difficulty toward `target_shares_per_min`.

Bounded by [min_diff, max_diff] to avoid runaway in either direction:
  - too low: pool floods with submits (DoS amp)
  - too high: low-hashrate worker submits nothing for hours; pool can't tell
    if they're alive

Hysteresis: don't retarget unless the observed rate is >30% off the goal.
Keeps difficulty from oscillating on bursty miners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class VardiffPolicy:
    initial_diff: int = 1 << 20          # 1,048,576 — matches alphapool default
    min_diff: int = 1 << 16              # 65,536
    max_diff: int = 1 << 24              # 16,777,216
    target_shares_per_min: float = 6.0   # 1 share per 10s/worker, alpha-pool convention
    retarget_interval_secs: float = 60.0
    hysteresis_frac: float = 0.30        # only retarget if observed is >30% off goal
    max_step_multiplier: float = 4.0     # never adjust by more than 4× in one step


@dataclass
class VardiffState:
    """One per connection."""

    current_diff: int = 0
    last_retarget_ts: float = 0.0
    shares_since_retarget: int = 0
    # The diff we suggested via mining.set_difficulty most recently.
    # We track this so callers can decide whether to actually push a new value.
    last_pushed_diff: int = 0

    def init(self, policy: VardiffPolicy, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        self.current_diff = policy.initial_diff
        self.last_retarget_ts = now
        self.shares_since_retarget = 0
        self.last_pushed_diff = policy.initial_diff
        return self.current_diff

    def note_share(self) -> None:
        self.shares_since_retarget += 1


def maybe_retarget(
    state: VardiffState, policy: VardiffPolicy, now: float | None = None
) -> int | None:
    """Decide whether to push a new difficulty. Returns the new diff if a
    `mining.set_difficulty` push is warranted, else None.

    Mutates `state` to reset the window counters when a retarget fires.
    """
    now = now if now is not None else time.time()
    elapsed = now - state.last_retarget_ts
    if elapsed < policy.retarget_interval_secs:
        return None

    observed_per_min = (state.shares_since_retarget / elapsed) * 60.0 if elapsed > 0 else 0.0
    goal = policy.target_shares_per_min
    # Hysteresis: skip if observed is within the deadband.
    if observed_per_min > 0:
        ratio = observed_per_min / goal
        if (1 - policy.hysteresis_frac) < ratio < (1 + policy.hysteresis_frac):
            # Within deadband — reset window, don't push.
            state.last_retarget_ts = now
            state.shares_since_retarget = 0
            return None

    # If observed is 2× goal, double the diff (cuts share rate to ~goal).
    # If observed is 0.5× goal, halve the diff. Clamp by max_step.
    if observed_per_min <= 0:
        # Zero submits in window — drop diff hard to invite activity / detect alive.
        factor = 1.0 / policy.max_step_multiplier
    else:
        factor = observed_per_min / goal
        # Clamp
        factor = max(1.0 / policy.max_step_multiplier, min(policy.max_step_multiplier, factor))

    new_diff = int(state.current_diff * factor)
    new_diff = max(policy.min_diff, min(policy.max_diff, new_diff))

    state.current_diff = new_diff
    state.last_retarget_ts = now
    state.shares_since_retarget = 0

    # Only push if the wire-visible value actually changed.
    if new_diff != state.last_pushed_diff:
        state.last_pushed_diff = new_diff
        return new_diff
    return None
