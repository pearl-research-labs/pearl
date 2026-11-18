"""Consecutive FP8 lottery tiles and the fixed sixteen-chunk extractor order.

Sizes are committed in the job; starts are chosen by each proof and must be
multiples of the size. A compact row-major output tile is split into sixteen
consecutive equal-sized chunks.
"""

from __future__ import annotations

# The extracted lottery message is LANES x u32 = 64 bytes: one BLAKE3 block.
LANES = 16

# Per-chunk element bound. Mirrors MAX_SUBTILE_ELEMS in
# zk-pow/src/v4/api/layout.rs.
MAX_SUBTILE_ELEMS = 256

# Lower bound on the per-chunk element count:
# each extractor word must fold at least this many elements.
MIN_SUBTILE_ELEMS = 16

# Minimum rows in a tile: the number of
# strips opened from the activations matrix.
MIN_TILE_ROWS = 4

# Minimum cols in a tile: the number of strips opened from the weight matrix.
MIN_TILE_COLS = 16

# Upper bound on the whole lottery tile (h * w).
MAX_TILE_ELEMS = 2048


def lane_assignment(h: int, w: int) -> list[list[int]]:
    """Flat cell indices for the sixteen consecutive row-major fold chunks."""
    if (h * w) % LANES:
        raise ValueError(f"tile element count must be divisible by {LANES}")
    subtile_elems = h * w // LANES
    return [list(range(lane * subtile_elems, (lane + 1) * subtile_elems)) for lane in range(LANES)]
