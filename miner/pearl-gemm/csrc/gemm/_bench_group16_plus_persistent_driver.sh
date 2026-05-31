#!/usr/bin/env bash
# Wave-4 driver: bench group-of-16 swizzle × persistent-over-output-tiles combo
# across the production sweep AND the large 65536²+ shapes where the L2 cliff
# was previously hypothesized to live.
#
# The PersistentSwizzledTileScheduler is the production sm_89 scheduler — every
# variant here uses it. We vary the (swizzle_width, n_major) pair.
#
# Variants per shape:
#   auto      — current default (adaptive S=4/32, longer axis major)
#   g16       — alpha-style: S=16, N-major  (= PEARL_SM89_GROUP16_SWIZZLE=1)
#   16xN      — S=16, N-major (same as g16 but via PEARL_SM89_SWIZZLE)
#   16xM      — S=16, M-major
#   32xN      — S=32, N-major
#   32xM      — S=32, M-major
#   4xN       — S=4,  N-major (skinny default re-tested at large shapes)
#   4xM       — S=4,  M-major
#
# Driver writes CSV to $OUT_CSV with one line per (shape, variant).
#
# Run on bench rig (CPU01/CPU02/mini28/AI01); mfarm-agent + alpha-miner must be
# stopped first.
#
# Args:
#   $1 = device (default: 0)
#   $2 = output CSV path (default: /tmp/bench_group16_plus_persistent.csv)
#   $3 = bench binary path (default: /tmp/bench_group16_plus_persistent)
#   $4 = "docker" or "" — if "docker", wrap each invocation through pearl-ab.
set -euo pipefail
DEV="${1:-0}"
OUT_CSV="${2:-/tmp/bench_group16_plus_persistent.csv}"
BENCH="${3:-/tmp/bench_group16_plus_persistent}"
WRAP="${4:-}"

run_bench() {
  # $1.. = env-var assignments + bench args
  if [ "$WRAP" = "docker" ]; then
    docker exec pearl-ab bash -c "$*"
  else
    bash -c "$*"
  fi
}

SHAPES=(
  # Production sweep (regression guard)
  "2048  2048  2048 30"
  "4096  4096  4096 30"
  "8192  8192  4096 20"
  # Mid (where wave-3 saw the largest loss)
  "16384 4096  4096 20"
  "4096  16384 4096 20"
  # Large — the L2 cliff regime
  "16384 16384 4096 15"
  "32768 32768 4096 8"
  "65536 16384 4096 5"
  "16384 65536 4096 5"
  # Alpha-style skinny
  "131072 4096 4096 5"
  "4096 131072 4096 5"
)

V_ORDER=(auto g16 16xN 16xM 32xN 32xM 4xN 4xM)
declare -A V_ENV=(
  [auto]=""
  [g16]="PEARL_SM89_GROUP16_SWIZZLE=1"
  [16xN]="PEARL_SM89_SWIZZLE=16 PEARL_SM89_SWIZZLE_NMAJ=1"
  [16xM]="PEARL_SM89_SWIZZLE=16 PEARL_SM89_SWIZZLE_NMAJ=0"
  [32xN]="PEARL_SM89_SWIZZLE=32 PEARL_SM89_SWIZZLE_NMAJ=1"
  [32xM]="PEARL_SM89_SWIZZLE=32 PEARL_SM89_SWIZZLE_NMAJ=0"
  [4xN]="PEARL_SM89_SWIZZLE=4 PEARL_SM89_SWIZZLE_NMAJ=1"
  [4xM]="PEARL_SM89_SWIZZLE=4 PEARL_SM89_SWIZZLE_NMAJ=0"
)

echo "M,N,K,variant,median_us,tops" > "$OUT_CSV"

for shape in "${SHAPES[@]}"; do
  read -r M N K I <<< "$shape"
  for variant in "${V_ORDER[@]}"; do
    env_extra="${V_ENV[$variant]}"
    cmd="${env_extra} $BENCH $DEV $M $N $K $I"
    echo "=== shape=${M}x${N}x${K} iters=${I} variant=${variant} ===" >&2
    out="$(run_bench "$cmd")"
    csv_line=$(echo "$out" | grep '^CSV,' || true)
    if [ -n "$csv_line" ]; then
      med=$(echo "$csv_line" | awk -F',' '{print $6}')
      tops=$(echo "$csv_line" | awk -F',' '{print $7}')
      echo "${M},${N},${K},${variant},${med},${tops}" >> "$OUT_CSV"
      echo "  -> ${variant}: med=${med}us tops=${tops}" >&2
    else
      echo "  -> ${variant}: NO CSV OUTPUT" >&2
      echo "$out" | head -5 >&2
    fi
  done
done

echo "---"
echo "Done. CSV at $OUT_CSV:"
cat "$OUT_CSV"
