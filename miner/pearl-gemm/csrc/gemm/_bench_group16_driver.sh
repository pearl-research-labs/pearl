#!/usr/bin/env bash
# Drives bench_group16 across the canonical shape set, twice — once with
# PEARL_SM89_GROUP16_SWIZZLE unset (adaptive S=4/32) and once with it set
# (alpha-miner-style fixed S=16, N-major). Writes CSV to $OUT_CSV.
#
# Run on the rig under test (CPU01 / CPU02 / mini28 / AI01), NOT a production
# rig. mfarm-agent + alpha-miner MUST be stopped before invocation; the script
# does not stop them itself (see memo: feedback_dont_bench_on_production_rigs).

set -euo pipefail
DEV="${1:-0}"
OUT_CSV="${2:-/tmp/bench_group16.csv}"
BENCH="${3:-/tmp/bench_group16}"

# Shapes: (M, N, K, iters)
# Iters chosen so each shape takes ~1-3 s wall on AD103. Match the Tier 1a
# memo's bench set so direct apples-to-apples comparison is possible.
SHAPES=(
  "1024 1024 1024 50"
  "2048 2048 2048 30"
  "4096 4096 4096 20"
  "4096 4096 8192 15"
  "8192 8192 4096 10"
  "4096 16384 4096 10"
  "16384 4096 4096 10"
)

echo "M,N,K,variant,median_us,tops" > "$OUT_CSV"

for shape in "${SHAPES[@]}"; do
  read -r M N K I <<< "$shape"
  for variant in adaptive group16; do
    env_extra=""
    if [ "$variant" = "group16" ]; then
      env_extra="PEARL_SM89_GROUP16_SWIZZLE=1"
    fi
    out="$(env $env_extra "$BENCH" "$DEV" "$M" "$N" "$K" "$I")"
    echo "$out" | tee /dev/stderr | grep '^CSV,' | sed 's/^CSV,//' >> "$OUT_CSV"
  done
done

echo "---"
echo "Done. CSV at $OUT_CSV:"
cat "$OUT_CSV"
