#!/usr/bin/env bash
# Sweep PEARL_SM89_SWIZZLE across {2, 4, 8, 16, 32, 64} to triangulate the
# group-of-16 result. The base case (no env) uses the adaptive picker (S=4
# skinny, S=32 balanced).
#
# Run on a bench rig (CPU01) with mfarm-agent stopped + GPU compute apps paused.

set -euo pipefail
DEV="${1:-0}"
OUT_CSV="${2:-/tmp/bench_swizzle_sweep.csv}"
BENCH="${3:-/tmp/bench_group16}"

# A representative subset of production-relevant shapes.
SHAPES=(
  "4096 4096 4096 20"
  "4096 4096 8192 15"
  "8192 8192 4096 10"
)

SWIZZLES=("0" "2" "4" "8" "16" "32" "64")  # 0 = adaptive baseline

echo "M,N,K,swizzle,median_us,tops" > "$OUT_CSV"

for shape in "${SHAPES[@]}"; do
  read -r M N K I <<< "$shape"
  for S in "${SWIZZLES[@]}"; do
    if [ "$S" = "0" ]; then
      env_extra=""
      label="adaptive"
    else
      env_extra="PEARL_SM89_SWIZZLE=$S"
      label="$S"
    fi
    out="$(env $env_extra "$BENCH" "$DEV" "$M" "$N" "$K" "$I")"
    echo "$out" >&2
    # parse: CSV,M,N,K,variant,med_us,tops — replace `variant` with our label
    csv_line="$(echo "$out" | grep '^CSV,' | head -1)"
    # we control the label here, replace what bench_group16 wrote
    med_us=$(echo "$csv_line" | awk -F, '{print $6}')
    tops=$(echo "$csv_line" | awk -F, '{print $7}')
    echo "$M,$N,$K,$label,$med_us,$tops" >> "$OUT_CSV"
  done
done

echo "---"
cat "$OUT_CSV"
