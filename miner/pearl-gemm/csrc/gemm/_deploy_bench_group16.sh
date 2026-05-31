#!/usr/bin/env bash
# Deploy bench_group16 + driver to a bench rig (CPU01/CPU02) and run.
# Usage: ./_deploy_bench_group16.sh <rig_ip> [<dev>]
#
# Caller must have ALREADY pulled the rig off the production flight sheet via
# CatStack — see memo feedback_dont_bench_on_production_rigs.md.

set -euo pipefail
RIG="${1:-192.168.71.252}"  # default CPU01
DEV="${2:-0}"
SRC_BIN="/mnt/c/Source/pearl/miner/pearl-gemm/build-sm89/tier1a/bench_group16"
SRC_SH="/mnt/c/Source/pearl/miner/pearl-gemm/csrc/gemm/_bench_group16_driver.sh"

# Sanity check: confirm binary exists locally.
[ -f "$SRC_BIN" ] || { echo "ERROR: $SRC_BIN not found"; exit 1; }
[ -f "$SRC_SH" ]  || { echo "ERROR: $SRC_SH not found"; exit 1; }

echo "==> Pre-flight check on $RIG"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$RIG" \
  'systemctl stop mfarm-agent 2>/dev/null || true; pkill -9 alpha-miner 2>/dev/null || true; sleep 1; nvidia-smi -L; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'

echo "==> Copy artifacts"
scp -o StrictHostKeyChecking=no "$SRC_BIN" "root@$RIG:/tmp/bench_group16"
scp -o StrictHostKeyChecking=no "$SRC_SH"  "root@$RIG:/tmp/bench_group16_driver.sh"

echo "==> Run bench on $RIG GPU $DEV"
ssh -o StrictHostKeyChecking=no root@"$RIG" \
  "chmod +x /tmp/bench_group16 /tmp/bench_group16_driver.sh; \
   /tmp/bench_group16_driver.sh $DEV /tmp/bench_group16.csv /tmp/bench_group16"

echo "==> Pull CSV back"
scp -o StrictHostKeyChecking=no \
  "root@$RIG:/tmp/bench_group16.csv" \
  /mnt/c/Source/pearl-investigation/bench_group16_swizzle_2026_05_17.csv
echo "Wrote bench_group16_swizzle_2026_05_17.csv"
