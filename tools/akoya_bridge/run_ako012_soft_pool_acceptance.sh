#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="${PEARL_OPS_DIR:-/home/bereket/pearl-ops}"
LAST_ENV="$OPS_DIR/last_instance.env"
if [[ ! -f "$LAST_ENV" ]]; then
  echo "missing $LAST_ENV; start or rent a GPU instance first" >&2
  exit 2
fi

source "$LAST_ENV"

MAX_RUN_USD="${MAX_RUN_USD:-8}"
REMOTE_REPO_DIR="${REPO_DIR:-/workspace/pearl-src}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/workspace/pearl-runs}"
LOCAL_REPO_DIR="${AKO012_LOCAL_REPO_DIR:-/home/bereket/pearl-miner-benchmark-tooling}"
TIMEOUT_SECONDS="${AKO012_TIMEOUT_SECONDS:-1800}"
AUTO_DESTROY="${PEARL_AUTO_DESTROY_AFTER_AKO012:-yes}"
REPO_REMOTE="${AKO012_REPO_REMOTE:-https://github.com/bket7/pearl-miner-benchmark-tooling.git}"
REPO_REF="${AKO012_REPO_REF:-codex/pearl-20260515-handoff}"
SOFT_NBITS="${AKO012_SOFT_NBITS:-0x1e3fffff}"
MAX_ATTEMPTS="${AKO012_MAX_ATTEMPTS:-40}"
DURATION_SECONDS="${AKO012_DURATION_SECONDS:-420}"
PORT="${AKO012_PORT:-3334}"
SERVER_TIMEOUT_SECONDS="${AKO012_SERVER_TIMEOUT_SECONDS:-$((DURATION_SECONDS + 300))}"
TILE_SIZE_N="${AKO012_TILE_SIZE_N:-256}"
SWIZZLE_N_MAJ="${AKO012_SWIZZLE_N_MAJ:-false}"
KERNEL_CONFIG_MODULE="${AKO012_KERNEL_CONFIG_MODULE:-}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="ako012_soft_pool_${TS}_${INSTANCE_ID}"
LOCAL_RUN_DIR="$OPS_DIR/artifacts/ako012-soft-pool/$RUN_ID"
LOG_DIR="$OPS_DIR/logs"
mkdir -p "$LOCAL_RUN_DIR" "$LOG_DIR"
LOG="$LOG_DIR/${RUN_ID}.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

estimate_cost_guard() {
  local row rate
  row="$(vastai show instance "$INSTANCE_ID" 2>/dev/null | tail -1 || true)"
  rate="$(awk '{print $12}' <<<"$row")"
  python3 - "$MAX_RUN_USD" "$TIMEOUT_SECONDS" "$rate" <<'PY'
import sys
max_usd = float(sys.argv[1])
seconds = int(sys.argv[2])
try:
    rate = float(sys.argv[3])
except Exception:
    rate = 999.0
cost = rate * seconds / 3600.0
print(f"estimated_max_cost_usd={cost:.4f} rate_usd_per_hour={rate:.4f} timeout_seconds={seconds}")
if cost > max_usd:
    raise SystemExit(f"refusing: estimated ${cost:.2f} exceeds max ${max_usd:.2f}")
PY
}

SSH_BASE=(
  ssh
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -p "$SSH_PORT"
  "$SSH_TARGET"
)

sync_artifacts() {
  mkdir -p "$LOCAL_RUN_DIR/remote"
  rsync -az -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p '$SSH_PORT'" \
    "$SSH_TARGET:$REMOTE_RUN_ROOT/$RUN_ID/" "$LOCAL_RUN_DIR/remote/" >>"$LOG" 2>&1 || true
}

destroy_instance() {
  if [[ "$AUTO_DESTROY" == "yes" ]]; then
    log "destroying Vast instance $INSTANCE_ID"
    vastai destroy instance "$INSTANCE_ID" >>"$LOG" 2>&1 || true
  else
    log "auto-destroy disabled; instance $INSTANCE_ID remains running"
  fi
}

cleanup() {
  sync_artifacts
  destroy_instance
}
trap cleanup EXIT

log "starting AKO-012 private soft-pool run_id=$RUN_ID instance_id=$INSTANCE_ID soft_nbits=$SOFT_NBITS"
estimate_cost_guard | tee -a "$LOG"

if [[ ! -d "$LOCAL_REPO_DIR" ]]; then
  echo "missing local repo at $LOCAL_REPO_DIR" >&2
  exit 2
fi

log "deploying local repo snapshot from $LOCAL_REPO_DIR"
"${SSH_BASE[@]}" "rm -rf '$REMOTE_REPO_DIR'; mkdir -p '$REMOTE_REPO_DIR' '$REMOTE_RUN_ROOT'"
tar -C "$LOCAL_REPO_DIR" \
  --exclude=.git \
  --exclude=.venv \
  --exclude='target' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  -czf - . | "${SSH_BASE[@]}" "tar -xzf - -C '$REMOTE_REPO_DIR'"

log "remote preflight"
"${SSH_BASE[@]}" "set -euo pipefail; hostname; date -u; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; test -d '$REMOTE_REPO_DIR'; cd '$REMOTE_REPO_DIR'; git rev-parse --short HEAD 2>/dev/null || echo nogit-rsync-tree; (command -v uv && uv --version) || true" \
  2>&1 | tee "$LOCAL_RUN_DIR/remote_preflight.txt" | tee -a "$LOG"

log "running remote AKO-012 soft-pool acceptance"
set +e
"${SSH_BASE[@]}" \
  "RUN_ID='$RUN_ID' REMOTE_REPO_DIR='$REMOTE_REPO_DIR' REMOTE_RUN_ROOT='$REMOTE_RUN_ROOT' REPO_REMOTE='$REPO_REMOTE' REPO_REF='$REPO_REF' SOFT_NBITS='$SOFT_NBITS' MAX_ATTEMPTS='$MAX_ATTEMPTS' DURATION_SECONDS='$DURATION_SECONDS' SERVER_TIMEOUT_SECONDS='$SERVER_TIMEOUT_SECONDS' TILE_SIZE_N='$TILE_SIZE_N' SWIZZLE_N_MAJ='$SWIZZLE_N_MAJ' KERNEL_CONFIG_MODULE='$KERNEL_CONFIG_MODULE' PORT='$PORT' timeout '$TIMEOUT_SECONDS' bash -s" \
  >"$LOCAL_RUN_DIR/remote_stdout.log" 2>"$LOCAL_RUN_DIR/remote_stderr.log" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

RUN_DIR="$REMOTE_RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update >"$RUN_DIR/apt_update.log" 2>&1
  apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3-dev \
    python3.12-dev \
    rsync \
    >"$RUN_DIR/apt_install.log" 2>&1
fi

cd "$REMOTE_REPO_DIR"
if [[ -d .git ]]; then
  git config --global --add safe.directory "$REMOTE_REPO_DIR" || true
  git remote set-url origin "$REPO_REMOTE" || true
  git fetch origin "$REPO_REF" --depth=1
  git checkout -B "$REPO_REF" "origin/$REPO_REF"
  git submodule update --init --recursive
else
  echo "remote tree has no .git; assuming caller deployed code via rsync" >"$RUN_DIR/deploy_mode.txt"
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

export PEARL_GEMM_DISABLE_R64=TRUE
if [[ -n "$KERNEL_CONFIG_MODULE" ]]; then
  export PEARL_GEMM_KERNEL_CONFIG_MODULE="$KERNEL_CONFIG_MODULE"
fi
if [[ ! -d .venv ]]; then
  export UV_TORCH_BACKEND=cu129
  uv sync --package vllm-miner --package pearl-gemm --no-editable --refresh >"$RUN_DIR/uv_sync.log" 2>&1
fi

TORCH_LIB="$REMOTE_REPO_DIR/.venv/lib/python3.12/site-packages/torch/lib"
if [[ -d "$TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
fi

{
  echo "run_id=$RUN_ID"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REMOTE_REPO_DIR"
  git rev-parse HEAD 2>/dev/null || echo "nogit-rsync-tree"
  git status --short 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "soft_nbits=$SOFT_NBITS"
  echo "max_attempts=$MAX_ATTEMPTS"
  echo "duration_seconds=$DURATION_SECONDS"
  echo "server_timeout_seconds=$SERVER_TIMEOUT_SECONDS"
  echo "tile_size_n=$TILE_SIZE_N"
  echo "swizzle_n_maj=$SWIZZLE_N_MAJ"
  echo "kernel_config_module=${KERNEL_CONFIG_MODULE:-default}"
  echo "PEARL_GEMM_DISABLE_R64=$PEARL_GEMM_DISABLE_R64"
} >"$RUN_DIR/provenance.txt"

.venv/bin/python -m py_compile \
  tools/akoya_bridge/akoya_soft_pool_server.py \
  tools/akoya_bridge/direct_gpu_akoya_submit.py \
  tools/akoya_bridge/verify_captured_share.py

SERVER_OUT="$RUN_DIR/server_result.json"
.venv/bin/python tools/akoya_bridge/akoya_soft_pool_server.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --share-difficulty "$SOFT_NBITS" \
  --network-nbits 0x207fffff \
  --timeout "$SERVER_TIMEOUT_SECONDS" \
  --out "$SERVER_OUT" \
  >"$RUN_DIR/server_stdout.log" 2>"$RUN_DIR/server_stderr.log" &
SERVER_PID=$!
echo "$SERVER_PID" >"$RUN_DIR/server.pid"

cleanup_remote() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_remote EXIT
sleep 1

SWIZZLE_ARGS=()
if [[ "$SWIZZLE_N_MAJ" == "true" ]]; then
  SWIZZLE_ARGS+=(--swizzle-n-maj)
else
  SWIZZLE_ARGS+=(--no-swizzle-n-maj)
fi

set +e
.venv/bin/python tools/akoya_bridge/direct_gpu_akoya_submit.py \
  --torch-noising \
  --fast-sideband \
  --tile-size-n "$TILE_SIZE_N" \
  "${SWIZZLE_ARGS[@]}" \
  --a-refresh-mode nonce-prefix \
  --a-nonce-bytes 12 \
  --submit \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-attempts "$MAX_ATTEMPTS" \
  --duration-seconds "$DURATION_SECONDS" \
  --timeout 20 \
  --wallet ako012-private-wallet \
  --worker ako012-soft \
  --out-dir "$RUN_DIR/direct_run" \
  >"$RUN_DIR/direct_stdout.log" 2>"$RUN_DIR/direct_stderr.log"
DIRECT_RC=$?
set -e

wait "$SERVER_PID"
SERVER_RC=$?
trap - EXIT

python3 - "$RUN_DIR" "$DIRECT_RC" "$SERVER_RC" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
direct_rc = int(sys.argv[2])
server_rc = int(sys.argv[3])

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}

direct = load(run_dir / "direct_run" / "summary.json")
server = load(run_dir / "server_result.json")
accepted = bool(server.get("accepted")) and direct.get("status") == "accepted"
summary = {
    "schema": "ako012_soft_pool_acceptance.v1",
    "accepted": accepted,
    "direct_rc": direct_rc,
    "server_rc": server_rc,
    "direct_status": direct.get("status"),
    "server_accepted": server.get("accepted"),
    "server_message": server.get("message"),
    "attempt_count": len(direct.get("attempts", [])) if isinstance(direct.get("attempts"), list) else None,
    "direct_summary": str(run_dir / "direct_run" / "summary.json"),
    "server_result": str(run_dir / "server_result.json"),
}
(run_dir / "ako012_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if not accepted:
    raise SystemExit(2)
PY
REMOTE
REMOTE_RC=$?
set -e

sync_artifacts
if [[ -f "$LOCAL_RUN_DIR/remote/ako012_summary.json" ]]; then
  cp "$LOCAL_RUN_DIR/remote/ako012_summary.json" "$LOCAL_RUN_DIR/ako012_summary.json"
fi
log "remote AKO-012 rc=$REMOTE_RC artifacts=$LOCAL_RUN_DIR"
exit "$REMOTE_RC"
