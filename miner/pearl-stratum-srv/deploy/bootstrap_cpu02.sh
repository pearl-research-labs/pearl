#!/usr/bin/env bash
# Bootstrap the solo pool on CPU02 (or any Linux box).
#
# Assumes:
#   - This repo is already cloned to /opt/pearl
#   - pearld + oyster binaries already built (`task build:pearld` etc.)
#   - `uv` is on PATH
#   - You have sudo
#
# Idempotent: re-running is safe.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/opt/pearl}
POOL_ROOT="$REPO_ROOT/miner/pearl-stratum-srv"
ENV_DIR=/etc/pearl-stratum-srv
LOG_DIR=/var/log/pearl-stratum-srv
SYSTEMD_UNIT=/etc/systemd/system/pearl-stratum-srv.service

if [[ ! -d "$POOL_ROOT" ]]; then
  echo "FATAL: $POOL_ROOT does not exist. Set REPO_ROOT or clone first." >&2
  exit 1
fi

echo "== 1/6  Ensure 'pearl' system user"
if ! id pearl &>/dev/null; then
  sudo useradd --system --no-create-home --shell /usr/sbin/nologin pearl
fi

echo "== 2/6  Create log dir + env dir (owner: pearl)"
sudo install -d -m 0750 -o pearl -g pearl "$LOG_DIR"
sudo install -d -m 0750 -o root  -g pearl "$ENV_DIR"

if [[ ! -f "$ENV_DIR/env" ]]; then
  sudo install -m 0640 -o root -g pearl "$POOL_ROOT/deploy/env.example" "$ENV_DIR/env"
  echo "  -> seeded $ENV_DIR/env from env.example — EDIT IT BEFORE STARTING"
else
  echo "  -> $ENV_DIR/env already present, leaving alone"
fi

echo "== 3/6  uv sync (creates .venv with pearl-stratum-srv + deps)"
(cd "$POOL_ROOT" && uv sync)

echo "== 4/6  Run unit tests as a smoke check"
(cd "$POOL_ROOT" && uv run pytest -q)

echo "== 5/6  Install systemd unit"
sudo install -m 0644 "$POOL_ROOT/deploy/pearl-stratum-srv.service" "$SYSTEMD_UNIT"
sudo systemctl daemon-reload

echo "== 6/6  Done. Next steps:"
cat <<EOF

  1. Edit $ENV_DIR/env and set:
       PEARL_SRV_RPC_PASSWORD=<your pearld rpcpass>
       PEARL_SRV_MINING_ADDRESS=<\$(oyster-cli getnewaddress)>

  2. sudo systemctl enable --now pearl-stratum-srv

  3. Verify health:
       curl -sf http://localhost:9101/health      # → "ok"
       curl    http://localhost:9101/metrics      # → Prometheus exposition

  4. Watch logs:
       journalctl -u pearl-stratum-srv -f

  5. Point a pilot rig at this host's :5566 via mfarm flight sheet.

EOF
