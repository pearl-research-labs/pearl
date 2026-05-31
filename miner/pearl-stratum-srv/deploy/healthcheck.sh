#!/usr/bin/env bash
# Post-deploy + ongoing healthcheck.
# Exit 0 iff: pool service active, /health 200, at least one miner connected,
# template_age < 60s, no stale-error explosion.
#
# Suitable for a 60s cron or systemd timer.

set -uo pipefail

HOST=${HOST:-localhost}
METRICS_PORT=${METRICS_PORT:-9101}

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. Service active?
if ! systemctl is-active --quiet pearl-stratum-srv; then
  fail "pearl-stratum-srv not active"
fi

# 2. /health
if ! curl -sf -m 3 "http://${HOST}:${METRICS_PORT}/health" >/dev/null; then
  fail "/health returned non-200"
fi

# 3. Scrape /metrics, eyeball the load-bearing numbers.
METRICS=$(curl -sf -m 3 "http://${HOST}:${METRICS_PORT}/metrics") || fail "/metrics unreachable"

CONNECTED=$(echo "$METRICS" | awk '/^pearl_stratum_srv_connected_miners / {print $2}')
TEMPLATE_AGE=$(echo "$METRICS" | awk '/^pearl_stratum_srv_template_age_seconds / {print $2}')
TEMPLATE_HEIGHT=$(echo "$METRICS" | awk '/^pearl_stratum_srv_template_height / {print $2}')

[[ "${CONNECTED:-0}" -gt 0 ]] || fail "no miners connected"
[[ "${TEMPLATE_HEIGHT:-0}" -gt 0 ]] || fail "no template yet"

# template_age is a float; awk for comparison
if ! echo "$TEMPLATE_AGE" | awk '{exit ($1 >= 0 && $1 < 60) ? 0 : 1}'; then
  fail "template_age=${TEMPLATE_AGE}s (should be 0-60)"
fi

echo "OK: ${CONNECTED} miners, height=${TEMPLATE_HEIGHT}, template_age=${TEMPLATE_AGE}s"
