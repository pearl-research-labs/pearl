#!/usr/bin/env bash
#
# End-to-end smoke tests for the Pearl RPC proxy sidecar.
#
# Connects the proxy to an existing node (local or remote), then verifies
# TLS termination, auth passthrough, rate limiting, and GBT caching.
#
# Usage:
#   ./proxy/proxy_test.sh --node-rpc=node1.testnet.pearlresearch.ai:44111 \
#                         --rpc-user=admin --rpc-pass=pass
#
# All flags can also be set via environment variables:
#   NODE_RPC, RPC_USER, RPC_PASS, PROXY_PORT, RATE_LIMIT

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

NODE_RPC="${NODE_RPC:-}"
RPC_USER="${RPC_USER:-}"
RPC_PASS="${RPC_PASS:-}"
PROXY_PORT="${PROXY_PORT:-44111}"
RATE_LIMIT="${RATE_LIMIT:-100}"

for arg in "$@"; do
    case "$arg" in
        --node-rpc=*)   NODE_RPC="${arg#*=}" ;;
        --rpc-user=*)   RPC_USER="${arg#*=}" ;;
        --rpc-pass=*)   RPC_PASS="${arg#*=}" ;;
        --proxy-port=*) PROXY_PORT="${arg#*=}" ;;
        --rate-limit=*) RATE_LIMIT="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 --node-rpc=HOST:PORT --rpc-user=USER --rpc-pass=PASS"
            echo ""
            echo "Options:"
            echo "  --node-rpc=HOST:PORT   RPC endpoint of the target node (required)"
            echo "  --rpc-user=USER        RPC username (required)"
            echo "  --rpc-pass=PASS        RPC password (required)"
            echo "  --proxy-port=PORT      Local port for the proxy (default: 44111)"
            echo "  --rate-limit=N         Expected rate limit events/min (default: 100)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ -z "$NODE_RPC" ] || [ -z "$RPC_USER" ] || [ -z "$RPC_PASS" ]; then
    echo "Error: --node-rpc, --rpc-user, and --rpc-pass are required."
    echo "Run with --help for usage."
    exit 1
fi

CONTAINER_NAME="pearld-proxy-test-$$"
CA_CERT=""
PASS_COUNT=0
FAIL_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "  FAIL: $1"; }

cleanup() {
    echo ""
    echo "[teardown] Removing test container..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    rm -f "$CA_CERT" "$CADDYFILE" "$HEADER_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# ============================================================
# Setup
# ============================================================

echo "=== Pearl RPC Proxy Smoke Tests ==="
echo "  Node:  $NODE_RPC"
echo "  Proxy: localhost:$PROXY_PORT"
echo ""

echo "[setup] Verifying node is reachable..."
DIRECT=$(curl -sf --max-time 5 -u "${RPC_USER}:${RPC_PASS}" \
    -d '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":0}' \
    "http://${NODE_RPC}" 2>&1) || { echo "Error: cannot reach node at $NODE_RPC"; exit 1; }
echo "  Node block height: $(echo "$DIRECT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"])' 2>/dev/null || echo 'unknown')"
echo ""

CADDYFILE=$(mktemp /tmp/pearl-caddyfile-XXXXXX)
cat > "$CADDYFILE" <<EOF
localhost:${PROXY_PORT} {
	tls internal

	rate_limit {
		zone rpc {
			key    {remote_host}
			events ${RATE_LIMIT}
			window 1m
		}
	}

	jsonrpc_cache {
		cache getblocktemplate 3s
	}

	reverse_proxy http://${NODE_RPC}
}
EOF

echo "[setup] Building proxy image..."
docker build -t pearld-proxy proxy/ -q 2>&1

echo "[setup] Starting proxy container..."
docker run -d --name "$CONTAINER_NAME" \
    -p "${PROXY_PORT}:${PROXY_PORT}" \
    -v "${CADDYFILE}:/etc/caddy/Caddyfile:ro" \
    pearld-proxy > /dev/null 2>&1

echo "[setup] Waiting for proxy to be ready..."
sleep 3

CA_CERT=$(mktemp /tmp/pearl-ca-XXXXXX.crt)
docker cp "${CONTAINER_NAME}:/data/caddy/pki/authorities/local/root.crt" "$CA_CERT" 2>/dev/null || true

echo ""

# ============================================================
# Tests
# ============================================================

RPC_URL="https://localhost:${PROXY_PORT}"

rpc() {
    curl -sf --max-time 5 --cacert "$CA_CERT" \
        -u "${RPC_USER}:${RPC_PASS}" \
        -d "$1" "$RPC_URL" 2>/dev/null
}

rpc_code() {
    curl -s --max-time 5 --cacert "$CA_CERT" \
        -o /dev/null -w "%{http_code}" \
        -u "${RPC_USER}:${RPC_PASS}" \
        -d "$1" "$RPC_URL" 2>/dev/null || echo "000"
}

HEADER_FILE=$(mktemp /tmp/pearl-headers-XXXXXX)

rpc_headers() {
    curl -s --max-time 10 --cacert "$CA_CERT" \
        -D "$HEADER_FILE" -o /dev/null \
        -u "${RPC_USER}:${RPC_PASS}" \
        -d "$1" "$RPC_URL" 2>/dev/null || true
    cat "$HEADER_FILE"
}

# --- 1: TLS handshake ---
echo "[test] TLS handshake with Caddy CA cert"
if RESP=$(rpc '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":1}') && echo "$RESP" | grep -q '"result"'; then
    pass "TLS handshake succeeds, getblockcount returns result"
else
    fail "TLS handshake or RPC call failed"
fi

# --- 2: Certificate issuer ---
echo "[test] Certificate issued by Caddy internal CA"
CERT_INFO=$(echo | openssl s_client -connect "localhost:${PROXY_PORT}" -servername localhost 2>/dev/null)
ISSUER=$(echo "$CERT_INFO" | openssl x509 -noout -issuer 2>/dev/null || echo "")
if echo "$ISSUER" | grep -qi "caddy"; then
    pass "Certificate issuer contains 'Caddy'"
else
    fail "Unexpected certificate issuer: ${ISSUER}"
fi

# --- 3: Plain HTTP rejected ---
echo "[test] Plain HTTP is rejected"
HTTP_RESP=$(curl -s --max-time 3 \
    -d '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":1}' \
    "http://localhost:${PROXY_PORT}" 2>&1 || echo "")
if echo "$HTTP_RESP" | grep -qi "HTTPS"; then
    pass "HTTP request rejected with HTTPS error"
else
    fail "HTTP request was not rejected: ${HTTP_RESP}"
fi

# --- 4: Auth passthrough (wrong creds) ---
echo "[test] Wrong credentials rejected (HTTP 401)"
CODE=$(curl -s --max-time 5 --cacert "$CA_CERT" \
    -o /dev/null -w "%{http_code}" \
    -u "wrong:credentials" \
    -d '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":1}' \
    "$RPC_URL" 2>/dev/null || echo "000")
if [ "$CODE" = "401" ]; then
    pass "Wrong credentials return HTTP 401"
else
    fail "Wrong credentials returned HTTP ${CODE}, expected 401"
fi

# --- 5: GBT cache miss (first request) ---
echo "[test] GBT cache miss on first request"
HDRS=$(rpc_headers '{"jsonrpc":"1.0","method":"getblocktemplate","params":[{"rules":["segwit"]}],"id":10}')
if echo "$HDRS" | grep -qi "x-jsonrpc-cache.*HIT"; then
    fail "First GBT request was a cache HIT (expected MISS)"
else
    pass "First GBT request is a cache MISS"
fi

# --- 6: GBT cache hit (second request) ---
echo "[test] GBT cache hit on second request"
HDRS=$(rpc_headers '{"jsonrpc":"1.0","method":"getblocktemplate","params":[{"rules":["segwit"]}],"id":11}')
if echo "$HDRS" | grep -qi "x-jsonrpc-cache.*HIT"; then
    pass "Second GBT request is a cache HIT"
else
    fail "Second GBT request was not a cache HIT"
fi

# --- 7: Cache preserves JSON-RPC id ---
echo "[test] Cache preserves JSON-RPC request id"
RESP=$(rpc '{"jsonrpc":"1.0","method":"getblocktemplate","params":[{"rules":["segwit"]}],"id":99999}')
if echo "$RESP" | grep -q '"id":99999'; then
    pass "Cached response has caller's request id (99999)"
else
    fail "Response id mismatch: $(echo "$RESP" | head -c 100)"
fi

# --- 8: Non-cached method has no cache header ---
echo "[test] Non-cached method skips cache"
HDRS=$(rpc_headers '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":1}')
if echo "$HDRS" | grep -qi "x-jsonrpc-cache"; then
    fail "getblockcount unexpectedly has cache header"
else
    pass "getblockcount has no cache header"
fi

# --- 9: Cache TTL expiry ---
echo "[test] Cache expires after TTL (3s)"
rpc '{"jsonrpc":"1.0","method":"getblocktemplate","params":[{"rules":["segwit"]}],"id":20}' > /dev/null
sleep 4
HDRS=$(rpc_headers '{"jsonrpc":"1.0","method":"getblocktemplate","params":[{"rules":["segwit"]}],"id":21}')
if echo "$HDRS" | grep -qi "x-jsonrpc-cache.*HIT"; then
    fail "GBT request after TTL was still a HIT"
else
    pass "GBT request after TTL is a MISS (cache expired)"
fi

# --- 10: Rate limiting (run last — exhausts the rate limit window) ---
echo "[test] Rate limiting (${RATE_LIMIT} req/min)"
RATE_429=0
for _ in $(seq 1 $((RATE_LIMIT + 10))); do
    CODE=$(rpc_code '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":1}')
    if [ "$CODE" = "429" ]; then
        RATE_429=$((RATE_429 + 1))
    fi
done
if [ "$RATE_429" -gt 0 ]; then
    pass "Rate limiter returned 429 for ${RATE_429} excess requests"
else
    fail "Rate limiter did not return any 429 responses"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== Results: ${PASS_COUNT} passed, ${FAIL_COUNT} failed ==="
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
