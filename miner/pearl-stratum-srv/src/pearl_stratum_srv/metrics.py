"""Prometheus-text /metrics + /health HTTP endpoint.

Plain stdlib (no prometheus_client dep) — emits Prometheus exposition format
on GET /metrics, returns 200 OK on GET /health when the pool has fresh state.

What we expose:
  pearl_stratum_srv_connected_miners            gauge
  pearl_stratum_srv_template_age_seconds        gauge — seconds since the
                                                latest template was minted;
                                                a runaway value means pearld
                                                stopped delivering work
  pearl_stratum_srv_template_height             gauge
  pearl_stratum_srv_shares_total{worker,outcome="accepted|stale|malformed"} counter
  pearl_stratum_srv_blocks_total{outcome}        counter
  pearl_stratum_srv_jobs_in_registry            gauge

The per-worker `shares_total{worker=..}` series is the load-bearing fleet-
liveness signal: scrape every 15s, alert if any worker label drops to zero
rate for >120s.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

_LOGGER = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Plain in-memory counters/gauges. Single-threaded asyncio access only."""

    shares_total: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """(worker, outcome) → count. outcome ∈ {accepted,stale,malformed}."""

    blocks_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """outcome → count. outcome ∈ {accepted, rejected, error}."""

    connected_miners: int = 0
    template_height: int = 0
    template_minted_at: float = 0.0
    jobs_in_registry: int = 0

    def record_share(self, worker: str, outcome: str) -> None:
        self.shares_total[(worker or "anonymous", outcome)] += 1

    def record_block(self, outcome: str) -> None:
        self.blocks_total[outcome] += 1

    def render_prometheus(self) -> str:
        """Render in Prometheus text exposition format (v0.0.4)."""
        now = time.time()
        template_age = (now - self.template_minted_at) if self.template_minted_at else -1.0
        lines: list[str] = []

        lines += [
            "# HELP pearl_stratum_srv_connected_miners Currently connected stratum clients.",
            "# TYPE pearl_stratum_srv_connected_miners gauge",
            f"pearl_stratum_srv_connected_miners {self.connected_miners}",
            "",
            "# HELP pearl_stratum_srv_template_age_seconds Seconds since the latest mined job.",
            "# TYPE pearl_stratum_srv_template_age_seconds gauge",
            f"pearl_stratum_srv_template_age_seconds {template_age:.3f}",
            "",
            "# HELP pearl_stratum_srv_template_height Height of the latest broadcast template.",
            "# TYPE pearl_stratum_srv_template_height gauge",
            f"pearl_stratum_srv_template_height {self.template_height}",
            "",
            "# HELP pearl_stratum_srv_jobs_in_registry Recent jobs cached for submit lookup.",
            "# TYPE pearl_stratum_srv_jobs_in_registry gauge",
            f"pearl_stratum_srv_jobs_in_registry {self.jobs_in_registry}",
            "",
            "# HELP pearl_stratum_srv_shares_total Shares received, by worker and outcome.",
            "# TYPE pearl_stratum_srv_shares_total counter",
        ]
        for (worker, outcome), count in sorted(self.shares_total.items()):
            w = _escape_label(worker)
            o = _escape_label(outcome)
            lines.append(f'pearl_stratum_srv_shares_total{{worker="{w}",outcome="{o}"}} {count}')

        lines += [
            "",
            "# HELP pearl_stratum_srv_blocks_total Blocks attempted, by outcome.",
            "# TYPE pearl_stratum_srv_blocks_total counter",
        ]
        for outcome, count in sorted(self.blocks_total.items()):
            o = _escape_label(outcome)
            lines.append(f'pearl_stratum_srv_blocks_total{{outcome="{o}"}} {count}')

        return "\n".join(lines) + "\n"

    def is_healthy(self, max_template_age_seconds: float = 60.0) -> tuple[bool, str]:
        """Pool is healthy iff template was minted recently. Returns (ok, reason)."""
        if self.template_minted_at == 0.0:
            return False, "no template yet"
        age = time.time() - self.template_minted_at
        if age > max_template_age_seconds:
            return False, f"template age {age:.1f}s exceeds {max_template_age_seconds}s"
        return True, "ok"


def _escape_label(value: str) -> str:
    # Prometheus label escaping: \\ → \\\\, " → \\", newline → \\n
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


async def serve_http(
    metrics: Metrics,
    host: str,
    port: int,
    max_template_age_seconds: float = 60.0,
    history=None,
    server=None,
) -> asyncio.base_events.Server:
    """Start an HTTP listener on (host, port). Routes:
      GET  /metrics       Prometheus exposition
      GET  /health        200 / 503 based on template freshness
      GET  /              built-in dashboard HTML (vanilla JS, no deps)
      GET  /api/stats     JSON snapshot for the dashboard
      GET  /api/history   JSON: last N samples for chart rendering

    `history` (pearl_stratum_srv.dashboard.History) is optional; if absent,
    /api/history returns an empty list.
    """
    import json

    from pearl_stratum_srv.dashboard import DASHBOARD_HTML, stats_snapshot

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            parts = request_line.decode("ascii", errors="replace").split()
            method, path = (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "/")

            if method != "GET":
                await _write(writer, 405, "text/plain", "method not allowed\n")
            elif path == "/metrics":
                await _write(writer, 200, "text/plain; version=0.0.4", metrics.render_prometheus())
            elif path == "/health":
                ok, reason = metrics.is_healthy(max_template_age_seconds)
                code = 200 if ok else 503
                await _write(writer, code, "text/plain", reason + "\n")
            elif path == "/" or path == "/index.html":
                await _write(writer, 200, "text/html; charset=utf-8", DASHBOARD_HTML)
            elif path == "/api/stats":
                body = json.dumps(stats_snapshot(metrics))
                await _write(writer, 200, "application/json", body)
            elif path == "/api/history":
                body = json.dumps(history.as_json() if history is not None else {"samples": []})
                await _write(writer, 200, "application/json", body)
            elif path == "/api/alerts":
                if server is not None and getattr(server, "alerter", None) is not None:
                    body = json.dumps(server.alerter.as_json())
                else:
                    body = json.dumps({"active": [], "count": 0})
                await _write(writer, 200, "application/json", body)
            elif path.startswith("/api/miner"):
                # /api/miner?addr=prl1... — per-address stats. Public; address acts as key.
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(path).query)
                addr = qs.get("addr", [""])[0]
                if not addr or server is None:
                    await _write(writer, 400, "application/json", '{"error":"addr required"}')
                else:
                    body = json.dumps(await _miner_stats(server, addr))
                    await _write(writer, 200, "application/json", body)
            elif path.startswith("/api/op"):
                # Operator-only — gated by shared-secret token in settings.
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(path).query)
                token = qs.get("token", [""])[0]
                if (server is None
                    or not server.settings.operator_dashboard_token
                    or token != server.settings.operator_dashboard_token):
                    await _write(writer, 403, "application/json", '{"error":"forbidden"}')
                else:
                    body = json.dumps(await _operator_stats(server))
                    await _write(writer, 200, "application/json", body)
            else:
                await _write(writer, 404, "text/plain", "not found\n")
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        except Exception:
            _LOGGER.exception("metrics handler crashed")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return await asyncio.start_server(handle, host=host, port=port)


async def _miner_stats(server, addr: str) -> dict:
    """Per-address slice: recent shares + total contribution."""
    import time as _time

    if server.share_db is None:
        return {"error": "public-pool mode required"}
    since = _time.time() - 86400.0
    shares = await server.share_db.shares_for_worker(addr, since)
    recent = [{"ts": ts, "outcome": outcome, "difficulty": diff, "label": label}
              for (ts, outcome, diff, label) in shares[:100]]
    total_diff_24h = sum(d for (_ts, o, d, _l) in shares if o == "accepted")
    by_outcome = {"accepted": 0, "stale": 0, "malformed": 0}
    for (_ts, o, _d, _l) in shares:
        by_outcome[o] = by_outcome.get(o, 0) + 1
    return {
        "address": addr,
        "recent_shares": recent,
        "total_difficulty_24h": total_diff_24h,
        "share_counts_24h": by_outcome,
    }


async def _operator_stats(server) -> dict:
    """Operator-only fleet view: pending payouts, block count, revenue."""
    if server.share_db is None:
        return {"error": "public-pool mode required"}
    pending = await server.share_db.pending_payouts()
    block_count = await server.share_db.block_count()
    pending_total = sum(amount for (_id, _r, amount, _sc, _ts) in pending)
    return {
        "blocks_found_total": block_count,
        "pending_payout_count": len(pending),
        "pending_payout_total_sats": pending_total,
        "pending_payouts": [
            {"id": pid, "recipient": r, "amount_sats": a, "share_count": sc, "created_at": ts}
            for (pid, r, a, sc, ts) in pending[:50]
        ],
    }


async def _write(writer: asyncio.StreamWriter, status: int, ctype: str, body: str) -> None:
    reason = {200: "OK", 404: "Not Found", 405: "Method Not Allowed", 503: "Service Unavailable"}.get(
        status, "Status"
    )
    payload = body.encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {ctype}\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(head + payload)
    await writer.drain()
