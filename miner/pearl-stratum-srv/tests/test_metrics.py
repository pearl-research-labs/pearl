"""Metrics + /metrics + /health HTTP endpoint tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from pearl_stratum_srv.metrics import Metrics, serve_http


# ----------------------------------------------------------- render tests


def test_render_includes_help_and_type_lines():
    m = Metrics()
    out = m.render_prometheus()
    assert "# HELP pearl_stratum_srv_connected_miners" in out
    assert "# TYPE pearl_stratum_srv_connected_miners gauge" in out
    assert "# TYPE pearl_stratum_srv_shares_total counter" in out


def test_render_per_worker_share_series():
    m = Metrics()
    m.record_share("rig04.gpu0", "accepted")
    m.record_share("rig04.gpu0", "accepted")
    m.record_share("rig04.gpu1", "stale")
    out = m.render_prometheus()
    assert 'pearl_stratum_srv_shares_total{worker="rig04.gpu0",outcome="accepted"} 2' in out
    assert 'pearl_stratum_srv_shares_total{worker="rig04.gpu1",outcome="stale"} 1' in out


def test_render_template_age_is_negative_when_no_template():
    m = Metrics()
    out = m.render_prometheus()
    # negative sentinel signals "never seen one"
    assert "pearl_stratum_srv_template_age_seconds -1.000" in out


def test_render_template_age_increases_with_time():
    m = Metrics()
    m.template_minted_at = time.time() - 10.0
    out = m.render_prometheus()
    # Match the metric, parse, assert ≥ 10.
    line = next(line for line in out.splitlines() if "template_age_seconds" in line and "#" not in line)
    age = float(line.split()[-1])
    assert age >= 10.0


def test_label_value_escaping():
    m = Metrics()
    m.record_share('weird"worker\\name', "accepted")
    out = m.render_prometheus()
    assert 'worker="weird\\"worker\\\\name"' in out


def test_block_counter_records_outcomes():
    m = Metrics()
    m.record_block("accepted")
    m.record_block("error")
    m.record_block("accepted")
    out = m.render_prometheus()
    assert 'pearl_stratum_srv_blocks_total{outcome="accepted"} 2' in out
    assert 'pearl_stratum_srv_blocks_total{outcome="error"} 1' in out


# ----------------------------------------------------------- health logic


def test_health_false_when_no_template():
    m = Metrics()
    ok, reason = m.is_healthy()
    assert not ok and "no template yet" in reason


def test_health_true_when_template_fresh():
    m = Metrics()
    m.template_minted_at = time.time()
    ok, reason = m.is_healthy(max_template_age_seconds=60.0)
    assert ok and reason == "ok"


def test_health_false_when_template_stale():
    m = Metrics()
    m.template_minted_at = time.time() - 120.0
    ok, reason = m.is_healthy(max_template_age_seconds=60.0)
    assert not ok and "exceeds 60" in reason


# ----------------------------------------------------------- HTTP endpoint


async def _http_get(host: str, port: int, path: str) -> tuple[int, str, str]:
    """Minimal HTTP/1.1 GET. Returns (status, content_type, body)."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=2.0)
    finally:
        writer.close()
        await writer.wait_closed()

    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    ctype = ""
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-type:"):
            ctype = line.split(b":", 1)[1].strip().decode()
    return status, ctype, body.decode("utf-8")


@pytest.fixture
async def http_server():
    m = Metrics()
    srv = await serve_http(m, host="127.0.0.1", port=0, max_template_age_seconds=10.0)
    port = srv.sockets[0].getsockname()[1]
    try:
        yield m, port
    finally:
        srv.close()
        await srv.wait_closed()


async def test_metrics_endpoint_returns_prometheus_text(http_server):
    metrics, port = http_server
    metrics.connected_miners = 5
    status, ctype, body = await _http_get("127.0.0.1", port, "/metrics")
    assert status == 200
    assert ctype.startswith("text/plain")
    assert "version=0.0.4" in ctype
    assert "pearl_stratum_srv_connected_miners 5" in body


async def test_health_endpoint_returns_503_when_no_template(http_server):
    _, port = http_server
    status, _, body = await _http_get("127.0.0.1", port, "/health")
    assert status == 503
    assert "no template" in body


async def test_health_endpoint_returns_200_when_fresh(http_server):
    metrics, port = http_server
    metrics.template_minted_at = time.time()
    status, _, body = await _http_get("127.0.0.1", port, "/health")
    assert status == 200
    assert body.strip() == "ok"


async def test_unknown_path_returns_404(http_server):
    _, port = http_server
    # `/` now serves the dashboard (added in dashboard.py); use a bogus path.
    status, _, _ = await _http_get("127.0.0.1", port, "/definitely-not-a-route")
    assert status == 404


async def test_non_get_returns_405(http_server):
    _, port = http_server
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(b"POST /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=2.0)
    finally:
        writer.close()
        await writer.wait_closed()
    assert b"405" in raw
