"""Dashboard: HTML page + JSON endpoints + history ring."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pearl_stratum_srv.dashboard import (
    DASHBOARD_HTML,
    HISTORY_LEN,
    History,
    stats_snapshot,
)
from pearl_stratum_srv.metrics import Metrics, serve_http


# ----------------------------------------------------------- history ring


def test_history_initially_empty():
    h = History()
    assert h.as_json() == {"samples": []}


def test_history_records_one_sample():
    h = History()
    m = Metrics()
    m.connected_miners = 3
    m.template_height = 100
    m.template_minted_at = time.time()
    m.record_share("rig01", "accepted")
    m.record_share("rig01", "accepted")
    m.record_block("accepted")

    h.record(m)
    samples = h.as_json()["samples"]
    assert len(samples) == 1
    s = samples[0]
    assert s["connected_miners"] == 3
    assert s["template_height"] == 100
    assert s["blocks_accepted"] == 1
    assert s["shares_accepted_by_worker"] == {"rig01": 2}
    assert s["template_age"] >= 0


def test_history_evicts_when_capacity_exceeded():
    h = History(capacity=5)
    m = Metrics()
    for _ in range(10):
        h.record(m)
    assert len(h.as_json()["samples"]) == 5


def test_history_capacity_default_is_60_minutes():
    assert HISTORY_LEN == 60


def test_history_sample_excludes_stale_and_malformed_from_per_worker():
    h = History()
    m = Metrics()
    m.record_share("rig01", "accepted")
    m.record_share("rig01", "stale")
    m.record_share("rig01", "malformed")
    h.record(m)
    s = h.as_json()["samples"][0]
    assert s["shares_accepted_by_worker"] == {"rig01": 1}


# ----------------------------------------------------------- stats snapshot


def test_stats_snapshot_shape():
    m = Metrics()
    m.connected_miners = 2
    m.template_height = 50_000
    m.template_minted_at = time.time() - 5.0
    m.jobs_in_registry = 8
    m.record_share("rig04.gpu0", "accepted")
    m.record_share("rig04.gpu0", "accepted")
    m.record_share("rig04.gpu1", "stale")
    m.record_block("accepted")

    s = stats_snapshot(m)
    assert s["connected_miners"] == 2
    assert s["template_height"] == 50_000
    assert 4.0 < s["template_age_seconds"] < 7.0
    assert s["jobs_in_registry"] == 8
    assert s["blocks"]["accepted"] == 1
    assert s["blocks"]["error"] == 0
    assert s["shares_total"]["accepted"] == 2
    assert s["shares_total"]["stale"] == 1
    assert s["shares_total"]["malformed"] == 0
    assert s["workers"] == {
        "rig04.gpu0": {"accepted": 2, "stale": 0, "malformed": 0},
        "rig04.gpu1": {"accepted": 0, "stale": 1, "malformed": 0},
    }


def test_stats_snapshot_template_age_none_when_no_template():
    m = Metrics()
    s = stats_snapshot(m)
    assert s["template_age_seconds"] is None


# ----------------------------------------------------------- HTTP routes


async def _http_get(host, port, path):
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
async def dashboard_server():
    m = Metrics()
    h = History()
    srv = await serve_http(m, host="127.0.0.1", port=0, history=h)
    port = srv.sockets[0].getsockname()[1]
    try:
        yield m, h, port
    finally:
        srv.close()
        await srv.wait_closed()


async def test_root_serves_html_dashboard(dashboard_server):
    _, _, port = dashboard_server
    status, ctype, body = await _http_get("127.0.0.1", port, "/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert "<title>Pearl solo pool</title>" in body
    # Auto-refresh hook present so the UI updates without a reload.
    assert "setInterval(refreshStats" in body


async def test_api_stats_returns_json(dashboard_server):
    metrics, _, port = dashboard_server
    metrics.connected_miners = 7
    metrics.template_height = 123
    metrics.record_share("rig01", "accepted")
    status, ctype, body = await _http_get("127.0.0.1", port, "/api/stats")
    assert status == 200
    assert ctype == "application/json"
    obj = json.loads(body)
    assert obj["connected_miners"] == 7
    assert obj["template_height"] == 123
    assert obj["workers"]["rig01"]["accepted"] == 1


async def test_api_history_returns_samples(dashboard_server):
    metrics, history, port = dashboard_server
    history.record(metrics)
    metrics.record_share("rig01", "accepted")
    history.record(metrics)

    status, ctype, body = await _http_get("127.0.0.1", port, "/api/history")
    assert status == 200
    assert ctype == "application/json"
    obj = json.loads(body)
    assert len(obj["samples"]) == 2
    # Second sample reflects the share we recorded between snapshots.
    assert obj["samples"][1]["shares_accepted_by_worker"] == {"rig01": 1}


async def test_api_history_empty_when_no_history_passed(dashboard_server):
    """If serve_http is called without history (legacy path), endpoint
    still returns valid empty JSON rather than 500."""
    m = Metrics()
    srv = await serve_http(m, host="127.0.0.1", port=0)  # no history
    port = srv.sockets[0].getsockname()[1]
    try:
        status, _, body = await _http_get("127.0.0.1", port, "/api/history")
        assert status == 200
        assert json.loads(body) == {"samples": []}
    finally:
        srv.close()
        await srv.wait_closed()


async def test_root_still_unknown_paths_404(dashboard_server):
    _, _, port = dashboard_server
    status, _, _ = await _http_get("127.0.0.1", port, "/bogus")
    assert status == 404


async def test_metrics_endpoint_still_works_with_dashboard_wired(dashboard_server):
    """Regression: adding dashboard routes shouldn't break existing /metrics."""
    metrics, _, port = dashboard_server
    metrics.connected_miners = 4
    status, ctype, body = await _http_get("127.0.0.1", port, "/metrics")
    assert status == 200
    assert "version=0.0.4" in ctype
    assert "pearl_stratum_srv_connected_miners 4" in body
