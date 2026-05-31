"""Alerter: detection rules + fired/resolved diff + delivery."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import pytest

from pearl_stratum_srv.alerts import (
    SEV_CRIT,
    SEV_WARN,
    Alert,
    Alerter,
    AlertRule,
    _LogFileDelivery,
    _rule_block_error,
    _rule_malformed_flood,
    _rule_no_miners,
    _rule_rig_idle,
    _rule_template_never,
    _rule_template_stale,
)
from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.dashboard import History, Sample
from pearl_stratum_srv.metrics import Metrics


@dataclass
class _S:
    """Minimal Settings-shape for rule tests."""
    public_pool: bool = True
    alert_template_age_seconds: float = 120.0
    alert_template_never_seconds: float = 300.0
    alert_no_miners_seconds: float = 180.0
    alert_rig_idle_seconds: float = 180.0
    alert_malformed_total_threshold: int = 10


def _empty_history() -> History:
    return History()


# ----------------------------------------------------------- rule: template_stale


def test_template_stale_quiet_when_no_template_yet():
    m = Metrics()
    assert _rule_template_stale(m, _empty_history(), _S()) == []


def test_template_stale_quiet_when_template_is_fresh():
    m = Metrics()
    m.template_minted_at = time.time() - 30.0
    assert _rule_template_stale(m, _empty_history(), _S()) == []


def test_template_stale_fires_when_old():
    m = Metrics()
    m.template_minted_at = time.time() - 300.0
    alerts = _rule_template_stale(m, _empty_history(), _S(alert_template_age_seconds=120.0))
    assert len(alerts) == 1
    assert alerts[0].key == "template_stale"
    assert alerts[0].severity == SEV_CRIT
    assert "pearld" in alerts[0].message


# ----------------------------------------------------------- rule: template_never


def test_template_never_quiet_under_threshold():
    m = Metrics()
    m._alerter_uptime_secs = 30.0
    assert _rule_template_never(m, _empty_history(), _S(alert_template_never_seconds=300.0)) == []


def test_template_never_fires_after_long_uptime_no_template():
    m = Metrics()
    m._alerter_uptime_secs = 500.0
    alerts = _rule_template_never(m, _empty_history(), _S(alert_template_never_seconds=300.0))
    assert len(alerts) == 1
    assert alerts[0].key == "template_never"
    assert alerts[0].severity == SEV_CRIT


def test_template_never_quiet_once_template_arrives():
    m = Metrics()
    m._alerter_uptime_secs = 9999.0
    m.template_minted_at = time.time()
    assert _rule_template_never(m, _empty_history(), _S()) == []


# ----------------------------------------------------------- rule: no_miners


def test_no_miners_quiet_in_solo_mode():
    m = Metrics()
    m.template_minted_at = time.time() - 1000.0
    m.connected_miners = 0
    assert _rule_no_miners(m, _empty_history(), _S(public_pool=False)) == []


def test_no_miners_quiet_when_template_is_brand_new():
    m = Metrics()
    m.template_minted_at = time.time() - 30.0
    m.connected_miners = 0
    assert _rule_no_miners(m, _empty_history(), _S()) == []


def test_no_miners_quiet_when_at_least_one_connected():
    m = Metrics()
    m.template_minted_at = time.time() - 1000.0
    m.connected_miners = 1
    assert _rule_no_miners(m, _empty_history(), _S()) == []


def test_no_miners_fires_when_long_uptime_zero_connected():
    m = Metrics()
    m.template_minted_at = time.time() - 1000.0
    m.connected_miners = 0
    alerts = _rule_no_miners(m, _empty_history(), _S(alert_no_miners_seconds=180.0))
    assert len(alerts) == 1
    assert alerts[0].key == "no_miners"


# ----------------------------------------------------------- rule: block_error


def test_block_error_quiet_when_zero():
    m = Metrics()
    assert _rule_block_error(m, _empty_history(), _S()) == []


def test_block_error_fires_on_any_increment():
    m = Metrics()
    m.record_block("error")
    alerts = _rule_block_error(m, _empty_history(), _S())
    assert len(alerts) == 1
    assert alerts[0].severity == SEV_CRIT


# ----------------------------------------------------------- rule: rig_idle


def test_rig_idle_needs_at_least_two_samples():
    m = Metrics()
    h = History()
    h.record(m)
    assert _rule_rig_idle(m, h, _S()) == []


def test_rig_idle_fires_when_worker_stopped_submitting():
    m = Metrics()
    h = History()
    # Older sample — worker had 5 shares.
    h.samples.append(Sample(ts=time.time() - 500.0, connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={"rigA": 5}))
    # Newer sample — still 5 shares (no growth).
    h.samples.append(Sample(ts=time.time(), connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={"rigA": 5}))
    alerts = _rule_rig_idle(m, h, _S(alert_rig_idle_seconds=180.0))
    assert len(alerts) == 1
    assert alerts[0].key == "rig_idle:rigA"


def test_rig_idle_quiet_when_new_joiner_with_no_baseline():
    """Worker that JUST joined (older snapshot has 0, newer has 0) doesn't
    look IDLE — they just have no shares yet."""
    m = Metrics()
    h = History()
    h.samples.append(Sample(ts=time.time() - 500.0, connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={}))
    h.samples.append(Sample(ts=time.time(), connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={"rigB": 0}))
    assert _rule_rig_idle(m, h, _S()) == []


def test_rig_idle_quiet_when_worker_active():
    m = Metrics()
    h = History()
    h.samples.append(Sample(ts=time.time() - 500.0, connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={"rigA": 5}))
    h.samples.append(Sample(ts=time.time(), connected_miners=1, template_height=1,
                            template_age=0.0, blocks_accepted=0,
                            shares_accepted_by_worker={"rigA": 15}))
    assert _rule_rig_idle(m, h, _S()) == []


# ----------------------------------------------------------- rule: malformed_flood


def test_malformed_flood_quiet_below_threshold():
    m = Metrics()
    for _ in range(5):
        m.record_share("rigX", "malformed")
    assert _rule_malformed_flood(m, _empty_history(), _S(alert_malformed_total_threshold=10)) == []


def test_malformed_flood_fires_above_threshold():
    m = Metrics()
    for _ in range(20):
        m.record_share("rigX", "malformed")
    alerts = _rule_malformed_flood(m, _empty_history(), _S(alert_malformed_total_threshold=10))
    assert len(alerts) == 1
    assert alerts[0].severity == SEV_WARN


# ============================================================ Alerter tick


def _make_settings(**over):
    return Settings(
        rpc_url="http://stub", rpc_user="x", rpc_password="y",
        mining_address="prl1stub",
        public_pool=True,
        alert_log_path="",  # disable file delivery in tests
        alert_webhook_url="",
        alert_template_age_seconds=60.0,
        alert_no_miners_seconds=120.0,
        **over,
    )


async def test_alerter_tick_fires_then_resolves_cleanly():
    settings = _make_settings()
    m = Metrics()
    m.connected_miners = 1  # don't also fire no_miners
    h = History()
    alerter = Alerter(settings, m, h, tick_secs=0.01)

    # Initially nothing fires.
    await alerter.tick()
    assert alerter.active_alerts == {}

    # Trigger template_stale.
    m.template_minted_at = time.time() - 999.0
    await alerter.tick()
    assert list(alerter.active_alerts.keys()) == ["template_stale"]

    # Resolve it (fresh template).
    m.template_minted_at = time.time()
    await alerter.tick()
    assert alerter.active_alerts == {}


async def test_alerter_dedupes_ongoing_alerts():
    """When the same alert is still active across ticks, we don't re-fire it."""
    settings = _make_settings()
    m = Metrics()
    m.connected_miners = 1
    h = History()
    alerter = Alerter(settings, m, h)

    delivered = []

    class _StubDelivery:
        async def send(self, action, alert):
            delivered.append((action, alert.key))

    alerter._deliveries = [_StubDelivery()]

    m.template_minted_at = time.time() - 999.0
    await alerter.tick()  # fires
    await alerter.tick()  # no change → no new delivery
    await alerter.tick()
    assert delivered == [("fired", "template_stale")]


async def test_alerter_delivers_resolved_event_when_condition_clears():
    settings = _make_settings()
    m = Metrics()
    m.connected_miners = 1
    h = History()
    alerter = Alerter(settings, m, h)
    delivered = []

    class _StubDelivery:
        async def send(self, action, alert):
            delivered.append((action, alert.key))

    alerter._deliveries = [_StubDelivery()]

    m.template_minted_at = time.time() - 999.0
    await alerter.tick()
    m.template_minted_at = time.time()  # fresh
    await alerter.tick()
    assert delivered == [("fired", "template_stale"), ("resolved", "template_stale")]


async def test_alerter_as_json_shape():
    settings = _make_settings()
    m = Metrics()
    m.connected_miners = 1
    h = History()
    alerter = Alerter(settings, m, h)
    m.template_minted_at = time.time() - 999.0
    await alerter.tick()
    j = alerter.as_json()
    assert j["count"] == 1
    assert j["active"][0]["key"] == "template_stale"
    assert "age_seconds" in j["active"][0]


# ============================================================ log file delivery


async def test_log_file_delivery_writes_json_line(tmp_path):
    log_path = tmp_path / "alerts.log"
    delivery = _LogFileDelivery(log_path)
    alert = Alert(key="template_stale", severity=SEV_CRIT, message="boom",
                  fired_at=1779338000.0)
    await delivery.send("fired", alert)
    contents = log_path.read_text().splitlines()
    assert len(contents) == 1
    obj = json.loads(contents[0])
    assert obj["action"] == "fired"
    assert obj["key"] == "template_stale"
    assert obj["severity"] == SEV_CRIT
    assert obj["message"] == "boom"


async def test_log_file_delivery_appends_not_truncates(tmp_path):
    log_path = tmp_path / "alerts.log"
    delivery = _LogFileDelivery(log_path)
    for i in range(3):
        await delivery.send("fired", Alert(key=f"k{i}", severity=SEV_WARN, message="m"))
    assert len(log_path.read_text().splitlines()) == 3
