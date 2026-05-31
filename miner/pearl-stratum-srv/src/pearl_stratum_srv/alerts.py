"""Pool alerter — detects bad patterns + delivers notifications.

Runs as a background asyncio task ticked every `tick_secs`. Each tick:
  1. Evaluates each `AlertRule.check(metrics, history)` against the live state
  2. For rules that fired, compares against last-fired state to decide whether
     this is a NEW alert (state changed) or an ongoing one (suppress dupes)
  3. Delivers new alerts via:
       - Python logging (ERROR level, captured by journald)
       - Dedicated alerts logfile (one line per alert, JSON-encoded)
       - Optional Discord/Slack-style webhook (HTTP POST {"content": "..."})
  4. Updates an in-memory `Alerter.active_alerts` dict that the HTTP layer
     can serve via /api/alerts

Alerts auto-clear when their condition reverses. Cleared alerts are also
logged + webhooked ("RESOLVED: ...") so on-call sees recovery.

Detection rules (all configurable thresholds in Settings):
  - template_stale     pool template age > threshold (pearld stopped delivering)
  - no_miners          connected_miners == 0 (public-pool mode only)
  - block_error        blocks_total{outcome=error} incremented since last tick
  - rig_idle           per-worker accepted-share rate == 0 in last N minutes
  - malformed_flood    shares_total{outcome=malformed} climbing
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pearl_stratum_srv.dashboard import History
    from pearl_stratum_srv.metrics import Metrics

_LOGGER = logging.getLogger(__name__)

# Severity affects icon + how loudly we yell.
SEV_INFO = "info"
SEV_WARN = "warn"
SEV_CRIT = "crit"


@dataclass
class Alert:
    key: str               # stable identifier (e.g. "template_stale", "rig_idle:tprl1...gpu0")
    severity: str          # SEV_INFO / SEV_WARN / SEV_CRIT
    message: str           # human-readable
    fired_at: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "severity": self.severity,
            "message": self.message,
            "fired_at": self.fired_at,
            "age_seconds": time.time() - self.fired_at,
        }


@dataclass
class AlertRule:
    """A predicate over (metrics, history) → list[Alert]. Returns the alerts
    that should be ACTIVE right now. The alerter diffs against last tick to
    figure out which ones are new vs. resolved."""
    name: str
    check: Callable  # (metrics, history, settings) -> list[Alert]


# ===================================================================== rules


def _rule_template_stale(metrics, history, settings) -> list[Alert]:
    if metrics.template_minted_at == 0.0:
        return []  # no template ever — handled by template_never below
    age = time.time() - metrics.template_minted_at
    if age <= settings.alert_template_age_seconds:
        return []
    return [Alert(
        key="template_stale",
        severity=SEV_CRIT,
        message=f"pool template age {age:.0f}s > {settings.alert_template_age_seconds}s — pearld stopped delivering work",
    )]


def _rule_template_never(metrics, history, settings) -> list[Alert]:
    """Fires when the pool has been up for `alert_template_never_seconds` but
    has never received a template from pearld. Catches:
      - pearld is wedged in IBD and refusing getblocktemplate
      - RPC creds wrong (silent auth failures)
      - pearld unreachable / wrong port
    Crit-level because no template = pool can't dispatch jobs to miners."""
    if metrics.template_minted_at != 0.0:
        return []
    # Use the alerter's start_ts (held in settings reflection); we track uptime
    # via Alerter._started_at — see Alerter.__init__. The settings carries the
    # threshold knob; the alerter passes itself's uptime by storing it on
    # `metrics` as `_alerter_uptime_secs` (set by the alerter every tick).
    uptime = getattr(metrics, "_alerter_uptime_secs", 0.0)
    if uptime < settings.alert_template_never_seconds:
        return []
    return [Alert(
        key="template_never",
        severity=SEV_CRIT,
        message=f"pool up {uptime:.0f}s but pearld hasn't returned a template once — IBD, RPC creds, or wedged node",
    )]


def _rule_no_miners(metrics, history, settings) -> list[Alert]:
    if not settings.public_pool:
        return []  # solo deploys may legitimately have 0 connected
    if metrics.connected_miners > 0:
        return []
    # Use template_minted_at as a proxy for "pool has been up a while"
    if metrics.template_minted_at == 0.0:
        return []
    uptime = time.time() - metrics.template_minted_at
    if uptime < settings.alert_no_miners_seconds:
        return []
    return [Alert(
        key="no_miners",
        severity=SEV_WARN,
        message="no miners connected — flight sheet wrong, or firewall, or fleet down",
    )]


def _rule_block_error(metrics, history, settings) -> list[Alert]:
    errors = metrics.blocks_total.get("error", 0)
    if errors <= 0:
        return []
    return [Alert(
        key="block_error",
        severity=SEV_CRIT,
        message=f"{errors} block submission(s) errored — possibly lost block, see pool logs immediately",
    )]


def _rule_rig_idle(metrics, history, settings) -> list[Alert]:
    """Per-worker share rate dropped to zero for N minutes.

    Compares the oldest vs. newest sample in the history ring. If a worker
    had >0 shares in the older snapshot and the same count in the newest,
    they're idle. Requires at least `alert_rig_idle_seconds` of history
    (smaller windows don't have enough data to be confident).
    """
    samples = list(history.samples)
    if len(samples) < 2:
        return []
    older = samples[0]
    newest = samples[-1]
    elapsed = newest.ts - older.ts
    window = settings.alert_rig_idle_seconds
    if elapsed < window:
        return []  # not enough lookback to claim someone's idle
    alerts = []
    for worker, newest_count in newest.shares_accepted_by_worker.items():
        older_count = older.shares_accepted_by_worker.get(worker, 0)
        if older_count > 0 and newest_count - older_count <= 0:
            alerts.append(Alert(
                key=f"rig_idle:{worker}",
                severity=SEV_WARN,
                message=f"worker {worker} submitted 0 shares in last {elapsed:.0f}s — likely IDLE",
            ))
    return alerts


def _rule_malformed_flood(metrics, history, settings) -> list[Alert]:
    """Total malformed-share count rising. Per-IP auto-ban already kicks in at
    50/5min; this rule fires earlier as an operator heads-up at 10+ total."""
    total = sum(c for (_w, o), c in metrics.shares_total.items() if o == "malformed")
    if total < settings.alert_malformed_total_threshold:
        return []
    return [Alert(
        key="malformed_flood",
        severity=SEV_WARN,
        message=f"{total} malformed share(s) seen — protocol drift, bad miner build, or attack",
    )]


DEFAULT_RULES = [
    AlertRule("template_stale", _rule_template_stale),
    AlertRule("template_never", _rule_template_never),
    AlertRule("no_miners", _rule_no_miners),
    AlertRule("block_error", _rule_block_error),
    AlertRule("rig_idle", _rule_rig_idle),
    AlertRule("malformed_flood", _rule_malformed_flood),
]


# ================================================================== delivery


class _LogFileDelivery:
    """One-line-per-alert JSON appended to a logfile. Operator can tail -f."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def send(self, action: str, alert: Alert) -> None:
        line = json.dumps({"action": action, **alert.to_json()})
        await asyncio.to_thread(self._append, line + "\n")

    def _append(self, line: str) -> None:
        with self.path.open("a") as f:
            f.write(line)


class _WebhookDelivery:
    """POST a Discord/Slack-style {"content": "..."} payload."""

    def __init__(self, url: str):
        self.url = url

    async def send(self, action: str, alert: Alert) -> None:
        import aiohttp
        emoji = {SEV_INFO: "ℹ️", SEV_WARN: "⚠️", SEV_CRIT: "🚨"}.get(alert.severity, "•")
        action_word = "FIRED" if action == "fired" else "RESOLVED"
        content = f"{emoji} **Pearl pool alert {action_word}**: `{alert.key}`\n{alert.message}"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(self.url, json={"content": content}, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status >= 400:
                        _LOGGER.warning("webhook delivery failed: HTTP %s", r.status)
        except Exception as e:
            _LOGGER.warning("webhook delivery error: %s", e)


# ================================================================== alerter


class Alerter:
    """Owns the active-alert state + delivery. Tick from a background task."""

    def __init__(self, settings, metrics: "Metrics", history: "History",
                 rules=None, tick_secs: float = 15.0):
        self.settings = settings
        self.metrics = metrics
        self.history = history
        self.rules = rules if rules is not None else DEFAULT_RULES
        self.tick_secs = tick_secs
        self.active_alerts: dict[str, Alert] = {}
        self._started_at = time.time()
        # Built lazily so tests can swap in their own
        self._deliveries: list = []

    def configure_deliveries(self) -> None:
        if self.settings.alert_log_path:
            self._deliveries.append(_LogFileDelivery(self.settings.alert_log_path))
        if self.settings.alert_webhook_url:
            self._deliveries.append(_WebhookDelivery(self.settings.alert_webhook_url))

    async def run_forever(self) -> None:
        """Background loop: tick alerter every `tick_secs`."""
        # First evaluation happens immediately, then on the interval.
        try:
            while True:
                try:
                    await self.tick()
                except Exception:
                    _LOGGER.exception("alerter tick failed")
                await asyncio.sleep(self.tick_secs)
        except asyncio.CancelledError:
            raise

    async def tick(self) -> None:
        """One evaluation pass: compute active alerts, diff vs. last tick,
        deliver fired/resolved events."""
        # Surface uptime to rules that need it (template_never).
        self.metrics._alerter_uptime_secs = time.time() - self._started_at
        current: dict[str, Alert] = {}
        for rule in self.rules:
            for alert in rule.check(self.metrics, self.history, self.settings):
                current[alert.key] = alert

        # Newly fired = in current, not in previous
        for key, alert in current.items():
            if key not in self.active_alerts:
                _LOGGER.error("ALERT FIRED: %s — %s", alert.key, alert.message)
                await self._deliver("fired", alert)

        # Resolved = in previous, not in current
        for key, prev_alert in list(self.active_alerts.items()):
            if key not in current:
                _LOGGER.warning("ALERT RESOLVED: %s — %s",
                                prev_alert.key, prev_alert.message)
                await self._deliver("resolved", prev_alert)

        self.active_alerts = current

    async def _deliver(self, action: str, alert: Alert) -> None:
        for d in self._deliveries:
            try:
                await d.send(action, alert)
            except Exception:
                _LOGGER.exception("delivery channel raised")

    def as_json(self) -> dict:
        return {
            "active": [a.to_json() for a in self.active_alerts.values()],
            "count": len(self.active_alerts),
        }
