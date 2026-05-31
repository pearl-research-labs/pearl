"""Built-in dashboard served on the same HTTP port as /metrics.

Routes added (on top of /metrics + /health from metrics.py):
  GET  /              — single-page HTML dashboard (vanilla JS, no deps)
  GET  /api/stats     — JSON snapshot: connected miners, template state, share counters
  GET  /api/history   — JSON: last 60 samples (1/min) of accepted-shares rate per worker

History is kept in memory in a ring buffer; ticked every 60s by a background task
in the PoolServer. No persistence — restart clears history.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pearl_stratum_srv.metrics import Metrics

HISTORY_LEN = 60  # 60 samples × 60s = last hour


@dataclass
class Sample:
    ts: float
    connected_miners: int
    template_height: int
    template_age: float
    blocks_accepted: int
    # Per-worker accepted-share totals at this sample's instant.
    # We diff successive samples to get rate; UI does that math.
    shares_accepted_by_worker: dict[str, int] = field(default_factory=dict)


class History:
    """Bounded ring buffer of Metrics snapshots, sampled by a background task."""

    def __init__(self, capacity: int = HISTORY_LEN):
        self.samples: deque[Sample] = deque(maxlen=capacity)

    def record(self, metrics: "Metrics") -> None:
        now = time.time()
        template_age = (now - metrics.template_minted_at) if metrics.template_minted_at else -1.0
        # Sum per-worker accepted shares into a dict keyed by worker name.
        per_worker: dict[str, int] = {}
        for (worker, outcome), count in metrics.shares_total.items():
            if outcome == "accepted":
                per_worker[worker] = per_worker.get(worker, 0) + count
        self.samples.append(
            Sample(
                ts=now,
                connected_miners=metrics.connected_miners,
                template_height=metrics.template_height,
                template_age=template_age,
                blocks_accepted=metrics.blocks_total.get("accepted", 0),
                shares_accepted_by_worker=per_worker,
            )
        )

    def as_json(self) -> dict:
        return {
            "samples": [
                {
                    "ts": s.ts,
                    "connected_miners": s.connected_miners,
                    "template_height": s.template_height,
                    "template_age": s.template_age,
                    "blocks_accepted": s.blocks_accepted,
                    "shares_accepted_by_worker": s.shares_accepted_by_worker,
                }
                for s in self.samples
            ]
        }


def stats_snapshot(metrics: "Metrics") -> dict:
    """Current point-in-time view."""
    now = time.time()
    template_age = (now - metrics.template_minted_at) if metrics.template_minted_at else None

    # Per-worker totals + outcomes
    workers: dict[str, dict[str, int]] = {}
    for (worker, outcome), count in metrics.shares_total.items():
        workers.setdefault(worker, {"accepted": 0, "stale": 0, "malformed": 0})
        workers[worker][outcome] = count

    return {
        "now": now,
        "connected_miners": metrics.connected_miners,
        "template_height": metrics.template_height,
        "template_age_seconds": template_age,
        "jobs_in_registry": metrics.jobs_in_registry,
        "blocks": {
            "accepted": metrics.blocks_total.get("accepted", 0),
            "error": metrics.blocks_total.get("error", 0),
        },
        "shares_total": {
            "accepted": sum(c for (_, o), c in metrics.shares_total.items() if o == "accepted"),
            "stale": sum(c for (_, o), c in metrics.shares_total.items() if o == "stale"),
            "malformed": sum(c for (_, o), c in metrics.shares_total.items() if o == "malformed"),
        },
        "workers": workers,
    }


# ----------------------------------------------------------- HTML page

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pearl solo pool</title>
<style>
* { box-sizing: border-box; }
body {
  font: 14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  background: #0b0e14; color: #cdd6f4; margin: 0; padding: 24px;
}
h1 { font-size: 18px; margin: 0 0 8px; color: #89b4fa; letter-spacing: 0.5px; }
.sub { color: #6c7086; font-size: 12px; margin-bottom: 24px; }
.grid {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 24px;
}
.card {
  background: #181c25; border: 1px solid #313244; border-radius: 6px;
  padding: 14px 16px;
}
.card .label { color: #6c7086; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 28px; font-weight: 600; margin-top: 4px; color: #cdd6f4; }
.card .value.ok { color: #a6e3a1; }
.card .value.warn { color: #f9e2af; }
.card .value.crit { color: #f38ba8; }
.card .sub { font-size: 11px; color: #6c7086; margin-top: 2px; }
section { margin-bottom: 24px; }
section h2 {
  font-size: 13px; font-weight: 600; color: #89b4fa; text-transform: uppercase;
  letter-spacing: 0.5px; margin: 0 0 12px;
}
table {
  width: 100%; border-collapse: collapse; font-size: 13px;
  background: #181c25; border: 1px solid #313244; border-radius: 6px; overflow: hidden;
}
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #313244; }
th { background: #1e2230; color: #89b4fa; font-weight: 500; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.5px; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; color: #cdd6f4; }
td.worker { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #94e2d5; }
.empty { padding: 24px; text-align: center; color: #6c7086; }
.chart-wrap {
  background: #181c25; border: 1px solid #313244; border-radius: 6px; padding: 12px;
}
canvas { width: 100%; height: 200px; display: block; }
.footer { color: #6c7086; font-size: 11px; margin-top: 32px; text-align: center; }
.footer a { color: #89b4fa; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
.alert {
  padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
  font-size: 13px; display: flex; align-items: center; gap: 10px;
  border: 1px solid;
}
.alert.crit { background: #2a1418; border-color: #f38ba8; color: #f38ba8; }
.alert.warn { background: #2a2515; border-color: #f9e2af; color: #f9e2af; }
.alert.info { background: #15252a; border-color: #89b4fa; color: #89b4fa; }
.alert .key { font-family: ui-monospace, Menlo, monospace; font-size: 11px;
              padding: 2px 6px; border-radius: 3px;
              background: rgba(0,0,0,0.25); }
.alert .age { margin-left: auto; opacity: 0.7; font-size: 11px; }
</style>
</head>
<body>

<h1>Pearl solo pool</h1>
<div class="sub" id="header-sub">Loading…</div>

<div id="alerts"></div>
<div class="grid" id="kpis"></div>

<section>
  <h2>Workers</h2>
  <div id="workers"></div>
</section>

<section>
  <h2>Shares per minute (last hour)</h2>
  <div class="chart-wrap"><canvas id="chart" width="800" height="200"></canvas></div>
</section>

<div class="footer">
  pearl-stratum-srv ·
  <a href="/metrics">/metrics</a> ·
  <a href="/health">/health</a> ·
  <a href="/api/stats">/api/stats</a> ·
  <a href="/api/history">/api/history</a>
</div>

<script>
const fmt = {
  int: n => (n == null ? '—' : n.toLocaleString()),
  age: s => s == null || s < 0 ? '—' : s < 60 ? s.toFixed(1)+'s' : (s/60).toFixed(1)+'m',
  rate: n => n == null ? '—' : n.toFixed(2) + '/min',
  worker: w => w === 'anonymous' ? '<i>(anonymous)</i>' : w,
};

function ageClass(s) {
  if (s == null || s < 0) return 'crit';
  if (s < 60) return 'ok';
  if (s < 120) return 'warn';
  return 'crit';
}

function minersClass(n) {
  if (n === 0) return 'crit';
  if (n < 30) return 'warn';
  return 'ok';
}

function card(label, value, cls='', sub='') {
  return `<div class="card">
    <div class="label">${label}</div>
    <div class="value ${cls}">${value}</div>
    ${sub ? `<div class="sub">${sub}</div>` : ''}
  </div>`;
}

async function refreshAlerts() {
  try {
    const r = await fetch('/api/alerts');
    const a = await r.json();
    const el = document.getElementById('alerts');
    if (!a.active || a.active.length === 0) { el.innerHTML = ''; return; }
    const icon = sev => ({crit:'🚨', warn:'⚠️', info:'ℹ️'}[sev] || '•');
    el.innerHTML = a.active.map(x =>
      `<div class="alert ${x.severity}">${icon(x.severity)}
        <span class="key">${x.key}</span>
        <span>${x.message}</span>
        <span class="age">fired ${fmt.age(x.age_seconds)} ago</span>
       </div>`
    ).join('');
  } catch (e) { /* leave previous render in place */ }
}

async function refreshStats() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();

    document.getElementById('header-sub').textContent =
      'Network template height ' + fmt.int(s.template_height) +
      ' · ' + s.jobs_in_registry + ' jobs cached · auto-refresh 5s';

    document.getElementById('kpis').innerHTML =
      card('Connected miners', fmt.int(s.connected_miners), minersClass(s.connected_miners)) +
      card('Template age', fmt.age(s.template_age_seconds), ageClass(s.template_age_seconds)) +
      card('Template height', fmt.int(s.template_height)) +
      card('Blocks found', fmt.int(s.blocks.accepted), s.blocks.accepted > 0 ? 'ok' : '',
           s.blocks.error > 0 ? `<span style="color:#f38ba8">${s.blocks.error} errors</span>` : '') +
      card('Shares accepted', fmt.int(s.shares_total.accepted)) +
      card('Shares stale', fmt.int(s.shares_total.stale),
           s.shares_total.stale > s.shares_total.accepted * 0.1 ? 'warn' : '') +
      card('Shares malformed', fmt.int(s.shares_total.malformed),
           s.shares_total.malformed > 0 ? 'crit' : 'ok');

    const workerNames = Object.keys(s.workers).sort();
    if (workerNames.length === 0) {
      document.getElementById('workers').innerHTML =
        '<div class="card empty">No workers connected yet. Point a rig at this pool to see it here.</div>';
    } else {
      let html = '<table><thead><tr><th>Worker</th><th>Accepted</th><th>Stale</th><th>Malformed</th></tr></thead><tbody>';
      for (const w of workerNames) {
        const ws = s.workers[w];
        html += `<tr><td class="worker">${fmt.worker(w)}</td>
                 <td class="num">${fmt.int(ws.accepted)}</td>
                 <td class="num" style="${ws.stale > 10 ? 'color:#f9e2af' : ''}">${fmt.int(ws.stale)}</td>
                 <td class="num" style="${ws.malformed > 0 ? 'color:#f38ba8' : ''}">${fmt.int(ws.malformed)}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('workers').innerHTML = html;
    }
  } catch (e) {
    document.getElementById('header-sub').textContent = 'Connection failed: ' + e.message;
  }
}

async function refreshChart() {
  try {
    const r = await fetch('/api/history');
    const h = await r.json();
    drawChart(h.samples);
  } catch (e) { /* ignore */ }
}

function drawChart(samples) {
  const canvas = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, hh = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = hh * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, hh);

  if (samples.length < 2) {
    ctx.fillStyle = '#6c7086'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Collecting samples — first datapoint at 60s, then once per minute…', w/2, hh/2);
    return;
  }

  // Build per-minute share counts (diffs between consecutive samples), summed across workers.
  const points = [];
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i-1], cur = samples[i];
    let delta = 0;
    for (const worker in cur.shares_accepted_by_worker) {
      const a = cur.shares_accepted_by_worker[worker] || 0;
      const b = prev.shares_accepted_by_worker[worker] || 0;
      delta += Math.max(0, a - b);
    }
    points.push({ts: cur.ts, count: delta});
  }

  const maxY = Math.max(1, ...points.map(p => p.count));
  const padL = 36, padR = 12, padT = 12, padB = 24;
  const cw = w - padL - padR, ch = hh - padT - padB;

  // Axes
  ctx.strokeStyle = '#313244'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+ch); ctx.lineTo(padL+cw, padT+ch);
  ctx.stroke();

  // Y gridlines + labels
  ctx.fillStyle = '#6c7086'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = padT + ch - (ch * i / 4);
    const v = (maxY * i / 4);
    ctx.fillText(v.toFixed(v < 10 ? 1 : 0), padL - 4, y + 3);
    ctx.strokeStyle = '#1e2230';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL+cw, y); ctx.stroke();
  }

  // X labels (first, mid, last)
  const fmtT = ts => new Date(ts*1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  ctx.textAlign = 'left';
  ctx.fillText(fmtT(points[0].ts), padL, hh - 6);
  ctx.textAlign = 'center';
  ctx.fillText(fmtT(points[Math.floor(points.length/2)].ts), padL + cw/2, hh - 6);
  ctx.textAlign = 'right';
  ctx.fillText(fmtT(points[points.length-1].ts), padL + cw, hh - 6);

  // Line
  ctx.strokeStyle = '#89b4fa'; ctx.lineWidth = 2;
  ctx.fillStyle = 'rgba(137,180,250,0.15)';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const x = padL + (cw * i / (points.length-1));
    const y = padT + ch - (ch * points[i].count / maxY);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Fill under
  ctx.lineTo(padL + cw, padT + ch);
  ctx.lineTo(padL, padT + ch);
  ctx.closePath();
  ctx.fill();
}

refreshStats(); refreshChart(); refreshAlerts();
setInterval(refreshStats, 5000);
setInterval(refreshAlerts, 5000);
setInterval(refreshChart, 60000);
</script>
</body>
</html>
"""
