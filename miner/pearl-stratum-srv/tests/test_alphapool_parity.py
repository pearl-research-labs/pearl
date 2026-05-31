"""Parity check: our PoolServer must emit frames that match alphapool byte-for-byte
in shape and semantics so alpha-miner v1.5 can't tell us apart.

Drives a real subscribe handshake against our server using a fake template whose
field values match the captured alphapool sample. Reads `fixtures/alphapool_capture_2026_05_18.json`
and diffs frame-by-frame:

  - subscribe response: shape, extranonce1=="" + extranonce2_size==0, subscription
    method names match.
  - pearl.set_mining_params: EVERY scalar field of the dict must match the capture
    (m, n, k, rank, rows_pattern, cols_pattern, mma_type). This is the field most
    likely to drift — if alphapool ever upgrades to rank=256 or pattern stride
    [0,16,32,48], we'll see CI fail here before deploying.
  - mining.notify: arity, types, hex-format conventions of each positional param.

This test runs offline. To refresh the fixture from a live pool, run
`tools/capture_alphapool.py` from a Linux deploy box; see its docstring.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

# pearl_mining shim is installed in conftest.py.
from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.server import PoolServer

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_BASELINE_PATH = _FIXTURE_DIR / "alphapool_capture_2026_05_18.json"


@pytest.fixture(scope="module")
def capture() -> dict:
    return json.loads(_BASELINE_PATH.read_text())


# ----------------------------------------------------------- support


@dataclass
class _Header:
    timestamp: int
    target_bits: int
    previous_block_hash: bytes

    def serialize_without_proof_commitment(self) -> bytes:
        return b"\xfe" * 76  # arbitrary; we don't inspect bytes in this test


@dataclass
class _Template:
    height: int
    header: _Header


class _Node:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


class _WorkCache:
    async def update_template(self, t):
        pass


class _Submission:
    async def submit_plain_proof(self, p, t):
        return {"status": "rejected: dont-care"}


def _build_template_from_capture(capture: dict) -> _Template:
    """Reconstruct a template whose stratum-visible fields match the capture."""
    notify_params = capture["notify_sample"]["params"]
    job_id_hex = notify_params[0]
    height_hex = job_id_hex.split("-", 1)[0]
    height = int(height_hex, 16)
    prev_hash = bytes.fromhex(notify_params[1])
    ntime = int(notify_params[4], 16)
    nbits = int(notify_params[5], 16)
    return _Template(height=height, header=_Header(ntime, nbits, prev_hash))


@pytest.fixture
async def server_with_capture_template(capture):
    template = _build_template_from_capture(capture)
    settings = Settings(
        rpc_url="http://stub",
        rpc_user="x",
        rpc_password="y",
        mining_address="prl1stub",
        listen_host="127.0.0.1",
        listen_port=0,
    )
    server = PoolServer(
        settings=settings,
        node=_Node(),
        work_cache=_WorkCache(),
        submission=_Submission(),
    )
    port = await server.start_listener(port=0)
    await server.ingest_template(template)
    try:
        yield server, port, template
    finally:
        await server.stop_listener()


async def _drive_subscribe(port: int) -> list[dict]:
    """Connect, subscribe, collect the 4 expected frames (reply + 3 pushes)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(b'{"id":47,"method":"mining.subscribe","params":["alpha-miner/0.1"]}\n')
        await writer.drain()
        frames = []
        for _ in range(4):
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            frames.append(json.loads(line))
        return frames
    finally:
        writer.close()
        await writer.wait_closed()


# ------------------------------------------------------------- parity tests


async def test_subscribe_response_matches_capture_shape(server_with_capture_template, capture):
    _, port, _ = server_with_capture_template
    frames = await _drive_subscribe(port)

    reply = frames[0]
    template = capture["subscribe_response_template"]
    assert reply["jsonrpc"] == "2.0"
    assert reply["error"] is None
    # result = [subscriptions, extranonce1, extranonce2_size]
    assert isinstance(reply["result"], list) and len(reply["result"]) == 3
    assert reply["result"][1] == template["result"][1]  # extranonce1 == ""
    assert reply["result"][2] == template["result"][2]  # extranonce2_size == 0
    # Subscription method names match
    expected_methods = {row[0] for row in template["result"][0]}
    got_methods = {row[0] for row in reply["result"][0]}
    assert got_methods == expected_methods


async def test_set_mining_params_payload_matches_capture_exactly(
    server_with_capture_template, capture
):
    """The most load-bearing parity check. Pool consensus on these scalars is
    mainnet-wide; drift here means miner rejects our pool."""
    _, port, _ = server_with_capture_template
    frames = await _drive_subscribe(port)

    pushed = next(f for f in frames if f.get("method") == "pearl.set_mining_params")
    captured = capture["set_mining_params_push"]

    # Top-level structure
    assert pushed["method"] == captured["method"]
    assert "jsonrpc" not in pushed  # alphapool notifications omit jsonrpc
    assert pushed["id"] is None
    # Payload is a single-element list of a dict (alphapool's convention)
    assert isinstance(pushed["params"], list) and len(pushed["params"]) == 1
    got = pushed["params"][0]
    want = captured["params"][0]

    # Every field must match — this is the byte-for-byte parity gate
    assert got == want, f"set_mining_params drift!\n  got:    {got}\n  wanted: {want}"


async def test_mining_notify_field_arity_types_and_formats(
    server_with_capture_template, capture
):
    _, port, template = server_with_capture_template
    frames = await _drive_subscribe(port)
    notify = next(f for f in frames if f.get("method") == "mining.notify")
    captured = capture["notify_sample"]

    assert notify["method"] == captured["method"]
    assert "jsonrpc" not in notify
    assert notify["id"] is None

    p_got = notify["params"]
    p_want = captured["params"]

    # Arity
    assert len(p_got) == len(p_want), f"notify arity {len(p_got)} != {len(p_want)}"

    # Field 0: job_id format HHHHHHHH-SSSS (8 hex + dash + 4 hex)
    assert isinstance(p_got[0], str)
    parts = p_got[0].split("-")
    assert len(parts) == 2
    assert len(parts[0]) == 8 and len(parts[1]) == 4
    int(parts[0], 16)  # parses as hex
    int(parts[1], 16)
    # Same height-prefix as the template we ingested
    assert p_got[0].startswith(f"{template.height:08x}-")

    # Field 1: prevhash, 64 hex chars
    assert isinstance(p_got[1], str) and len(p_got[1]) == 64
    bytes.fromhex(p_got[1])

    # Field 2: incomplete_header_bytes, hex string
    assert isinstance(p_got[2], str)
    bytes.fromhex(p_got[2])

    # Field 3: opaque int (alphapool uses a seq counter; we send 0; the alpha
    # miner doesn't validate it, so any int is fine)
    assert isinstance(p_got[3], int)

    # Field 4: ntime, 8 hex chars (32-bit big-endian unix epoch in hex)
    assert isinstance(p_got[4], str) and len(p_got[4]) == 8
    int(p_got[4], 16)

    # Field 5: nbits, 8 hex chars
    assert isinstance(p_got[5], str) and len(p_got[5]) == 8
    int(p_got[5], 16)
    # And it must match the template's nbits (alphapool capture: 0x1a0ffff0)
    assert int(p_got[5], 16) == template.header.target_bits

    # Field 6: clean_jobs bool
    assert isinstance(p_got[6], bool)


async def test_no_jsonrpc_field_on_pushes_and_present_on_replies(
    server_with_capture_template,
):
    """alphapool convention: server notifications omit `jsonrpc`, replies include it.
    Drift here breaks JSON-RPC strict clients."""
    _, port, _ = server_with_capture_template
    frames = await _drive_subscribe(port)

    reply = frames[0]
    assert reply.get("jsonrpc") == "2.0"

    for f in frames[1:]:
        assert "jsonrpc" not in f, f"push frame leaked jsonrpc field: {f}"
        assert "method" in f
        assert f["id"] is None


async def test_extranonce_is_empty_string_size_zero(server_with_capture_template):
    """Pearl/v1 has no client-side extranonce rolling. Server must return
    extranonce1="" and extranonce2_size=0; any other value confuses miners
    that try to roll a nonce locally."""
    _, port, _ = server_with_capture_template
    frames = await _drive_subscribe(port)
    reply = frames[0]
    extranonce1, extranonce2_size = reply["result"][1], reply["result"][2]
    assert extranonce1 == ""
    assert extranonce2_size == 0


# ============================================================ drift detection
#
# When the operator runs `tools/capture_alphapool.py --out fixtures/alphapool_capture_YYYY_MM_DD.json`
# to pull a fresh capture from the live pool, the tests below diff that
# capture against our canonical baseline and FAIL on any load-bearing drift.
#
# Workflow:
#   1. python tools/capture_alphapool.py --out tests/fixtures/alphapool_capture_$(date +%Y_%m_%d).json
#   2. pytest tests/test_alphapool_parity.py::test_no_drift_in_latest_capture
#   3a. green = alphapool wire format unchanged; ship our pool.
#   3b. red = drift detected. Investigate: maybe alphapool upgraded rank to 256,
#       or added a new field. Decide whether to update our server, then rotate
#       the baseline (rename the new capture to alphapool_capture_2026_05_18.json
#       or update _BASELINE_PATH).


def _list_fresh_captures() -> list[Path]:
    """All capture fixtures except the baseline, newest-first."""
    captures = sorted(
        (p for p in _FIXTURE_DIR.glob("alphapool_capture_*.json") if p != _BASELINE_PATH),
        reverse=True,
    )
    return captures


# Fields whose drift would BREAK our pool's wire-compat with alpha-miner.
# These come from the captured `set_mining_params.params[0]` dict.
_LOAD_BEARING_PARAMS_FIELDS = ("m", "n", "k", "rank", "mma_type", "rows_pattern", "cols_pattern")


def _diff_set_mining_params(baseline: dict, fresh: dict) -> list[str]:
    """Return human-readable list of drifted fields, or [] if identical."""
    drifts: list[str] = []
    b_params = baseline["set_mining_params_push"]["params"][0]
    f_params = fresh["set_mining_params_push"]["params"][0]
    for field in _LOAD_BEARING_PARAMS_FIELDS:
        if b_params.get(field) != f_params.get(field):
            drifts.append(f"  {field}: baseline={b_params.get(field)!r} fresh={f_params.get(field)!r}")
    # Any wholly new field is also drift — alphapool added something we don't know about.
    new_fields = set(f_params) - set(b_params)
    for nf in sorted(new_fields):
        drifts.append(f"  +{nf}: {f_params[nf]!r}  (new field — alphapool added; update our pool?)")
    return drifts


def _diff_notify_field_shape(baseline: dict, fresh: dict) -> list[str]:
    """Diff arity + field types/lengths of mining.notify between captures."""
    drifts: list[str] = []
    b = baseline["notify_sample"]["params"]
    f = fresh["notify_sample"]["params"]
    if len(b) != len(f):
        drifts.append(f"  arity changed: baseline={len(b)}, fresh={len(f)}")
        return drifts
    for i, (bv, fv) in enumerate(zip(b, f)):
        if type(bv) is not type(fv):
            drifts.append(f"  field [{i}]: type changed {type(bv).__name__} → {type(fv).__name__}")
        elif isinstance(bv, str) and len(bv) != len(fv):
            # job_id and incomplete_header_bytes vary in length across captures,
            # but prevhash (idx 1), ntime (idx 4), nbits (idx 5) must keep their
            # canonical lengths.
            if i in (1, 4, 5) and len(bv) != len(fv):
                drifts.append(f"  field [{i}] hex length changed: {len(bv)} → {len(fv)}")
    return drifts


def test_no_drift_in_latest_capture():
    """Diff the newest capture in tests/fixtures/ against the baseline.
    SKIPS if no fresh capture exists (the operator hasn't run the capture tool yet)."""
    baseline = json.loads(_BASELINE_PATH.read_text())
    fresh_captures = _list_fresh_captures()
    if not fresh_captures:
        pytest.skip(
            "no fresh capture in tests/fixtures/; run "
            "`python tools/capture_alphapool.py --out tests/fixtures/alphapool_capture_$(date +%Y_%m_%d).json` "
            "from a Linux deploy box to pull one"
        )
    fresh_path = fresh_captures[0]
    fresh = json.loads(fresh_path.read_text())

    params_drift = _diff_set_mining_params(baseline, fresh)
    notify_drift = _diff_notify_field_shape(baseline, fresh)

    if params_drift or notify_drift:
        msg = [f"alphapool wire format drift detected (fresh capture: {fresh_path.name}):"]
        if params_drift:
            msg.append("set_mining_params changes:")
            msg.extend(params_drift)
        if notify_drift:
            msg.append("mining.notify shape changes:")
            msg.extend(notify_drift)
        msg.append(
            "\nFIX OPTIONS:\n"
            "  a) Update pearl_stratum_srv to match (config.py, connection.py)\n"
            "  b) If change is acceptable, rotate the baseline:\n"
            f"     mv {fresh_path} {_BASELINE_PATH.name} (then update _BASELINE_PATH)"
        )
        pytest.fail("\n".join(msg))


def test_baseline_capture_is_well_formed():
    """Lock the baseline structure — guards against accidental fixture edits."""
    baseline = json.loads(_BASELINE_PATH.read_text())
    assert "set_mining_params_push" in baseline
    assert "notify_sample" in baseline
    smp = baseline["set_mining_params_push"]
    assert smp["method"] == "pearl.set_mining_params"
    assert smp["params"][0]["rank"] == 128  # mainnet invariant
    assert smp["params"][0]["mma_type"] == "Int7xInt7ToInt32"
    notify = baseline["notify_sample"]["params"]
    assert len(notify) == 7
    assert notify[5] == "1a0ffff0"  # nbits captured at testnet diff 1


def test_drift_diff_helpers_detect_planted_drift():
    """Smoke test: rank=256 in fresh capture → drift detected; identical → no drift."""
    baseline = json.loads(_BASELINE_PATH.read_text())
    identical = json.loads(_BASELINE_PATH.read_text())
    assert _diff_set_mining_params(baseline, identical) == []
    assert _diff_notify_field_shape(baseline, identical) == []

    # Plant a drift
    drifted = json.loads(_BASELINE_PATH.read_text())
    drifted["set_mining_params_push"]["params"][0]["rank"] = 256
    drifted["set_mining_params_push"]["params"][0]["new_param"] = "surprise"
    drifts = _diff_set_mining_params(baseline, drifted)
    assert any("rank" in d for d in drifts)
    assert any("+new_param" in d for d in drifts)
