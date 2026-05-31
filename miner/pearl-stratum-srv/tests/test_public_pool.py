"""Tests for the public-pool modules: auth, vardiff, payouts, challenge, share_db."""

from __future__ import annotations

import time

import pytest

from pearl_stratum_srv.auth import IpLimiter, IpQuotas, parse_worker_name, validate_pearl_address
from pearl_stratum_srv.challenge import Challenge, _has_leading_zero_bits
from pearl_stratum_srv.payouts import PayoutPolicy, compute_pplns_payouts
from pearl_stratum_srv.share_db import ShareDb
from pearl_stratum_srv.vardiff import VardiffPolicy, VardiffState, maybe_retarget


# ============================================================ auth: addresses


VALID_ADDR = "prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg"


def test_validate_pearl_address_accepts_known_good():
    assert validate_pearl_address(VALID_ADDR)


@pytest.mark.parametrize("hrp", ["prl1", "tprl1", "sprl1", "rprl1"])
def test_validate_pearl_address_accepts_all_pearl_networks(hrp):
    """mainnet, testnet, simnet, regtest all share the same bech32 charset."""
    addr = hrp + "pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkq"
    # Re-pad to >=50 chars if the shorter HRPs need it
    while len(addr) < 50:
        addr += "q"
    assert validate_pearl_address(addr)


def test_parse_worker_name_accepts_testnet_prefix():
    addr = "tprl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg"
    parsed, label = parse_worker_name(f"{addr}.testrig")
    assert parsed == addr
    assert label == "testrig"


@pytest.mark.parametrize("bad", [
    "", "x", "btc1qwxyz", "prl1!!!",
    "prl1" + "z" * 200,                  # too long
    "prl1abc",                            # too short
    "prl1PGK8J7VJ0XKXPPZUX5VQGQUR9T03K9ZVMM5QKAM5HZAAVZS69VJKQZZ28WG",  # bech32 charset is lowercase-only
    None, 123, b"prl1...",
])
def test_validate_pearl_address_rejects_bad(bad):
    assert not validate_pearl_address(bad)


def test_parse_worker_name_splits_addr_and_label():
    addr, label = parse_worker_name(f"{VALID_ADDR}.rig04.gpu0")
    assert addr == VALID_ADDR
    assert label == "rig04.gpu0"


def test_parse_worker_name_defaults_label_when_missing():
    addr, label = parse_worker_name(VALID_ADDR)
    assert addr == VALID_ADDR
    assert label == "default"


def test_parse_worker_name_returns_none_addr_for_invalid():
    addr, label = parse_worker_name("badaddr.worker")
    assert addr is None
    assert label == "badaddr.worker"


# ============================================================ auth: rate limits


def test_ip_limiter_accepts_below_concurrent_cap():
    lim = IpLimiter(IpQuotas(max_concurrent=3, max_new_per_minute=100))
    for _ in range(3):
        ok, _ = lim.try_accept("1.2.3.4")
        assert ok
        lim.note_open("1.2.3.4")
    ok, reason = lim.try_accept("1.2.3.4")
    assert not ok and "concurrent" in reason


def test_ip_limiter_resets_after_close():
    lim = IpLimiter(IpQuotas(max_concurrent=2, max_new_per_minute=100))
    for _ in range(2):
        ok, _ = lim.try_accept("1.2.3.4")
        assert ok
        lim.note_open("1.2.3.4")
    lim.note_close("1.2.3.4")
    ok, _ = lim.try_accept("1.2.3.4")
    assert ok


def test_ip_limiter_caps_new_connections_per_minute():
    lim = IpLimiter(IpQuotas(max_concurrent=1000, max_new_per_minute=2))
    now = 1000.0
    for _ in range(2):
        assert lim.try_accept("1.2.3.4", now=now)[0]
        lim.note_open("1.2.3.4", now=now)
        lim.note_close("1.2.3.4")
        now += 0.1
    ok, reason = lim.try_accept("1.2.3.4", now=now)
    assert not ok and "rate" in reason


def test_ip_limiter_other_ips_unaffected():
    lim = IpLimiter(IpQuotas(max_concurrent=1, max_new_per_minute=1))
    assert lim.try_accept("1.2.3.4")[0]
    lim.note_open("1.2.3.4")
    assert lim.try_accept("5.6.7.8")[0]  # different IP, fresh budget


# ============================================================ vardiff


def test_vardiff_initial_diff_set():
    s = VardiffState()
    p = VardiffPolicy(initial_diff=2048)
    assert s.init(p) == 2048
    assert s.current_diff == 2048


def test_vardiff_no_retarget_inside_interval():
    s = VardiffState()
    p = VardiffPolicy(initial_diff=1024, retarget_interval_secs=60.0)
    s.init(p, now=0.0)
    assert maybe_retarget(s, p, now=30.0) is None  # not yet time


def test_vardiff_no_retarget_when_within_hysteresis():
    s = VardiffState()
    p = VardiffPolicy(
        initial_diff=1024, target_shares_per_min=6.0,
        retarget_interval_secs=60.0, hysteresis_frac=0.30,
    )
    s.init(p, now=0.0)
    # 6 shares in 60s = exactly goal; no retarget.
    for _ in range(6):
        s.note_share()
    assert maybe_retarget(s, p, now=60.0) is None


def test_vardiff_doubles_diff_when_shares_are_2x_goal():
    s = VardiffState()
    p = VardiffPolicy(
        initial_diff=1024, target_shares_per_min=6.0,
        retarget_interval_secs=60.0, hysteresis_frac=0.10,
        max_step_multiplier=4.0, min_diff=1, max_diff=1 << 30,
    )
    s.init(p, now=0.0)
    for _ in range(12):                       # 12 shares in 60s = 2× goal
        s.note_share()
    new_diff = maybe_retarget(s, p, now=60.0)
    assert new_diff == 2048                    # 1024 × 2


def test_vardiff_drops_diff_when_zero_shares():
    s = VardiffState()
    p = VardiffPolicy(
        initial_diff=1024, target_shares_per_min=6.0,
        retarget_interval_secs=60.0, max_step_multiplier=4.0, min_diff=1,
    )
    s.init(p, now=0.0)
    # No shares submitted — drop hard.
    new_diff = maybe_retarget(s, p, now=60.0)
    assert new_diff == 256                     # 1024 / 4


def test_vardiff_clamps_to_max_step():
    s = VardiffState()
    p = VardiffPolicy(initial_diff=1024, target_shares_per_min=1.0,
                     retarget_interval_secs=60.0, max_step_multiplier=4.0,
                     min_diff=1, max_diff=1 << 30, hysteresis_frac=0.10)
    s.init(p, now=0.0)
    for _ in range(1000):                      # absurdly over-goal
        s.note_share()
    new_diff = maybe_retarget(s, p, now=60.0)
    assert new_diff == 4096                    # clamped to 4× not 1000×


def test_vardiff_respects_min_and_max():
    s = VardiffState()
    p = VardiffPolicy(initial_diff=8, min_diff=8, max_diff=100,
                     retarget_interval_secs=60.0, max_step_multiplier=4.0)
    s.init(p, now=0.0)
    # Zero shares → would drop to 2, clamped to min=8.
    new_diff = maybe_retarget(s, p, now=60.0)
    # 8 is current diff already; clamp may not yield a new value (we expect None).
    assert new_diff is None or new_diff >= 8


# ============================================================ PPLNS payouts


OPERATOR = "prl1operator0000000000000000000000000000000000000000000000000000"
A = "prl1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa00"
B = "prl1bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb00"
C = "prl1cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc00"


def test_payouts_pure_solo_no_shares_all_to_operator():
    p = PayoutPolicy(fee_percent=1.0, operator_address=OPERATOR)
    r = compute_pplns_payouts(1_000_000_000, [], p)
    assert sum(e.amount_sats for e in r.entries) == 1_000_000_000
    assert r.entries[0].recipient == OPERATOR


def test_payouts_proportional_split_minus_fee():
    p = PayoutPolicy(fee_percent=1.0, operator_address=OPERATOR, min_payout_sats=0)
    # A contributed 30, B contributed 70 (total 100). Reward 1,000,000.
    # Fee = 1% = 10,000. Miners pool = 990,000.
    # A gets 297,000, B gets 693,000.
    r = compute_pplns_payouts(1_000_000, [(A, 30), (B, 70)], p)
    total = sum(e.amount_sats for e in r.entries)
    assert total == 1_000_000                  # no sats lost
    by = {e.recipient: e.amount_sats for e in r.entries}
    # Fee always goes to operator; miners get proportional split.
    assert by[OPERATOR] >= 10_000              # fee at minimum
    assert by[A] == 297_000
    assert by[B] == 693_000


def test_payouts_rounding_remainder_lands_somewhere():
    p = PayoutPolicy(fee_percent=0.0, operator_address=OPERATOR, min_payout_sats=0)
    # Awkward split: 1,000,000 sats / 3 shares each = 333,333.33...
    r = compute_pplns_payouts(1_000_000, [(A, 1), (B, 1), (C, 1)], p)
    assert sum(e.amount_sats for e in r.entries) == 1_000_000  # exact


def test_payouts_min_payout_dust_rolls_to_operator():
    p = PayoutPolicy(fee_percent=0.0, operator_address=OPERATOR, min_payout_sats=400_000)
    # B's share is 100, A's is 1 → A gets ~9,900 which is below the min.
    r = compute_pplns_payouts(1_000_000, [(A, 1), (B, 99)], p)
    by = {e.recipient: e.amount_sats for e in r.entries}
    # B is over min, A is under and rolls into operator.
    assert A not in by
    assert by.get(OPERATOR, 0) > 0
    assert sum(by.values()) == 1_000_000


def test_payouts_rejects_invalid_fee_percent():
    p = PayoutPolicy(fee_percent=101.0, operator_address=OPERATOR)
    with pytest.raises(ValueError):
        compute_pplns_payouts(1_000_000, [(A, 1)], p)


def test_payouts_requires_operator_address():
    p = PayoutPolicy(fee_percent=1.0, operator_address="")
    with pytest.raises(ValueError):
        compute_pplns_payouts(1_000_000, [(A, 1)], p)


# ============================================================ challenge


def test_challenge_issue_has_64_hex_seed():
    ch = Challenge.issue(difficulty=8)
    assert len(ch.seed_hex) == 64
    assert ch.difficulty == 8
    assert int(ch.seed_hex, 16) >= 0  # valid hex


def test_challenge_to_notification_params_shape():
    ch = Challenge(seed_hex="ab" * 32, difficulty=16)
    p = ch.to_notification_params()
    assert p == {"seed": "ab" * 32, "difficulty": 16}


def test_challenge_verify_finds_nonce_for_low_difficulty():
    # diff=8 = 1 byte zero prefix; on average 256 tries.
    ch = Challenge.issue(difficulty=8)
    import blake3 as b3
    seed = bytes.fromhex(ch.seed_hex)
    for nonce in range(10_000):
        nonce_le = nonce.to_bytes(8, "little")
        if b3.blake3(seed + nonce_le).digest()[0] == 0:
            assert ch.verify(ch.seed_hex, f"{nonce:016x}")
            return
    pytest.fail("failed to find nonce for diff=8 in 10k tries (extremely unlikely)")


def test_challenge_verify_rejects_wrong_seed():
    ch = Challenge.issue(difficulty=4)
    assert not ch.verify("ff" * 32, "0000000000000000")


def test_challenge_verify_rejects_unsolved_nonce():
    ch = Challenge(seed_hex="00" * 32, difficulty=32)
    # nonce=0 → blake3(00..00 || 00..00). Vanishingly unlikely to be all-zero prefix.
    assert not ch.verify(ch.seed_hex, "0000000000000000")


@pytest.mark.parametrize("data,n,expected", [
    (b"\x00\x00\x00\x00", 32, True),
    (b"\x00\x00\x00\x00", 33, False),
    (b"\x00\x00\x00\x80", 24, True),
    (b"\x00\x00\x00\x80", 25, False),
    (b"\x80", 0, True),
    (b"\x00\xff", 8, True),
])
def test_has_leading_zero_bits(data, n, expected):
    assert _has_leading_zero_bits(data, n) is expected


# ============================================================ share_db


async def test_share_db_insert_and_window_query(tmp_path):
    async with ShareDb(tmp_path / "db.sqlite3") as db:
        t0 = time.time()
        for i in range(5):
            await db.insert_share(
                worker_addr=A, worker_label="gpu0", job_id=f"j{i}",
                difficulty=1024, outcome="accepted", ip="1.2.3.4", ts=t0 + i,
            )
        await db.insert_share(
            worker_addr=B, worker_label="gpu0", job_id="x",
            difficulty=2048, outcome="accepted", ip="5.6.7.8", ts=t0 + 6,
        )
        # Stales don't count toward PPLNS.
        await db.insert_share(
            worker_addr=A, worker_label="gpu0", job_id="stale",
            difficulty=1024, outcome="stale", ip="1.2.3.4", ts=t0 + 7,
        )
        # Explicit until_ts past our future-dated rows; default uses time.time()
        # which is barely > t0 (loop finishes in microseconds) and would exclude
        # t0+1..t0+6.
        rows = await db.shares_in_window(since_ts=t0 - 1, until_ts=t0 + 100)
        d = dict(rows)
        assert d[A] == 5 * 1024
        assert d[B] == 2048


async def test_share_db_payouts_lifecycle(tmp_path):
    async with ShareDb(tmp_path / "db.sqlite3") as db:
        block_id = await db.insert_block(height=100, finder_addr=A, reward_total=2_750_000_000_000)
        await db.insert_payouts(block_id, [(A, 1_000_000_000, 100), (B, 750_000_000, 75)])
        pending = await db.pending_payouts()
        assert len(pending) == 2
        # Mark one sent.
        await db.mark_payout_sent(pending[0][0], txid="deadbeef")
        pending2 = await db.pending_payouts()
        assert len(pending2) == 1


async def test_share_db_ip_ban_lifecycle(tmp_path):
    async with ShareDb(tmp_path / "db.sqlite3") as db:
        assert not await db.is_banned("1.2.3.4")
        await db.ban_ip("1.2.3.4", reason="malformed flood", duration_secs=3600.0)
        assert await db.is_banned("1.2.3.4")
        await db.unban_ip("1.2.3.4")
        assert not await db.is_banned("1.2.3.4")


async def test_share_db_expired_ban_is_not_active(tmp_path):
    async with ShareDb(tmp_path / "db.sqlite3") as db:
        # duration=0.01 → ban expires almost immediately
        await db.ban_ip("9.9.9.9", reason="test", duration_secs=0.01)
        import asyncio
        await asyncio.sleep(0.05)
        assert not await db.is_banned("9.9.9.9")


async def test_share_db_malformed_rate_for_ip(tmp_path):
    async with ShareDb(tmp_path / "db.sqlite3") as db:
        now = time.time()
        for i in range(7):
            await db.insert_share(A, "x", "j", 1, "malformed", ip="1.1.1.1", ts=now - i)
        for i in range(2):
            await db.insert_share(A, "x", "j", 1, "malformed", ip="2.2.2.2", ts=now - i)
        assert await db.malformed_rate_for_ip("1.1.1.1", window_secs=60.0) == 7
        assert await db.malformed_rate_for_ip("2.2.2.2", window_secs=60.0) == 2
        assert await db.malformed_rate_for_ip("3.3.3.3", window_secs=60.0) == 0
