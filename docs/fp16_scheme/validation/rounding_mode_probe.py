#!/usr/bin/env python3
"""Empirically resolve the A100/sm_80 HMMA.16816.F32 rounding mode on silicon.

The FP16 scheme's model (`zk-pow/src/api/fp16/accumulate.rs`) asserts the device
rounds **toward zero** (RZ), at two places: (1) each product/accumulator is
truncated toward zero onto the per-group `2^(eta-24)` alignment grid, and (2) the
per-group integer sum is rounded toward zero to FP32. The natural alternative is
round-to-nearest-even (RNE) at either place. If the real device used RNE where
the model uses RZ, miner and verifier would still agree with each other (both run
the model) and no in-tree check would catch it -- so this must be settled against
the hardware.

Method: build four model variants (prod ∈ {rz,rne} × group ∈ {rz,rne}), search
random vectors for ones where the variants disagree, capture those on the real
tensor core, and report which variant the device matches. A clean result is: the
device matches (rz,rz) on every discriminating vector and none of the others.

Usage:  python3 rounding_mode_probe.py           (needs the built capture tool + sm_80 GPU)
"""
from __future__ import annotations

import os
import subprocess
import sys
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "miner", "pearl-gemm", "tests", "helpers"),
)
import a100_fp16_reference as ref  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_BIN = os.path.join(HERE, "a100_hmma_capture")


def shift_round(x: int, s: int, mode: str) -> int:
    """`x * 2^s` as an integer. Left shift if s>=0. Right shift (s<0) either
    truncates toward zero ('rz') or rounds half-to-even ('rne')."""
    if s >= 0:
        return x << s
    r = -s
    if r >= 200:
        return 0
    neg = x < 0
    a = -x if neg else x
    if mode == "rz":
        q = a >> r
    else:  # rne, half-to-even
        q = a >> r
        rem = a - (q << r)
        half = 1 << (r - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    return -q if neg else q


def to_f32_bits(s: int, unit: int, mode: str) -> int:
    """Round integer `s * 2^unit` to FP32 (24 sig bits; subnormal 2^-149 grid),
    toward zero ('rz') or half-to-even ('rne'); return u32 bits."""
    if s == 0:
        return 0
    neg = s < 0
    a = -s if neg else s
    nb = a.bit_length() - 1
    keep = max(nb + unit - 23, ref._FP32_MIN_EXP)
    drop = min(max(keep - unit, 0), 200)
    if mode == "rz":
        m = a >> drop
    else:
        m = a >> drop
        if drop > 0:
            rem = a - (m << drop)
            half = 1 << (drop - 1)
            if rem > half or (rem == half and (m & 1)):
                m += 1
    truncated = m << drop
    val = np.float64(-1.0 if neg else 1.0) * np.float64(truncated) * np.float64(2.0) ** np.float64(unit)
    return int(np.float32(val).view(np.uint32))


def dot_mode(a, b, c_bits, prod_mode, group_mode) -> int:
    """a100_dot with selectable product-alignment and group-final rounding."""
    cur = c_bits & 0xFFFFFFFF
    k = len(a)
    g0 = 0
    while g0 < k:
        g1 = min(g0 + ref.GROUP, k)
        cs, cm, cel, culp = ref._acc_parts(cur)
        eta = cel
        for u in range(g0, g1):
            _, ma, ea = ref.decompose_fp16(a[u])
            _, mb, eb = ref.decompose_fp16(b[u])
            if ma and mb:
                eta = max(eta, ea + eb)
        if eta == ref._NEG:
            g0 = g1
            continue
        unit = eta - ref.W
        total = 0
        for u in range(g0, g1):
            sa, ma, ea = ref.decompose_fp16(a[u])
            sb, mb, eb = ref.decompose_fp16(b[u])
            if not ma or not mb:
                continue
            prod = ma * mb
            total += (sa * sb) * shift_round(prod, (ea + eb) - 20 - unit, prod_mode)
        total += cs * shift_round(cm, culp - unit, prod_mode)
        nb = to_f32_bits(total, unit, group_mode)
        if (nb >> 23) & 0xFF == 0xFF:
            return -1  # overflow
        cur = nb
        g0 = g1
    return cur & 0xFFFFFFFF


def run_capture(records):
    lines = []
    for k, a, b, c in records:
        lines.append(" ".join([str(k)] + [str(x) for x in a] + [str(x) for x in b] + [str(c & 0xFFFFFFFF)]))
    proc = subprocess.run([CAPTURE_BIN], input="\n".join(lines) + "\n", capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"capture failed: {proc.stderr}")
    return [int(x) for x in proc.stdout.split()]


MODES = [("rz", "rz"), ("rz", "rne"), ("rne", "rz"), ("rne", "rne")]


def main() -> int:
    rng = np.random.default_rng(0x2026)
    # Find vectors that discriminate the rounding modes. Wide exponent spread
    # within groups maximizes truncated low bits, so modes diverge often.
    disc = []
    tries = 0
    while len(disc) < 400 and tries < 200000:
        tries += 1
        k = int(rng.choice([8, 16, 64, 256]))
        a, b = [], []
        for _ in range(k):
            ea = int(rng.integers(1, 31)); ma = int(rng.integers(0, 1024))
            eb = int(rng.integers(1, 31)); mb = int(rng.integers(0, 1024))
            a.append((int(rng.integers(0, 2)) << 15) | (ea << 10) | ma)
            b.append((int(rng.integers(0, 2)) << 15) | (eb << 10) | mb)
        c = int(rng.integers(0, 1 << 32)) if rng.integers(0, 2) else 0
        if (c >> 23) & 0xFF == 0xFF:
            continue
        vals = [dot_mode(a, b, c, pm, gm) for pm, gm in MODES]
        if -1 in vals:
            continue
        if len(set(vals)) > 1:  # the modes disagree -> discriminating
            disc.append((k, a, b, c, vals))

    if not disc:
        sys.exit("found no discriminating vectors (unexpected)")
    records = [(k, a, b, c) for k, a, b, c, _ in disc]
    device = run_capture(records)

    # Tally which mode the device matches.
    match = {m: 0 for m in MODES}
    device_matches_committed = 0
    device_matches_other = 0
    examples = []
    for (k, a, b, c, vals), dev in zip(disc, device):
        matched_here = []
        for m, v in zip(MODES, vals):
            if v == dev:
                match[m] += 1
                matched_here.append(m)
        if ("rz", "rz") in matched_here:
            device_matches_committed += 1
            # does any NON-(rz,rz) mode give a DIFFERENT value than the device? (yes by construction on some)
        if ("rz", "rz") not in matched_here:
            device_matches_other += 1
            if len(examples) < 10:
                examples.append((k, hex(c), {f"{pm},{gm}": hex(v) for (pm, gm), v in zip(MODES, vals)}, hex(dev)))

    n = len(disc)
    print(f"discriminating vectors captured on silicon: {n} (from {tries} tries)")
    print("device-matches-mode tally (a vector can match several modes when they happen to agree):")
    for m in MODES:
        tag = "  <-- committed model" if m == ("rz", "rz") else ""
        print(f"  prod={m[0]:3} group={m[1]:3}: {match[m]:4}/{n}{tag}")
    print(f"\ndevice matched committed (rz,rz) on {device_matches_committed}/{n} discriminating vectors")
    print(f"device matched (rz,rz) on NONE: {device_matches_other}/{n}")
    if examples:
        print("\n=== vectors where the device did NOT match (rz,rz) -- would indicate the model is wrong ===")
        for e in examples:
            print("  ", e)
    # A clean RZ result: every discriminating vector matches (rz,rz), and at least
    # one other mode is excluded (i.e., (rz,rz) is not vacuously always-matching).
    rz_rz = match[("rz", "rz")]
    strictly_excluded = any(match[m] < n for m in MODES if m != ("rz", "rz"))
    ok = (rz_rz == n) and strictly_excluded
    print(f"\nVERDICT: device is round-toward-zero (RZ) at both stages: {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    if ok:
        other = [f"{pm}/{gm}={match[(pm,gm)]}/{n}" for pm, gm in MODES if (pm, gm) != ("rz", "rz")]
        print(f"  (committed (rz,rz) matched all {n}; alternatives excluded on some vectors: {', '.join(other)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
