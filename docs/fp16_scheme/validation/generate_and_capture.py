#!/usr/bin/env python3
"""Generate an edge-case FP16/A100 dot-product corpus, capture it on real sm_80
silicon, and cross-check the software model against the device.

This is the reproducible hardware oracle for the FP16 scheme
(docs/fp16_scheme). It closes the gap that the committed
`zk-pow/src/api/fp16/testdata/a100_dot_vectors.txt` corpus tops out at k=256 and
carries no explicitly-constructed subnormal / cancellation / overflow edges, and
that nothing in-tree checked the model against the device rather than against
another copy of itself.

Pipeline, per generated vector `(k, a[k], b[k], c_bits)`:
  1. the software model (`a100_fp16_reference.a100_dot_bits`, the authoritative
     RZ port of `zk-pow/src/api/fp16/accumulate.rs`) computes the expected d_bits
     (or flags overflow);
  2. the standalone capture tool runs the REAL `mma.sync.m16n8k16.f32.f16` on the
     GPU and reports the device d_bits;
  3. the two are compared bit-for-bit. Any disagreement is a model/silicon
     divergence and is reported (and fails the run).

Finite, agreeing vectors are written to `a100_dot_vectors_k1024.txt` in the same
format as the committed corpus, so they can be added to it and consumed by the
Rust cross-check test (`accumulate.rs`).

Usage:
    # build the capture tool once:
    nvcc -arch=sm_80 -O2 -o a100_hmma_capture a100_hmma_capture.cu
    # then:
    python3 generate_and_capture.py --out a100_dot_vectors_k1024.txt

Requires: an sm_80 GPU (A100 / GA100 / CMP 170HX), numpy. Deterministic (seeded).
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import numpy as np

# The authoritative RZ software model (bit-exact port of accumulate.rs).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "miner", "pearl-gemm", "tests", "helpers"
    ),
)
import a100_fp16_reference as ref  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_BIN = os.path.join(HERE, "a100_hmma_capture")


def fp16_normal(rng: np.random.Generator) -> int:
    """A finite normal/zero FP16 bit pattern (exp field in [0, 30]; never inf/NaN)."""
    exp = int(rng.integers(0, 31))  # 0 = subnormal/zero, 1..30 = normal
    man = int(rng.integers(0, 1024))
    sign = int(rng.integers(0, 2)) << 15
    return sign | (exp << 10) | man


def fp16_subnormal(rng: np.random.Generator) -> int:
    man = int(rng.integers(1, 1024))
    sign = int(rng.integers(0, 2)) << 15
    return sign | man  # exp field 0


def fp16_large(rng: np.random.Generator) -> int:
    """Near-max-magnitude FP16 (exp 29..30), for the overflow-edge probe."""
    exp = int(rng.integers(29, 31))
    man = int(rng.integers(0, 1024))
    sign = int(rng.integers(0, 2)) << 15
    return sign | (exp << 10) | man


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def gen_vectors(seed: int = 0xA100):
    """Yield (category, k, a_list, b_list, c_bits) edge-case vectors."""
    rng = np.random.default_rng(seed)

    # 1. Dense random at the requested top dimension k=1024 (the study's k, and
    #    the one the committed corpus omits), several accumulator regimes.
    for _ in range(64):
        k = 1024
        a = [fp16_normal(rng) for _ in range(k)]
        b = [fp16_normal(rng) for _ in range(k)]
        c = 0 if rng.integers(0, 2) else f32_bits(float(rng.standard_normal()) * 2.0 ** int(rng.integers(-10, 20)))
        yield ("random_k1024", k, a, b, c)

    # 2. A sweep of k up to 1024 (fills the 256->1024 gap).
    for k in (16, 32, 64, 128, 256, 512, 768, 1024):
        for _ in range(8):
            a = [fp16_normal(rng) for _ in range(k)]
            b = [fp16_normal(rng) for _ in range(k)]
            yield ("sweep_k", k, a, b, 0)

    # 3. Subnormal FP16 operands (exp field 0) -- the per-product decode's
    #    subnormal branch, dense.
    for _ in range(32):
        k = int(rng.choice([64, 256, 1024]))
        a = [fp16_subnormal(rng) for _ in range(k)]
        b = [fp16_subnormal(rng) if rng.integers(0, 2) else fp16_normal(rng) for _ in range(k)]
        yield ("subnormal_operands", k, a, b, 0)

    # 4. Subnormal FP32 accumulator carry-in (exp field 0, nonzero mantissa).
    for _ in range(32):
        k = int(rng.choice([16, 64, 256]))
        a = [fp16_subnormal(rng) for _ in range(k)]
        b = [fp16_subnormal(rng) for _ in range(k)]
        c = int(rng.integers(1, 1 << 23)) | (int(rng.integers(0, 2)) << 31)  # subnormal f32
        yield ("subnormal_acc", k, a, b, c)

    # 5. Subnormal FP32 OUTPUT (the MA13 subnormal-encode branch the ZK circuit
    #    audit flagged as silicon-untested). With FP16 operands (min |product|
    #    = 2^-48) a nonzero sub-2^-126 result is only reachable when a subnormal
    #    FP32 carry-in survives products that are zero or exactly cancel -- so we
    #    construct exactly that: a subnormal carry-in with (a) all-zero operands
    #    (pure subnormal passthrough) or (b) exactly-cancelling +x/-x product
    #    pairs. These keep the output in the subnormal grid, exercising the
    #    device's subnormal FP32 encode. (That nonzero subnormal outputs are
    #    otherwise unreachable is itself a reportable property -- see README.)
    for i in range(32):
        k = int(rng.choice([16, 64, 256]))
        sub_c = int(rng.integers(1, 1 << 23))  # subnormal f32 bits (exp field 0)
        if i % 2 == 0:
            a = [0] * k  # all-zero operands: output == carry-in (subnormal)
            b = [fp16_subnormal(rng) for _ in range(k)]
        else:
            # +x / -x pairs that cancel exactly inside each group of 8.
            a, b = [], []
            for u in range(k):
                x = fp16_subnormal(rng)
                a.append(x if u % 2 == 0 else (x ^ 0x8000))  # flip sign every other
                b.append(1 << 10)  # 2^-14 (smallest normal) so product = +/- x * 2^-14
        yield ("subnormal_output", k, a, b, sub_c)

    # 6. Grid-boundary products: operands whose product lands exactly on / just
    #    off the 2^(eta-24) alignment grid.
    for _ in range(32):
        k = int(rng.choice([8, 16, 64]))
        a, b = [], []
        for _u in range(k):
            # one big, rest small -> wide exponent spread within a group => dense truncation
            if rng.integers(0, 4) == 0:
                a.append(fp16_large(rng)); b.append(fp16_large(rng))
            else:
                a.append(fp16_subnormal(rng)); b.append(fp16_normal(rng))
        yield ("grid_boundary", k, a, b, 0)

    # 7. Overflow edge: max-magnitude operands + a near-max-finite carry-in.
    for _ in range(16):
        k = 1024
        a = [fp16_large(rng) for _ in range(k)]
        b = [fp16_large(rng) for _ in range(k)]
        c = int(rng.choice([0, 0x7F7FFFFF, 0xFF7FFFFF]))  # 0, +max, -max finite f32
        yield ("overflow_edge", k, a, b, c)


def run_capture(records: list[tuple[int, list[int], list[int], int]]) -> list[int]:
    """Feed (k,a,b,c_bits) records to the silicon capture tool; return d_bits."""
    if not os.path.exists(CAPTURE_BIN):
        sys.exit(f"capture tool not built: {CAPTURE_BIN}\n  nvcc -arch=sm_80 -O2 -o a100_hmma_capture a100_hmma_capture.cu")
    lines = []
    for k, a, b, c in records:
        toks = [str(k)] + [str(x) for x in a] + [str(x) for x in b] + [str(c & 0xFFFFFFFF)]
        lines.append(" ".join(toks))
    proc = subprocess.run(
        [CAPTURE_BIN], input="\n".join(lines) + "\n", capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.exit(f"capture tool failed (rc={proc.returncode}):\n{proc.stderr}")
    out = [int(x) for x in proc.stdout.split()]
    if len(out) != len(records):
        sys.exit(f"capture returned {len(out)} results for {len(records)} records")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    # Canonical location: the Rust cross-check test include_str!'s it from testdata.
    default_out = os.path.join(
        HERE, "..", "..", "..", "zk-pow", "src", "api", "fp16", "testdata", "a100_dot_vectors_k1024.txt"
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0xA100)
    args = ap.parse_args()

    cats = {}
    records = []
    meta = []  # (category, model_dbits_or_None_if_overflow)
    for cat, k, a, b, c in gen_vectors(args.seed):
        try:
            model = ref.a100_dot_bits(a, b, c)
            overflow = False
        except OverflowError:
            model, overflow = None, True
        records.append((k, a, b, c))
        meta.append((cat, model, overflow))
        cats[cat] = cats.get(cat, 0) + 1

    print(f"generated {len(records)} vectors: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    device = run_capture(records)

    agree = mism = overflow_finite = overflow_both = 0
    mismatches = []
    corpus = []
    for (k, a, b, c), (cat, model, overflow), dev in zip(records, meta, device):
        dev_nonfinite = ((dev >> 23) & 0xFF) == 0xFF
        if overflow:
            # model aborts; silicon should be non-finite (inf) too.
            if dev_nonfinite:
                overflow_both += 1
            else:
                overflow_finite += 1
                mismatches.append((cat, k, "model-overflow but device finite", hex(dev)))
            continue
        if dev == model:
            agree += 1
            if not dev_nonfinite:
                corpus.append((k, a, b, c, dev))
        else:
            mism += 1
            mismatches.append((cat, k, f"model={hex(model)}", f"device={hex(dev)}"))

    print(f"\nfinite vectors: {agree} agree, {mism} MISMATCH")
    print(f"overflow-flagged: {overflow_both} device-also-nonfinite, {overflow_finite} device-finite(!)")
    if mismatches:
        print("\n=== DIVERGENCES (model vs silicon) ===")
        for m in mismatches[:50]:
            print("  ", m)

    # Write the agreeing finite vectors as a committed corpus extension.
    with open(args.out, "w") as f:
        f.write("# A100 FP16 dot-product reference vectors (hardware-captured on sm_80).\n")
        f.write(f"# Generated by docs/fp16_scheme/validation/generate_and_capture.py --seed {hex(args.seed)}\n")
        f.write("# Device: run on NVIDIA sm_80 (A100/GA100/CMP 170HX) via real mma.sync.m16n8k16.f32.f16.\n")
        f.write("# Covers k up to 1024 plus subnormal-operand/subnormal-acc/cancellation/grid-boundary edges.\n")
        f.write("# format: k  a_bits[k]  b_bits[k]  c_bits(u32 f32)  d_bits(u32 f32)\n")
        for k, a, b, c, d in corpus:
            toks = [str(k)] + [str(x) for x in a] + [str(x) for x in b] + [str(c & 0xFFFFFFFF), str(d & 0xFFFFFFFF)]
            f.write(" ".join(toks) + "\n")
    print(f"\nwrote {len(corpus)} finite silicon-verified vectors to {args.out}")
    return 1 if (mism or overflow_finite) else 0


if __name__ == "__main__":
    raise SystemExit(main())
