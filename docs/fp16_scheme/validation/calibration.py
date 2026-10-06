#!/usr/bin/env python3
"""Reproducible calibration / adversarial harness for the FP16 jackpot policy.

Backs the whitepaper's Appendix "Empirical validation": for a battery of input
strategies (realistic and adversarial), it computes the per-tile breakpoint
density `f_bp` and certified-work ratio `rho` (bit-matched to
`zk-pow/src/api/fp16/policy.rs`), and the reproduction rate of the two cheap
"shortcut" predictors (exact-sum-rounded-to-FP32; group-sums-without-per-product
truncation). It then checks them against the policy gate (`f_bp >= 0.30`,
`rho >= 1.2`) and prints the min `rho` / `f_bp` observed per strategy.

What this harness IS: a reproducible reimplementation of the policy-metric
distribution study, so a reviewer can re-run the calibration and see where
honest and adversarial tiles land relative to the gate. The accumulation model
is now silicon-validated (see `generate_and_capture.py` / `rounding_mode_probe.py`
and the README), so computing the census from the model is sound.

What this harness is NOT: the concrete truncation-correction *attack cost*
(the "6-13x honest" figure). That is a separate hardware micro-benchmark, not a
policy-metric computation, and is left as the documented remaining piece.

Noise: an independent, faithful reimplementation of the mandated rank-32,
delta=1/2 low-rank noise added before FP16 quantization (mirrors the SHAPE of
`api/fp16/noise.rs` + `quantization.rs`, not its seed-exact bytes). Final
consensus numbers should be reproduced through the Rust path; this is for the
statistical distribution.

Usage:  python3 calibration.py            (CPU only; no GPU needed)
"""
from __future__ import annotations

import argparse
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
GROUP = ref.GROUP
NOISE_RANK = 32
DELTA = 0.5
MIN_FBP = 0.30
MIN_RHO = 1.2


# ---- FP16 <-> f32 (round-to-nearest-even cast, saturating, matches dtype.rs) ----
def f32_to_fp16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint16)


def fp16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint16).view(np.float16).astype(np.float32)


# ---- census (mirrors accumulate.rs PolicyStep + a100_dot breakpoint logic) ----
def dot_with_census(a, b, c_bits):
    """Returns (d_bits, list_of_steps) where each step is (nonempty, breakpoint,
    products_truncated). Bit-identical census to accumulate.rs::a100_dot."""
    a = [int(x) for x in a]
    b = [int(x) for x in b]
    cur = int(c_bits) & 0xFFFFFFFF
    k = len(a)
    steps = []
    g0 = 0
    while g0 < k:
        g1 = min(g0 + GROUP, k)
        cs, cm, cel, culp = ref._acc_parts(cur)
        eta = cel
        for u in range(g0, g1):
            _, ma, ea = ref.decompose_fp16(a[u])
            _, mb, eb = ref.decompose_fp16(b[u])
            if ma and mb:
                eta = max(eta, ea + eb)
        if eta == ref._NEG:
            steps.append((False, False, 0))
            g0 = g1
            continue
        unit = eta - ref.W
        total = 0
        ptc = 0
        for u in range(g0, g1):
            sa, ma, ea = ref.decompose_fp16(a[u])
            sb, mb, eb = ref.decompose_fp16(b[u])
            if not ma or not mb:
                continue
            prod = ma * mb
            sh = (ea + eb) - 20 - unit
            if sh < 0 and (prod & ((1 << min(-sh, 126)) - 1)) != 0:
                ptc += 1
            total += (sa * sb) * ref._shift(prod, sh)
        csh = culp - unit
        acc_trunc = csh < 0 and cm != 0 and (cm & ((1 << min(-csh, 63)) - 1)) != 0
        total += cs * ref._shift(cm, csh)
        nb = ref._rz_to_f32_bits(total, unit)
        # rz dropped a nonzero bit iff the encoded value != exact sum*2^unit
        enc = np.float64(np.float32(np.uint32(nb).view(np.float32)))
        exact = np.float64(total) * np.float64(2.0) ** np.float64(unit)
        rz_dropped = enc != exact
        steps.append((True, bool(acc_trunc or rz_dropped), ptc))
        cur = nb
        g0 = g1
    return cur & 0xFFFFFFFF, steps


def tile_metrics(A_bits, B_bits, h, w, k):
    """f_bp and rho over an h x w tile (bit patterns, row-major, B transposed)."""
    breakpoints = 0
    total_steps = 0
    numerator = 0
    for i in range(h):
        ai = A_bits[i * k:(i + 1) * k]
        for j in range(w):
            bj = B_bits[j * k:(j + 1) * k]
            _, steps = dot_with_census(list(ai), list(bj), 0)
            n_bp = sum(1 for s in steps if s[1])
            n_pt = sum(s[2] for s in steps if not s[1])  # truncated products outside breakpoints
            # maximal runs of non-breakpoint *nonempty* steps
            n_runs = 0
            in_run = False
            for s in steps:
                if s[0] and not s[1]:
                    if not in_run:
                        n_runs += 1
                        in_run = True
                else:
                    in_run = False
            numerator += GROUP * n_bp + NOISE_RANK * n_runs + n_pt
            breakpoints += n_bp
            total_steps += len(steps)
    f_bp = breakpoints / total_steps
    rho = numerator / (h * w * k)
    return f_bp, rho


def shortcut_rates(A_bits, B_bits, h, w, k):
    """Fraction of cells reproduced by the two cheap predictors."""
    A = fp16_bits_to_f32(np.asarray(A_bits, np.uint16)).reshape(h, k)
    B = fp16_bits_to_f32(np.asarray(B_bits, np.uint16)).reshape(w, k)
    exact_hit = group_hit = 0
    for i in range(h):
        for j in range(w):
            d_bits, _ = dot_with_census(list(A_bits[i * k:(i + 1) * k]), list(B_bits[j * k:(j + 1) * k]), 0)
            # predictor 1: exact FP32 dot product (no per-group grid), then it IS the rz-to-f32 of exact sum
            exact = np.float32(np.dot(A[i].astype(np.float64), B[j].astype(np.float64)))
            if exact.view(np.uint32) == d_bits:
                exact_hit += 1
            # predictor 2: group sums without per-product truncation (sum each group exactly, rz per group)
            cur = np.float32(0.0)
            for g0 in range(0, k, GROUP):
                s = np.float64(cur) + np.dot(A[i, g0:g0 + GROUP].astype(np.float64), B[j, g0:g0 + GROUP].astype(np.float64))
                cur = np.float32(s)  # RNE cast; approximates "no per-product trunc"
            if cur.view(np.uint32) == d_bits:
                group_hit += 1
    cells = h * w
    return exact_hit / cells, group_hit / cells


# ---- seed-exact noise via the real Rust consensus pipeline ----
RUST_NOISE_TOOL = os.path.join(
    HERE, "..", "..", "..", "zk-pow", "target", "release", "examples", "fp16_noise_tool"
)


def rust_noise(A_bits, B_bits, h, w, k):
    """Noise A/B raw FP16 codes through the real `fp16_noised_operands` pipeline
    (root-derived seeds -> BLAKE3 noise lines -> noisy_quantize). Returns the
    seed-exact noised A, B code arrays."""
    if not os.path.exists(RUST_NOISE_TOOL):
        sys.exit(
            f"rust noise tool not built: {RUST_NOISE_TOOL}\n"
            "  (cd zk-pow && cargo build --release --example fp16_noise_tool)"
        )
    payload = (
        f"{h} {w} {k}\n"
        + " ".join(str(int(x)) for x in A_bits) + "\n"
        + " ".join(str(int(x)) for x in B_bits) + "\n"
    )
    proc = subprocess.run([RUST_NOISE_TOOL], input=payload, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"rust noise tool failed: {proc.stderr}")
    out = [int(x) for x in proc.stdout.split()]
    if len(out) != h * k + w * k:
        sys.exit(f"rust noise tool returned {len(out)} codes, expected {h*k + w*k}")
    return np.array(out[: h * k], np.uint16), np.array(out[h * k:], np.uint16)


# ---- mandated low-rank noise + FP16 quantization (shape-faithful fallback) ----
def noisy_quantize(X: np.ndarray, rng, k):
    """X: (rows, k) f32. Add rank-32 delta=1/2 noise, then FP16-cast. Returns bits."""
    rows = X.shape[0]
    E = rng.standard_normal((rows, NOISE_RANK)).astype(np.float32)
    F = rng.standard_normal((k, NOISE_RANK)).astype(np.float32)
    N = (E @ F.T).astype(np.float32)
    # per-row scales: noise carries relative Euclidean weight delta
    xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    beta = DELTA * xn / nn
    # alpha range-normalizes to the FP16 finite range (target magnitude ~ 65504)
    noised = X + beta * N
    mx = np.max(np.abs(noised), axis=1, keepdims=True) + 1e-30
    alpha = 60000.0 / mx
    noised = (alpha * noised).astype(np.float32)
    return f32_to_fp16_bits(noised).reshape(-1)


# ---- adversarial / realistic input strategies ----
def make_operand(strategy, rows, k, rng):
    if strategy == "gaussian_outlier":
        X = rng.standard_normal((rows, k)).astype(np.float32)
        X[:, rng.integers(0, k, size=max(1, k // 64))] *= 30.0
    elif strategy == "heavy_tailed":
        X = (rng.standard_t(2.0, size=(rows, k))).astype(np.float32)
    elif strategy == "zeros":
        X = np.zeros((rows, k), np.float32)
    elif strategy == "rank_two":
        U = rng.standard_normal((rows, 2)).astype(np.float32)
        V = rng.standard_normal((2, k)).astype(np.float32)
        X = (U @ V).astype(np.float32)
    elif strategy == "flat":
        X = np.ones((rows, k), np.float32) * rng.standard_normal((rows, 1)).astype(np.float32)
    elif strategy == "few_bits":
        X = rng.integers(-4, 5, size=(rows, k)).astype(np.float32)
    elif strategy == "sparse":
        X = rng.standard_normal((rows, k)).astype(np.float32)
        X *= (rng.random((rows, k)) < 0.05)
    elif strategy == "spiked":
        X = (rng.random((rows, k)) < 0.02).astype(np.float32) * 1000.0
    elif strategy == "geometric":
        X = (2.0 ** rng.integers(-8, 8, size=(rows, k))).astype(np.float32) * rng.choice([-1, 1], (rows, k))
    else:
        raise ValueError(strategy)
    return X


STRATEGIES = [
    "gaussian_outlier", "heavy_tailed", "rank_two", "flat",
    "few_bits", "sparse", "spiked", "geometric",
]


def self_test():
    """The accumulate.rs breakpoint example: 2^24 + 1 drops a bit -> breakpoint."""
    a = [f32_to_fp16_bits(np.array([1.0], np.float32))[0]]
    b = [f32_to_fp16_bits(np.array([1.0], np.float32))[0]]
    c_bits = np.float32(16_777_216.0).view(np.uint32)
    d, steps = dot_with_census(a, b, int(c_bits))
    assert steps[0][1], "census must flag the RZ-drop breakpoint"
    assert d == int(np.float32(16_777_216.0).view(np.uint32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rust-noise", action="store_true",
        help="derive noise through the real Rust fp16_noised_operands pipeline "
             "(seed-exact consensus bytes) instead of the shape-faithful Python noise",
    )
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--k", type=int, default=1024)
    args = ap.parse_args()

    self_test()
    rng = np.random.default_rng(0xBEEF)
    h, w, k = args.h, args.w, args.k  # the study's tile
    noise_src = "rust (seed-exact consensus pipeline)" if args.rust_noise else "python (shape-faithful)"
    print(f"tile h={h} w={w} k={k}, rank={NOISE_RANK}, delta={DELTA}; gate f_bp>={MIN_FBP}, rho>={MIN_RHO}")
    print(f"noise source: {noise_src}\n")
    print(f"{'strategy':20} {'f_bp':>8} {'rho':>8} {'exact%':>8} {'group%':>8} {'gate':>6}")
    worst_fbp, worst_rho = 1e9, 1e9
    all_pass = True
    for strat in STRATEGIES:
        A = make_operand(strat, h, k, rng)
        B = make_operand(strat, w, k, rng)
        if args.rust_noise:
            # Commit raw FP16 codes, then noise through the real pipeline.
            A_raw = f32_to_fp16_bits(np.clip(A, -65504, 65504).astype(np.float32)).reshape(-1)
            B_raw = f32_to_fp16_bits(np.clip(B, -65504, 65504).astype(np.float32)).reshape(-1)
            A_bits, B_bits = rust_noise(A_raw, B_raw, h, w, k)
        else:
            A_bits = noisy_quantize(A, rng, k)
            B_bits = noisy_quantize(B, rng, k)
        f_bp, rho = tile_metrics(A_bits, B_bits, h, w, k)
        ex, gr = shortcut_rates(A_bits, B_bits, h, w, k)
        gate = f_bp >= MIN_FBP and rho >= MIN_RHO
        all_pass &= gate
        worst_fbp = min(worst_fbp, f_bp)
        worst_rho = min(worst_rho, rho)
        print(f"{strat:20} {f_bp:8.3f} {rho:8.3f} {ex*100:7.2f}% {gr*100:7.2f}% {'PASS' if gate else 'FAIL':>6}")
    print(f"\nworst observed: f_bp={worst_fbp:.3f} (gate {MIN_FBP}), rho={worst_rho:.3f} (gate {MIN_RHO})")
    print("whitepaper claims (noised adversarial): f_bp in [0.36,0.51], rho>=1.38, shortcut<=~1.6%")
    print(f"all strategies pass the gate: {all_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
