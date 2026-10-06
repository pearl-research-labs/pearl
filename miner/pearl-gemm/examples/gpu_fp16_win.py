"""On-GPU FP16/A100 winning-tile driver + end-to-end plaintext consensus verify.

Runs the real sm_80 lottery search on the local CMP 170HX at the committed-fixture
geometry (h=4, w=64, k=256; M=8, N=128) with EASY nbits and the honest_fixture
header, decodes the latched winner, assembles the Fp16PlainProof from the GPU tile,
and verifies it through pearl_mining.verify_fp16_plain_proof (the consensus
plaintext oracle). Also checks byte-identity to the committed plaintext fixture and
dumps (header | u32 len | plain_proof_bytes) for the Rust ZK verify.
"""
import struct, sys
from pathlib import Path
import numpy as np
import torch

from pearl_gemm.fp16_miner import search_block, Fp16JobParams, Fp16OperandParams, difficulty_bound
from pearl_gemm.fp16_pipeline import AxisPattern

M, N, K, R = 8, 128, 256, 32
NBITS = 0x207FFFFF
GEN_SEED = 0xDEADBEEF0BADF00D
U64 = (1 << 64) - 1
BLAKE, FOLD = 2, 1  # DimType codes used by pearl_gemm.AxisPattern.new


class Gen:
    """Byte-identical to Rust Gen / miner_base test _Gen (xorshift64 + f32->f16)."""
    def __init__(self, seed): self.s = seed & U64
    def _next(self):
        self.s ^= (self.s << 13) & U64
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & U64
        return self.s
    def operand(self, n):
        out = np.empty(n, np.float16)
        for i in range(n):
            r = self._next()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = ((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(np.float32(sign * mant * (2.0 ** exp)))
        return out


def honest_header():
    return (struct.pack("<I", 0) + bytes(range(31, -1, -1)) + bytes(range(0x5F, 0x3F, -1))
            + struct.pack("<II", 0x66666666, NBITS))


def main():
    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}  compute_cap={cap}  torch={torch.__version__}")
    assert cap == (8, 0), cap

    g = Gen(GEN_SEED)
    a = g.operand(M * K).reshape(M, K)
    b = g.operand(N * K).reshape(N, K)
    header = honest_header()
    a_dev = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    b_dev = torch.from_numpy(np.ascontiguousarray(b)).cuda()

    rp = AxisPattern.new([(4, BLAKE)])               # h = 4
    cp = AxisPattern.new([(4, BLAKE), (16, FOLD)])   # w = 64
    h, w = rp.tile_size(), cp.tile_size()
    op_a = Fp16OperandParams(num_rows=M, pattern=rp, chunk_len=1024)
    op_b = Fp16OperandParams(num_rows=N, pattern=cp, chunk_len=1024)
    params = Fp16JobParams(k=K, a=op_a, b=op_b, nbits=NBITS)

    print(f"\n=== GPU lottery search  (h={h} w={w} k={K}  M={M} N={N}  nbits={NBITS:#010x} EASY) ===")
    wt = search_block(header, a_dev, b_dev, params)
    assert wt.found, "EASY nbits must latch a winner on the real GPU"

    bound = difficulty_bound(NBITS, h, w, K)
    ticket_le = int.from_bytes(wt.ticket, "little")
    print(f"WINNER latched at tile (tr, tc) = ({wt.tile_row}, {wt.tile_col})")
    print(f"  opened A rows (global): {list(range(wt.tile_row*h, wt.tile_row*h+h))}")
    print(f"  opened B cols (global): {wt.tile_col*w}..{wt.tile_col*w+w-1}")
    print(f"  opened_a_rows shape={tuple(wt.opened_a_rows.shape)} dtype={wt.opened_a_rows.dtype}")
    print(f"  opened_b_rows shape={tuple(wt.opened_b_rows.shape)}")
    print(f"  e_a {tuple(wt.e_a.shape)}  f_a {tuple(wt.f_a.shape)}  e_b {tuple(wt.e_b.shape)}  f_b {tuple(wt.f_b.shape)}")
    print(f"  policy: f_bp={wt.report.f_bp:.6f}  rho={wt.report.rho:.6f}  accept={wt.report.accept}")
    print(f"          gates: f_bp>=0.30 -> {wt.report.f_bp>=0.30} ; rho>=1.2 -> {wt.report.rho>=1.2}")
    print(f"  jackpot ticket = {wt.ticket.hex()}")
    print(f"  ticket(LE int) <= bound ? {ticket_le <= bound}   (clears EASY difficulty)")
    print(f"  seed_a  = {wt.seed_a.hex()}")
    print(f"  pow_key = {wt.seeds.pow_key.hex()}")
    print(f"  root_a  = {wt.seeds.root_a.hex()}")
    print(f"  root_b  = {wt.seeds.root_b.hex()}")

    # Dump a few operand values for the record.
    print(f"  opened_a_rows[0,:6] = {wt.opened_a_rows[0,:6].cpu().numpy().tolist()}")
    print(f"  opened_b_rows[0,:6] = {wt.opened_b_rows[0,:6].cpu().numpy().tolist()}")

    # ---- Assemble + verify the plaintext consensus certificate (path b) ----
    import pearl_mining
    from miner_base.fp16_block_submission import create_fp16_proof
    from miner_base.layout import AxisPattern as MbPattern, DimType
    from miner_base.params import HashId

    mb_rp = MbPattern(((4, DimType.BLAKE),))
    mb_cp = MbPattern(((4, DimType.BLAKE), (16, DimType.FOLD)))
    proof = create_fp16_proof(
        header, a.view(np.uint16), b.view(np.uint16), k=K, m=M, n=N,
        rows_pattern=mb_rp, cols_pattern=mb_cp,
        tile_row=wt.tile_row, tile_col=wt.tile_col, hash_id=HashId.BLAKE3_CHUNK_1024,
    )
    print(f"\n=== Plaintext certificate (Fp16PlainProof) from the GPU tile ===")
    print(f"  values_a.row_indices = {list(proof.values_a.row_indices)}")
    print(f"  values_b.row_indices = {list(proof.values_b.row_indices)[:4]}..{list(proof.values_b.row_indices)[-2:]}")
    print(f"  a.num_rows={proof.a.num_rows} b.num_rows={proof.b.num_rows} k={proof.k} r={proof.r}")

    ok, msg = pearl_mining.verify_fp16_plain_proof(
        pearl_mining.IncompleteBlockHeader.from_bytes(header), proof, None)
    print(f"  verify_fp16_plain_proof -> ok={ok}  msg={msg!r}")
    assert ok, f"consensus plaintext verifier REJECTED the GPU tile: {msg}"

    # Byte-identity to the committed Go/Rust fixture (proves the GPU found the
    # canonical honest winner).
    fx = Path("/home/jon/projects/pearl/node/zkpow/testdata/fp16_plain_proof_a100.bin")
    pbytes = bytes(proof.to_bytes())
    if fx.exists():
        blob = fx.read_bytes()
        plen = struct.unpack_from("<I", blob, 76)[0]
        same = pbytes == blob[80:80 + plen] and blob[:76] == header
        print(f"  committed fixture match: {same}  (fixture plain_proof {plen} B, GPU {len(pbytes)} B)")
    else:
        print("  committed fixture absent")

    # Dump (header | u32 len | plain_proof) for the Rust ZK prove/verify.
    out = Path("/mnt/raid/projects/pearl-scratch/gpu_fp16_plain_proof.bin")
    buf = header + struct.pack("<I", len(pbytes)) + pbytes
    out.write_bytes(buf)
    print(f"\n  wrote GPU plain-proof bundle ({len(buf)} B) -> {out}")
    print("\nRESULT: GPU-found FP16/A100 tile WINS and is ACCEPTED by the consensus plaintext verifier.")


if __name__ == "__main__":
    sys.exit(main())
