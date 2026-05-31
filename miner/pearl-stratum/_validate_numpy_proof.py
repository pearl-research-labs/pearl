"""Offline bit-exactness gate for the torch-free numpy proof builder.

Runs on a rig (py3.10, no torch): mine a HIT against the captured easy-nbits
header, build the PlainProof via pearl_proof_numpy, and assert
pearl_mining.verify_plain_proof accepts it. No pool, no torch.
Usage: PYTHONPATH=/tmp/pearlbench python3 _validate_numpy_proof.py
"""
import base64
import json
import os
import subprocess
import sys

import numpy as np
import pearl_mining as pm
import pearl_proof_numpy as ppn
from run_canary import _splitmix64_fill, B_SEED_MIX

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/tmp/pearlbench"
D = os.environ.get("PEARL_SHAREDUMP", "/tmp/pearlbench")
BIN = os.environ.get("PEARL_MINER_SM89_BIN", "/tmp/pearlbench/pearl_miner_sm89_sm89")

hdr = open(os.path.join(D, "header.bin"), "rb").read()
cfg = open(os.path.join(D, "mining_config.bin"), "rb").read()
mc = pm.MiningConfiguration.from_bytes(cfg)
m = n = 131072
k = int(mc.common_dim)
r = int(getattr(mc, "rank", 256))

FF = "f" * 64
inp = (f"header={hdr.hex()} config={cfg.hex()} mode=mine "
       f"m={m} n={n} k={k} r={r} nonce_start=0 nonce_count=1 target={FF}\n")
env = {**os.environ, "PEARL_SM89_SWIZZLE": "24"}
out = subprocess.run([BIN], input=inp, capture_output=True, text=True, timeout=180, env=env)
hits = [l for l in out.stdout.splitlines() if l.startswith("HIT ")]
if not hits:
    print("NO HIT FROM BINARY; stderr tail:")
    print("\n".join(out.stderr.splitlines()[-8:]))
    sys.exit(2)
hit = json.loads(hits[0][4:])
seed = int(hit["seed"])
a_rows = list(map(int, hit["a_rows"]))
b_cols = list(map(int, hit["b_cols"]))
print(f"HIT seed={seed} a_rows={a_rows[:4]}... b_cols={b_cols[:4]}...")

A = _splitmix64_fill(m * k, seed).reshape(m, k)
B_t = _splitmix64_fill(n * k, seed ^ B_SEED_MIX).reshape(n, k)
opened = ppn.OpenedBlockInfo(A_row_indices=a_rows, B_column_indices=b_cols,
                             A=A, B_t=B_t, commitment_hash=None, noise_rank=r)
proof = ppn.create_proof(opened, hdr)
proof_bytes = base64.b64decode(proof.to_base64())
print(f"built proof: {len(proof_bytes)} bytes")

ok = pm.verify_plain_proof(hdr, proof_bytes)
print(f"verify_plain_proof = {ok}")
print("NUMPY-PROOF GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
