"""Validate the patched binary: run jobmine (real_commit=1, on-GPU roots) with the
h=2/w=64 config, and assert the HIT's pm.dump_jackpot <= share_bound. A pass means
the host readback now reconstructs the verifier's transcript -> real accepted shares.
"""
import sys, json, subprocess
sys.path.insert(0, ".")
import numpy as np
from pearl_mining import IncompleteBlockHeader, MiningConfiguration, dump_jackpot
from pearl_proof_numpy import OpenedBlockInfo, create_proof

CONFIG_HEX = "00100000000100000701000000000001031f" + "00" * 34   # h=2 w=64, 52 bytes
config = bytes.fromhex(CONFIG_HEX)
STDIN = open("/tmp/last_stdin.txt").read()
def kv(s, k):
    for t in s.split():
        if t.startswith(k + "="): return t[len(k)+1:]
header = bytes.fromhex(kv(STDIN, "header"))
target = int.from_bytes(bytes.fromhex(kv(STDIN, "target")), "big")
M=131072; N=131072; K=4096; R=256
cfg = MiningConfiguration.from_bytes(config)
dot = cfg.common_dim - cfg.common_dim % cfg.rank
share_bound = target * cfg.hash_tile_h * cfg.hash_tile_w * dot
bh = IncompleteBlockHeader.from_bytes(header)
print(f"config h={cfg.hash_tile_h} w={cfg.hash_tile_w}  share_bound=2^{share_bound.bit_length()-1}", flush=True)

GOLDEN=np.uint64(0x9E3779B97F4A7C15);C1=np.uint64(0xBF58476D1CE4E5B9);C2=np.uint64(0x94D049BB133111EB)
def fill(n, seed):
    out=np.empty(n,dtype=np.int8);CH=1<<26;su=np.uint64(seed)
    with np.errstate(over="ignore"):
        for s in range(0,n,CH):
            e=min(s+CH,n);z=np.arange(s+1,e+1,dtype=np.uint64);z*=GOLDEN;z+=su
            z=(z^(z>>np.uint64(30)))*C1;z=(z^(z>>np.uint64(27)))*C2;z^=z>>np.uint64(31)
            out[s:e]=(z%np.uint64(127)).astype(np.int64)-63
    return out

BIN = "/tmp/pearlbench/pearl_miner_sm89_sm89"
stdin = (f"header={header.hex()} config={config.hex()} target={target.to_bytes(32,'big').hex()} "
         f"m={M} n={N} k={K} r={R} mode=jobmine real_commit=1 nonce_start=0 nonce_count=96 dev=0")
print("running jobmine (real_commit=1, nonce_count=96) ...", flush=True)
p = subprocess.run([BIN], input=stdin.encode(), capture_output=True, timeout=300)
out = p.stdout.decode(errors="replace").strip()
if not out.startswith("HIT"):
    print("NOHIT. stderr:", p.stderr.decode(errors="replace")[-300:]); sys.exit(1)
hit = json.loads(out[3: out.find("}")+1])
ab = int(hit.get("ab_seed", hit.get("seed")))
a_rows = list(map(int, hit["a_rows"])); b_cols = list(map(int, hit["b_cols"]))
print(f"HIT ab_seed={ab} #a_rows={len(a_rows)} #b_cols={len(b_cols)} gpu_hash={hit['gpu_hash'][:20]}", flush=True)
A = fill(M*K, ab).reshape(M, K); B_t = fill(N*K, ab ^ 0xD1B54A32D192ED03).reshape(N, K)
proof = create_proof(OpenedBlockInfo(a_rows, b_cols, A, B_t, None, R), header)
j = int.from_bytes(dump_jackpot(bh, proof)[0], "little")
ok = j <= share_bound
print(f"dump_jackpot=2^{j.bit_length()-1}  share_bound=2^{share_bound.bit_length()-1}  VALID={ok}", flush=True)
print("\n*** FIX CONFIRMED: jobmine HIT verifies <= share_bound ***" if ok
      else "\n!!! still invalid — host readback/config still mismatched")
