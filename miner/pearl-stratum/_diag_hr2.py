"""Host-roots validation with the PATCHED binary (2x64 readback) + h2w64 config:
real_commit=0 + correct pearl_mining roots so the GPU searches under the VERIFIER's
commitment. On HIT, assert dump_jackpot <= share_bound.
"""
import sys, json, subprocess
sys.path.insert(0, ".")
import numpy as np
from blake3 import blake3
from pearl_mining import IncompleteBlockHeader, MiningConfiguration, dump_jackpot
from pearl_proof_numpy import MatrixMerkleTree, OpenedBlockInfo, create_proof

CONFIG_HEX = "00100000000100000701000000000001031f" + "00" * 34
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
job_key = blake3(header + config).digest()
print(f"h={cfg.hash_tile_h} w={cfg.hash_tile_w} share_bound=2^{share_bound.bit_length()-1}", flush=True)

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
for ab in range(0, 64):   # ab_seed=54 is a known on-GPU HIT; cover around it
    A = fill(M*K, ab).reshape(M, K); B_t = fill(N*K, ab ^ 0xD1B54A32D192ED03).reshape(N, K)
    a_root = MatrixMerkleTree(A, job_key).root; b_root = MatrixMerkleTree(B_t, job_key).root
    line = (f"header={header.hex()} config={config.hex()} target={target.to_bytes(32,'big').hex()} "
            f"m={M} n={N} k={K} r={R} mode=jobmine real_commit=0 aroot={a_root.hex()} broot={b_root.hex()} "
            f"nonce_start={ab} nonce_count=1 dev=0")
    p = subprocess.run([BIN], input=line.encode(), capture_output=True, timeout=90)
    out = p.stdout.decode(errors="replace").strip()
    if not out.startswith("HIT"):
        continue
    hit = json.loads(out[3: out.find("}")+1])
    a_rows = list(map(int, hit["a_rows"])); b_cols = list(map(int, hit["b_cols"]))
    proof = create_proof(OpenedBlockInfo(a_rows, b_cols, A, B_t, None, R), header)
    j = int.from_bytes(dump_jackpot(bh, proof)[0], "little")
    ok = j <= share_bound
    print(f"ab={ab} HIT dump_jackpot=2^{j.bit_length()-1} VALID={ok}", flush=True)
    if ok:
        print(f"\n*** FIX CONFIRMED: host-roots + patched readback -> VERIFIER-VALID share (ab={ab}) ***")
        print(f"    a_rows={a_rows}\n    b_cols[:8]={b_cols[:8]}")
        break
