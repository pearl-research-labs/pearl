"""Compare the GPU's on-device merkle a_root (real_commit=1) to pearl_mining's
MerkleTree root for ab_seed=0. DEBUG_ROOT prints on the FIRST attempt (no hit
needed), so any target works.
"""
import sys, subprocess, re
sys.path.insert(0, ".")
import numpy as np
from blake3 import blake3
from pearl_proof_numpy import MatrixMerkleTree

CONFIG_HEX = "00100000000100000701000000000001031f" + "00" * 34
config = bytes.fromhex(CONFIG_HEX)
STDIN = open("/tmp/last_stdin.txt").read()
def kv(s, k):
    for t in s.split():
        if t.startswith(k + "="): return t[len(k)+1:]
header = bytes.fromhex(kv(STDIN, "header"))
target = kv(STDIN, "target")
M=131072; N=131072; K=4096; R=256
job_key = blake3(header + config).digest()

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
line = (f"header={header.hex()} config={config.hex()} target={target} m={M} n={N} k={K} r={R} "
        f"mode=jobmine real_commit=1 nonce_start=0 nonce_count=1 dev=0")
p = subprocess.run([BIN], input=line.encode(), capture_output=True, timeout=120)
err = p.stderr.decode(errors="replace")
m = re.search(r"DEBUG_ROOT ab_seed=(\d+) a_chunks=\d+ job_key=([0-9a-f]+) a_root=([0-9a-f]+) b_root=([0-9a-f]+)", err)
if not m:
    print("no DEBUG_ROOT line. stderr tail:", err[-400:]); sys.exit(1)
ab = int(m.group(1)); gpu_jk = m.group(2); gpu_aroot = m.group(3); gpu_broot = m.group(4)
print("job_key match:", gpu_jk == job_key.hex())
A = fill(M*K, ab).reshape(M, K)
py_aroot = MatrixMerkleTree(A, job_key).root
b3_aroot = blake3(A.tobytes(), key=job_key).digest()
print("GPU a_root      :", gpu_aroot)
print("MerkleTree a_root:", py_aroot.hex())
print("blake3lib a_root :", b3_aroot.hex())
print("GPU == MerkleTree:", gpu_aroot == py_aroot.hex())
print("MerkleTree == blake3lib:", py_aroot == b3_aroot)
