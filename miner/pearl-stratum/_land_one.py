"""Direct land-share: raw stratum connect -> get job -> jobmine (real_commit=1) ->
verify dump_jackpot<=bound -> mining.submit -> report accepted/rejected.
Bypasses run_canary's asyncio land-share loop (which hangs on the pool's
authorize->immediate-notify handshake)."""
import socket, json, time, base64, subprocess, sys
sys.path.insert(0, ".")
import numpy as np
from pearl_mining import IncompleteBlockHeader, MiningConfiguration, dump_jackpot
from pearl_proof_numpy import OpenedBlockInfo, create_proof

WALLET = "prl1pc3a98vtz20t3szzq9qclprrwgtq60wlm582cvfpt33y8vv6g8gzs2xa3dz"
CONFIG_HEX = "00100000000100000701000000000001031f" + "00" * 34
config = bytes.fromhex(CONFIG_HEX)
cfg = MiningConfiguration.from_bytes(config)
M=131072; N=131072; K=4096; R=256
dot = cfg.common_dim - cfg.common_dim % cfg.rank
BIN = "/tmp/pearlbench/pearl_miner_sm89_sm89"
GOLDEN=np.uint64(0x9E3779B97F4A7C15);C1=np.uint64(0xBF58476D1CE4E5B9);C2=np.uint64(0x94D049BB133111EB)
def fill(n, seed):
    out=np.empty(n,dtype=np.int8);CH=1<<26;su=np.uint64(seed)
    with np.errstate(over="ignore"):
        for s in range(0,n,CH):
            e=min(s+CH,n);z=np.arange(s+1,e+1,dtype=np.uint64);z*=GOLDEN;z+=su
            z=(z^(z>>np.uint64(30)))*C1;z=(z^(z>>np.uint64(27)))*C2;z^=z>>np.uint64(31)
            out[s:e]=(z%np.uint64(127)).astype(np.int64)-63
    return out

s = socket.create_connection(("pearl-ca1.luckypool.io", 3360), timeout=10)
s.settimeout(35)
f = s.makefile("rwb")
f.write((json.dumps({"id":1,"method":"mining.authorize","params":{"wallet":WALLET,"worker":"cnry-land1","agent":"lpminer/0.1.9-552bdfe"}})+"\n").encode()); f.flush()

def next_job():
    for _ in range(20):
        line = f.readline()
        if not line: return None
        try: msg = json.loads(line)
        except Exception: continue
        if msg.get("method") == "mining.notify":
            return msg["params"]
    return None

for attempt in range(4):
    job = next_job()
    if not job: print("no job"); break
    header = bytes.fromhex(job["header"]); wire = int(job["target"], 16); job_id = job["job_id"]
    bound = wire * cfg.hash_tile_h * cfg.hash_tile_w * dot
    bh = IncompleteBlockHeader.from_bytes(header)
    print(f"[{attempt}] job {job_id} wire=2^{wire.bit_length()-1} bound=2^{bound.bit_length()-1}", flush=True)
    line = (f"header={header.hex()} config={config.hex()} target={job['target']} m={M} n={N} k={K} r={R} "
            f"mode=jobmine real_commit=1 nonce_start=0 nonce_count=160 dev=0")
    t0 = time.time()
    p = subprocess.run([BIN], input=line.encode(), capture_output=True, timeout=180)
    out = p.stdout.decode(errors="replace").strip()
    if not out.startswith("HIT"):
        print(f"  NOHIT ({time.time()-t0:.0f}s) {p.stderr.decode(errors='replace')[-120:]}"); continue
    hit = json.loads(out[3: out.find("}")+1])
    ab = int(hit["ab_seed"]); a_rows = list(map(int, hit["a_rows"])); b_cols = list(map(int, hit["b_cols"]))
    A = fill(M*K, ab).reshape(M, K); B_t = fill(N*K, ab ^ 0xD1B54A32D192ED03).reshape(N, K)
    proof = create_proof(OpenedBlockInfo(a_rows, b_cols, A, B_t, None, R), header)
    pb = base64.b64decode(proof.to_base64())
    j = int.from_bytes(dump_jackpot(bh, proof)[0], "little")
    print(f"  HIT ab={ab} ({time.time()-t0:.0f}s) dump_jackpot=2^{j.bit_length()-1} valid={j<=bound}", flush=True)
    if j > bound: print("  not valid (skip)"); continue
    b64 = base64.b64encode(pb).decode()
    f.write((json.dumps({"id":2+attempt,"method":"mining.submit","params":{"job_id":job_id,"plain_proof":b64,"hs":0.0}})+"\n").encode()); f.flush()
    for _ in range(20):
        line = f.readline()
        if not line: break
        try: msg = json.loads(line)
        except Exception: continue
        if msg.get("id") == 2 + attempt:
            print(f"  SUBMIT RESPONSE: {msg}", flush=True)
            if msg.get("result") and not msg.get("error"):
                print("\n*** SHARE ACCEPTED ON LUCKYPOOL ***"); sys.exit(0)
            else:
                print(f"  rejected: error={msg.get('error')}")
            break
print("done (no accept)")
