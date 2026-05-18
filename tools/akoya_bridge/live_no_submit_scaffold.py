#!/usr/bin/env python3
"""Akoya live job scaffold for P1K-131.

Default behavior is safe: register, receive one live job, write decoded job
state, and exit without submitting. With --mine-wrong-jackpot, it also builds a
structurally valid type-3 frame from local Pearl proof material and saves it to
disk, still without submitting it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from akoya_protocol import (  # noqa: E402
    JobAssignment,
    RegisterMiner,
)
from akoya_pool_client import AkoyaPoolSession, mining_job_dict_for_akoya  # noqa: E402
from akoya_share_builder import build_share_from_pearl_plain_proof  # noqa: E402


DEFAULT_WALLET = "prl1pylgkp99zxwpvc9mzu7yw6f73tcswln7fw67tte074haukjkvpsdscan2q9"
AKOYA_MINING_CONFIG_HEX = (
    "00080000800000000701000000000001031f0000"
    "0000000000000000000000000000000000000000000000000000000000000000"
)
AKOYA_M = 8192
AKOYA_N = 32768
AKOYA_K = 2048


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _hex(data: bytes) -> str:
    return data.hex()


def json_default(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported json value: {type(value).__name__}")


def import_pearl():
    try:
        import pearl_mining as pm  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pearl_mining. Use:\n"
            "PYTHONPATH=/home/bereket/.local/lib/python3.12/site-packages "
            ".venv/bin/python tools/akoya_bridge/live_no_submit_scaffold.py --mine-wrong-jackpot"
        ) from exc
    return pm


def maybe_build_share(args, out_dir: Path, job: JobAssignment) -> dict[str, object]:
    if not args.mine_wrong_jackpot:
        return {"built": False, "reason": "--mine-wrong-jackpot not set"}

    pm = import_pearl()
    header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
    mining_config_bytes = bytes.fromhex(AKOYA_MINING_CONFIG_HEX)
    mining_config = pm.MiningConfiguration.from_bytes(mining_config_bytes)

    plain_proof = pm.mine(
        AKOYA_M,
        AKOYA_N,
        AKOYA_K,
        header,
        mining_config,
        signal_range=(-64, 64),
        wrong_jackpot_hash=True,
    )
    share = build_share_from_pearl_plain_proof(
        pm=pm,
        plain_proof=plain_proof,
        header=header,
        header_bytes=job.header_bytes,
        seed_hash=job.seed_hash,
        share_difficulty=job.share_difficulty,
        mining_config_bytes=mining_config_bytes,
        share_id=str(uuid.uuid4()),
    )

    frame_path = out_dir / "nosubmit_type3_plainproofshare.frame"
    msgpack_path = out_dir / "nosubmit_type3_plainproofshare.msgpack"
    frame = share.frame()
    frame_path.write_bytes(frame)
    msgpack_path.write_bytes(frame[4:])

    ok, message = pm.verify_plain_proof_with_nbits(header, plain_proof, job.share_difficulty)
    return {
        "built": True,
        "submitted": False,
        "share_id": share.share_id,
        "frame_path": frame_path,
        "msgpack_path": msgpack_path,
        "frame_len": len(frame),
        "payload_len": len(frame) - 4,
        "local_verify_with_share_difficulty": ok,
        "local_verify_message": message,
        "warning": "This frame was intentionally not submitted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="pool.akoyapool.com")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--wallet", default=DEFAULT_WALLET)
    parser.add_argument("--worker", default=f"codex-nosubmit-{utc_stamp()}")
    parser.add_argument("--gpu-name", default="Codex Local NoSubmit")
    parser.add_argument("--common-dim", type=int, default=AKOYA_K)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--git-sha", default="codex-nosubmit")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--mine-wrong-jackpot", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/bereket/pearl-ops/akoya_recon") / f"live_scaffold_{utc_stamp()}",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    register = RegisterMiner(
        session_uuid=str(uuid.uuid4()),
        wallet_address=args.wallet,
        worker_name=args.worker,
        gpu_name=args.gpu_name,
        common_dim=args.common_dim,
        miner_version=args.version,
        git_sha=args.git_sha,
    )

    with AkoyaPoolSession(args.host, args.port, args.timeout) as session:
        context = session.register_and_wait_job(register)
        ack = context.ack
        job = context.job

    summary: dict[str, object] = {
        "endpoint": f"{args.host}:{args.port}",
        "submitted": False,
        "register": register.to_fields(),
        "ack": {
            "accepted": ack.accepted,
            "pool_uuid": ack.pool_uuid,
            "share_difficulty": ack.share_difficulty,
            "miner_id": ack.miner_id,
        },
        "job": {
            "job_uuid": job.job_uuid,
            "header_len": len(job.header_bytes),
            "header_hex": _hex(job.header_bytes),
            "share_difficulty": job.share_difficulty,
            "height": job.height,
            "seed_hash_hex": _hex(job.seed_hash),
            "reserved_hex": _hex(job.reserved),
            "network_nbits": job.network_nbits,
            "gateway_compatible_mining_job": mining_job_dict_for_akoya(job),
        },
    }
    summary["share_build"] = maybe_build_share(args, args.out_dir, job)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
