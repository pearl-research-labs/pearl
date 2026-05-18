#!/usr/bin/env python3
"""Bounded P1K-134 supervisor for direct_gpu_akoya_submit.py.

This intentionally supervises from outside the submitter.  The child owns pool
registration, share construction, share_id generation, and share_id/result
matching.  This wrapper only restarts short child sessions that ended before a
share submission outcome existed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CHILD = THIS_DIR / "direct_gpu_akoya_submit.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "missing_summary"}
    except json.JSONDecodeError as exc:
        return {"status": "invalid_summary_json", "exception": {"type": type(exc).__name__, "message": str(exc)}}


def attempts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw = summary.get("attempts")
    return raw if isinstance(raw, list) else []


def has_submission_outcome(summary: dict[str, Any]) -> bool:
    for attempt in attempts(summary):
        if "submission" in attempt or "submission_exception" in attempt:
            return True
    return False


def exception_type(summary: dict[str, Any]) -> str | None:
    exc = summary.get("exception")
    if isinstance(exc, dict):
        value = exc.get("type")
        return value if isinstance(value, str) else None
    return None


def classify_child(summary: dict[str, Any], returncode: int) -> str:
    status = summary.get("status")
    if status == "accepted":
        return "accepted"
    if status in {"rejected", "rejected_or_submit_failed"}:
        return "hard_stop"
    if has_submission_outcome(summary):
        return "hard_stop"
    if status == "failed_exception" and exception_type(summary) == "EOFError":
        return "retry_eof_before_submission"
    if status in {"timeout", "no_accept_after_attempts"}:
        return "retry_bounded_no_submission"
    if returncode == 0:
        return "complete_no_accept"
    return "hard_stop"


def child_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.child),
        "--torch-noising",
        "--proof-only",
        "--submit",
        "--duration-seconds",
        str(args.session_seconds),
        "--max-attempts",
        str(args.child_max_attempts),
        "--out-dir",
        str(out_dir),
    ]
    cmd.extend(args.child_arg)
    return cmd


def write_supervisor_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "supervisor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    out_root = args.out_root / f"p1k134_supervised_{utc_stamp()}"
    summary: dict[str, Any] = {
        "schema": "supervise_direct_gpu_akoya_submit.v1",
        "started_at": utc_stamp(),
        "child": str(args.child),
        "max_sessions": args.max_sessions,
        "session_seconds": args.session_seconds,
        "child_max_attempts": args.child_max_attempts,
        "sessions": [],
        "status": "running",
    }
    exit_code = 3
    write_supervisor_summary(out_root, summary)

    for session_idx in range(1, args.max_sessions + 1):
        session_dir = out_root / f"session_{session_idx:03d}"
        cmd = child_command(args, session_dir)
        started = time.monotonic()
        proc = subprocess.run(cmd, check=False)
        child_summary = read_summary(session_dir / "summary.json")
        verdict = classify_child(child_summary, proc.returncode)
        session_record = {
            "session": session_idx,
            "out_dir": str(session_dir),
            "returncode": proc.returncode,
            "elapsed_s": time.monotonic() - started,
            "child_status": child_summary.get("status"),
            "child_exception_type": exception_type(child_summary),
            "child_attempts": len(attempts(child_summary)),
            "has_submission_outcome": has_submission_outcome(child_summary),
            "verdict": verdict,
        }
        summary["sessions"].append(session_record)

        if verdict == "accepted":
            summary["status"] = "accepted"
            exit_code = 0
            write_supervisor_summary(out_root, summary)
            return exit_code
        if verdict == "hard_stop":
            summary["status"] = "hard_stop"
            exit_code = proc.returncode if proc.returncode else 2
            write_supervisor_summary(out_root, summary)
            return exit_code
        if verdict == "complete_no_accept":
            summary["status"] = "complete_no_accept"
            exit_code = 0
            write_supervisor_summary(out_root, summary)
            return exit_code

        summary["status"] = "retrying"
        write_supervisor_summary(out_root, summary)
        if args.retry_sleep_seconds > 0:
            time.sleep(args.retry_sleep_seconds)

    summary["status"] = "exhausted"
    exit_code = 3
    write_supervisor_summary(out_root, summary)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--out-root", type=Path, default=Path("akoya_direct_supervised_runs"))
    parser.add_argument("--max-sessions", type=int, default=12)
    parser.add_argument("--session-seconds", type=float, default=300.0)
    parser.add_argument("--child-max-attempts", type=int, default=25)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--child-arg",
        action="append",
        default=[],
        help="Extra argument passed through to direct_gpu_akoya_submit.py; repeat for multiple args.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
