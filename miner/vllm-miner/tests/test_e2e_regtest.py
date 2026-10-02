"""End-to-end mining against a real regtest node.

vLLM serves a real model with ``--quantization pearl``; its mined prefills submit
FP8 proofs to ``pearl-gateway``, which ZK-proves them and submits blocks to a
``pearld --regtest`` node that verifies the V4 certificate before accepting.
"""

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
PEARLD_BIN = Path(os.environ.get("PEARLD_BIN", REPO_ROOT / "bin" / "pearld"))
MODEL = os.environ.get("E2E_MODEL", "Qwen/Qwen3-0.6B")
MINING_ADDRESS = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2"
RPC_USER, RPC_PASS = "user", "pass"
BLOCK_TIMEOUT_S = 1800


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, body: dict, auth: str | None = None, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    if auth:
        request.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_until(predicate, timeout: float, procs: dict[str, subprocess.Popen], what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, proc in procs.items():
            assert proc.poll() is None, (
                f"{name} exited with {proc.returncode} while waiting for {what}"
            )
        try:
            if predicate():
                return
        except OSError:
            pass
        time.sleep(2)
    pytest.fail(f"timed out after {timeout}s waiting for {what}")


@pytest.mark.skipif(not PEARLD_BIN.exists(), reason=f"pearld binary not found at {PEARLD_BIN}")
def test_vllm_mines_block_verified_by_regtest_node(tmp_path):
    rpc_url = f"http://127.0.0.1:{_free_port()}"
    vllm_url = f"http://127.0.0.1:{_free_port()}"
    socket_path = tmp_path / "pearlgw.sock"
    env = {
        **os.environ,
        "PEARLD_RPC_URL": rpc_url,
        "PEARLD_RPC_USER": RPC_USER,
        "PEARLD_RPC_PASSWORD": RPC_PASS,
        "PEARLD_MINING_ADDRESS": MINING_ADDRESS,
        "MINER_RPC_SOCKET_PATH": str(socket_path),
        "MINER_NO_GATEWAY": "false",
        "MINER_NO_MINING": "false",
    }

    def node_rpc(method: str, *params):
        reply = _post(
            rpc_url,
            {"jsonrpc": "1.0", "id": 0, "method": method, "params": list(params)},
            auth=f"{RPC_USER}:{RPC_PASS}",
            timeout=30,
        )
        assert reply.get("error") is None, reply
        return reply["result"]

    procs: dict[str, subprocess.Popen] = {}

    def start(name: str, cmd: list[str]) -> None:
        with open(tmp_path / f"{name}.log", "w") as log:
            procs[name] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    try:
        start(
            "pearld",
            [
                str(PEARLD_BIN),
                "--regtest",
                "--nolisten",
                "--notls",
                f"--datadir={tmp_path / 'pearld'}",
                f"--logdir={tmp_path / 'pearld'}",
                f"--rpcuser={RPC_USER}",
                f"--rpcpass={RPC_PASS}",
                f"--rpclisten={rpc_url.removeprefix('http://')}",
                f"--miningaddr={MINING_ADDRESS}",
            ],
        )
        _wait_until(lambda: node_rpc("getblockcount") == 0, 60, procs, "pearld RPC")

        start("gateway", [sys.executable, "-m", "pearl_gateway.cli", "start"])
        _wait_until(socket_path.is_socket, 300, procs, "gateway socket")

        start(
            "vllm",
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                MODEL,
                "--quantization",
                "pearl",
                "--port",
                vllm_url.rsplit(":", 1)[1],
                "--max-model-len",
                "8192",
                "--gpu-memory-utilization",
                "0.5",
            ],
        )
        _wait_until(
            lambda: urllib.request.urlopen(f"{vllm_url}/health").status == 200,
            1200,
            procs,
            "vLLM server",
        )

        # Each prompt is well above PEARL_MIN_MINING_TOKENS, so its prefill is mined.
        prompt = "Summarize the history of cryptography in detail. " * 300
        deadline = time.monotonic() + BLOCK_TIMEOUT_S
        while node_rpc("getblockcount") < 1:
            for name, proc in procs.items():
                assert proc.poll() is None, f"{name} exited with {proc.returncode}"
            assert time.monotonic() < deadline, "no block accepted by the regtest node"
            reply = _post(
                f"{vllm_url}/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 16}
            )
            assert reply["choices"][0]["text"]
            time.sleep(5)
    finally:
        for proc in reversed(procs.values()):
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in sorted(tmp_path.glob("*.log")):
            print(
                f"===== {log.name} (tail) =====\n" + "\n".join(log.read_text().splitlines()[-200:])
            )
