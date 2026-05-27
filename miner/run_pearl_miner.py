#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

DEFAULT_MODEL = "pearl-ai/Llama-3.3-70B-Instruct-pearl"
DEFAULT_IMAGE = "vllm_miner:latest"
DEFAULT_CACHE_MOUNT = str(Path.home() / ".cache" / "huggingface")


@dataclass(frozen=True)
class MinerRunConfig:
    mode: str = "docker"
    image: str = DEFAULT_IMAGE
    model: str = DEFAULT_MODEL
    gpus: tuple[str, ...] = ("all",)
    tensor_parallel_size: int = 0
    pearld_rpc_url: str = "http://127.0.0.1:44107"
    pearld_rpc_user: str = "user"
    pearld_rpc_password: str = "pass"
    mining_address: str = ""
    hf_token: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True
    shm_size: str = "8g"
    network_host: bool = True
    cache_mount: str = DEFAULT_CACHE_MOUNT
    lora_adapter: str = ""
    lora_name: str = "pearl_ft"
    extra_vllm_args: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = False


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request_stop(self, _signum: int | None = None, _frame: FrameType | None = None) -> None:
        self.requested = True


def build_docker_run_command(config: MinerRunConfig) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--gpus",
        "all",
    ]
    if config.network_host:
        command.extend(["--network", "host"])
    else:
        command.extend(
            ["-p", f"{config.api_port}:{config.api_port}", "-p", "8337:8337", "-p", "8339:8339"]
        )
    command.extend(["--shm-size", config.shm_size])
    for env_name, env_value in docker_env(config).items():
        if env_value != "":
            command.extend(["-e", f"{env_name}={env_value}"])
    if config.cache_mount:
        command.extend(["-v", f"{config.cache_mount}:/root/.cache/huggingface"])
    if config.lora_adapter:
        command.extend(["-v", f"{config.lora_adapter}:{container_lora_path(config)}:ro"])
    command.append(config.image)
    command.extend(build_vllm_args(config, docker=True))
    return command


def build_local_commands(config: MinerRunConfig) -> tuple[dict[str, str], list[str], list[str]]:
    env = local_env(config)
    return (
        env,
        ["pearl-gateway", "start"],
        ["vllm", "serve", *build_vllm_args(config, docker=False)],
    )


def docker_env(config: MinerRunConfig) -> dict[str, str]:
    env = base_env(config)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(config.gpus)
    env["MINER_RPC_TRANSPORT"] = "tcp"
    return env


def local_env(config: MinerRunConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(base_env(config))
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(config.gpus)
    return env


def base_env(config: MinerRunConfig) -> dict[str, str]:
    return {
        "PEARLD_RPC_URL": config.pearld_rpc_url,
        "PEARLD_RPC_USER": config.pearld_rpc_user,
        "PEARLD_RPC_PASSWORD": config.pearld_rpc_password,
        "PEARLD_MINING_ADDRESS": config.mining_address,
        "HF_TOKEN": config.hf_token,
    }


def build_vllm_args(config: MinerRunConfig, docker: bool) -> list[str]:
    args = [
        config.model,
        "--host",
        config.api_host,
        "--port",
        str(config.api_port),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(effective_tensor_parallel_size(config)),
    ]
    if config.enforce_eager:
        args.append("--enforce-eager")
    if config.lora_adapter:
        args.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{config.lora_name}={container_lora_path(config) if docker else config.lora_adapter}",
            ]
        )
    args.extend(config.extra_vllm_args)
    return args


def container_lora_path(config: MinerRunConfig) -> str:
    return f"/models/{config.lora_name}"


def effective_tensor_parallel_size(config: MinerRunConfig) -> int:
    if config.tensor_parallel_size > 0:
        return config.tensor_parallel_size
    return 1 if config.gpus == ("all",) else len(config.gpus)


def cuda_visible_devices(gpus: tuple[str, ...]) -> str:
    if gpus == ("all",):
        return ""
    return ",".join(gpus)


def parse_gpus(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "all"}:
        return ("all",)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def run_docker(config: MinerRunConfig) -> int:
    command = build_docker_run_command(config)
    if config.dry_run:
        print_command(command)
        return 0
    return subprocess.call(command)


def run_local(config: MinerRunConfig) -> int:
    env, gateway_command, vllm_command = build_local_commands(config)
    if config.dry_run:
        print(
            " ".join(
                f"{key}={shlex.quote(value)}"
                for key, value in sorted(base_env(config).items())
                if value
            )
        )
        print_command(gateway_command)
        print_command(vllm_command)
        return 0

    stop = StopFlag()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop.request_stop)
    gateway = subprocess.Popen(gateway_command, env=env)
    try:
        time.sleep(2)
        vllm = subprocess.Popen(vllm_command, env=env)
        while not stop.requested:
            code = vllm.poll()
            if code is not None:
                return code
            time.sleep(1)
        vllm.terminate()
        return vllm.wait(timeout=30)
    finally:
        gateway.terminate()
        try:
            gateway.wait(timeout=30)
        except subprocess.TimeoutExpired:
            gateway.kill()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the official Pearl vLLM miner for single-node multi-GPU or Docker runs."
    )
    parser.add_argument("--mode", choices=("docker", "local"), default="docker")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--gpus", default="all", help="all or a comma-separated list, for example 0,1"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--pearld-rpc-url", default="http://127.0.0.1:44107")
    parser.add_argument("--pearld-rpc-user", default="user")
    parser.add_argument("--pearld-rpc-password", default="pass")
    parser.add_argument("--mining-address", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--shm-size", default="8g")
    parser.add_argument("--no-network-host", action="store_true")
    parser.add_argument("--cache-mount", default=DEFAULT_CACHE_MOUNT)
    parser.add_argument("--lora-adapter", default="")
    parser.add_argument("--lora-name", default="pearl_ft")
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MinerRunConfig:
    gpus = parse_gpus(args.gpus)
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size <= 0:
        tensor_parallel_size = 1 if gpus == ("all",) else len(gpus)
    return MinerRunConfig(
        mode=args.mode,
        image=args.image,
        model=args.model,
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        pearld_rpc_url=args.pearld_rpc_url,
        pearld_rpc_user=args.pearld_rpc_user,
        pearld_rpc_password=args.pearld_rpc_password,
        mining_address=args.mining_address,
        hf_token=args.hf_token,
        api_host=args.api_host,
        api_port=args.api_port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        shm_size=args.shm_size,
        network_host=not args.no_network_host,
        cache_mount=args.cache_mount,
        lora_adapter=args.lora_adapter,
        lora_name=args.lora_name,
        extra_vllm_args=tuple(args.extra_vllm_arg),
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(parse_args(sys.argv[1:] if argv is None else argv))
    if config.mode == "docker":
        return run_docker(config)
    return run_local(config)


if __name__ == "__main__":
    raise SystemExit(main())
