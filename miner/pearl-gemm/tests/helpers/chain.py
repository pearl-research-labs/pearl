"""Small cert-v4 chain helpers shared by the isolated kernel tests.

The A side of the chain, from the two committed blobs to the three A keys
(``seedA || noise-line key A || jackpot key``) the prep and GEMM kernels
consume, plus the reference ``OperandNoiser`` for either side. Protocol
composition over a real block header is owned by the production miner;
these helpers only keep the fixture boilerplate of the per-kernel tests
in one place.
"""

from dataclasses import dataclass

import torch
from blake3 import blake3
from miner_base.commitment import Device
from miner_base.commitment_hash import AKeys, noise_line_key
from miner_base.hardware import hardware_for
from miner_base.noise import OperandNoiser, Side

from pearl_gemm import (
    TensorHashConfig,
    tensor_hash_plus_stats,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.protocol_constants import R

from .preprocess import default_config

# Fixed test material: keyA and noise seedB stand in for the header-derived
# key and the B side's commitment chain.
KEY_A = blake3(b"pearl-gemm-tests/key-a").digest()
SEED_B = blake3(b"pearl-gemm-tests/seed-b").digest()


def device_bytes(data: bytes, device="cuda") -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


def p_a_for(m: int, k: int) -> bytes:
    """The dense ``pA`` of the default committed layout for an ``(m, k)`` activation."""
    return default_config(k).p_a(m)


@dataclass
class CommittedA:
    """One activation's commit stage: roots, fused stats and the finalized A keys."""

    root_codes: torch.Tensor
    root_scales: torch.Tensor
    commit_stats: torch.Tensor
    a_keys_dev: torch.Tensor  # (96,) u8 on device

    @property
    def a_keys(self) -> AKeys:
        return AKeys.from_bytes(bytes(self.a_keys_dev.cpu().numpy()))

    @property
    def seed_a(self) -> bytes:
        return self.a_keys.seed_a

    @property
    def noise_key_a_dev(self) -> torch.Tensor:
        return self.a_keys_dev[32:64]

    @property
    def pow_key_dev(self) -> torch.Tensor:
        return self.a_keys_dev[64:96]

    def noise_a(self, k: int, compute=None) -> OperandNoiser:
        noiser = OperandNoiser(
            self.seed_a, Side.A, R, k, compute or hardware_for(Device.BLACKWELL).compute
        )
        assert noiser._key == noise_line_key(self.seed_a)
        return noiser


def commit_a(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    key_a: bytes = KEY_A,
    seed_b: bytes = SEED_B,
    p_a: bytes | None = None,
    config: TensorHashConfig | None = None,
) -> CommittedA:
    """Commit ``pre_quant``'s two blobs on device and finalize the A keys."""
    m, k = codes.shape
    device = codes.device
    hash_config = config or TensorHashConfig()
    committed = CommittedA(
        root_codes=torch.zeros(32, dtype=torch.uint8, device=device),
        root_scales=torch.zeros(32, dtype=torch.uint8, device=device),
        commit_stats=torch.zeros(2 * (m * k // 512), dtype=torch.float32, device=device),
        a_keys_dev=torch.zeros(96, dtype=torch.uint8, device=device),
    )
    roots = torch.zeros(
        tensor_hash_workspace_bytes(m, k, hash_config), dtype=torch.uint8, device=device
    )
    tensor_hash_plus_stats(
        codes,
        scales,
        device_bytes(key_a, device),
        device_bytes(seed_b, device),
        committed.root_codes,
        committed.root_scales,
        roots,
        committed.a_keys_dev,
        committed.commit_stats,
        p_a=p_a if p_a is not None else p_a_for(m, k),
        config=hash_config,
    )
    torch.cuda.synchronize()
    return committed


def noise_b(seed_b: bytes = SEED_B, k: int = 512, compute=None) -> OperandNoiser:
    return OperandNoiser(seed_b, Side.B, R, k, compute or hardware_for(Device.BLACKWELL).compute)


def noise_key_b_dev(seed_b: bytes = SEED_B, device="cuda") -> torch.Tensor:
    return device_bytes(noise_line_key(seed_b), device)
