"""Host launch for the bit-exact A100 (``sm_80``) FP16 operand commitment.

Reproduces ``zk-pow/src/api/fp16/commitment.rs::commit_operand`` -- i.e.
``pearl_blake3::MerkleTree::with_chunk_len(hash_id.pad(rows_to_bytes(rows)),
key, hash_id.chunk_len())`` -- on real GA100 silicon, returning the 32-byte
keyed-BLAKE3 Merkle root bit-for-bit against the verifier.

The committed byte image is the FP16 rows as little-endian ``u16`` row-major,
zero-padded up to a multiple of ``chunk_len`` (``HashId::pad``). That padded
image is uploaded as ``u32`` words and reduced to the root entirely on the GPU:
one keyed-BLAKE3 non-root chunk CV per ``chunk_len``-byte leaf (chunk counter =
leaf index), keyed non-root parent compressions up the tree (lone odd nodes
promoted), and a ROOT-finalized final combine -- with the single-leaf tree
special-cased to the ROOT-finalized hash of its one chunk, exactly as the
reference's ``MerkleTree::with_chunk_len``.

The BLAKE3 compression + tree reduction run in a small hand-written CUDA
extension compiled with ``nvcc -arch=sm_80`` (loaded through
``torch.utils.cpp_extension``), exactly like ``fp16_gemm`` /
``fp16_noisy_quant`` / ``fp16_noise_lines``; the compression is the one agent B
validated bit-identical to the ``blake3`` crate, reused verbatim.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch

from .._utils._arch import Arch, require_arch

_SUPPORTED_ARCHS = (Arch.SM80,)

# Leaf sizes the reference tree accepts (``pearl_blake3::ALLOWED_CHUNK_LENS`` /
# ``HashId::chunk_len``); the FP16 scheme's operand ``HashId`` is one of these.
ALLOWED_CHUNK_LENS = (128, 256, 512, 1024)
# The operand ``HashId`` the FP16 certificate/verifier default to
# (``HashId::Blake3Chunk1024``); operands may use any allowed length.
DEFAULT_CHUNK_LEN = 1024


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_commit_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


def _key_words(key: bytes, device: torch.device) -> torch.Tensor:
    """The 8 little-endian ``u32`` words of the 32-byte ``key``, on ``device``."""
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")
    words = [int.from_bytes(key[4 * i : 4 * i + 4], "little") for i in range(8)]
    signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
    return torch.tensor(signed, dtype=torch.int32, device=device)


def _rows_to_u16(rows, num_rows: int, k: int) -> np.ndarray:
    """``num_rows x k`` FP16 bit patterns as a flat ``uint16`` array."""
    if isinstance(rows, torch.Tensor):
        arr = rows.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    else:
        arr = np.asarray(rows)
    arr = arr.reshape(-1).astype(np.uint16, copy=False)
    if arr.size != num_rows * k:
        raise ValueError(f"operand is not num_rows x k ({arr.size} != {num_rows}*{k})")
    return arr


def _padded_u32(rows_u16: np.ndarray, chunk_len: int) -> tuple[np.ndarray, int]:
    """Little-endian byte image zero-padded to a ``chunk_len`` multiple, as ``u32``.

    Returns ``(words, num_leaves)``.
    """
    raw = rows_u16.astype("<u2").tobytes()
    padded_len = -(-len(raw) // chunk_len) * chunk_len  # div_ceil * chunk_len
    padded_len = max(padded_len, chunk_len)  # at least one leaf
    buf = bytearray(padded_len)
    buf[: len(raw)] = raw
    words = np.frombuffer(bytes(buf), dtype="<u4").astype(np.uint32)
    return words, padded_len // chunk_len


def commit_operand(
    rows,
    num_rows: int,
    k: int,
    key: bytes,
    chunk_len: int = DEFAULT_CHUNK_LEN,
    device: torch.device | int | None = None,
) -> bytes:
    """Keyed-BLAKE3 Merkle root over the FP16 operand rows (32 bytes).

    ``rows`` is a ``num_rows x k`` FP16 operand (torch ``int16``/``uint16`` view
    or any array of ``u16`` bit patterns), row-major; ``key`` is the 32-byte
    per-tree BLAKE3 key; ``chunk_len`` is the operand ``HashId::chunk_len`` (one
    of 128/256/512/1024). Returns the root bit-exact to ``commit_operand(rows,
    num_rows, k, HashId::from(chunk_len), key)``.
    """
    if chunk_len not in ALLOWED_CHUNK_LENS:
        raise ValueError(f"chunk_len must be one of {ALLOWED_CHUNK_LENS}, got {chunk_len}")
    dev = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else (device or torch.device("cuda"))
    )
    require_arch("fp16_commit", dev, *_SUPPORTED_ARCHS)

    rows_u16 = _rows_to_u16(rows, num_rows, k)
    words, num_leaves = _padded_u32(rows_u16, chunk_len)

    data32 = torch.from_numpy(words.view(np.int32)).to(dev)
    key_t = _key_words(key, dev)
    out = _extension().fp16_commit_root(data32, key_t, int(num_leaves), int(chunk_len))
    root_words = out.cpu().numpy().view(np.uint32)
    return b"".join(int(w).to_bytes(4, "little") for w in root_words)
