"""``tensor_hash_plus_stats`` vs the reference: the A-keys chain + commit stats.

Bit-exact gates: the finalized ``seedA || noise-line key A || jackpot key``
equals the reference chain over ``HA = commit_planes([codes, scales], keyA)``,
``seedB`` and ``pA`` (``miner_base.commitment_hash.a_keys``), where each blob
carries its own keyed Merkle tree (its digest is the root), and
the per-block ``(sumsq, absmax)`` partials match the factored protocol
definition over codes and scales (``PrequantMatrix.exact_norms``) --
``absmax`` exactly (it equals the opened rows' absmax by monotonicity),
per-group ``sumsq`` terms exactly, with only the fp32 cross-group sum order
free (asserted exactly on integer-valued data; ``alpha``/``beta``
bit-exactness over random data is covered in test_noisy_quant).

Raw ``tensor_hash`` coverage follows q8 ``test_commitment`` spirit: irregular
byte lengths (single-chunk, multi-block, trailing remainders), dtype
invariance, and determinism.

The last sections cover the ``chunk_size`` / ``sync_loads`` /
``chunks_per_thread`` / ``mad_rot`` knobs and the automatic launch-shape
optimizations
(fused stage-2 tail, concurrent ragged fold), whose digests are checked
against ``pearl_mining.MerkleTree`` (pearl-blake3). The sync (no-TMA) kernel
must agree with the TMA kernel at every configuration, which also
cross-checks the two load paths. The quick subset runs in the PR suite; the
full matrix is ``slow`` (merge-queue only).
"""

import faulthandler
import multiprocessing
import threading

import pearl_mining
import pytest
import torch
from blake3 import blake3
from miner_base.commitment import HashId, commit_planes
from miner_base.commitment_hash import a_keys as ref_a_keys
from miner_base.prequant import PrequantMatrix

from pearl_gemm import (
    TensorHashConfig,
    pre_quant,
    pre_quant_output_shapes,
    tensor_hash,
    tensor_hash_plus_stats,
    tensor_hash_scratchpad_bytes,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.tensor_hash_plus_stats import _blake3, _host, _merkle_host
from pearl_gemm.tensor_hash_plus_stats._merkle_host import (
    tensor_hash_launch,
    tensor_hash_pair_launch,
)
from tests.helpers.chain import p_a_for

_KEY = bytes(range(32))
_LEAF = HashId.BLAKE3_CHUNK_1024
# Constraint: k % 512 == 0 (whole stats blocks per row; a row is then k code
# bytes and k/4 scale bytes). 32x512 is the minimal shape, 1536 a non-power-of-2
# k whose scales rows straddle 1024-byte hash chunks, and 4096x8192 (32 MiB of
# codes) spans multiple MT blocks so the reduce_roots stage runs too.
_SHAPES = [(32, 512), (256, 2048), (128, 4096), (512, 1536), (4096, 8192)]
_HASH_CONFIGS = [
    (128, 2, 64, 1),
    (128, 2, 128, 1),
    (256, 3, 128, 1),
    (512, 2, 128, 257),
    (128, 2, 512, 1),
]
_CHUNK_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
# PR-suite smoke: the default leaf, one smaller, one bigger, the bounds, and
# one non-power-of-2 leaf. The full matrix is merge-queue only; once autotune
# lands per-shape leaf choices, the selected configurations join this set.
_SMOKE_CHUNKS = [
    (1024, True),  # default
    (256, False),  # smaller
    (2048, False),  # bigger
    (64, True),  # lower bound
    (4096, True),  # upper bound
    (192, False),  # non-power-of-2 (3 blocks per leaf)
]


@pytest.fixture
def experimental_leaves(monkeypatch):
    """Explicitly allow non-1024 leaves. The wrappers already default to
    that (miners overlay a committed leaf); the fixture keeps older tests
    that opted in from depending on the module default."""
    monkeypatch.setattr(_host, "_REQUIRE_PROTOCOL_LEAF", False)


def _make_input(m, k, seed):
    torch.manual_seed(seed)
    return torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 1.5


def _data_bytes(length: int) -> bytes:
    return bytes((i * 7 + 3) % 251 for i in range(length))


def _chunk_config(chunk_size: int, **fields) -> TensorHashConfig:
    """A config for ``chunk_size`` (loads must divide the leaf)."""
    fields.setdefault("thread_load_size", 128 if chunk_size % 128 == 0 else 64)
    return TensorHashConfig(chunk_size=chunk_size, **fields)


def _blake3_1024(data: bytes) -> bytes:
    """The C library's keyed digest of ``data`` padded to a whole 1024 leaf."""
    return blake3(data + bytes((-len(data)) % 1024), key=_KEY).digest()


def _blobs_gpu(a):
    """Reference-quantized ``(codes, scales)`` blobs, moved to the GPU."""
    ref = PrequantMatrix.encode(a.cpu())
    return ref.int_values.cuda().contiguous(), ref.scales.cuda().contiguous()


def _buffers(m, k, key_a, seed_b, config=None):
    """Caller-owned buffers for one ``tensor_hash_plus_stats`` call (``key_a``
    is keyA, ``seed_b`` the B side's noise seed; ``pA`` is the default layout's)."""
    config = TensorHashConfig() if config is None else config
    device = "cuda"
    return {
        "config": config,
        "key": torch.frombuffer(bytearray(key_a), dtype=torch.uint8).to(device),
        "seed_b": torch.frombuffer(bytearray(seed_b), dtype=torch.uint8).to(device),
        "root_codes": torch.zeros(32, dtype=torch.uint8, device=device),
        "root_scales": torch.zeros(32, dtype=torch.uint8, device=device),
        "roots": torch.zeros(
            tensor_hash_workspace_bytes(m, k, config),
            dtype=torch.uint8,
            device=device,
        ),
        "a_keys": torch.zeros(96, dtype=torch.uint8, device=device),
        "stats": torch.zeros(2 * (m * k // 512), dtype=torch.float32, device=device),
        "p_a": p_a_for(m, k),
    }


def _launch(codes, scales, buffers):
    tensor_hash_plus_stats(codes, scales, **buffers)
    return buffers["a_keys"]


def _gpu_root(data: bytes, config: TensorHashConfig) -> bytes:
    """The raw ``tensor_hash`` root of ``data`` under ``config``."""
    data_gpu = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    key = torch.frombuffer(bytearray(_KEY), dtype=torch.uint8).cuda()
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(len(data), config), dtype=torch.uint8, device="cuda"
    )
    tensor_hash(data_gpu, key, root, roots, config=config)
    torch.cuda.synchronize()
    return bytes(root.cpu().numpy())


def _assert_commitment(buffers, codes, scales, key_a, seed_b):
    """Both roots and the finalized A keys match the reference two-tree chain."""
    comm = commit_planes([codes.cpu(), scales.cpu()], key_a, _LEAF)
    assert bytes(buffers["root_codes"].cpu().numpy()) == comm.parts[0].tree.root
    assert bytes(buffers["root_scales"].cpu().numpy()) == comm.parts[1].tree.root
    expected = ref_a_keys(comm.digest, seed_b, key_a, buffers["p_a"]).to_bytes()
    assert bytes(buffers["a_keys"].cpu().numpy()) == expected


def _exact_stats_input(m, k):
    """Integer-valued BF16 whose dequantized squares sum exactly in fp32.

    Every scale group carries a 127 peak, so ``scale == 1`` exactly, the
    codes equal the inputs, and each chunk's sum of squares stays below
    2^24 -- exact in fp32 regardless of accumulation order.
    """
    values = (torch.arange(m * k, device="cuda", dtype=torch.int32) % 251) - 125
    values = values.reshape(-1, 8)
    values[:, 0] = 127
    return values.reshape(m, k).to(torch.bfloat16)


def _reference_chunk_stats(a_q) -> torch.Tensor:
    """Factored per-512-chunk (sumsq, absmax) from codes and scales (the
    protocol definition; see PrequantMatrix.exact_norms). absmax also
    equals the opened rows' absmax bit-exactly (monotone BF16 rounding)."""
    m, k = a_q.int_values.shape
    q = a_q.int_values.reshape(m, k // 8, 8).to(torch.float32)
    s = a_q.scales.to(torch.float32)
    group_sumsq = (s * s) * q.square().sum(dim=-1)
    group_absmax = (s * q.abs().amax(dim=-1)).to(torch.bfloat16).float()
    sumsq = group_sumsq.reshape(-1, 64).sum(dim=1)
    absmax = group_absmax.reshape(-1, 64).amax(dim=1)
    return torch.stack((sumsq, absmax), dim=1).flatten()


# q8 commitment-style lengths: single sub-chunk, exact chunks, trailing
# remainders under 64 B, and multi-block irregular matrices (flattened).
_RAW_HASH_LENGTHS = [
    1,
    3,
    63,
    64,
    65,
    256,  # (16, 16)
    1023,
    1024,  # (32, 32)
    1025,
    1056,  # (1, 1056): two chunks, 32 B remainder
    2080,  # (1, 2080): three chunks, 32 B remainder
    4096,  # (64, 64)
    16384,  # (128, 128)
    131044,  # (362, 362): largest single block @128 threads
    131769,  # (363, 363): first size needing two blocks
    16 * 1024 * 1024,  # 16 MiB multi-block
]


@pytest.mark.parametrize("length", _RAW_HASH_LENGTHS)
def test_raw_tensor_hash_matches_padded_blake3(length):
    data = (torch.arange(length, dtype=torch.int64, device="cuda") % 251).to(torch.uint8)
    key = torch.arange(32, dtype=torch.uint8, device="cuda")
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(length),
        dtype=torch.uint8,
        device="cuda",
    )
    tensor_hash(data, key, root, roots)
    torch.cuda.synchronize()

    raw = bytes(data.cpu().numpy())
    expected = blake3(raw + bytes((-length) % 1024), key=_KEY).digest()
    assert bytes(root.cpu().numpy()) == expected


_MATRIX_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (777, 1024),
    (1337, 512),
    (2000, 2048),
    (2048, 3072),
    (4096, 2048),
    (8192, 1024),
]


@pytest.mark.parametrize(
    "shape",
    _MATRIX_SHAPES,
    ids=["x".join(map(str, s)) for s in _MATRIX_SHAPES],
)
def test_raw_tensor_hash_matrix_shapes(shape):
    """Flattened matrix shapes mirroring q8 ``test_tensor_hash_shapes``."""
    matrix = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")
    num_bytes = matrix.numel()
    key = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(num_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    tensor_hash(matrix, key, root, roots)
    torch.cuda.synchronize()

    raw = bytes(matrix.reshape(-1).cpu().numpy())
    expected = blake3(raw + bytes((-num_bytes) % 1024), key=bytes(key.cpu().numpy())).digest()
    assert bytes(root.cpu().numpy()) == expected


def test_raw_tensor_hash_deterministic():
    """Repeated launches over the same bytes produce the same digest."""
    length = 1024 * 1024  # 1 MiB: multi-block without multi-second stress
    data = torch.randint(0, 256, (length,), dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(tensor_hash_scratchpad_bytes(length), dtype=torch.uint8, device="cuda")
    base = torch.empty(32, dtype=torch.uint8, device="cuda")
    tensor_hash(data, key, base, roots)
    torch.cuda.synchronize()
    for _ in range(32):
        out = torch.empty(32, dtype=torch.uint8, device="cuda")
        tensor_hash(data, key, out, roots)
        torch.cuda.synchronize()
        assert torch.equal(base, out)


@pytest.mark.slow
def test_raw_tensor_hash_deterministic_large():
    """q8-style large-matrix determinism stress (64 MiB)."""
    length = 8192 * 8192
    data = torch.randint(0, 256, (length,), dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(tensor_hash_scratchpad_bytes(length), dtype=torch.uint8, device="cuda")
    base = torch.empty(32, dtype=torch.uint8, device="cuda")
    tensor_hash(data, key, base, roots)
    torch.cuda.synchronize()
    for _ in range(8):
        out = torch.empty(32, dtype=torch.uint8, device="cuda")
        tensor_hash(data, key, out, roots)
        torch.cuda.synchronize()
        assert torch.equal(base, out)


@pytest.mark.parametrize(
    "dtype,numel",
    [
        (torch.uint8, 1057),
        (torch.int32, 265),
        (torch.bfloat16, 529),
    ],
)
def test_raw_tensor_hash_is_dtype_agnostic(dtype, numel):
    num_bytes = numel * torch.empty((), dtype=dtype).element_size()
    raw = (torch.arange(num_bytes, dtype=torch.int64, device="cuda") % 251).to(torch.uint8)
    data = raw.view(dtype)
    key = torch.arange(32, dtype=torch.uint8, device="cuda")
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(num_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    tensor_hash(data, key, root, roots)
    torch.cuda.synchronize()

    expected = blake3(bytes(raw.cpu().numpy()) + bytes((-num_bytes) % 1024), key=_KEY).digest()
    assert bytes(root.cpu().numpy()) == expected


@pytest.mark.parametrize(
    "threads_per_block,num_stages,thread_load_size,m",
    _HASH_CONFIGS,
    ids=lambda value: str(value),
)
def test_geometry_digest(threads_per_block, num_stages, thread_load_size, m):
    k = 512
    jk, c_b = blake3(b"geometry").digest(), blake3(b"geometry-cb").digest()
    config = TensorHashConfig(
        threads_per_block=threads_per_block,
        num_stages=num_stages,
        leaves_per_mt_block=256,
        thread_load_size=thread_load_size,
    )
    a = _exact_stats_input(m, k)
    codes, scales = _blobs_gpu(a)
    buffers = _buffers(m, k, jk, c_b, config)
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()

    _assert_commitment(buffers, codes, scales, jk, c_b)
    # The fused stats must be geometry-independent (exact input -> exact fp32).
    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    torch.testing.assert_close(buffers["stats"].cpu(), ref_stats, rtol=0, atol=0)


def test_geometry_multiblock_digest():
    # 257 * 512 code bytes = 128.5 chunks => two roots CTAs plus a partial
    # chunk; the 257 * 128 scale bytes are 32.125 chunks, one CTA.
    m, k = 257, 512
    jk = blake3(b"multiblock").digest()
    c_b = blake3(b"multiblock-cb").digest()
    config = TensorHashConfig(
        threads_per_block=128,
        num_stages=2,
        leaves_per_mt_block=256,
        thread_load_size=256,
    )
    a = _exact_stats_input(m, k)
    codes, scales = _blobs_gpu(a)
    buffers = _buffers(m, k, jk, c_b, config)
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()

    _assert_commitment(buffers, codes, scales, jk, c_b)
    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    torch.testing.assert_close(buffers["stats"].cpu(), ref_stats, rtol=0, atol=0)


@pytest.mark.parametrize("m", [4, 8, 32])
def test_small_scales_blob_digest(m):
    """The scales blob is a fifth of the codes, so it hits its own edge cases.

    At k=512 a row is 128 scale bytes, so m=4 is half a BLAKE3 chunk (padded,
    no full chunk), m=8 is exactly one chunk (the single-chunk root path), and
    m=32 is four chunks. The codes blob is four chunks or more throughout.
    """
    jk, c_b = blake3(b"small").digest(), blake3(b"small-cb").digest()
    k = 512
    a = _exact_stats_input(m, k)
    codes, scales = _blobs_gpu(a)
    buffers = _buffers(m, k, jk, c_b)
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()

    _assert_commitment(buffers, codes, scales, jk, c_b)
    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    torch.testing.assert_close(buffers["stats"].cpu(), ref_stats, rtol=0, atol=0)


@pytest.mark.parametrize("m,k", _SHAPES)
def test_stats_bit_exact(m, k):
    """Fused (sumsq, absmax) partials over random data: absmax exactly;
    sumsq to accumulation order (absorbed by the protocol's l2 grid
    rounding downstream)."""
    a = _make_input(m, k, seed=2)
    a[:, :16] = 0  # exercise the zero-scale group path
    codes, scales = _blobs_gpu(a)
    buffers = _buffers(m, k, blake3(b"stats").digest(), blake3(b"stats-cb").digest())
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()

    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    stats = buffers["stats"].cpu()
    assert torch.equal(stats[1::2], ref_stats[1::2])
    torch.testing.assert_close(stats[0::2], ref_stats[0::2], rtol=1e-6, atol=0)


@pytest.mark.parametrize("m,k", _SHAPES)
def test_hash_bit_exact(m, k):
    jk = blake3(b"job").digest()
    cB = blake3(b"cb").digest()
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))

    buffers = _buffers(m, k, jk, cB)
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()

    # H(A): one keyed Merkle chain per blob, digests combined into HA.
    _assert_commitment(buffers, codes, scales, jk, cB)


def _launch_b(codes, scales, buffers):
    """The weights-path wrapper: same pair grid, no device finalize."""
    from pearl_gemm import tensor_hash_plus_stats_b

    tensor_hash_plus_stats_b(
        codes,
        scales,
        buffers["key"],
        buffers["root_codes"],
        buffers["root_scales"],
        buffers["roots"],
        buffers["stats"],
        config=buffers["config"],
    )


@pytest.mark.parametrize("m,k", _SHAPES)
def test_b_roots_match_reference_planes(m, k):
    """Each weight plane's root equals the reference ``commit_planes`` part, so
    the caller's host-side shape binding / ``hash_b`` / ``cB`` combines are
    unchanged (they consume the same roots the reference chain produces)."""
    jk = blake3(b"job-b").digest()
    codes, scales = _blobs_gpu(_make_input(m, k, seed=3))
    buffers = _buffers(m, k, jk, blake3(b"unused").digest())
    _launch_b(codes, scales, buffers)
    torch.cuda.synchronize()

    comm = commit_planes([codes.cpu(), scales.cpu()], jk, _LEAF)
    assert bytes(buffers["root_codes"].cpu().numpy()) == comm.parts[0].tree.root
    assert bytes(buffers["root_scales"].cpu().numpy()) == comm.parts[1].tree.root


@pytest.mark.parametrize("m,k", _SHAPES)
def test_b_stats_bit_exact(m, k):
    """The weights-path fused (sumsq, absmax) partials match the activation
    path's contract: absmax exactly; sumsq to accumulation order (oracle:
    ``PrequantMatrix.exact_norms``, absorbed by the l2 grid rounding in
    ``noisy_quant_b``)."""
    a = _make_input(m, k, seed=4)
    a[:, :16] = 0  # exercise the zero-scale group path
    codes, scales = _blobs_gpu(a)
    buffers = _buffers(m, k, blake3(b"stats-b").digest(), blake3(b"unused").digest())
    _launch_b(codes, scales, buffers)
    torch.cuda.synchronize()

    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    stats = buffers["stats"].cpu()
    assert torch.equal(stats[1::2], ref_stats[1::2])
    torch.testing.assert_close(stats[0::2], ref_stats[0::2], rtol=1e-6, atol=0)


def test_b_stats_are_optional():
    """``stats=None`` skips the fused sweep and leaves both roots intact."""
    m, k = 128, 4096
    jk = blake3(b"job-b").digest()
    codes, scales = _blobs_gpu(_make_input(m, k, seed=3))
    with_stats = _buffers(m, k, jk, blake3(b"unused").digest())
    _launch_b(codes, scales, with_stats)
    without = _buffers(m, k, jk, blake3(b"unused").digest())
    without["stats"] = None
    _launch_b(codes, scales, without)
    torch.cuda.synchronize()

    for name in ("root_codes", "root_scales"):
        assert torch.equal(without[name], with_stats[name])


@pytest.mark.parametrize("m,k", _SHAPES)
def test_stats_are_optional(m, k):
    """``stats=None`` skips the fused sweep and leaves the commitment intact.

    This is the verifier's path (it recomputes cA and needs no partials) and
    the ``commit_clean`` benchmark stage, which isolates the fusion's cost.
    """
    jk, cB = blake3(b"job").digest(), blake3(b"cb").digest()
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))

    with_stats = _buffers(m, k, jk, cB)
    _launch(codes, scales, with_stats)
    without = _buffers(m, k, jk, cB)
    without["stats"] = None
    _launch(codes, scales, without)
    torch.cuda.synchronize()

    _assert_commitment(without, codes, scales, jk, cB)
    for name in ("root_codes", "root_scales", "a_keys"):
        assert torch.equal(without[name], with_stats[name])


@pytest.mark.parametrize("m,k", _SHAPES)
def test_hash_tracks_content(m, k):
    jk, cB = blake3(b"j").digest(), blake3(b"c").digest()
    blobs1 = _blobs_gpu(_make_input(m, k, seed=0))
    blobs2 = _blobs_gpu(_make_input(m, k, seed=1))
    buffers = _buffers(m, k, jk, cB)
    cA1 = bytes(_launch(*blobs1, buffers).cpu().numpy())
    cA2 = bytes(_launch(*blobs2, buffers).cpu().numpy())
    assert cA1 != cA2


def test_hash_tracks_the_scales_blob():
    """A change confined to the scales must move the A keys (the second tree is bound)."""
    m, k = 32, 512
    jk, cB = blake3(b"j2").digest(), blake3(b"c2").digest()
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    buffers = _buffers(m, k, jk, cB)
    cA1 = bytes(_launch(codes, scales, buffers).cpu().numpy())
    perturbed = scales.clone()
    perturbed[0, 0] = perturbed[0, 1]
    cA2 = bytes(_launch(codes, perturbed, buffers).cpu().numpy())
    assert cA1 != cA2


def test_activation_finalize_uses_current_stream():
    """The whole op, finalize included, follows the caller's stream.

    A finalize left to pick its own stream still passes here on an idle GPU;
    it takes SM contention to separate the two. ``scripts/probe_finalize_
    stream.py`` reproduces that deterministically under load.
    """
    m, k = 32, 512
    jk, c_b = blake3(b"stream-job").digest(), blake3(b"stream-cb").digest()
    a = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    buffers = _buffers(m, k, jk, c_b)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        a.fill_(1.5)
        pre_quant(a, codes, scales)
        _launch(codes, scales, buffers)
    stream.synchronize()

    _assert_commitment(buffers, codes, scales, jk, c_b)


def test_rejects_short_keys():
    codes, scales = _blobs_gpu(_make_input(32, 512, seed=0))
    buffers = _buffers(32, 512, b"\x01" * 32, b"\x02" * 32)
    buffers["key"] = buffers["key"][:-1]
    with pytest.raises(ValueError, match="shape"):
        _launch(codes, scales, buffers)


def test_rejects_mismatched_scales_blob():
    codes, scales = _blobs_gpu(_make_input(32, 512, seed=0))
    buffers = _buffers(32, 512, b"\x01" * 32, b"\x02" * 32)
    with pytest.raises(ValueError, match="scales"):
        _launch(codes, scales[:, :-8], buffers)
    with pytest.raises(ValueError, match="codes"):
        _launch(codes.to(torch.uint8), scales, buffers)


def test_raw_tensor_hash_rejects_empty_input_and_short_workspace():
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(32, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="between 1"):
        tensor_hash(torch.empty(0, dtype=torch.uint8, device="cuda"), key, root, roots)

    data = torch.zeros(129 * 1024, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="at least"):
        tensor_hash(data, key, root, roots)


# --------------------------------------------------------------------------
# The chunk_size / sync_loads knobs, against pearl_mining.MerkleTree
# --------------------------------------------------------------------------


def _reference_root(data: bytes, chunk_size: int) -> bytes:
    """Keyed Merkle root from pearl-blake3, padded to a whole leaf."""
    padded = data + b"\x00" * (-len(data) % chunk_size)
    try:
        return pearl_mining.MerkleTree(data=padded, key=_KEY, chunk_size=chunk_size).root
    except TypeError:
        if chunk_size != 1024:
            pytest.skip("installed pearl_mining predates chunk_size support")
        return pearl_mining.MerkleTree(data=padded, key=_KEY).root


@pytest.mark.parametrize("chunk_size,sync_loads", _SMOKE_CHUNKS, ids=lambda value: str(value))
def test_chunk_size_and_sync_smoke(chunk_size, sync_loads):
    """PR-suite smoke: one partial-chunk and one multi-CTA payload per knob."""
    config = _chunk_config(chunk_size, sync_loads=sync_loads)
    for length in (1025, 200_000):
        data = _data_bytes(length)
        assert _gpu_root(data, config) == _reference_root(data, chunk_size)


@pytest.mark.slow
@pytest.mark.parametrize("sync_loads", [False, True], ids=["tma", "sync"])
@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
def test_chunk_sync_digest_matrix(chunk_size, sync_loads):
    """Full matrix vs pearl_mining.MerkleTree, covering the tree edge cases:
    sub-block, partial-block, exact single chunk, single-CTA non-power-of-2
    leaves (6), and a ragged multi-CTA payload whose last CTA holds a
    non-power-of-2 leaf count (131 chunks -> 3 leaves after the first CTA's
    128)."""
    config = _chunk_config(chunk_size, sync_loads=sync_loads)
    lengths = (
        1,
        1023,
        chunk_size,
        chunk_size + 1,
        5 * chunk_size + 17,
        130 * chunk_size + 5,
    )
    for length in lengths:
        data = _data_bytes(length)
        assert _gpu_root(data, config) == _reference_root(data, chunk_size), f"length={length}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "threads_per_block,num_stages,thread_load_size",
    [(256, 3, 128), (512, 2, 128)],
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("chunk_size", _CHUNK_SIZES)
def test_chunk_size_across_geometries(chunk_size, threads_per_block, num_stages, thread_load_size):
    """Non-default CTA geometries (including the dual-pipeline 512) keep the
    digest at every chunk size, on both load paths."""
    length = 2 * threads_per_block * chunk_size + 999
    data = _data_bytes(length)
    expected = _reference_root(data, chunk_size)
    for sync_loads in (False, True):
        config = _chunk_config(
            chunk_size,
            threads_per_block=threads_per_block,
            num_stages=num_stages,
            thread_load_size=min(thread_load_size, chunk_size),
            sync_loads=sync_loads,
        )
        assert _gpu_root(data, config) == expected, f"sync_loads={sync_loads}"


@pytest.mark.slow
def test_multi_mt_block_payload():
    """A payload spanning multiple MT blocks (stage 3 runs). At 1024 both load
    paths are anchored to the C library; at larger chunks the two paths must
    agree with each other (stage 2/3 are chunk-size independent)."""
    length = 256 * 128 * 1024 + 12345  # > leaves_per_mt_block * tpb chunks
    data = _data_bytes(length)
    expected = _blake3_1024(data)
    assert _gpu_root(data, TensorHashConfig()) == expected
    assert _gpu_root(data, TensorHashConfig(sync_loads=True)) == expected
    # Small leaves multiply the root count (chunk 64 runs stage 3 with 65 of
    # its 128-root capacity here).
    assert _gpu_root(data, _chunk_config(64)) == _reference_root(data, 64)
    for chunk_size in (2048, 4096):
        tma = _gpu_root(data, TensorHashConfig(chunk_size=chunk_size))
        sync = _gpu_root(data, TensorHashConfig(chunk_size=chunk_size, sync_loads=True))
        assert tma == sync, f"chunk_size={chunk_size}"


@pytest.mark.slow
def test_plus_stats_sync_loads_matches_tma():
    """The activation commitment (stats=None) is digest-identical on the sync
    path -- the verifier could run either."""
    m, k = 257, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    outputs = {}
    for name, config in (("tma", TensorHashConfig()), ("sync", TensorHashConfig(sync_loads=True))):
        buffers = _buffers(m, k, _KEY, blake3(b"cb").digest(), config)
        buffers["stats"] = None
        _launch(codes, scales, buffers)
        torch.cuda.synchronize()
        outputs[name] = buffers
    for field in ("root_codes", "root_scales", "a_keys"):
        assert torch.equal(outputs["tma"][field], outputs["sync"][field]), field


def test_invalid_chunk_size_rejected():
    for chunk_size in (96, 32, 8192):  # not a block multiple; below/above bounds
        with pytest.raises(ValueError, match="chunk_size"):
            TensorHashConfig(chunk_size=chunk_size)
    # Loads must divide the leaf (the default 128 does not divide 192).
    with pytest.raises(ValueError, match="thread_load_size"):
        TensorHashConfig(chunk_size=192)
    with pytest.raises(ValueError, match="thread_load_size"):
        TensorHashConfig(chunk_size=64, thread_load_size=128)
    TensorHashConfig(chunk_size=192, thread_load_size=64)  # valid
    # Reject non-int numeric values and bools before range checks.
    with pytest.raises(ValueError, match="chunk_size must be an int"):
        TensorHashConfig(chunk_size=1024.0)
    with pytest.raises(ValueError, match="chunk_size must be an int"):
        TensorHashConfig(chunk_size=True)


def test_small_chunk_capacity_guard():
    """A payload whose small-leaf root count exceeds the single-CTA final
    reduction is rejected instead of silently corrupting."""
    config = _chunk_config(64)
    n = 300 * 1024 * 1024  # 4.9M leaves -> 150 MT blocks > 128 threads
    data = torch.zeros(n, dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(tensor_hash_scratchpad_bytes(n, config), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="final reduction"):
        tensor_hash(data, key, root, roots, config=config)


def test_plus_stats_non_default_chunk_size(experimental_leaves):
    """The activation commitment runs at non-default leaves (the leaf is a
    digest-changing protocol decision): each blob's root matches
    pearl_mining.MerkleTree, TMA and sync agree on the whole chain, and the
    fused partials are leaf- and load-path-invariant (both reductions fold
    the same balanced in-order tree)."""
    chunk_size = 256
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))

    def commit(config):
        buffers = _buffers(m, k, _KEY, blake3(b"cb").digest(), config)
        _launch(codes, scales, buffers)
        torch.cuda.synchronize()
        return buffers

    tma = commit(_chunk_config(chunk_size, sync_loads=False))
    sync = commit(_chunk_config(chunk_size, sync_loads=True))
    codes_bytes = bytes(codes.cpu().view(torch.uint8).reshape(-1).numpy())
    scales_bytes = bytes(scales.cpu().view(torch.uint8).reshape(-1).numpy())
    assert bytes(tma["root_codes"].cpu().numpy()) == _reference_root(codes_bytes, chunk_size)
    assert bytes(tma["root_scales"].cpu().numpy()) == _reference_root(scales_bytes, chunk_size)
    for field in ("root_codes", "root_scales", "a_keys", "stats"):
        assert torch.equal(tma[field], sync[field]), field

    default = commit(TensorHashConfig())
    assert torch.equal(tma["stats"], default["stats"])


def test_plus_stats_b_non_default_chunk_size(experimental_leaves):
    """The weights commitment forwards the same digest-changing knobs: a
    wrapper that dropped ``chunk_size`` / ``sync_loads`` would emit a
    default-leaf root here while sizing the workspace for the requested one."""
    chunk_size = 256
    n, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(n, k, seed=0))

    def commit(config):
        buffers = _buffers(n, k, _KEY, blake3(b"unused").digest(), config)
        _launch_b(codes, scales, buffers)
        torch.cuda.synchronize()
        return buffers

    tma = commit(_chunk_config(chunk_size, sync_loads=False))
    sync = commit(_chunk_config(chunk_size, sync_loads=True))
    codes_bytes = bytes(codes.cpu().view(torch.uint8).reshape(-1).numpy())
    scales_bytes = bytes(scales.cpu().view(torch.uint8).reshape(-1).numpy())
    assert bytes(tma["root_codes"].cpu().numpy()) == _reference_root(codes_bytes, chunk_size)
    assert bytes(tma["root_scales"].cpu().numpy()) == _reference_root(scales_bytes, chunk_size)
    for field in ("root_codes", "root_scales", "stats"):
        assert torch.equal(tma[field], sync[field]), field

    default = commit(TensorHashConfig())
    assert torch.equal(tma["stats"], default["stats"])
    sync_default = commit(TensorHashConfig(sync_loads=True))
    assert torch.equal(sync_default["stats"], default["stats"])
    for field in ("root_codes", "root_scales"):
        assert torch.equal(sync_default[field], default[field]), field


# Stats-legal leaves: sub-block leaves (cross-thread smem fold) and whole
# multiples of a stats block (per-thread carry). The PR-suite smoke covers
# both reductions and both load paths; 192 exercises chunks that straddle
# stats-block boundaries mid-chunk, 448 the widest sub-block leaf.
_STATS_CHUNKS = (*range(64, 512, 64), *range(512, 4096 + 1, 512))
_STATS_SMOKE_CHUNKS = [(64, True), (192, False), (448, True), (512, False), (1536, True)]


def _assert_stats_leaf_invariant(chunk_size, sync_loads):
    """Roots match pearl_mining.MerkleTree; the fused partials match the
    protocol reference and are bit-identical to the default-leaf kernel's
    (both reductions fold the same balanced in-order tree, so the leaf and
    the load path never move the summation order)."""
    m, k = 257, 512  # partial chunks at wide leaves, ragged multi-CTA at small
    a = _make_input(m, k, seed=6)
    codes, scales = _blobs_gpu(a)
    config = _chunk_config(chunk_size, sync_loads=sync_loads)
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest(), config)
    _launch_b(codes, scales, buffers)
    default = _buffers(m, k, _KEY, blake3(b"unused").digest())
    _launch_b(codes, scales, default)
    torch.cuda.synchronize()

    codes_bytes = bytes(codes.cpu().view(torch.uint8).reshape(-1).numpy())
    scales_bytes = bytes(scales.cpu().view(torch.uint8).reshape(-1).numpy())
    assert bytes(buffers["root_codes"].cpu().numpy()) == _reference_root(codes_bytes, chunk_size)
    assert bytes(buffers["root_scales"].cpu().numpy()) == _reference_root(scales_bytes, chunk_size)
    assert torch.equal(buffers["stats"], default["stats"])
    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(a.cpu()))
    stats = buffers["stats"].cpu()
    assert torch.equal(stats[1::2], ref_stats[1::2])
    torch.testing.assert_close(stats[0::2], ref_stats[0::2], rtol=1e-6, atol=0)


@pytest.mark.parametrize("chunk_size,sync_loads", _STATS_SMOKE_CHUNKS, ids=lambda value: str(value))
def test_stats_chunk_size_smoke(chunk_size, sync_loads, experimental_leaves):
    _assert_stats_leaf_invariant(chunk_size, sync_loads)


@pytest.mark.slow
@pytest.mark.parametrize("sync_loads", [False, True], ids=["tma", "sync"])
@pytest.mark.parametrize("chunk_size", _STATS_CHUNKS)
def test_stats_chunk_size_matrix(chunk_size, sync_loads, experimental_leaves):
    _assert_stats_leaf_invariant(chunk_size, sync_loads)


def test_stats_alignment_matches_the_prep_consumers():
    """The producer's ``stats`` contract is the consumers': ``noisy_quant`` /
    ``noisy_quant_b`` fetch a row's pairs with LDG.128, so a pooled slice this
    accepted at fp32x2 would be rejected one stage later."""
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    pool = torch.zeros(2 * (m * k // 512) + 4, dtype=torch.float32, device="cuda")

    # Two fp32 in: contiguous and 8B-aligned, but not 16B-aligned.
    buffers["stats"] = pool[2:-2]
    with pytest.raises(ValueError, match="align"):
        _launch_b(codes, scales, buffers)

    buffers["stats"] = pool[4:]
    _launch_b(codes, scales, buffers)
    torch.cuda.synchronize()
    assert torch.count_nonzero(buffers["stats"]) > 0


def test_stats_reject_ragged_leaf():
    """A leaf above one stats block that is not a whole multiple splits
    blocks across threads at ragged offsets; neither reduction supports it."""
    codes = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(256, dtype=torch.uint8, device="cuda")
    stats = torch.zeros(16, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="stats requires"):
        tensor_hash_launch(
            codes,
            key,
            root,
            roots,
            threads_per_block=128,
            num_stages=2,
            leaves_per_mt_block=256,
            thread_load_size=64,
            chunk_size=576,
            stats=stats,
            scales=scales,
        )


def test_stats_operand_contract_on_the_single_blob_path():
    """Malformed stats companions (wrong device, non-contiguous, wrong dtype
    or size) are rejected before the raw byte views reach the fused kernel."""
    codes = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(256, dtype=torch.uint8, device="cuda")
    stats = torch.zeros(16, dtype=torch.float32, device="cuda")

    def launch(scales=scales, stats=stats):
        tensor_hash_launch(
            codes,
            key,
            root,
            roots,
            threads_per_block=128,
            num_stages=2,
            leaves_per_mt_block=256,
            thread_load_size=64,
            stats=stats,
            scales=scales,
        )

    with pytest.raises(TypeError, match="scales must be a torch.Tensor"):
        launch(scales=[1, 2])
    with pytest.raises(ValueError, match="scales must be on the same CUDA device"):
        launch(scales=scales.cpu())
    with pytest.raises(ValueError, match="scales must be contiguous"):
        launch(scales=torch.zeros(2048, dtype=torch.uint8, device="cuda")[::2])
    with pytest.raises(ValueError, match="stats must be float32"):
        launch(stats=stats.view(torch.int32))
    with pytest.raises(ValueError, match="pair per stats block"):
        launch(stats=torch.zeros(8, dtype=torch.float32, device="cuda"))
    launch()  # the well-formed contract still launches
    torch.cuda.synchronize()


def test_stats_reject_wrong_scales_dtype():
    """An fp32 scales tensor with the right byte count must not be silently
    reinterpreted as packed BF16."""
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(tensor_hash_scratchpad_bytes(m * k), dtype=torch.uint8, device="cuda")
    stats = torch.zeros(2 * (m * k // 512), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="packed BF16"):
        tensor_hash_launch(
            codes.view(torch.uint8).reshape(-1),
            key,
            root,
            roots,
            threads_per_block=128,
            num_stages=2,
            leaves_per_mt_block=256,
            stats=stats,
            scales=scales.view(torch.float32),  # same bytes, wrong dtype
        )


def test_commitment_rejects_aliased_operands():
    """Shape-valid operands that share storage must be rejected: the pair
    launch writes the codes root and then the scales root (a shared digest
    tensor silently corrupts the A keys), and the grid writes the roots workspace
    and stats concurrently."""
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    buffers["root_scales"] = buffers["root_codes"]
    with pytest.raises(ValueError, match="must not overlap"):
        _launch_b(codes, scales, buffers)

    # A stats view over the roots workspace: both correctly shaped, one pool.
    pool = torch.zeros(512, dtype=torch.uint8, device="cuda")
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    buffers["roots"] = pool[: tensor_hash_workspace_bytes(m, k)]
    buffers["stats"] = pool[: 8 * (m * k // 512)].view(torch.float32)
    with pytest.raises(ValueError, match="must not overlap"):
        _launch_b(codes, scales, buffers)

    # The finalize operands against a read-only input: the A keys aliasing
    # the key (or seed_b a digest) would overwrite the caller's material.
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    buffers["a_keys"] = torch.cat([buffers["key"], buffers["key"], buffers["key"]])
    buffers["key"] = buffers["a_keys"][:32]
    with pytest.raises(ValueError, match="must not overlap"):
        _launch(codes, scales, buffers)
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    buffers["seed_b"] = buffers["root_codes"]
    with pytest.raises(ValueError, match="must not overlap"):
        _launch(codes, scales, buffers)


def test_commitment_accepts_disjoint_slices_of_one_pool():
    """Pooling stays supported: operands carved from one allocation pass as
    long as their byte ranges are disjoint (only true overlap is rejected)."""
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    workspace_bytes = tensor_hash_workspace_bytes(m, k)
    stats_bytes = 8 * (m * k // 512)
    pool = torch.zeros(workspace_bytes + stats_bytes + 3 * 32, dtype=torch.uint8, device="cuda")
    buffers = _buffers(m, k, _KEY, blake3(b"unused").digest())
    offset = 0

    def carve(num_bytes):
        nonlocal offset
        slice_ = pool[offset : offset + num_bytes]
        offset += num_bytes
        return slice_

    buffers["roots"] = carve(workspace_bytes)
    buffers["stats"] = carve(stats_bytes).view(torch.float32)
    buffers["root_codes"] = carve(32)
    buffers["root_scales"] = carve(32)
    _launch_b(codes, scales, buffers)
    torch.cuda.synchronize()
    comm = commit_planes([codes.cpu(), scales.cpu()], _KEY, _LEAF)
    assert bytes(buffers["root_codes"].cpu().numpy()) == comm.parts[0].tree.root
    assert bytes(buffers["root_scales"].cpu().numpy()) == comm.parts[1].tree.root


def test_single_flight_compile_runs_the_factory_once():
    """A cold key raced by two threads runs the factory once and hands both
    the same object: a losing duplicate would be a second CUDA module owned
    only by a launch-local variable, free to unload mid-launch."""
    from pearl_gemm._utils._compile import single_flight_compile

    calls = []
    entered = threading.Event()
    release = threading.Event()

    @single_flight_compile
    def build(key):
        calls.append(key)
        entered.set()
        assert release.wait(timeout=30)
        return object()

    results = [None, None]
    first = threading.Thread(target=lambda: results.__setitem__(0, build("cold")))
    second = threading.Thread(target=lambda: results.__setitem__(1, build("cold")))
    first.start()
    assert entered.wait(timeout=30)  # the factory is mid-flight...
    second.start()  # ...when the second caller misses the cache
    release.set()
    first.join(timeout=30)
    second.join(timeout=30)
    assert calls == ["cold"]
    assert results[0] is not None
    assert results[0] is results[1]


def _run_cold_compile_race():
    faulthandler.enable()
    faulthandler.dump_traceback_later(60)
    config = TensorHashConfig(
        threads_per_block=256, num_stages=4, thread_load_size=128, leaves_per_mt_block=512
    )
    data = _data_bytes(200_000)
    expected = _blake3_1024(data)
    barrier = threading.Barrier(2)
    results = [None, None]
    errors = [None, None]

    def worker(idx):
        try:
            stream = torch.cuda.Stream()
            barrier.wait(timeout=120)
            with torch.cuda.stream(stream):
                results[idx] = _gpu_root(data, config)
        except Exception as error:  # noqa: BLE001 -- re-raised by the main thread
            errors[idx] = error

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
    assert errors == [None, None]
    assert results == [expected, expected]

    next_data = _data_bytes(4096)
    next_config = TensorHashConfig(threads_per_block=512, num_stages=3)
    assert _gpu_root(next_data, next_config) == _blake3_1024(next_data)
    faulthandler.cancel_dump_traceback_later()


def test_cold_compile_race_shares_one_module():
    """Concurrent first launches preserve digests and allow a later module to initialize."""
    process = multiprocessing.get_context("spawn").Process(target=_run_cold_compile_race)
    process.start()
    try:
        process.join(timeout=90)
        assert not process.is_alive(), "concurrent CUDA library initialization hung"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)


def test_smem_estimator_brackets_the_launch_boundary():
    """The largest-footprint config the shared estimator admits must launch,
    and the smallest it rejects must fail, so tune-space generation and
    record legality track the device cap from both sides."""
    admitted = TensorHashConfig(threads_per_block=512, num_stages=3, thread_load_size=128)
    rejected = TensorHashConfig(threads_per_block=512, num_stages=4, thread_load_size=128)
    assert _merkle_host.tensor_hash_smem_fits(512, 3, 128)
    assert not _merkle_host.tensor_hash_smem_fits(512, 4, 128)
    data = _data_bytes(4096)
    assert _gpu_root(data, admitted) == _blake3_1024(data)
    with pytest.raises(Exception):  # noqa: B017 -- the DSL's smem-cap error type is its own
        _gpu_root(data, rejected)


def test_single_blob_sync_stats_match_tma():
    """The single-blob launcher fuses stats on the sync path too, at wide and
    sub-block leaves (the pair kernels are covered by the plus-stats tests)."""
    m, k = 64, 512
    codes, scales = _blobs_gpu(_exact_stats_input(m, k))
    data = codes.view(torch.uint8).reshape(-1)
    scales_blob = scales.view(torch.uint8).reshape(-1)
    key = torch.frombuffer(bytearray(_KEY), dtype=torch.uint8).cuda()
    ref_stats = _reference_chunk_stats(PrequantMatrix.encode(_exact_stats_input(m, k).cpu()))

    for chunk_size in (256, 1024):
        outputs = {}
        for sync_loads in (False, True):
            root = torch.zeros(32, dtype=torch.uint8, device="cuda")
            roots = torch.zeros(
                tensor_hash_scratchpad_bytes(data.numel(), _chunk_config(chunk_size)),
                dtype=torch.uint8,
                device="cuda",
            )
            stats = torch.zeros(2 * (m * k // 512), dtype=torch.float32, device="cuda")
            tensor_hash_launch(
                data,
                key,
                root,
                roots,
                threads_per_block=128,
                num_stages=2,
                leaves_per_mt_block=256,
                thread_load_size=128,
                chunk_size=chunk_size,
                sync_loads=sync_loads,
                stats=stats,
                scales=scales_blob,
            )
            torch.cuda.synchronize()
            outputs[sync_loads] = (root, stats)
            torch.testing.assert_close(stats.cpu(), ref_stats, rtol=0, atol=0)
        assert torch.equal(outputs[False][0], outputs[True][0])
        assert torch.equal(outputs[False][1], outputs[True][1])


# Roots-kernel optimization coverage.


def test_fused_tail_digests():
    """Fused-tail and separate-stage payloads match the BLAKE3 reference."""
    config = TensorHashConfig()
    for length in (100, 1024, 128 * 1024, 128 * 1024 + 1, 200 * 1024, 256 * 1024, 300 * 1024):
        data = _data_bytes(length)
        assert _gpu_root(data, config) == _blake3_1024(data), f"length={length}"


def test_fused_tail_concurrent_streams():
    """Overlapping fused-tail launches preserve every per-stream digest."""
    iters = 64
    hold_cycles = 200_000_000  # ~0.1 s, ample time to queue both streams
    config = TensorHashConfig()
    key = torch.frombuffer(bytearray(_KEY), dtype=torch.uint8).cuda()
    cases = []
    for length in (200 * 1024, 250 * 1024):  # 2 CTAs each -> fused tail
        data = _data_bytes(length)
        cases.append(
            {
                "data": torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda(),
                "digests": torch.empty(32 * iters, dtype=torch.uint8, device="cuda"),
                "roots": torch.empty(
                    tensor_hash_scratchpad_bytes(length, config),
                    dtype=torch.uint8,
                    device="cuda",
                ),
                "expected": _blake3_1024(data),
                "stream": torch.cuda.Stream(),
            }
        )
    # Compile (and create this device's counters) before the overlap, so the
    # timed region holds nothing but launches.
    for case in cases:
        tensor_hash(case["data"], key, case["digests"][:32], case["roots"], config=config)
    torch.cuda.synchronize()

    # Hold both streams while the whole run is queued: submitted one at a time
    # the launches never overlap, and a shared ticket then goes unnoticed.
    for case in cases:
        with torch.cuda.stream(case["stream"]):
            torch.cuda._sleep(hold_cycles)
    # Each iteration keeps its own digest slot; a corrupted middle launch would
    # otherwise be overwritten by a healthy last one.
    for i in range(iters):
        for case in cases:
            with torch.cuda.stream(case["stream"]):
                tensor_hash(
                    case["data"],
                    key,
                    case["digests"][32 * i : 32 * (i + 1)],
                    case["roots"],
                    config=config,
                )
    torch.cuda.synchronize()

    for case in cases:
        digests = bytes(case["digests"].cpu().numpy())
        for i in range(iters):
            assert digests[32 * i : 32 * (i + 1)] == case["expected"], f"iteration {i}"


def test_fused_tail_many_sequential_streams():
    """Fused-tail hashes keep working after many streams have come and gone.

    Counter slots are never reclaimed, so a fixed slot budget would start
    rejecting valid calls once enough distinct stream handles had been seen --
    torch cycles a bounded pool of them, so this is reachable in a loop.
    """
    config = TensorHashConfig()
    data = _data_bytes(200 * 1024)  # 2 CTAs -> fused tail
    data_gpu = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    key = torch.frombuffer(bytearray(_KEY), dtype=torch.uint8).cuda()
    root = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(len(data), config), dtype=torch.uint8, device="cuda"
    )
    expected = _blake3_1024(data)
    for i in range(40):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            tensor_hash(data_gpu, key, root, roots, config=config)
        torch.cuda.synchronize()
        assert bytes(root.cpu().numpy()) == expected, f"stream {i}"


@pytest.mark.slow
def test_fused_tail_dual_pipeline():
    """The dual-pipeline fused tail matches the BLAKE3 reference."""
    data = _data_bytes(2 * 512 * 1024)
    assert _gpu_root(data, TensorHashConfig(threads_per_block=512)) == _blake3_1024(data)


def test_concurrent_ragged_fold():
    """Ragged last-CTA leaf counts preserve the BLAKE3 digest."""
    config = TensorHashConfig()
    for rem in (3, 97, 127):  # popcounts 2, 3, 7
        data = _data_bytes((128 + rem) * 1024 - 500)
        assert _gpu_root(data, config) == _blake3_1024(data), f"rem={rem}"


@pytest.mark.slow
def test_concurrent_fold_in_later_stages():
    """Ragged stage-2 and stage-3 reductions preserve the BLAKE3 digest."""
    config = TensorHashConfig()
    # 259 CTAs -> stage 2's last group reduces 3 roots.
    stage2 = _data_bytes(259 * 128 * 1024 - 700)
    assert _gpu_root(stage2, config) == _blake3_1024(stage2)
    # 515 CTAs -> 3 MT blocks -> stage 3 reduces 3 roots.
    stage3 = _data_bytes(515 * 128 * 1024 - 700)
    assert _gpu_root(stage3, config) == _blake3_1024(stage3)


@pytest.mark.parametrize("chunks_per_thread", [2, 8])
def test_coarsening_smoke(chunks_per_thread):
    """Coarsening and its k=1 fallback preserve the BLAKE3 digest."""
    config = TensorHashConfig(chunks_per_thread=chunks_per_thread)
    group = 1024 * chunks_per_thread
    # Eligible, ineligible (silent k=1 fallback), and the eligibility floor of
    # two leaf groups, where ROOT sits directly above two private folds.
    for length in (group * 300, group * 300 + 512, group * 2):
        data = _data_bytes(length)
        assert _gpu_root(data, config) == _blake3_1024(data), f"length={length}"


@pytest.mark.slow
@pytest.mark.parametrize("sync_loads", [False, True], ids=["tma", "sync"])
@pytest.mark.parametrize("chunk_size", [1024, 2048, 4096])
@pytest.mark.parametrize("chunks_per_thread", [2, 4, 8])
def test_coarsening_matrix(chunks_per_thread, chunk_size, sync_loads):
    """Coarsening preserves pearl-blake3 digests across load paths."""
    config = _chunk_config(chunk_size, chunks_per_thread=chunks_per_thread, sync_loads=sync_loads)
    # 131 groups -> two CTAs, 3 leaf groups in the last one (non-power-of-2).
    data = _data_bytes(chunk_size * chunks_per_thread * 131)
    assert _gpu_root(data, config) == _reference_root(data, chunk_size)


@pytest.mark.slow
def test_coarsening_across_geometries():
    """256- and dual-pipeline 512-thread CTAs keep the coarsened digest."""
    for threads_per_block in (256, 512):
        config = TensorHashConfig(threads_per_block=threads_per_block, chunks_per_thread=4)
        data = _data_bytes(1024 * 4 * (2 * threads_per_block + 3))
        assert _gpu_root(data, config) == _blake3_1024(data), f"tpb={threads_per_block}"


def test_stats_coarsened_matches_k1(dispatched_keys):
    """Coarsening rides the stats path at whole-block leaves: bit-identical
    partials and the reference commitment, with the coarsened variant really
    dispatched (the scales blob sits exactly at the two-group floor)."""
    m, k = 32, 512
    codes, scales = _blobs_gpu(_make_input(m, k, seed=5))
    outputs = {}
    for chunks_per_thread in (1, 2):
        config = TensorHashConfig(chunks_per_thread=chunks_per_thread)
        buffers = _buffers(m, k, _KEY, blake3(b"cb").digest(), config)
        _launch(codes, scales, buffers)
        torch.cuda.synchronize()
        outputs[chunks_per_thread] = buffers
    for field in ("root_codes", "root_scales", "a_keys", "stats"):
        assert torch.equal(outputs[1][field], outputs[2][field]), field
    _assert_commitment(outputs[2], codes, scales, _KEY, blake3(b"cb").digest())
    assert {key.chunks_per_thread for key in dispatched_keys} == {1, 2}


@pytest.mark.slow
def test_pair_commitment_coarsened():
    """Coarsened and stats-fused pair commitments match the reference roots
    and keep the partials bit-identical to the uncoarsened kernel's."""
    m, k = 4096, 8192
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    c_b = blake3(b"cb").digest()
    comm = commit_planes([codes.cpu(), scales.cpu()], _KEY, _LEAF)
    baseline = _buffers(m, k, _KEY, c_b)
    _launch(codes, scales, baseline)

    for chunks_per_thread, with_stats in ((4, False), (4, True), (8, False), (8, True)):
        config = TensorHashConfig(chunks_per_thread=chunks_per_thread)
        buffers = _buffers(m, k, _KEY, c_b, config)
        if not with_stats:
            buffers["stats"] = None
        _launch(codes, scales, buffers)
        torch.cuda.synchronize()
        label = f"k={chunks_per_thread} stats={with_stats}"
        assert bytes(buffers["root_codes"].cpu().numpy()) == comm.parts[0].tree.root, label
        assert bytes(buffers["root_scales"].cpu().numpy()) == comm.parts[1].tree.root, label
        if with_stats:
            assert torch.equal(buffers["stats"], baseline["stats"]), label


def test_invalid_chunks_per_thread_rejected():
    for chunks_per_thread in (0, 3, 16):
        with pytest.raises(ValueError, match="chunks_per_thread"):
            TensorHashConfig(chunks_per_thread=chunks_per_thread)


def test_coarsening_field_is_last_positionally():
    """``chunks_per_thread`` and ``mad_rot`` sit after the pre-existing
    fields, so positional construction keeps binding them as before."""
    config = TensorHashConfig(128, 2, 256, 128, 1024, True)
    assert (config.sync_loads, config.chunks_per_thread, config.mad_rot) == (True, 1, False)


def test_launchers_reject_non_tensor_operands():
    """The centralized TypeError, not an AttributeError from the length reads
    the coarsening gate and the workspace split do first."""
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    out = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(64, dtype=torch.uint8, device="cuda")
    launch_kwargs = {"threads_per_block": 128, "num_stages": 2, "leaves_per_mt_block": 256}
    with pytest.raises(TypeError, match="data must be a torch.Tensor"):
        tensor_hash_launch([1, 2, 3], key, out, roots, **launch_kwargs)
    with pytest.raises(TypeError, match="scales must be a torch.Tensor"):
        tensor_hash_pair_launch(
            torch.zeros(512, dtype=torch.uint8, device="cuda"),
            None,
            key,
            out,
            out,
            roots,
            **launch_kwargs,
        )


# CLMAD rotate-offload coverage.


@pytest.fixture
def dispatched_keys(monkeypatch):
    """The compile-cache key of every Merkle launch in the test.

    ``mad_rot`` is digest-invariant, so a dropped flag or broken gate still
    yields bit-exact digests; only the dispatched variant key can tell.
    Spying on the compile entry points also sees LRU cache hits, which
    inspecting the cache contents after the fact cannot."""
    keys = []

    def install(name, key_type):
        original = getattr(_merkle_host, name)

        def spy(*args):
            keys.append(key_type(*args))
            return original(*args)

        monkeypatch.setattr(_merkle_host, name, spy)

    install("_compile_variant", _merkle_host._VariantKey)
    install("_compile_pair_variant", _merkle_host._PairVariantKey)
    return keys


def test_mad_rot_digest_ungated(monkeypatch, dispatched_keys):
    """The offloaded kernel is bit-exact on small payloads (gate lifted):
    sub-chunk, partial chunk, multi-CTA ragged, and a coarsened payload --
    and the offloaded variant really is the one dispatched."""
    monkeypatch.setattr(_blake3, "MAD_ROT_MIN_BYTES", 0)
    config = TensorHashConfig(mad_rot=True)
    for length in (100, 1025, 200 * 1024 + 5):
        data = _data_bytes(length)
        assert _gpu_root(data, config) == _blake3_1024(data), f"length={length}"
    coarse = _data_bytes(1024 * 4 * 300)
    coarse_config = TensorHashConfig(mad_rot=True, chunks_per_thread=4)
    assert _gpu_root(coarse, coarse_config) == _blake3_1024(coarse)
    assert {key.mad_rot for key in dispatched_keys} == {True}


def test_mad_rot_below_gate_stays_on_shf(dispatched_keys):
    """Under the natural gate the plain-SHF variant dispatches (and the
    digest is trivially unchanged)."""
    data = _data_bytes(200 * 1024)
    assert _gpu_root(data, TensorHashConfig(mad_rot=True)) == _blake3_1024(data)
    assert {key.mad_rot for key in dispatched_keys} == {False}


@pytest.mark.slow
def test_mad_rot_digest_above_gate(dispatched_keys):
    """A payload above the natural size gate takes the offloaded kernel."""
    data = _data_bytes(_blake3.MAD_ROT_MIN_BYTES + 12345)
    assert _gpu_root(data, TensorHashConfig(mad_rot=True)) == _blake3_1024(data)
    assert {key.mad_rot for key in dispatched_keys} == {True}


@pytest.mark.slow
def test_mad_rot_commitment(dispatched_keys):
    """The activation commitment with the offload under the natural gate:
    the 32 MiB codes blob offloads, the 8 MiB scales blob stays on SHF --
    the exact mixed pair variant production would compile."""
    m, k = 4096, 8192
    assert m * k >= _blake3.MAD_ROT_MIN_BYTES > m * k // 4
    codes, scales = _blobs_gpu(_make_input(m, k, seed=0))
    c_b = blake3(b"cb").digest()
    buffers = _buffers(m, k, _KEY, c_b, TensorHashConfig(mad_rot=True))
    _launch(codes, scales, buffers)
    torch.cuda.synchronize()
    _assert_commitment(buffers, codes, scales, _KEY, c_b)
    assert {(key.codes_mad_rot, key.scales_mad_rot) for key in dispatched_keys} == {(True, False)}


@pytest.mark.parametrize("chunk_size", [0, 1024.0], ids=["zero", "non-integer"])
def test_pair_launch_rejects_bad_chunk_size(chunk_size):
    """Rejected before the workspace sizing, which would otherwise divide by
    (or slice with) the bad value first."""
    codes = torch.zeros(512, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(128, dtype=torch.uint8, device="cuda")
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root_codes = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root_scales = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(128, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="chunk_size"):
        tensor_hash_pair_launch(
            codes,
            scales,
            key,
            root_codes,
            root_scales,
            roots,
            threads_per_block=128,
            num_stages=2,
            leaves_per_mt_block=256,
            chunk_size=chunk_size,
        )
