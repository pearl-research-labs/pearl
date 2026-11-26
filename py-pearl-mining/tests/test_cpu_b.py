"""CPU B bindings: the buffer contract (bit-exactness is tested in zk-pow and
against the vLLM miner's torch chain)."""

from array import array

import pearl_mining
import pytest

R = pearl_mining.CPU_B_RANK
KEY = bytes(range(32))
ONE_BF16 = 0x3F80


def lines(side, factor, count, threads=2):
    out = array("B", bytes(count * R))
    pearl_mining.cpu_b_noise_lines(KEY, side, factor, out, threads)
    return out


def test_noise_lines_are_addressed_and_thread_independent():
    f_b = lines(1, 1, 300)
    assert f_b == lines(1, 1, 300, threads=7)
    assert f_b[: 10 * R] == lines(1, 1, 10)
    assert f_b != lines(0, 1, 300)
    assert all(b & 0x7F != 0x7F for b in f_b)


def test_noise_lines_reject_bad_buffers():
    with pytest.raises(ValueError, match="whole lines"):
        pearl_mining.cpu_b_noise_lines(KEY, 1, 1, array("B", bytes(R + 1)), 1)
    with pytest.raises(ValueError, match="writable"):
        pearl_mining.cpu_b_noise_lines(KEY, 1, 1, bytes(R), 1)
    with pytest.raises(ValueError, match="threads"):
        pearl_mining.cpu_b_noise_lines(KEY, 1, 1, array("B", bytes(R)), 0)


def prepare(n, k, threads=2, **override):
    rows = {
        "codes": array("b", [(i * 37) % 255 - 127 for i in range(n * k)]),
        "scales": array("H", [0x3C00] * (n * k // 8)),
        "alpha": array("H", [ONE_BF16] * n),
        "beta": array("H", [0x3A00] * n),
        "l2": array("H", [0x4300] * n),
        "e_b": lines(1, 0, n),
        "f_a": lines(0, 1, k),
        "f_b": lines(1, 1, k),
    }
    rows.update(override)
    b_prime, mid = array("B", bytes(n * k)), array("H", bytes(2 * n * R))
    pearl_mining.cpu_b_prepare_rows(
        *rows.values(),
        window_bits=26,
        strip_g=0.08,
        strip_g_small_rms=0.1,
        b_prime=b_prime,
        mid=mid,
        threads=threads,
    )
    return b_prime, mid


def test_prepare_rows_fills_both_outputs_independent_of_threads():
    b_prime, mid = prepare(20, 512)
    assert any(b_prime) and any(mid)
    assert (b_prime, mid) == prepare(20, 512, threads=5)


def test_prepare_rows_rejects_bad_shapes():
    with pytest.raises(ValueError, match="multiple of"):
        prepare(4, 128)
    with pytest.raises(ValueError, match="one value per row"):
        prepare(4, 256, beta=array("H", [ONE_BF16] * 3))
