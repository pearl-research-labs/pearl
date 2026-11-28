//! CPU B preparation ([`zk_pow::v4::api::cpu_b`], and an expert-parallel
//! rank's B subtree hashes) over caller-owned buffers.
//!
//! Every function writes into C-contiguous buffers the caller allocated (numpy
//! views of torch tensors, `array.array`, ...), releases the GIL, and runs on a
//! pool of `threads` workers: the caller sizes the work to the cores it was
//! given, independent of the module's global pool.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use zk_pow::v4::api::cpu_b::{self, BRows, DeviceArithmetic};

/// One pool per thread count. Workers inherit the creating thread's CPU
/// affinity, so a caller pins itself before its first call.
static POOLS: LazyLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = LazyLock::new(Default::default);

fn pool(threads: usize) -> PyResult<Arc<ThreadPool>> {
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pool) = pools.get(&threads) {
        return Ok(pool.clone());
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("cpu-b-{i}"))
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pools.entry(threads).or_insert(Arc::new(pool)).clone())
}

fn contiguous<T: Element>(buffer: &PyBuffer<T>, name: &str) -> PyResult<()> {
    if buffer.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn view<'a, T: Element>(buffer: &'a PyBuffer<T>, name: &str) -> PyResult<&'a [T]> {
    contiguous(buffer, name)?;
    // SAFETY: the buffer is C-contiguous with `item_count` items of `T` (its
    // format matched `T` on export), and the export outlives the borrow.
    Ok(unsafe { std::slice::from_raw_parts(buffer.buf_ptr().cast(), buffer.item_count()) })
}

#[allow(clippy::mut_from_ref)]
fn view_mut<'a, T: Element>(buffer: &'a PyBuffer<T>, name: &str) -> PyResult<&'a mut [T]> {
    contiguous(buffer, name)?;
    if buffer.readonly() {
        return Err(PyValueError::new_err(format!("{name} must be writable")));
    }
    // SAFETY: as in `view`; the buffer is writable, and the caller hands it to
    // this call alone.
    Ok(unsafe { std::slice::from_raw_parts_mut(buffer.buf_ptr().cast(), buffer.item_count()) })
}

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Fills `lines` (`count x RANK` E4M3 bytes) with the noise lines
/// `0..count` of one factor under the noise-line key `key`, at the address
/// `(side, factor)`.
#[pyfunction]
#[pyo3(signature = (key, side, factor, lines, threads))]
fn cpu_b_noise_lines(
    py: Python<'_>,
    key: [u8; 32],
    side: u8,
    factor: u8,
    lines: PyBuffer<u8>,
    threads: usize,
) -> PyResult<()> {
    let pool = pool(threads)?;
    let lines = view_mut(&lines, "lines")?;
    py.allow_threads(|| pool.install(|| cpu_b::noise_lines(&key, side, factor, lines)))
        .map_err(value_error)
}

/// Fills `b_prime` (`n x k` E4M3 bytes) and `mid` (`n x RANK` BF16 bits) for
/// every B row.
///
/// `codes` (`n x k` int8) and `scales` (`n x k/8` BF16 bits) are the rows'
/// prequant planes; `alpha`, `beta` and `l2` (`n` BF16 bits each) their row
/// scales; `e_b` (`n x RANK`), `f_a` and `f_b` (`RANK x k`) the job's E4M3
/// lines. `window_bits` and `(strip_g, strip_g_small_rms)` are the committed
/// device's noise-dot window and strip gains.
#[pyfunction]
#[pyo3(signature = (
    codes, scales, alpha, beta, l2, e_b, f_a, f_b, window_bits, strip_g, strip_g_small_rms, b_prime, mid, threads
))]
#[allow(clippy::too_many_arguments)]
fn cpu_b_prepare_rows(
    py: Python<'_>,
    codes: PyBuffer<i8>,
    scales: PyBuffer<u16>,
    alpha: PyBuffer<u16>,
    beta: PyBuffer<u16>,
    l2: PyBuffer<u16>,
    e_b: PyBuffer<u8>,
    f_a: PyBuffer<u8>,
    f_b: PyBuffer<u8>,
    window_bits: u16,
    strip_g: f32,
    strip_g_small_rms: f32,
    b_prime: PyBuffer<u8>,
    mid: PyBuffer<u16>,
    threads: usize,
) -> PyResult<()> {
    let pool = pool(threads)?;
    let rows = BRows {
        codes: view(&codes, "codes")?,
        scales: view(&scales, "scales")?,
        alpha: view(&alpha, "alpha")?,
        beta: view(&beta, "beta")?,
        l2: view(&l2, "l2")?,
        e_b: view(&e_b, "e_b")?,
        f_a: view(&f_a, "f_a")?,
        f_b: view(&f_b, "f_b")?,
    };
    let arithmetic = DeviceArithmetic {
        window_bits,
        strip_g: (strip_g, strip_g_small_rms),
    };
    let b_prime = view_mut(&b_prime, "b_prime")?;
    let mid = view_mut(&mid, "mid")?;
    py.allow_threads(|| pool.install(|| cpu_b::prepare_rows(&rows, arithmetic, b_prime, mid)))
        .map_err(value_error)
}

/// Fills `roots` (32 bytes per subtree) with the chaining values of the
/// `subtree_leaves`-leaf subtrees of the keyed Merkle tree that `data` is
/// made of ([`pearl_blake3::subtree_cvs`]).
///
/// `data` holds whole `chunk_len`-byte leaves of a larger tree, the first
/// being leaf `first_leaf` (which starts a subtree): an expert-parallel
/// rank's rows of a committed B plane, zero-padded to whole leaves.
#[pyfunction]
#[pyo3(signature = (key, data, first_leaf, chunk_len, subtree_leaves, roots, threads))]
#[allow(clippy::too_many_arguments)]
fn cpu_b_subtree_roots(
    py: Python<'_>,
    key: [u8; 32],
    data: PyBuffer<u8>,
    first_leaf: u64,
    chunk_len: usize,
    subtree_leaves: usize,
    roots: PyBuffer<u8>,
    threads: usize,
) -> PyResult<()> {
    let pool = pool(threads)?;
    let data = view(&data, "data")?;
    let roots = view_mut(&roots, "roots")?;
    if chunk_len == 0 || subtree_leaves == 0 {
        return Err(PyValueError::new_err(
            "chunk_len and subtree_leaves must be positive",
        ));
    }
    let subtrees = (data.len() / chunk_len).div_ceil(subtree_leaves);
    if roots.len() != subtrees * 32 {
        return Err(PyValueError::new_err(format!(
            "roots holds {} bytes; {subtrees} subtrees need {}",
            roots.len(),
            subtrees * 32
        )));
    }
    let cvs = py
        .allow_threads(|| {
            pool.install(|| {
                pearl_blake3::subtree_cvs(data, key, first_leaf, chunk_len, subtree_leaves)
            })
        })
        .map_err(value_error)?;
    for (out, cv) in roots.chunks_exact_mut(32).zip(&cvs) {
        out.copy_from_slice(cv);
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CPU_B_RANK", cpu_b::RANK)?;
    m.add("CPU_B_TILE_COLS", cpu_b::TILE_COLS)?;
    m.add_function(wrap_pyfunction!(cpu_b_noise_lines, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_b_prepare_rows, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_b_subtree_roots, m)?)?;
    Ok(())
}
