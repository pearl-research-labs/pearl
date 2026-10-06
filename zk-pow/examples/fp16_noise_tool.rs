//! Seed-exact FP16 noised-operand tool for offline calibration
//! (`docs/fp16_scheme/validation/calibration.py --rust-noise`).
//!
//! Drives the REAL consensus noise pipeline
//! ([`zk_pow::v5::circuit::driver::fp16_noised_operands`]: root-derived seeds ->
//! keyed-BLAKE3 noise lines -> `noisy_quantize`), so the calibration harness can
//! compute `f_bp`/`rho` over the exact bytes the verifier would, instead of a
//! shape-faithful reimplementation of the noise.
//!
//! Protocol (stdin, whitespace-separated integers):
//!   h w k
//!   a_codes[h*k]          (FP16 bit patterns, u16)
//!   b_codes[w*k]          (FP16 bit patterns, u16)
//! The opening keys and hash ids are fixed (the f_bp/rho distribution does not
//! depend on them, only on the derived noise being the real pipeline's).
//!
//! Output (stdout): `h*k` noised A codes then `w*k` noised B codes, space-separated.
//!
//! Run:  cargo run --release --example fp16_noise_tool < input.txt

use std::io::{self, Read, Write};

use zk_pow::v4::api::public_params::HashId;
use zk_pow::v5::circuit::driver::fp16_noised_operands;

const KEY_A: [u8; 32] = [0x11; 32];
const KEY_B: [u8; 32] = [0x22; 32];
const HASH_ID: HashId = HashId::Blake3Chunk1024;

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut it = input.split_whitespace().map(|t| t.parse::<i64>());
    let mut next = || -> anyhow::Result<i64> {
        it.next().ok_or_else(|| anyhow::anyhow!("unexpected end of input"))?.map_err(Into::into)
    };

    let h = next()? as usize;
    let w = next()? as usize;
    let k = next()? as usize;

    let mut a_codes = Vec::with_capacity(h * k);
    for _ in 0..h * k {
        a_codes.push(next()? as u16);
    }
    let mut b_codes = Vec::with_capacity(w * k);
    for _ in 0..w * k {
        b_codes.push(next()? as u16);
    }

    let (noised_a, noised_b) =
        fp16_noised_operands(h, w, k, &a_codes, &b_codes, KEY_A, KEY_B, HASH_ID, HASH_ID)?;

    let out = io::stdout();
    let mut w_out = io::BufWriter::new(out.lock());
    let strs: Vec<String> = noised_a.iter().chain(noised_b.iter()).map(|v| v.to_string()).collect();
    writeln!(w_out, "{}", strs.join(" "))?;
    Ok(())
}
