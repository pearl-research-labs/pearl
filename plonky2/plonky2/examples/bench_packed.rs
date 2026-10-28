// Quick benchmark: scalar vs packed batch Poseidon leaf hashing.
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::hashing::poseidon_hash_batch_no_pad;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;
use plonky2_field::packable::Packable;
use plonky2_field::packed::PackedField;
use plonky2_field::types::Field;

fn main() {
    type F = GoldilocksField;
    type P = <F as Packable>::Packing;
    println!("Packing WIDTH = {}", P::WIDTH);

    let widths = [237usize, 222, 339];
    let rows_count = 1 << 18;
    for &w in &widths {
        let rows: Vec<Vec<F>> = (0..rows_count)
            .map(|r| {
                (0..w)
                    .map(|c| F::from_canonical_u64(r as u64 * 31 + c as u64))
                    .collect()
            })
            .collect();

        let t0 = std::time::Instant::now();
        let scalar: Vec<_> = rows.iter().map(|r| PoseidonHash::hash_no_pad(r)).collect();
        let t_scalar = t0.elapsed();

        let t0 = std::time::Instant::now();
        let mut packed_out = Vec::with_capacity(rows.len());
        for block in rows.chunks(P::WIDTH) {
            let refs: Vec<&[F]> = block.iter().map(|r| &r[..]).collect();
            for h in poseidon_hash_batch_no_pad::<F, P>(&refs) {
                packed_out.push(h);
            }
        }
        let t_packed = t0.elapsed();

        // verify equal
        assert_eq!(scalar.len(), packed_out.len());
        for (a, b) in scalar.iter().zip(packed_out.iter()) {
            assert_eq!(a, b);
        }
        println!(
            "width={} rows={}: scalar {:?}  packed {:?}  speedup {:.2}x",
            w,
            rows_count,
            t_scalar,
            t_packed,
            t_scalar.as_secs_f64() / t_packed.as_secs_f64()
        );
    }
}
