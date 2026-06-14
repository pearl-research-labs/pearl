use crate::v1::{
    api::proof_utils::CompiledPublicParams,
    circuit::{
        pearl_noise::MMSlice,
        pearl_program::{JACKPOT_SIZE, LROT_PER_TILE},
    },
};

pub fn compute_jackpot(
    public_params: &CompiledPublicParams,
    secret_a: &[Vec<i8>],
    secret_b: &[Vec<i8>],
    noise: &MMSlice,
) -> [u32; JACKPOT_SIZE] {
    let h = public_params.h;
    let w = public_params.w;
    let k = public_params.k;
    let r = public_params.r;
    let mut jackpot = vec![vec![0i32; w]; h];
    let mut jackpot_msg: [u32; 16] = [0; JACKPOT_SIZE];
    for ll in (r..=k).step_by(r) {
        for u in 0..h {
            for v in 0..w {
                for l in ll - r..ll {
                    jackpot[u][v] +=
                        (secret_a[u][l] as i32 + noise.a[u][l] as i32) * (secret_b[v][l] as i32 + noise.b[v][l] as i32);
                }
            }
        }
        let xored_tile = jackpot.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
        let tid = (ll / r - 1) % JACKPOT_SIZE;
        jackpot_msg[tid] = jackpot_msg[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
    }
    jackpot_msg
}
