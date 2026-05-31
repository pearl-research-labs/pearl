/* pearl-c-driver — stratum client PLACEHOLDER.
 *
 * The SKELETON does NOT need stratum; bench mode drives synthetic
 * pow_target. This file exists to mark the next implementation
 * site, mirroring the structure agreed with the orchestrator.
 *
 * Reference protocol (alphapool v1.5):
 *   1. open TCP to us1.alphapool.tech:5566
 *   2. send `pearl.challenge.subscribe` -> receive challenge bytes
 *   3. shell out to /home/pearl-deploy/pearl-stratum-v15/src/pearl_stratum/pearl_challenge_solver_simd
 *      with the challenge -> stdin response
 *   4. send `pearl.challenge.submit` with response
 *   5. send `mining.subscribe` with worker UA
 *   6. send `mining.authorize` with decoy wallet
 *      `prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg`
 *   7. handle `pearl.set_mining_params` (M, N, K, R, batch) -> reconfigure buffers
 *   8. handle `mining.notify` -> {pow_target, pow_key} -> resume hot loop
 *   9. on host_signal_header_pinned `status==FOUND` -> mining.submit
 *
 * Estimated effort: ~600 LOC in pure C (cJSON + posix sockets). The pearl
 * challenge solver is already a standalone C SIMD binary (1 subprocess call,
 * ~20 ms once per session). The hardest piece is share derivation — the
 * Python driver pulls this from `pearl_gemm.commitment_hash_from_merkle_roots`
 * etc. (see _miner_driver_sm89_r128_w19r.py imports). Porting share derivation
 * to C is ~1-2 days.
 *
 * For the SKELETON we leave this stub; bench mode does not require it.
 */

/* Intentionally empty. Bench mode (--bench) drives the hot loop without
 * pool integration. See src/main.cu. */
