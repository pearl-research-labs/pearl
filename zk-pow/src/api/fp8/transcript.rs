//! FP8/v4 domain-separated BLAKE3 keys, noise seeds and lottery [`Ticket`] values.
//!
//! The caller supplies the proposed header. The proof carries the ancestor
//! header in `JobParams.ancestor_header`; the caller must authenticate it
//! within the proposed header's context window.
//!
//! [`PublicParams::commitment_keys`] and [`PublicParams::noise_seeds`] derive
//! the tree keys and noise seeds from the headers, parameters and commitments:
//!
//! ```text
//! keyA := H_"key-A"(proposed_header)            (A-side tree key, routing/offset keys)
//! keyB := H_"key-B"(ancestor_header)           (B-side tree key)
//! noise_seedB := H_"seed-B"(HB || keyB || pB)
//! noise_seedA := H_"seed-A"(HA || HR || HO || noise_seedB || keyA || pA)   (MoE)
//! noise_seedA := H_"seed-A"(HA || noise_seedB || keyA || pA)               (dense)
//! J := H_"jackpot"(z; noise_seedA)
//! ```
//!
//! `HA`/`HB` are operand commitments; `HR`/`HO` commit to routing and offsets.
//! `pA`/`pB` are the encoded parameter blocks. `z` is the 64-byte folded matmul
//! result, and `J` is its jackpot digest. Concatenation adds no tags or length prefixes.
//!
//! # Domain separation
//!
//! Each role label above has the prefix `pearl/v4/FP8/`. `H_label(message; parent)`
//! first hashes the full label with BLAKE3 to obtain a subkey, then hashes the
//! message with that subkey. The label hash is keyed by `parent` when supplied;
//! otherwise it uses unkeyed BLAKE3. Unkeyed mode is distinct from keyed mode
//! with a fixed key, including an all-zero key.

use pearl_blake3::blake3_digest;

use crate::api::fp8::jackpot_policy::JackpotMessage;
use crate::api::fp8::public_params::PublicParams;
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};

// Test vectors in the `tests` module below were generated independently with
// the Python `blake3` package:
//
//     PARENT = bytes(range(32))
//     default_key = blake3(label).digest()
//     keyed_key   = blake3(label, key=PARENT).digest()
//     hash        = blake3(message, key=default_key_or_keyed_key).digest()

/// Every v4 role label starts with this prefix.
const LABEL_PREFIX: &[u8] = b"pearl/v4/FP8/";

const LABEL_KEY_A: &[u8] = b"pearl/v4/FP8/key-A";
const LABEL_KEY_B: &[u8] = b"pearl/v4/FP8/key-B";
const LABEL_SEED_A: &[u8] = b"pearl/v4/FP8/seed-A";
const LABEL_SEED_B: &[u8] = b"pearl/v4/FP8/seed-B";
const LABEL_ZK_PUBLIC: &[u8] = b"pearl/v4/FP8/zk-public";

/// Line-key label used by [`crate::api::fp8::noise`].
pub(crate) const LABEL_NOISE_LINE: &[u8] = b"pearl/v4/FP8/noise-line";
/// Lottery-key label used by [`jackpot_key`] and [`compute_jackpot_ticket`].
pub(crate) const LABEL_JACKPOT: &[u8] = b"pearl/v4/FP8/jackpot";

/// Derive a 32-byte role key from a complete v4 label and an optional parent.
pub(crate) fn subkey(label: &'static [u8], key: Option<&Hash256>) -> Hash256 {
    debug_assert!(label.starts_with(LABEL_PREFIX), "v4 labels must start with pearl/v4/FP8/");
    blake3_digest(label, key.copied())
}

/// Hash `message` with a key derived from the role label and optional parent key.
pub(crate) fn hash_labelled(message: &[u8], label: &'static [u8], key: Option<&Hash256>) -> Hash256 {
    blake3_digest(message, Some(subkey(label, key)))
}

/// Derive A's tree key from the proposed header; also used for MoE routing and offsets.
pub(crate) fn key_a(proposed_header: &IncompleteBlockHeader) -> Hash256 {
    hash_labelled(&proposed_header.to_bytes(), LABEL_KEY_A, None)
}

/// Derive B's tree key from the ancestor header.
pub(crate) fn key_b(ancestor_header: &IncompleteBlockHeader) -> Hash256 {
    hash_labelled(&ancestor_header.to_bytes(), LABEL_KEY_B, None)
}

/// Derive the lottery hash key from A's noise seed.
pub(crate) fn jackpot_key(seed_a: &Hash256) -> Hash256 {
    subkey(LABEL_JACKPOT, Some(seed_a))
}

/// The recomputed ticket: the 64-byte extract fold and its keyed jackpot
/// digest (comparable to the difficulty bound).
pub struct Ticket {
    pub msg: JackpotMessage,
    pub jackpot: Hash256,
}

/// Hash the folded matmul message with A's derived lottery key and retain both in the ticket.
pub fn compute_jackpot_ticket(seed_a: &Hash256, msg: &JackpotMessage) -> Ticket {
    Ticket {
        msg: *msg,
        jackpot: hash_labelled(msg, LABEL_JACKPOT, Some(seed_a)),
    }
}

impl PublicParams {
    /// Hash the proposed header followed by [`PublicParams::to_bytes`] under
    /// the `zk-public` label to bind the statement into Fiat-Shamir.
    pub(crate) fn digest(&self, proposed_header: &IncompleteBlockHeader) -> Hash256 {
        let mut msg = proposed_header.to_bytes().to_vec();
        msg.extend_from_slice(&self.to_bytes());
        hash_labelled(&msg, LABEL_ZK_PUBLIC, None)
    }

    /// The A-side tree key, derived from the caller's proposed header.
    /// Also keys MoE routing and offsets.
    pub(crate) fn key_a(&self, proposed_header: &IncompleteBlockHeader) -> Hash256 {
        key_a(proposed_header)
    }

    /// The B-side tree key, derived from [`Self::ancestor_header`].
    pub(crate) fn key_b(&self) -> Hash256 {
        key_b(self.ancestor_header())
    }

    /// Derive the keys used to verify A's and B's Merkle openings.
    pub(crate) fn commitment_keys(&self, proposed_header: &IncompleteBlockHeader) -> Sides<Hash256> {
        Sides {
            a: self.key_a(proposed_header),
            b: self.key_b(),
        }
    }

    /// Derive the lottery key used to compute the jackpot ticket.
    pub(crate) fn jackpot_key(&self, proposed_header: &IncompleteBlockHeader) -> Hash256 {
        jackpot_key(&self.noise_seeds(proposed_header).a)
    }

    /// Derive seeds from [`Self::commitment_digests`] using the chain above.
    /// Native verification authenticates the claimed roots through the openings;
    /// recursive verification binds them through the proof.
    pub(crate) fn noise_seeds(&self, proposed_header: &IncompleteBlockHeader) -> Sides<Hash256> {
        let keys = self.commitment_keys(proposed_header);
        let roots = self.commitment_digests();
        let p_b = self.job().encode_p_b();
        let p_a = self.job().encode_p_a();
        let message_b = [roots.b.as_slice(), keys.b.as_slice(), p_b.as_slice()].concat();
        let seed_b = hash_labelled(&message_b, LABEL_SEED_B, None);
        let message_a = match self.moe_statement() {
            Some(moe) => [
                roots.a.as_slice(),
                moe.hash_routing.as_slice(),
                moe.hash_offsets.as_slice(),
                seed_b.as_slice(),
                keys.a.as_slice(),
                p_a.as_slice(),
            ]
            .concat(),
            None => [roots.a.as_slice(), seed_b.as_slice(), keys.a.as_slice(), p_a.as_slice()].concat(),
        };
        let seed_a = hash_labelled(&message_a, LABEL_SEED_A, None);
        Sides { a: seed_a, b: seed_b }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `blake3` pins (see the derivation comment at the top of this file).
    const PARENT: Hash256 = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    const ROLES: &[(&[u8], &str, &str)] = &[
        (
            LABEL_KEY_A,
            "8bd812084a635e5161f84d2fb34f5c50d4e979becc07437fd4d4f6f4896cbc9d",
            "e97059b9fd3a5c744225db08e9dc3e63d795a51ddce01e2ea82f5e9c06925165",
        ),
        (
            LABEL_KEY_B,
            "521d46f9dd031c281000380a382c3fdcd393ff59c71336c13b7347ad2c1a938f",
            "028107616c23fc723994b6e249e45b58f4e2e36aa1b4263be9c1041002ad8a58",
        ),
        (
            LABEL_SEED_A,
            "060fe1ea0482eb3b2cfb927fbcf7e5dd48118192fcb45dca34a58faec0469299",
            "9b454df4917a237f845a35fec59bda9e598b0c1222e09abe52cea8bf118890e6",
        ),
        (
            LABEL_SEED_B,
            "5472d30cc74f93d9273d7d3612bde3702b8f2f5958b4efc5e9235b0ec86b4e3a",
            "8330940458d1f040028c474dff27882dcf717656c5ed49bb9ca083bbff39faf1",
        ),
        (
            LABEL_NOISE_LINE,
            "7aadc28ccbbef879bd58974d4b38c9d538c33b21ee6460b3185e933df1179e90",
            "3f5176a01068c0b9e3c364a8670767a1b909dc5e2762ff99110edbcfe424c198",
        ),
        (
            LABEL_JACKPOT,
            "bce6af9f8fc1313d6904c8315a10e58b476a0cda822c1be77233c98eb1127eb5",
            "1abccbdca824c1fa9707f6456144fcd48a2567a143ac8dc85c7b6ff11f3dc28f",
        ),
        (
            LABEL_ZK_PUBLIC,
            "ffc5d44fcccba998161cc9890c2d2ceb6133a1689a4d6699d3552d646fa04ff3",
            "b6ab192fc0b2d2351210f87dfe8575353c1d9c6f03f20ec92762f1b2f5789aad",
        ),
    ];

    fn hex32(s: &str) -> Hash256 {
        assert_eq!(s.len(), 64, "expected 32-byte hex");
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    }

    #[test]
    fn every_role_has_pinned_default_and_keyed_subkeys() {
        for &(label, default_hex, keyed_hex) in ROLES {
            assert_eq!(subkey(label, None), hex32(default_hex), "{}", String::from_utf8_lossy(label));
            assert_eq!(
                subkey(label, Some(&PARENT)),
                hex32(keyed_hex),
                "{}",
                String::from_utf8_lossy(label)
            );
        }
    }

    #[test]
    fn hash_labelled_matches_python_default_and_keyed_messages() {
        let long: Vec<u8> = (0..64).collect();
        let messages = [
            (
                b"".as_slice(),
                "f9c774347cc10b0dd5729b13925f246e8d21487dce485be8ab4013727c2a2d15",
                "79311dd1640edf6873c8d7a956b501b2562e203e41a11bf24ff24adf1c1a32f9",
            ),
            (
                b"x".as_slice(),
                "7cdc56fbcde7155cbe3604c99a35831cca9fbeb53e1964d7b6f0cb7b5ce65c69",
                "66e8588103d496ebb3ee49e8f3d6a46309fe808f354bab28f8d058449f88df54",
            ),
            (
                long.as_slice(),
                "8cd0fb004b120832471a26ce8958f469c55cb19684b6dee0687bbc6dbf890bbb",
                "c9bfd8a818a577b8aa717b0b58294381fff818e0af65ed1283189e644f778d11",
            ),
        ];
        for &(message, default_hex, keyed_hex) in &messages {
            assert_eq!(hash_labelled(message, LABEL_SEED_A, None), hex32(default_hex));
            assert_eq!(hash_labelled(message, LABEL_SEED_A, Some(&PARENT)), hex32(keyed_hex));
        }
    }

    #[test]
    fn ticket_depends_only_on_seed_and_message() {
        let seed_a = [0x11u8; 32];
        let z = [0x5au8; 64];
        let ticket = compute_jackpot_ticket(&seed_a, &z);
        assert_eq!(ticket.msg, z);
        assert_eq!(ticket.jackpot, hash_labelled(&z, LABEL_JACKPOT, Some(&seed_a)));
        assert_ne!(ticket.jackpot, blake3_digest(&z, Some(seed_a)));
    }

    // ---- the PublicParams derivation chain, over the shared statement fixtures ----

    use crate::api::fp8::public_params::test_fixtures::{dense_params, moe_params};

    /// Replace the ancestor header at the start of `public_data`, then decode the statement.
    fn with_ancestor(params: &PublicParams, ancestor: IncompleteBlockHeader) -> PublicParams {
        let mut bytes = params.to_bytes();
        bytes[..IncompleteBlockHeader::SERIALIZED_SIZE].copy_from_slice(&ancestor.to_bytes());
        PublicParams::from_bytes(&bytes).expect("ancestor splice must round-trip")
    }

    #[test]
    fn digest_is_labelled_hash_of_proposed_header_and_public_data() {
        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let params = dense_params();
        let mut msg = header.to_bytes().to_vec();
        msg.extend_from_slice(&params.to_bytes());
        assert_eq!(params.digest(&header), hash_labelled(&msg, LABEL_ZK_PUBLIC, None));

        let other_header = IncompleteBlockHeader::new_for_test(0x1d00ffff);
        assert_ne!(params.digest(&header), params.digest(&other_header));

        // Changing the ancestor header changes `public_data` and its digest.
        let other_ancestor = with_ancestor(&params, other_header);
        assert_ne!(params.digest(&header), other_ancestor.digest(&header));
        assert_ne!(params.digest(&header), moe_params().digest(&header));
    }

    #[test]
    fn commitment_keys_bind_both_headers() {
        let proposed = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let ancestor = IncompleteBlockHeader {
            prev_block: [9; 32],
            ..proposed
        };
        let params = with_ancestor(&dense_params(), ancestor);

        let keys = params.commitment_keys(&proposed);
        // Checked against the raw formula (not `key_a`) so a shared bug in the
        // helper cannot mask itself.
        assert_eq!(keys.a, hash_labelled(&proposed.to_bytes(), LABEL_KEY_A, None));
        assert_eq!(keys.b, hash_labelled(&ancestor.to_bytes(), LABEL_KEY_B, None));

        // keyA follows the proposed header only; keyB follows the ancestor only.
        let other_proposed = IncompleteBlockHeader::new_for_test(0x1d00ffff);
        let keys2 = params.commitment_keys(&other_proposed);
        assert_ne!(keys.a, keys2.a);
        assert_eq!(keys.b, keys2.b);
    }

    #[test]
    fn moe_seed_a_folds_routing_and_offsets_into_a_commitment() {
        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);

        let dense = dense_params();
        let moe = moe_params();

        // seedB = H_seed-B(HB, keyB, pB), per the two statements.
        let seed_b_of = |params: &PublicParams| {
            let keys = params.commitment_keys(&header);
            hash_labelled(
                &[
                    params.commitment_digests().b.as_slice(),
                    keys.b.as_slice(),
                    params.job().encode_p_b().as_slice(),
                ]
                .concat(),
                LABEL_SEED_B,
                None,
            )
        };

        // Baseline: dense seedA = H_seed-A(HA, seedB, keyA, pA).
        let dense_keys = dense.commitment_keys(&header);
        let dense_message_a = [
            dense.commitment_digests().a.as_slice(),
            seed_b_of(&dense).as_slice(),
            dense_keys.a.as_slice(),
            dense.job().encode_p_a().as_slice(),
        ]
        .concat();
        assert_eq!(
            dense.noise_seeds(&header).a,
            hash_labelled(&dense_message_a, LABEL_SEED_A, None)
        );

        // MoE inserts HR and HO after HA: H_seed-A(HA, HR, HO, seedB, keyA, pA).
        let moe_keys = moe.commitment_keys(&header);
        let stmt = moe.moe_statement().unwrap().clone();
        let moe_message_a = [
            moe.commitment_digests().a.as_slice(),
            stmt.hash_routing.as_slice(),
            stmt.hash_offsets.as_slice(),
            seed_b_of(&moe).as_slice(),
            moe_keys.a.as_slice(),
            moe.job().encode_p_a().as_slice(),
        ]
        .concat();
        assert_eq!(moe.noise_seeds(&header).a, hash_labelled(&moe_message_a, LABEL_SEED_A, None));

        // Changing HR must change seedA (seed-A binds the routing commitment).
        let mut moe2 = moe.clone();
        moe2.moe_statement_mut().unwrap().hash_routing[0] ^= 1;
        assert_ne!(moe.noise_seeds(&header).a, moe2.noise_seeds(&header).a);

        // Changing HO must change seedA (seed-A binds the offset commitment).
        let mut moe3 = moe.clone();
        moe3.moe_statement_mut().unwrap().hash_offsets[0] ^= 1;
        assert_ne!(moe.noise_seeds(&header).a, moe3.noise_seeds(&header).a);
    }
}
