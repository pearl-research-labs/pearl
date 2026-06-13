#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::merkle_proofs::MerkleProof;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }

    pub fn digest(&self) -> HashOut<F> {
        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_element(F::from_canonical_usize(self.len()));
        challenger.observe_cap(self);
        challenger.get_hash()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: Vec<Vec<F>>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        // Base case: single leaf
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            //
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self
    where
        F: 'static,
        H: 'static,
    {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
        {
            // Check type match first (cheap, no ownership transfer)
            use core::any::TypeId;
            use crate::field::goldilocks_field::GoldilocksField;
            use crate::hash::poseidon::PoseidonHash;

            if TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
                && TypeId::of::<H>() == TypeId::of::<PoseidonHash>() {
                return Self::new_batched_goldilocks(leaves, cap_height);
            }
        }

        // Generic fallback path
        Self::new_generic(leaves, cap_height)
    }

    fn new_generic(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={cap_height} should be at most log2(leaves.len())={log2_leaves_len}"
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);

        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Try to construct using batched Poseidon hashing.
    /// Only succeeds for GoldilocksField + PoseidonHash on AVX-512 targets.
    /// Returns None otherwise, falling back to the generic path.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    fn new_batched_goldilocks(leaves: Vec<Vec<F>>, cap_height: usize) -> Self
    where
        F: 'static,
        H: 'static,
    {
        use crate::field::goldilocks_field::GoldilocksField;
        use crate::hash::merkle_tree::batched_merkle::merkle_tree_new_batched;

        // Safety: Caller verified F is GoldilocksField and H is PoseidonHash.
        // Since F and GoldilocksField are the same type, this transmute is safe.
        let goldilocks_leaves: Vec<Vec<GoldilocksField>> =
            unsafe { core::mem::transmute(leaves) };

        let (goldilocks_leaves, digests, cap) =
            merkle_tree_new_batched(goldilocks_leaves, cap_height);

        // Transmute the results back to the generic types
        let leaves: Vec<Vec<F>> = unsafe { core::mem::transmute(goldilocks_leaves) };
        let digests: Vec<H::Hash> = unsafe { core::mem::transmute(digests) };
        let cap: Vec<H::Hash> = unsafe { core::mem::transmute(cap) };

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

/// Batched Merkle tree construction for GoldilocksField + PoseidonHash.
/// Uses AVX-512 batched Poseidon to process 8 leaf hashes simultaneously.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
pub(crate) mod batched_merkle {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::poseidon::PoseidonHash;
    use crate::hash::poseidon_goldilocks::batched_poseidon::hash_batch_8;

    /// Build Merkle tree with batched Poseidon for GoldilocksField.
    /// Returns (leaves, digests, cap) in the same interleaved DFS format as the generic builder.
    pub fn merkle_tree_new_batched(
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> (Vec<Vec<GoldilocksField>>, Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>) {
        let n = leaves.len();
        let log2_n = log2_strict(n);
        assert!(cap_height <= log2_n);
        let num_subtrees = 1usize << cap_height;
        let subtree_size = n / num_subtrees;

        // Step 1: Batch-hash ALL leaves in batches of 8.
        // All leaves have the same number of elements (LDE rows).
        let leaf_hashes: Vec<HashOut<GoldilocksField>> = if !leaves.is_empty() && !leaves[0].is_empty() {
            leaves
                .par_chunks(256)
                .flat_map(|big_chunk| {
                    let mut results: Vec<HashOut<GoldilocksField>> = Vec::with_capacity(big_chunk.len());
                    for chunk in big_chunk.chunks(8) {
                        if chunk.len() == 8 {
                            let u64_inputs: [&[u64]; 8] = core::array::from_fn(|i| {
                                unsafe { core::slice::from_raw_parts(
                                    chunk[i].as_ptr() as *const u64,
                                    chunk[i].len()
                                ) }
                            });
                            let batch_results = hash_batch_8(u64_inputs);
                            for r in batch_results {
                                results.push(HashOut {
                                    elements: [
                                        GoldilocksField(r[0]),
                                        GoldilocksField(r[1]),
                                        GoldilocksField(r[2]),
                                        GoldilocksField(r[3]),
                                    ],
                                });
                            }
                        } else {
                            for leaf in chunk {
                                results.push(<PoseidonHash as Hasher<GoldilocksField>>::hash_or_noop(leaf));
                            }
                        }
                    }
                    results
                })
                .collect()
        } else {
            leaves.iter().map(|leaf| {
                <PoseidonHash as Hasher<GoldilocksField>>::hash_or_noop(leaf)
            }).collect()
        };

        // Step 2: Build internal tree from leaf hashes in DFS interleaved format.
        let num_digests = 2 * (n - num_subtrees);
        let mut digests = Vec::with_capacity(num_digests);
        let mut cap = Vec::with_capacity(num_subtrees);

        let subtree_digests_len = 2 * (subtree_size - 1);

        if subtree_digests_len == 0 {
            // All cap: each leaf is its own subtree root
            let cap_buf = capacity_up_to_mut(&mut cap, num_subtrees);
            cap_buf
                .par_iter_mut()
                .zip(leaf_hashes)
                .for_each(|(c, h)| {
                    c.write(h);
                });
        } else {
            let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
            let cap_buf = capacity_up_to_mut(&mut cap, num_subtrees);

            digests_buf
                .par_chunks_exact_mut(subtree_digests_len)
                .zip(cap_buf)
                .zip(leaf_hashes.par_chunks_exact(subtree_size))
                .for_each(|((sub_digests, sub_cap), sub_hashes)| {
                    let root = build_subtree_from_hashes(sub_digests, sub_hashes);
                    sub_cap.write(root);
                });
        }

        unsafe {
            digests.set_len(num_digests);
            cap.set_len(num_subtrees);
        }

        (leaves, digests, cap)
    }

    /// Build subtree from pre-computed leaf hashes in DFS interleaved format.
    /// Uses the same layout as fill_subtree: [left_digests | left_root | right_root | right_digests]
    fn build_subtree_from_hashes(
        digests_buf: &mut [MaybeUninit<HashOut<GoldilocksField>>],
        leaf_hashes: &[HashOut<GoldilocksField>],
    ) -> HashOut<GoldilocksField> {
        let n = leaf_hashes.len();
        assert_eq!(digests_buf.len(), 2 * (n - 1));

        if n == 1 {
            return leaf_hashes[0];
        }

        // Same split as fill_subtree: half/half, then take last of left and first of right
        let (left_buf, right_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_mem, left_buf) = left_buf.split_last_mut().unwrap();
        let (right_mem, right_buf) = right_buf.split_first_mut().unwrap();
        let (left_hashes, right_hashes) = leaf_hashes.split_at(n / 2);

        let (left_root, right_root) = plonky2_maybe_rayon::join(
            || build_subtree_from_hashes(left_buf, left_hashes),
            || build_subtree_from_hashes(right_buf, right_hashes),
        );

        left_mem.write(left_root);
        right_mem.write(right_root);

        <PoseidonHash as Hasher<GoldilocksField>>::two_to_one(left_root, right_root)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
