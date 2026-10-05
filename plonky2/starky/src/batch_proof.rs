//! Proof types for batched multi-STARK proofs, where all tables share a
//! single batched FRI argument.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2::field::extension::Extendable;
use plonky2::fri::proof::{FriProof, FriProofTarget};
use plonky2::hash::hash_types::{MerkleCapTarget, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::target::Target;
use plonky2::plonk::config::GenericConfig;
use serde::{Deserialize, Serialize};

use crate::proof::{StarkOpeningSet, StarkOpeningSetTarget};

/// A batched multi-STARK proof: per-table Merkle caps, per-table polynomial
/// openings, and a single batched FRI argument for all tables.
///
/// The preprocessed oracles' caps are *not* part of the proof: they are
/// committed at setup time and known to the verifier.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "")]
pub struct BatchStarkProof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    /// Merkle cap of each solo table that has online trace columns, in
    /// table-index order, then the grouped tables' shared cap (if the group
    /// has online columns).
    pub trace_caps: Vec<MerkleCap<F, C::Hasher>>,
    /// Merkle cap of each solo table that has auxiliary polynomials, in
    /// table-index order, then the grouped tables' shared cap; `None` if no
    /// table has them.
    pub auxiliary_polys_caps: Option<Vec<MerkleCap<F, C::Hasher>>>,
    /// Merkle cap of each solo table that has quotient polynomials, in
    /// table-index order, then the grouped tables' shared cap; `None` if no
    /// table has them.
    pub quotient_polys_caps: Option<Vec<MerkleCap<F, C::Hasher>>>,
    /// Purported values of each table's polynomials at the challenge points.
    /// `openings[t].local_values`/`next_values` hold the *full* column set of
    /// table `t` (online and preprocessed columns interleaved back in their
    /// original positions).
    pub openings: Vec<StarkOpeningSet<F, D>>,
    /// A single batched FRI argument for all openings of all tables.
    pub opening_proof: FriProof<F, C::Hasher, D>,
    /// The trace degree bits of each table, in table-index order.
    pub degree_bits: Vec<usize>,
}

/// A [`BatchStarkProof`] along with the public inputs of every table.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "")]
pub struct BatchStarkProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    /// The batched multi-STARK proof.
    pub proof: BatchStarkProof<F, C, D>,
    /// The public inputs of each table, in table order.
    pub public_inputs: Vec<Vec<F>>,
}

/// Circuit version of [`BatchStarkProof`].
#[derive(Debug, Clone)]
pub struct BatchStarkProofTarget<const D: usize> {
    /// `Target`s for the trace Merkle caps, ordered as
    /// [`BatchStarkProof::trace_caps`].
    pub trace_caps: Vec<MerkleCapTarget>,
    /// `Target`s for the auxiliary Merkle caps, ordered as
    /// [`BatchStarkProof::auxiliary_polys_caps`].
    pub auxiliary_polys_caps: Option<Vec<MerkleCapTarget>>,
    /// `Target`s for the quotient Merkle caps, ordered as
    /// [`BatchStarkProof::quotient_polys_caps`].
    pub quotient_polys_caps: Option<Vec<MerkleCapTarget>>,
    /// `Target`s for the purported values of each table's polynomials at the
    /// challenge points.
    pub openings: Vec<StarkOpeningSetTarget<D>>,
    /// `Target`s for the batched FRI argument of all tables.
    pub opening_proof: FriProofTarget<D>,
    /// The trace degree bits of each table, in table-index order. Batch proofs
    /// have a fixed shape, so these are compile-time constants rather than
    /// targets.
    pub degree_bits: Vec<usize>,
}

/// Circuit version of [`BatchStarkProofWithPublicInputs`].
#[derive(Debug, Clone)]
pub struct BatchStarkProofWithPublicInputsTarget<const D: usize> {
    /// `Target` version of the batched multi-STARK proof.
    pub proof: BatchStarkProofTarget<D>,
    /// `Target`s for the public inputs of each table, in table order.
    pub public_inputs: Vec<Vec<Target>>,
}
