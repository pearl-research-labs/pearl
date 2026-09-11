//! Proves every BLAKE3 compression used by FP8 and binds the resulting public hashes.
//!
//! # Compression rows
//!
//! One BLAKE3 compression consumes a 64-byte message (sixteen little-endian `u32` words), an
//! eight-word chaining value (CV), a counter, block length, and flags. It occupies eight trace
//! rows: seven apply BLAKE3's permutation rounds; the eighth applies feed-forward
//! finalization and exposes the eight-word output CV.
//!
//! [`Blake3Program`] is the public instruction schedule. For each compression it
//! selects the CV source, message source, counter/flag tweak and optional public-output slot.
//!
//! # Protocol hash graph
//!
//! Leaf and parent compressions build four keyed Merkle roots: int8 values and bf16 block-scale
//! codes for each of matrices A and B. A separate keyed tree binds mixture-of-experts (MoE)
//! routing data when present.
//!
//! Eight parallel lookups route an internal CV by packing each
//! `(source trace row, CV word value)` pair injectively; `cv_route_key_or_tweak` carries the
//! source-row pointer on fetch rows or the instruction's public tweak on initialization rows.
//!
//! Two keyed wrapper compressions ([`append_commit_fold`]) bind the four roots
//! to the public per-side commitment digests:
//!
//! ```text
//! HASH_A = blake3(A_values_root || A_scales_root, key=keyA)
//! HASH_B = blake3(B_values_root || B_scales_root, key=keyB)
//! ```
//!
//! Here `||` denotes byte concatenation. The individual roots remain internal CVs.
//!
//! After Matmul and XorFold, one final keyed compression hashes the folded lottery words under
//! `POW_KEY`; its output is the public jackpot hash. `HASH_A`, `HASH_B`, the optional
//! routing/offsets hashes, and the jackpot hash are public inputs.
//!
//! # Trace binding
//!
//! The verifier recomputes the schedule columns and checks their openings. Cross-table lookups
//! bind live int8/scale message blocks to InputQuantStark and lottery words to XorFoldStark;
//! byte and MoE-limb lookups range-check packed message data. Thus neither padding
//! compressions nor an alternative private instruction schedule can contribute to a public
//! binding.

use core::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;

use anyhow::{Result, ensure};

use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_ROOT, BLAKE3_MSG_LEN};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::lookup::{Column, Filter, Lookup};
use starky::stark::Stark;

use super::columns::{
    BLAKE3_COL_MAP, Blake3ColumnsView, Blake3StateCols, NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS, NUM_UINT8,
    NUM_UNPACK_FLAGS, PI_HASH_A, PI_HASH_B, PI_HASH_JACKPOT, PI_HASH_OFFSETS, PI_HASH_ROUTING, PI_JACKPOT_KEY, PI_KEY_A,
    PI_KEY_B,
};
use crate::api::fp8::prequant::BLOCK_SIZE;
use crate::api::fp8::public_params::{HashId, MoEStatement};
use crate::circuit::chip::blake3::logic::decode_is_msg_bits;
use crate::circuit::chip::blake3::program as chip_program;
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The BLAKE3 IV constants; the compression's fixed state words 8..12 (constraint 1).
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// The BLAKE3 message word permutation (applied between rounds).
const BLAKE3_MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rows per compression: 7 round rows + 1 finalization row.
pub const ROWS_PER_COMPRESSION: usize = 8;

/// Key factor of the CV-routing lookup: `key = word + 2^34 * trace_row_index`. The factor is
/// 2^34, not 2^32, because the round block's unchecked-add slack admits words up to about 2^34;
/// this keeps the packing injective for every value accepted by the constraints.
const CV_ROUTING_KEY_FACTOR: u64 = 1 << 34;

/// Permutes message words in place; generic so the AIR can also permute column handles.
fn blake3_permute<T: Copy>(msg: &mut [T; 16]) {
    let old = *msg;
    for i in 0..16 {
        msg[i] = old[BLAKE3_MSG_PERMUTATION[i]];
    }
}

// Public compression schedule

/// The six committed byte planes a compression's message bytes can come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneId {
    /// A-side int8 values plane (the sampled strip rows, `h * k` bytes row-major).
    AValues,
    /// A-side bf16 scales plane, `h * k / 4` bytes (one LE bf16 code per 8-element block).
    AScales,
    /// B-side int8 values plane, `w * k` bytes (B transposed row-major, matmul's layout).
    BValues,
    /// B-side bf16 scales plane, `w * k / 4` bytes.
    BScales,
    /// MoE routing plane: 16 u32 outer-index words per 64-byte block, each < 2^26.
    Routing,
    /// MoE offsets plane: the full zero-padded cumulative-count list `O`, 16 u32 words per
    /// 64-byte block, every block opened.
    Offsets,
}

const NUM_PLANES: usize = 6;

impl PlaneId {
    fn index(self) -> usize {
        match self {
            PlaneId::AValues => 0,
            PlaneId::AScales => 1,
            PlaneId::BValues => 2,
            PlaneId::BScales => 3,
            PlaneId::Routing => 4,
            PlaneId::Offsets => 5,
        }
    }
}

/// A parent's child CV: an earlier instruction's output or the witness root of an unopened subtree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvRef {
    /// `CV_OUT` of instruction `i` (must precede the referencing instruction).
    Instruction(usize),
    /// `aux_cvs[idx]` of the trace witness (32 bytes, prover-supplied).
    Auxiliary(usize),
}

/// The CV entering the compression (the constraint-3 mux source / the row-0 fetch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvSource {
    /// Keyed compression under `KEY_A` (A-plane/routing chunk starts and parents).
    KeyA,
    /// Keyed compression under `KEY_B` (B-plane chunk starts and parents).
    KeyB,
    /// `JACKPOT_KEY` (the lottery compression).
    JackpotKey,
    /// The BLAKE3 IV constants — unkeyed compression.
    Iv,
    /// `CV_OUT` of instruction `src`, fetched through the CV-routing lookup on row 0 (block 2+
    /// of a multi-block chunk).
    Chain(usize),
}

/// Message source for one 64-byte compression block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageSource {
    /// 64 bytes starting at `offset` in the chunk-padded plane, ingested eight per row.
    /// `ctl_base` is the first element/scale-block lookup key. Bytes beyond the raw plane
    /// length are hashed as padding but excluded from CTLs. A block may span adjacent
    /// opened strips, which are contiguous in the plane stream.
    PlaneBytes { plane: PlaneId, offset: usize, ctl_base: u64 },
    /// `aux_msgs[idx]`: an unopened 64-byte block, hashed without exporting lookup tuples.
    AuxBytes { idx: usize },
    /// A block crossing an opening boundary. Dwords `skip..skip+take` are opened plane
    /// bytes; `offset` and `ctl_base` refer to the first opened dword. Other bytes come
    /// from `aux_msgs[aux_idx]` and are hashed without exporting tuples.
    /// Used when values/scales rows end inside a 64-byte block.
    PlaneBytesSplit {
        plane: PlaneId,
        offset: usize,
        ctl_base: u64,
        skip: usize,
        take: usize,
        aux_idx: usize,
    },
    /// A Merkle parent: message = left child CV (words 0..8) | right child CV (words 8..16).
    /// In-table children arrive through the CV-routing lookup (window fetches on rows 3 / 7);
    /// auxiliary children are witness bytes ingested as 4 dword rows each.
    Parent { left: CvRef, right: CvRef },
    /// The 16 folded lottery words (witness bytes; pinned by the XorFold CTL against
    /// `BLAKE3_MSG` on the load row — see `super::ctl`).
    Lottery,
}

/// Which public input this compression's output binds to (constraint 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicBinding {
    HashA,
    HashB,
    HashRouting,
    HashOffsets,
    HashJackpot,
}

/// One eight-row compression: CV source, message source, tweak and optional public binding.
#[derive(Clone, Copy, Debug)]
pub struct Blake3Instruction {
    pub cv: CvSource,
    pub msg: MessageSource,
    /// BLAKE3 block counter (the chunk index for chunk blocks, 0 for parents), < 2^48.
    pub counter: u64,
    /// BLAKE3 block length (64 for full blocks), < 2^7.
    pub block_len: u32,
    /// BLAKE3 domain flags (`B3F_*`), < 2^8.
    pub flags: u32,
    pub bind: Option<PublicBinding>,
}

impl Blake3Instruction {
    /// Keyed compression of XorFold's 16 words under JACKPOT_KEY, bound to HASH_JACKPOT.
    pub fn lottery() -> Self {
        Self {
            cv: CvSource::JackpotKey,
            msg: MessageSource::Lottery,
            counter: 0,
            block_len: BLAKE3_MSG_LEN as u32,
            flags: (B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT | B3F_KEYED_HASH) as u32,
            bind: Some(PublicBinding::HashJackpot),
        }
    }

    /// Initialization tweak packed as `counter(48 bits) | flags(8) | block_len(7)`, stored on row 1.
    fn tweak_packed(&self) -> u64 {
        assert!(self.counter < 1 << 48 && self.flags < 1 << 8 && self.block_len < 1 << 7);
        self.counter + ((self.flags as u64) << 48) + ((self.block_len as u64) << 56)
    }

    /// CV-routing fetches of this compression: `(row within the compression, source
    /// instruction)`. Row 0 is the chunk-chaining fetch; rows 3 / 7 are the parent window
    /// fetches — the positions that land each fetched CV at message words 0..8 / 8..16 through
    /// the shift-by-2 buffer.
    fn fetches(&self) -> Vec<(usize, usize)> {
        let mut fetches = Vec::new();
        if let CvSource::Chain(src) = self.cv {
            fetches.push((0, src));
        }
        if let MessageSource::Parent { left, right } = self.msg {
            if let CvRef::Instruction(src) = left {
                fetches.push((3, src));
            }
            if let CvRef::Instruction(src) = right {
                fetches.push((7, src));
            }
        }
        fetches
    }
}

/// Appends `blake3(values_root || scales_root, key=key)`, bound to the side's public digest.
/// Both roots are earlier instruction indices fetched through CV routing; key is KeyA or KeyB.
pub fn append_commit_fold(
    instrs: &mut Vec<Blake3Instruction>,
    values_root: usize,
    scales_root: usize,
    bind: PublicBinding,
    key: CvSource,
) {
    debug_assert!(
        matches!(key, CvSource::KeyA | CvSource::KeyB),
        "commit-fold key must be KeyA/KeyB"
    );
    instrs.push(Blake3Instruction {
        cv: key,
        msg: MessageSource::Parent {
            left: CvRef::Instruction(values_root),
            right: CvRef::Instruction(scales_root),
        },
        counter: 0,
        block_len: BLAKE3_MSG_LEN as u32,
        flags: (B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT | B3F_KEYED_HASH) as u32,
        bind: Some(bind),
    });
}

/// The compiled schedule of one job's hash forest: the full per-plane commitment trees (with
/// auxiliary material at the opening boundary), the routing tree, and the lottery — one
/// instruction per compression, children before parents. Produced by the fp8 program
/// compiler; everything here is verifier-known (verifier-recomputable from public job geometry).
#[derive(Clone, Debug, Default)]
pub struct Blake3Program {
    pub instructions: Vec<Blake3Instruction>,
    /// Number of auxiliary 64-byte messages the schedule references (witness bound).
    pub num_aux_msgs: usize,
    /// Number of auxiliary sibling CVs the schedule references (witness bound).
    pub num_aux_cvs: usize,
    /// Sampled MoE entries `(stream_word_index, public_outer_index)`, sorted by stream
    /// position. Each public value is below 2^26. Selectors bind only sampled words;
    /// unsampled neighbors remain witness bytes bound by the hash and order chains.
    /// Empty for dense jobs.
    pub routing_pins: Vec<(usize, u32)>,
    /// The MoE scalar statement driving the offsets pins and the order chains; `None` for
    /// dense jobs. Verifier-known — public statement data.
    pub moe: Option<MoeSchedule>,
}

/// The public MoE scalars the offsets/routing schedule is derived from: the winner `w`, the
/// bracketing cumulative counts `O_{w-1}`/`O_w`, the total `O_{e-1}`, the expert count `e`,
/// and the A-row count `m` (the strict upper bound on routing values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeSchedule {
    pub w: u32,
    pub experts: u32,
    pub o_w_prev: u32,
    pub o_w: u32,
    pub o_last: u32,
    pub m: u32,
}

impl MoeSchedule {
    /// Assembles the schedule from the public MoE statement plus the expert count and `m`,
    /// failing closed on a statement whose scalars violate the schedule's invariants
    pub(crate) fn new(stmt: &MoEStatement, experts: u16, m: u32) -> Result<Self> {
        let (w, experts, o_w_prev, o_w, o_last) = (u32::from(stmt.w), u32::from(experts), stmt.o_w_prev, stmt.o_w, stmt.o_last);
        ensure!(experts >= 1 && w < experts, "malformed MoE schedule: w/e");
        ensure!(m >= 1, "malformed MoE schedule: m = 0");
        ensure!(
            o_w_prev <= o_w && o_w <= o_last,
            "malformed MoE schedule: O_{{w-1}} <= O_w <= O_{{e-1}}"
        );
        ensure!(w > 0 || o_w_prev == 0, "O_{{w-1}} must be 0 for w = 0");
        ensure!(w + 1 < experts || o_w == o_last, "O_w must equal O_{{e-1}} for w = e-1");
        Ok(Self {
            w,
            experts,
            o_w_prev,
            o_w,
            o_last,
            m,
        })
    }

    /// The pinned value of offsets-stream word `p`, `None` where the word is unpinned:
    /// - words `w-1`, `w` and `e-1` carry the public `O_{w-1}`, `O_w` and `O_{e-1}`;
    /// - every word past `e-1` is zero padding;
    /// - for `w = 0` there is no word to pin `O_{w-1}` to, and for `w = e-1` the `O_w` and
    ///   `O_{e-1}` pins coincide. The statement then enforces `O_{w-1} = 0` respectively
    ///   `O_w = O_{e-1}` natively.
    fn offsets_pin_at(&self, p: usize) -> Option<u64> {
        let (w, e) = (self.w as usize, self.experts as usize);
        if p >= e {
            return Some(0);
        }
        if p + 1 == e {
            return Some(self.o_last as u64);
        }
        if p == w {
            return Some(self.o_w as u64);
        }
        (w > 0 && p == w - 1).then_some(self.o_w_prev as u64)
    }

    /// Chain gates of the offsets row whose first stream word is `p`: `(intra, inter)` —
    /// intra enforces `O_p <= O_{p+1}` in-row, inter `O_{p-1} <= O_p` against the carry.
    /// Both stop at the pinned `O_{e-1}`; the zero padding is pinned, not chained.
    fn offsets_chain_at(&self, p: usize) -> (bool, bool) {
        let e = self.experts as usize;
        (p + 1 < e, p >= 1 && p < e)
    }

    /// First opened routing block (the routing stream's block base).
    fn routing_block_base(&self) -> usize {
        self.o_w_prev as usize * std::mem::size_of::<u32>() / BLAKE3_MSG_LEN
    }

    /// Chain gates of the routing row whose first *global* routing word is `g`:
    /// `(intra, inter, bound_first, bound_second)`. Intra/inter enforce strict increase over
    /// the winner slice `[O_{w-1}, O_w)`; the bound flags mark the slice's last word, whose
    /// row pins `m - 1` for the upper-bound check.
    fn routing_chain_at(&self, g: u64) -> (bool, bool, bool, bool) {
        let (lo, hi) = (u64::from(self.o_w_prev), u64::from(self.o_w));
        let in_slice = |x: u64| lo <= x && x < hi;
        (
            in_slice(g) && in_slice(g + 1),
            g > lo && in_slice(g),
            hi > lo && g == hi - 1,
            hi > lo && g + 1 == hi - 1,
        )
    }
}

impl Blake3Program {
    // TODO: Consolidate `chip_program::BlakeProgram` and this STARK-specific
    // `Blake3Program` into one canonical program representation.
    /// Translates the public `BlakeProgram` into this AIR's schedule, then appends
    /// per-side commitment folds and the lottery compression.
    ///
    /// - Keyed CVs select KeyA/KeyB/JackpotKey; chained CVs use the preceding instruction.
    /// - Matrix leaves become byte ranges in the concatenated opened strips. CTL keys
    ///   count values or scale blocks; B keys start after A's range.
    /// - Split leaves retain their opened dword window and auxiliary remainder.
    /// - Routing/offsets leaves use 64-byte blocks in their respective streams.
    /// - Parent/auxiliary CV indices retain the input witness indexing.
    /// - Routing/offsets roots bind directly; values/scales roots feed the commitment folds.
    ///
    /// `routing_pins` and `moe` supply the public sampled entries, offsets and order checks.
    /// Validation checks them against the schedule. Dense jobs use empty pins and no MoE data.
    /// Panics unless each A/B values/scales root is produced exactly once.
    pub fn from_blake_program(
        program: &chip_program::BlakeProgram,
        k: usize,
        routing_pins: Vec<(usize, u32)>,
        moe: Option<MoeSchedule>,
    ) -> Self {
        // Fail closed on the scheme using the four-plane shape; record the root indices the
        // fold wrappers reference.
        let mut plane_roots = [const { Vec::new() }; 4];
        for (i, instr) in program.instructions.iter().enumerate() {
            // Map compiler root roles to FP8 planes.
            let root_slot = match instr.out {
                chip_program::HashOut::A => 0,
                chip_program::HashOut::AScales => 1,
                chip_program::HashOut::B => 2,
                chip_program::HashOut::BScales => 3,
                _ => continue,
            };
            plane_roots[root_slot].push(i);
        }
        let [a_values_root, a_scales_root, b_values_root, b_scales_root] = plane_roots.map(|roots| {
            assert_eq!(
                roots.len(),
                1,
                "not a prequant four-plane program (A/B values + scales roots each produced once)"
            );
            roots[0]
        });

        let (h, w) = (program.num_a_rows, program.num_b_cols);
        let scale_row_bytes = 2 * (k / BLOCK_SIZE);
        let cv_ref = |cv: chip_program::CvType| match cv {
            chip_program::CvType::Instruction { idx } => CvRef::Instruction(idx),
            chip_program::CvType::Auxiliary { idx } => CvRef::Auxiliary(idx),
        };
        // One dword's plane and byte offset in that plane's stream (the concatenated opened
        // strips). Adjacent committed rows open to adjacent strips, so the stream offsets of
        // a block straddling a row boundary stay contiguous.
        let resolve_dword = |d: chip_program::MatDwordId| -> (PlaneId, usize) {
            let side_strips = if d.is_b_strip { w } else { h };
            let is_scales = d.strip_idx >= side_strips;
            let plane = match (d.is_b_strip, is_scales) {
                (false, false) => PlaneId::AValues,
                (false, true) => PlaneId::AScales,
                (true, false) => PlaneId::BValues,
                (true, true) => PlaneId::BScales,
            };
            let row_bytes = if is_scales { scale_row_bytes } else { k };
            (plane, (d.strip_idx % side_strips) * row_bytes + d.idx_in_strip)
        };
        let ctl_base = |plane: PlaneId, offset: usize| -> u64 {
            match plane {
                PlaneId::AValues => offset as u64,
                PlaneId::BValues => (h * k + offset) as u64,
                PlaneId::AScales => (offset / 2) as u64,
                PlaneId::BScales => (h * k / BLOCK_SIZE + offset / 2) as u64,
                PlaneId::Routing | PlaneId::Offsets => unreachable!(),
            }
        };

        // + 3: one commitment fold per side plus the lottery.
        let mut instructions = Vec::with_capacity(program.instructions.len() + 3);
        for (i, instr) in program.instructions.iter().enumerate() {
            let msg = match instr.msg {
                chip_program::MessageType::MatrixLeaf { mat_data } => {
                    let (plane, offset) = resolve_dword(mat_data.dwords[0]);
                    debug_assert!(
                        mat_data
                            .dwords
                            .iter()
                            .enumerate()
                            .all(|(j, dw)| { resolve_dword(*dw) == (plane, offset + j * chip_program::DWORD_SIZE) }),
                        "matrix message must be 64 stream-contiguous bytes of one plane"
                    );
                    MessageSource::PlaneBytes {
                        plane,
                        offset,
                        ctl_base: ctl_base(plane, offset),
                    }
                }
                chip_program::MessageType::SplitLeaf { mat_dwords, aux_idx } => {
                    let skip = mat_dwords.iter().position(|d| d.is_some()).expect("split leaf opens dwords");
                    let take = mat_dwords[skip..].iter().take_while(|d| d.is_some()).count();
                    assert!(
                        take < mat_dwords.len() && mat_dwords[skip + take..].iter().all(|d| d.is_none()),
                        "split leaf's opened dwords must be one proper contiguous window"
                    );
                    let (plane, offset) = resolve_dword(mat_dwords[skip].unwrap());
                    debug_assert!(
                        mat_dwords[skip..skip + take]
                            .iter()
                            .enumerate()
                            .all(|(j, dw)| { resolve_dword(dw.unwrap()) == (plane, offset + j * chip_program::DWORD_SIZE) }),
                        "split leaf's opened dwords must be stream-contiguous in one plane"
                    );
                    MessageSource::PlaneBytesSplit {
                        plane,
                        offset,
                        ctl_base: ctl_base(plane, offset),
                        skip,
                        take,
                        aux_idx,
                    }
                }
                chip_program::MessageType::RoutingLeaf { hotspot_idx } => MessageSource::PlaneBytes {
                    plane: PlaneId::Routing,
                    offset: BLAKE3_MSG_LEN * hotspot_idx,
                    ctl_base: 0,
                },
                chip_program::MessageType::OffsetsLeaf { block_idx } => MessageSource::PlaneBytes {
                    plane: PlaneId::Offsets,
                    offset: BLAKE3_MSG_LEN * block_idx,
                    ctl_base: 0,
                },
                chip_program::MessageType::AuxiliaryLeaf { idx } => MessageSource::AuxBytes { idx },
                chip_program::MessageType::Parent { cv_low, cv_high } => MessageSource::Parent {
                    left: cv_ref(cv_low),
                    right: cv_ref(cv_high),
                },
            };
            let bind = match instr.out {
                chip_program::HashOut::Routing => Some(PublicBinding::HashRouting),
                chip_program::HashOut::Offsets => Some(PublicBinding::HashOffsets),
                _ => None, // Plane roots stay internal: the fold wrappers consume their CVs.
            };
            assert!(
                instr.key_source != chip_program::KeySource::Prev || i > 0,
                "chained CV on the first instruction"
            );
            instructions.push(Blake3Instruction {
                cv: match instr.key_source {
                    chip_program::KeySource::KeyA => CvSource::KeyA,
                    chip_program::KeySource::KeyB => CvSource::KeyB,
                    chip_program::KeySource::Jackpot => CvSource::JackpotKey,
                    chip_program::KeySource::Prev => CvSource::Chain(i - 1),
                },
                msg,
                counter: u64::from(instr.tweak.counter_low) | (u64::from(instr.tweak.counter_high) << 32),
                block_len: instr.tweak.block_len,
                flags: instr.tweak.flags,
                bind,
            });
        }
        // Append folding instructions -- to get one hash per side (keyed by the side's opening key).
        append_commit_fold(
            &mut instructions,
            a_values_root,
            a_scales_root,
            PublicBinding::HashA,
            CvSource::KeyA,
        );
        append_commit_fold(
            &mut instructions,
            b_values_root,
            b_scales_root,
            PublicBinding::HashB,
            CvSource::KeyB,
        );
        instructions.push(Blake3Instruction::lottery());
        Self {
            instructions,
            num_aux_msgs: program.num_auxiliary_msgs,
            num_aux_cvs: program.num_auxiliary_cvs,
            routing_pins,
            moe,
        }
    }

    /// Live (unpadded) trace rows.
    pub fn num_live_rows(&self) -> usize {
        ROWS_PER_COMPRESSION * self.instructions.len()
    }

    /// The pinned outer-index value at stream word `pos`, `None` where the word is a free
    /// witness neighbor (`routing_pins` is sorted by position — `validate`).
    fn routing_pin_at(&self, pos: usize) -> Option<u64> {
        self.routing_pins
            .binary_search_by_key(&pos, |&(p, _)| p)
            .ok()
            .map(|i| self.routing_pins[i].1 as u64)
    }

    /// Trace height: live rows padded to a power of two.
    pub fn num_rows(&self) -> usize {
        self.num_live_rows().next_power_of_two()
    }

    /// Check instruction references, public bindings and MoE routing against the schedule.
    /// Panics on forward references, invalid witness indices, repeated bindings,
    /// or routing samples that cannot be authenticated by the scheduled blocks.
    fn validate(&self) {
        let mut seen_binds = Vec::new();
        let mut routing_blocks = Vec::new();
        let mut offsets_blocks = Vec::new();
        let mut lottery_count = 0;
        for (c, instr) in self.instructions.iter().enumerate() {
            // TODO: Remove this runtime validation once `Blake3Program` construction is
            // encapsulated and guarantees exactly one canonical lottery instruction.
            let has_lottery_role = matches!(instr.cv, CvSource::JackpotKey)
                || matches!(instr.msg, MessageSource::Lottery)
                || instr.bind == Some(PublicBinding::HashJackpot);
            if has_lottery_role {
                assert!(
                    matches!(instr.cv, CvSource::JackpotKey)
                        && matches!(instr.msg, MessageSource::Lottery)
                        && instr.counter == 0
                        && instr.block_len == BLAKE3_MSG_LEN as u32
                        && instr.flags == (B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT | B3F_KEYED_HASH) as u32
                        && instr.bind == Some(PublicBinding::HashJackpot),
                    "instruction {c}: malformed lottery compression"
                );
                lottery_count += 1;
            }
            if let CvSource::Chain(src) = instr.cv {
                assert!(src < c, "instruction {c}: CV chain source {src} is not earlier");
            }
            if instr.cv == CvSource::Iv {
                assert_eq!(
                    instr.flags & B3F_KEYED_HASH as u32,
                    0,
                    "instruction {c}: unkeyed compression carries the keyed flag"
                );
            }
            if let MessageSource::Parent { left, right } = instr.msg {
                for child in [left, right] {
                    match child {
                        CvRef::Instruction(src) => assert!(src < c, "instruction {c}: child {src} is not earlier"),
                        CvRef::Auxiliary(idx) => assert!(idx < self.num_aux_cvs, "instruction {c}: aux CV {idx} OOB"),
                    }
                }
            }
            if let MessageSource::AuxBytes { idx } = instr.msg {
                assert!(idx < self.num_aux_msgs, "instruction {c}: aux msg {idx} OOB");
            }
            if let MessageSource::PlaneBytesSplit {
                plane,
                skip,
                take,
                aux_idx,
                ..
            } = instr.msg
            {
                assert!(plane != PlaneId::Routing, "instruction {c}: routing blocks never split");
                assert!(
                    (1..ROWS_PER_COMPRESSION).contains(&take) && skip + take <= ROWS_PER_COMPRESSION,
                    "instruction {c}: a split block has both opened and unopened dwords"
                );
                assert!(aux_idx < self.num_aux_msgs, "instruction {c}: split aux msg {aux_idx} OOB");
            }
            if let MessageSource::PlaneBytes { plane, offset, .. } = instr.msg
                && (plane == PlaneId::Routing || plane == PlaneId::Offsets)
            {
                assert_eq!(offset % 64, 0, "instruction {c}: routing/offsets block not 64-byte aligned");
                if plane == PlaneId::Routing {
                    routing_blocks.push((c, offset / 64));
                } else {
                    offsets_blocks.push((c, offset / 64));
                }
            }
            if let Some(bind) = instr.bind {
                assert!(!seen_binds.contains(&bind), "binding {bind:?} set twice");
                seen_binds.push(bind);
            }
        }
        assert!(
            self.routing_pins.windows(2).all(|p| p[0].0 < p[1].0),
            "routing pins must be strictly ascending by stream position"
        );
        for &(pos, value) in &self.routing_pins {
            // Two-limb packing requires every sampled routing value to fit 26 bits.
            assert!(value < 1 << 26, "routing pin value {value} does not fit 26 bits");
            // Every public sample must occur in a block the trace actually hashes.
            // Blocks without samples are valid: their remaining words are private witness.
            assert!(
                routing_blocks.iter().any(|&(_, b)| b == pos / 16),
                "routing pin at stream word {pos} lies outside every scheduled routing block"
            );
        }

        assert_eq!(
            self.moe.is_some(),
            !offsets_blocks.is_empty(),
            "offsets sections and the MoE schedule imply each other"
        );
        if let Some(s) = &self.moe {
            // Both planes stream as blocks 0, 1, 2, ... in instruction order; the placement
            // inside the tree is pinned by the recomputed tweak/chain schedule.
            assert!(
                offsets_blocks.iter().enumerate().all(|(i, &(_, b))| b == i),
                "offsets blocks must be scheduled completely and in order"
            );
            assert!(
                routing_blocks.iter().enumerate().all(|(i, &(_, b))| b == i),
                "routing blocks must be scheduled completely and in order"
            );
            if let (Some(&(last_routing, _)), Some(&(first_offsets, _))) = (routing_blocks.last(), offsets_blocks.first()) {
                assert!(
                    last_routing < first_offsets,
                    "routing blocks must all precede the offsets blocks (shared chain carry)"
                );
            }
            assert!(
                s.experts as usize <= offsets_blocks.len() * 16,
                "offsets stream too short for e words"
            );
            if s.o_w > s.o_w_prev {
                assert!(
                    s.routing_block_base() + routing_blocks.len()
                        >= (s.o_w as usize * std::mem::size_of::<u32>()).div_ceil(BLAKE3_MSG_LEN),
                    "routing stream must cover the winner slice's block range"
                );
                let base = 16 * s.routing_block_base();
                for &(pos, _) in &self.routing_pins {
                    let g = (base + pos) as u64;
                    assert!(
                        u64::from(s.o_w_prev) <= g && g < u64::from(s.o_w),
                        "sampled routing pin at global word {g} lies outside the winner slice"
                    );
                }
            } else {
                assert!(routing_blocks.is_empty() && self.routing_pins.is_empty(), "empty slice");
            }
            assert!(
                seen_binds.contains(&PublicBinding::HashOffsets) && seen_binds.contains(&PublicBinding::HashRouting),
                "MoE programs must bind HASH_ROUTING and HASH_OFFSETS"
            );
        } else {
            assert!(
                routing_blocks.is_empty() && self.routing_pins.is_empty(),
                "dense programs schedule no routing blocks"
            );
        }
        assert_eq!(
            lottery_count, 1,
            "Blake3 program must contain exactly one lottery compression"
        );
    }

    /// Generates the Blake3Stark trace and public inputs. Bit-exact against native BLAKE3: with
    /// a correctly compiled schedule the bound tree roots equal `pearl_blake3::MerkleTree`
    /// roots over the chunk-padded full planes keyed with `key_a`/`key_b` (auxiliary CVs standing
    /// in for the unopened subtrees), and `HASH_JACKPOT` equals
    /// `blake3::keyed_hash(jackpot_key, lottery bytes)`.
    pub fn generate_trace<F: RichField>(
        &self,
        inputs: &Blake3TraceInputs<'_>,
    ) -> (Vec<[F; NUM_BLAKE3_COLUMNS]>, [F; NUM_BLAKE3_PUBLIC_INPUTS]) {
        self.validate();
        assert_eq!(
            inputs.aux_msgs.len(),
            self.num_aux_msgs,
            "aux msgs must match the compiled schedule"
        );
        assert_eq!(
            inputs.aux_cvs.len(),
            self.num_aux_cvs,
            "aux CVs must match the compiled schedule"
        );
        // The witness routing words at pinned positions must equal the public outer indices
        // (`parse_proof` guarantees it for an honest witness; failing fast here beats
        // emitting a trace the known-column check then rejects). Unpinned words are free
        // witness — any u32, no 26-bit cap.
        for &(pos, value) in &self.routing_pins {
            assert_eq!(
                inputs.routing_words.get(pos).copied(),
                Some(value),
                "witness routing word at pinned stream position {pos} differs from the public outer index"
            );
        }

        // Plane byte streams, chunk-padded (the padding bytes are hashed).
        let routing_bytes: Vec<u8> = inputs.routing_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let offsets_bytes: Vec<u8> = inputs.offsets_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let raw_lens: [usize; NUM_PLANES] = [
            inputs.a_values.len(),
            inputs.a_scales.len(),
            inputs.b_values.len(),
            inputs.b_scales.len(),
            routing_bytes.len(),
            offsets_bytes.len(),
        ];
        let hash_id = |plane: PlaneId| match plane {
            PlaneId::AValues | PlaneId::AScales => inputs.a_hash_id,
            PlaneId::BValues | PlaneId::BScales => inputs.b_hash_id,
            PlaneId::Routing => inputs.routing_hash_id,
            PlaneId::Offsets => inputs.offsets_hash_id,
        };
        let planes: [Vec<u8>; NUM_PLANES] = [
            hash_id(PlaneId::AValues).pad(inputs.a_values),
            hash_id(PlaneId::AScales).pad(inputs.a_scales),
            hash_id(PlaneId::BValues).pad(inputs.b_values),
            hash_id(PlaneId::BScales).pad(inputs.b_scales),
            hash_id(PlaneId::Routing).pad(&routing_bytes),
            hash_id(PlaneId::Offsets).pad(&offsets_bytes),
        ];

        // Evaluate messages and CV chains natively before filling rows.
        let aux_cv_words = |idx: usize| -> [u32; 8] {
            core::array::from_fn(|i| u32::from_le_bytes(inputs.aux_cvs[idx][4 * i..4 * i + 4].try_into().unwrap()))
        };
        let instrs = &self.instructions;
        let mut msgs: Vec<[u32; 16]> = Vec::with_capacity(instrs.len());
        let mut cvs_in: Vec<[u32; 8]> = Vec::with_capacity(instrs.len());
        let mut cvs_out: Vec<[u32; 8]> = Vec::with_capacity(instrs.len());
        for instr in instrs {
            let m: [u32; 16] = match instr.msg {
                MessageSource::PlaneBytes { plane, offset, .. } => {
                    let bytes = &planes[plane.index()][offset..offset + 64];
                    core::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
                }
                MessageSource::PlaneBytesSplit {
                    plane,
                    offset,
                    skip,
                    take,
                    aux_idx,
                    ..
                } => {
                    // Opened dwords from the plane stream, the rest from the auxiliary block.
                    let mut bytes = inputs.aux_msgs[aux_idx];
                    let stream = &planes[plane.index()][offset..offset + NUM_UINT8 * take];
                    bytes[NUM_UINT8 * skip..NUM_UINT8 * (skip + take)].copy_from_slice(stream);
                    core::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
                }
                MessageSource::AuxBytes { idx } => {
                    let bytes = &inputs.aux_msgs[idx];
                    core::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
                }
                MessageSource::Parent { left, right } => {
                    let side = |child: CvRef| -> [u32; 8] {
                        match child {
                            CvRef::Instruction(src) => cvs_out[src],
                            CvRef::Auxiliary(idx) => aux_cv_words(idx),
                        }
                    };
                    let (l, r) = (side(left), side(right));
                    core::array::from_fn(|i| if i < 8 { l[i] } else { r[i - 8] })
                }
                MessageSource::Lottery => inputs.lottery_words,
            };
            let cv: [u32; 8] = match instr.cv {
                CvSource::KeyA => inputs.key_a,
                CvSource::KeyB => inputs.key_b,
                CvSource::JackpotKey => inputs.jackpot_key,
                CvSource::Iv => BLAKE3_IV,
                CvSource::Chain(src) => cvs_out[src],
            };
            let state = native_compress_state(&cv, &m, instr.counter, instr.block_len, instr.flags);
            msgs.push(m);
            cvs_in.push(cv);
            cvs_out.push(core::array::from_fn(|i| state[i] ^ state[8 + i]));
        }

        // Fill trace rows from those compression results.
        let num_live = self.num_live_rows();
        let num_rows = self.num_rows();
        let mut rows: Vec<[F; NUM_BLAKE3_COLUMNS]> = vec![[F::ZERO; NUM_BLAKE3_COLUMNS]; num_rows];
        let mut freq = vec![0u64; num_rows];
        // Deferred slice-bound limbs for the row after IS_BOUND_SECOND (at most one per job).
        let mut bound_write: Option<(usize, [F; 2])> = None;

        for (c, instr) in instrs.iter().enumerate() {
            let r0 = ROWS_PER_COMPRESSION * c;
            let m = msgs[c];

            // Flags, program columns, message bytes (the buffer itself is filled globally).
            for j in 0..ROWS_PER_COMPRESSION {
                let row: &mut Blake3ColumnsView<F> = rows[r0 + j].borrow_mut();
                row.trace_row_index = F::from_canonical_usize(r0 + j);
                if j == 0 {
                    row.is_new_blake = F::ONE;
                    match instr.cv {
                        CvSource::KeyA => row.is_use_key_a = F::ONE,
                        CvSource::KeyB => row.is_use_key_b = F::ONE,
                        CvSource::JackpotKey => row.is_use_jackpot_key = F::ONE,
                        CvSource::Iv => row.is_use_iv = F::ONE,
                        CvSource::Chain(_) => {} // IS_CV_IN doubles as the mux selector (fetch below).
                    }
                    for i in 0..8 {
                        row.blake3_cv[i] = F::from_canonical_u32(cvs_in[c][i]);
                    }
                }
                if j == 1 {
                    row.cv_route_key_or_tweak = F::from_canonical_u64(instr.tweak_packed());
                }
                if j == ROWS_PER_COMPRESSION - 1 {
                    row.is_last_round = F::ONE;
                    match instr.bind {
                        Some(PublicBinding::HashA) => row.is_bind_hash_a = F::ONE,
                        Some(PublicBinding::HashB) => row.is_bind_hash_b = F::ONE,
                        Some(PublicBinding::HashRouting) => row.is_bind_routing_hash = F::ONE,
                        Some(PublicBinding::HashOffsets) => row.is_bind_offsets_hash = F::ONE,
                        _ => {}
                    }
                }
                if instr.bind == Some(PublicBinding::HashJackpot) {
                    row.is_bind_jackpot_hash = F::ONE; // All 8 rows (CTL filter; see columns.rs).
                }

                match instr.msg {
                    MessageSource::PlaneBytes { plane, offset, ctl_base } => {
                        let byte_offset = offset + NUM_UINT8 * j;
                        let is_moe_plane = plane == PlaneId::Routing || plane == PlaneId::Offsets;
                        set_msg_mode(row, if is_moe_plane { MODE_AUX } else { MODE_BYTES });
                        let bytes = &planes[plane.index()][byte_offset..byte_offset + NUM_UINT8];
                        for (i, &b) in bytes.iter().enumerate() {
                            row.uint8_data[i] = F::from_canonical_u8(b);
                        }
                        // MoE order chains: the gates and their limb witnesses.
                        // `set_moe_chain` fails fast if the witness words are out of order.
                        let stream_word =
                            |i: usize| u32::from_le_bytes(planes[plane.index()][4 * i..4 * i + 4].try_into().unwrap());
                        let live = byte_offset + NUM_UINT8 <= raw_lens[plane.index()];
                        match plane {
                            // One flag per channel serves both sides: `ctl_base` already
                            // carries the B-plane key offset, keeping the key spaces disjoint.
                            PlaneId::AValues | PlaneId::BValues if live => {
                                row.is_int8_message = F::ONE;
                                row.ctl_key_base = F::from_canonical_u64(ctl_base + (NUM_UINT8 * j) as u64);
                            }
                            PlaneId::AScales | PlaneId::BScales if live => {
                                row.is_scale_message = F::ONE;
                                row.ctl_key_base = F::from_canonical_u64(ctl_base + (NUM_UINT8 * j / 2) as u64);
                            }
                            PlaneId::Routing => {
                                let base = offset / 4 + 2 * j;
                                set_outer(row, self.routing_pin_at(base), self.routing_pin_at(base + 1));
                                let s = self.moe.as_ref().expect("routing rows imply a MoE schedule");
                                row.is_chain_data = F::ONE;
                                row.is_chain_strict = F::ONE;
                                let g = (16 * s.routing_block_base() + base) as u64;
                                let (intra, inter, bound_first, bound_second) = s.routing_chain_at(g);
                                if intra {
                                    row.is_chain_intra = F::ONE;
                                    set_moe_chain(&mut row.chain_intra_limbs, stream_word(base + 1), stream_word(base), 1);
                                }
                                if inter {
                                    row.is_chain_inter = F::ONE;
                                    set_moe_chain(&mut row.chain_inter_limbs, stream_word(base), stream_word(base - 1), 1);
                                }
                                if bound_first {
                                    // The slice's last value is `word0`: the intra gate is
                                    // off, so its limbs hold `m - 1 - word0` instead.
                                    debug_assert!(!intra, "the intra gate must be idle on the bound row");
                                    row.is_bound_first = F::ONE;
                                    row.word_pin_first = F::from_canonical_u32(s.m - 1);
                                    set_moe_chain(&mut row.chain_intra_limbs, s.m - 1, stream_word(base), 0);
                                }
                                if bound_second {
                                    // The slice's last value is `word1`: the next row's
                                    // inter gate is off, so its limbs hold `m - 1 - word1`.
                                    // Written after the fill loop, since that row may
                                    // belong to the next compression.
                                    row.is_bound_second = F::ONE;
                                    row.word_pin_second = F::from_canonical_u32(s.m - 1);
                                    let mut limbs = [F::ZERO; 2];
                                    set_moe_chain(&mut limbs, s.m - 1, stream_word(base + 1), 0);
                                    bound_write = Some((r0 + j + 1, limbs));
                                }
                            }
                            PlaneId::Offsets => {
                                let s = self.moe.as_ref().expect("offsets rows imply a MoE schedule");
                                row.is_chain_data = F::ONE;
                                let p = offset / 4 + 2 * j;
                                for (slot, flag, pin) in [
                                    (p, &mut row.is_word_pin_first, &mut row.word_pin_first),
                                    (p + 1, &mut row.is_word_pin_second, &mut row.word_pin_second),
                                ] {
                                    if let Some(v) = s.offsets_pin_at(slot) {
                                        assert_eq!(
                                            u64::from(stream_word(slot)),
                                            v,
                                            "offsets word {slot} differs from its public pin"
                                        );
                                        *flag = F::ONE;
                                        *pin = F::from_canonical_u64(v);
                                    }
                                }
                                let (intra, inter) = s.offsets_chain_at(p);
                                if intra {
                                    row.is_chain_intra = F::ONE;
                                    set_moe_chain(&mut row.chain_intra_limbs, stream_word(p + 1), stream_word(p), 0);
                                }
                                if inter {
                                    row.is_chain_inter = F::ONE;
                                    set_moe_chain(&mut row.chain_inter_limbs, stream_word(p), stream_word(p - 1), 0);
                                }
                            }
                            _ => {} // Chunk-padding row: channel filters stay off.
                        }
                    }
                    MessageSource::PlaneBytesSplit {
                        plane,
                        offset,
                        ctl_base,
                        skip,
                        take,
                        aux_idx,
                    } => {
                        if (skip..skip + take).contains(&j) {
                            // Opened dword: exactly a `PlaneBytes` row of the window.
                            let byte_offset = offset + NUM_UINT8 * (j - skip);
                            set_msg_mode(row, MODE_BYTES);
                            let bytes = &planes[plane.index()][byte_offset..byte_offset + NUM_UINT8];
                            for (i, &b) in bytes.iter().enumerate() {
                                row.uint8_data[i] = F::from_canonical_u8(b);
                            }
                            let live = byte_offset + NUM_UINT8 <= raw_lens[plane.index()];
                            match plane {
                                PlaneId::AValues | PlaneId::BValues if live => {
                                    row.is_int8_message = F::ONE;
                                    row.ctl_key_base = F::from_canonical_u64(ctl_base + (NUM_UINT8 * (j - skip)) as u64);
                                }
                                PlaneId::AScales | PlaneId::BScales if live => {
                                    row.is_scale_message = F::ONE;
                                    row.ctl_key_base = F::from_canonical_u64(ctl_base + (NUM_UINT8 * (j - skip) / 2) as u64);
                                }
                                _ => {}
                            }
                        } else {
                            // Unopened-neighbor dword: an auxiliary-bytes row, filters off.
                            set_msg_mode(row, MODE_AUX);
                            for i in 0..NUM_UINT8 {
                                row.uint8_data[i] = F::from_canonical_u8(inputs.aux_msgs[aux_idx][NUM_UINT8 * j + i]);
                            }
                        }
                    }
                    MessageSource::AuxBytes { idx } => {
                        set_msg_mode(row, MODE_AUX);
                        for i in 0..NUM_UINT8 {
                            row.uint8_data[i] = F::from_canonical_u8(inputs.aux_msgs[idx][NUM_UINT8 * j + i]);
                        }
                    }
                    MessageSource::Parent { left, right } => {
                        let (child, local) = if j < 4 { (left, j) } else { (right, j - 4) };
                        match child {
                            // An auxiliary sibling CV: its 32 bytes enter as 4 dword rows.
                            CvRef::Auxiliary(idx) => {
                                set_msg_mode(row, MODE_AUX);
                                for i in 0..NUM_UINT8 {
                                    row.uint8_data[i] = F::from_canonical_u8(inputs.aux_cvs[idx][NUM_UINT8 * local + i]);
                                }
                            }
                            // An in-table child: one CV-window fetch on the half's last row.
                            CvRef::Instruction(_) if local == 3 => set_msg_mode(row, MODE_CV),
                            CvRef::Instruction(_) => {}
                        }
                    }
                    MessageSource::Lottery => {
                        // The 16 folded words as witness bytes (BYTES2-checked like any block);
                        // the XorFold CTL pins BLAKE3_MSG on row 0.
                        set_msg_mode(row, MODE_AUX);
                        for i in 0..NUM_UINT8 {
                            row.uint8_data[i] = F::from_canonical_u8(m[2 * j + i / 4].to_le_bytes()[i % 4]);
                        }
                    }
                }
            }

            // CV-routing fetches.
            for (j, src) in instr.fetches() {
                let src_row = ROWS_PER_COMPRESSION * src + (ROWS_PER_COMPRESSION - 1);
                freq[src_row] += 1;
                let row: &mut Blake3ColumnsView<F> = rows[r0 + j].borrow_mut();
                row.is_cv_in = F::ONE;
                row.cv_route_key_or_tweak = F::from_canonical_usize(src_row);
                for i in 0..8 {
                    row.cv_in[i] = F::from_canonical_u32(cvs_out[src][i]);
                    // The mux (constraint 3) holds on every row: with IS_CV_IN set it forces
                    // BLAKE3_CV = CV_IN here too (harmless off IS_NEW_BLAKE rows).
                    row.blake3_cv[i] = F::from_canonical_u32(cvs_out[src][i]);
                }
                // Row-0 fetches are chunk chaining (mux path); rows 3/7 are parent window
                // fetches — the fetched CV words ARE message words, so the sliding buffer
                // below already carries them and the window constraint (5) closes.
            }

            // Message words: row 0 holds the message; each next row is the permutation of the
            // previous (permute^8 = id closes against the buffer at row 7).
            let mut msg_j = m;
            for j in 0..ROWS_PER_COMPRESSION {
                let row: &mut Blake3ColumnsView<F> = rows[r0 + j].borrow_mut();
                for i in 0..16 {
                    row.blake3_msg[i] = F::from_canonical_u32(msg_j[i]);
                }
                blake3_permute(&mut msg_j);
            }
            debug_assert_eq!(msg_j, m);

            // Round states (rows 0..6) and per-row CV_OUT.
            let mut state = init_state(&cvs_in[c], instr);
            let mut msg_j = m;
            for j in 0..ROWS_PER_COMPRESSION - 1 {
                let row: &mut Blake3ColumnsView<F> = rows[r0 + j].borrow_mut();
                write_state(&mut row.round[0], &state);
                let row_input = state;
                let states = compute_blake3_round(&mut state, &msg_j);
                for (t, s) in states[..3].iter().enumerate() {
                    write_state(&mut row.round[t + 1], s);
                }
                for i in 0..8 {
                    // The unconditional CV_OUT = finalize-expression identity, on round rows.
                    let v = if i < 4 {
                        states[0][4 + i] ^ states[0][12 + i]
                    } else {
                        row_input[i] ^ row_input[8 + i]
                    };
                    row.cv_out[i] = F::from_canonical_u32(v);
                }
                blake3_permute(&mut msg_j);
            }
            let row: &mut Blake3ColumnsView<F> = rows[r0 + ROWS_PER_COMPRESSION - 1].borrow_mut();
            write_state(&mut row.round[0], &state);
            for i in 0..8 {
                row.cv_out[i] = F::from_canonical_u32(state[i] ^ state[8 + i]);
                debug_assert_eq!(row.cv_out[i], F::from_canonical_u32(cvs_out[c][i]));
            }
        }

        // Padding rows: IS_NEW_BLAKE with the zero CV / zero tweak / zero message.
        for (r, row) in rows.iter_mut().enumerate().take(num_rows).skip(num_live) {
            let row: &mut Blake3ColumnsView<F> = row.borrow_mut();
            row.is_new_blake = F::ONE;
            row.trace_row_index = F::from_canonical_usize(r); // Constraint 8 spans padding too.
            for i in 0..4 {
                row.round[0].c_words[i] = F::from_canonical_u32(BLAKE3_IV[i]);
                // The finalize expression on padding rows: [IV[0..4] ^ 0, 0 ^ 0].
                row.cv_out[i] = F::from_canonical_u32(BLAKE3_IV[i]);
            }
        }

        // Sliding message buffer: a 16-word window over the concatenated message stream
        // (2 words appended per live row; padding rows append nothing and shift zeros
        // in). This makes the unconditional shift-by-2 (constraint 5a) hold everywhere,
        // and the CV-window pins (5b) are consistent automatically because the fetched
        // CVs are message words.
        let mut buffer = [0u32; 16];
        for r in 0..num_rows {
            buffer = shift_buffer(&buffer);
            if r < num_live {
                let (c, j) = (r / ROWS_PER_COMPRESSION, r % ROWS_PER_COMPRESSION);
                buffer[14] = msgs[c][2 * j];
                buffer[15] = msgs[c][2 * j + 1];
            }
            let row: &mut Blake3ColumnsView<F> = rows[r].borrow_mut();
            for i in 0..16 {
                row.blake3_msg_buffer[i] = F::from_canonical_u32(buffer[i]);
            }
            if r % ROWS_PER_COMPRESSION == ROWS_PER_COMPRESSION - 1 && r < num_live {
                debug_assert_eq!(buffer, msgs[r / ROWS_PER_COMPRESSION], "buffer(7) must be the message");
            }
        }
        // Wrap fixup: the unconditional shift also binds the last -> first row pair, so rows
        // 0..6 must carry the shifted-out residue of the last row's buffer in their prefixes
        // (the residue never reaches a compression's message).
        let mut fwd = buffer;
        for off in 1..=(ROWS_PER_COMPRESSION - 1) {
            fwd = shift_buffer(&fwd);
            let row: &mut Blake3ColumnsView<F> = rows[(num_rows - 1 + off) % num_rows].borrow_mut();
            for i in 0..16 - 2 * off {
                row.blake3_msg_buffer[i] = F::from_canonical_u32(fwd[i]);
            }
        }

        // Write the deferred slice bound into the row after IS_BOUND_SECOND: its inter
        // limbs are idle, since that row's word pair starts past the slice.
        if let Some((r, limbs)) = bound_write {
            let row: &mut Blake3ColumnsView<F> = rows[r].borrow_mut();
            assert_eq!(
                row.is_chain_inter,
                F::ZERO,
                "the bound target row must have its inter gate down"
            );
            row.chain_inter_limbs = limbs;
        }

        // Fill `chain_carry`: each row holds the last routing/offsets word before it.
        // Data rows overwrite it with their second word, all other rows copy it. The
        // first pass computes the end-of-trace value: the constraint is cyclic, so
        // row 0 checks against the last row.
        let carry_after = |row: &Blake3ColumnsView<F>, prev: F| -> F {
            if row.is_chain_data == F::ONE {
                F::from_canonical_u64((0..4).map(|i| row.uint8_data[4 + i].to_canonical_u64() << (8 * i)).sum())
            } else {
                prev
            }
        };
        let mut carry = F::ZERO;
        for row in &rows {
            carry = carry_after(row.borrow(), carry);
        }
        for row in rows.iter_mut() {
            let row: &mut Blake3ColumnsView<F> = row.borrow_mut();
            row.chain_carry = carry;
            carry = carry_after(row, carry);
        }

        // Postprocess rows whose next row starts a compression (real finalization rows,
        // padding rows, and the cyclic wrap): fill STATE1..3 so the unconditional
        // add2/add3 constraints hold and STATE1 carries the finalize bit decompositions.
        for r in 0..num_rows {
            let next = (r + 1) % num_rows;
            {
                let next_row: &Blake3ColumnsView<F> = rows[next].borrow();
                if next_row.is_new_blake != F::ONE {
                    continue;
                }
            }
            let (nr1, nr3, nr4) = {
                let next_row: &Blake3ColumnsView<F> = rows[next].borrow();
                (
                    read_words(&next_row.round[0].a_words),
                    read_words(&next_row.round[0].c_words),
                    read_bits(&next_row.round[0].d_bits),
                )
            };
            let row: &mut Blake3ColumnsView<F> = rows[r].borrow_mut();
            let msg: [u32; 16] = core::array::from_fn(|i| row.blake3_msg[i].to_canonical_u64() as u32);
            let r1 = read_words(&row.round[0].a_words);
            let r2 = read_bits(&row.round[0].b_bits);
            let r3 = read_words(&row.round[0].c_words);

            // STATE1: b_bits/d_bits repurposed as the finalize decompositions of a_words/c_words.
            let s1r1: [u32; 4] = core::array::from_fn(|i| r1[i].wrapping_add(r2[i]).wrapping_add(msg[2 * i]));
            let s1r3: [u32; 4] = core::array::from_fn(|i| r3[i].wrapping_add(r3[i]));
            write_words(&mut row.round[1].a_words, &s1r1);
            write_bits(&mut row.round[1].b_bits, &r1);
            write_words(&mut row.round[1].c_words, &s1r3);
            write_bits(&mut row.round[1].d_bits, &r3);

            // STATE2: bit halves zero; word halves from the unconditional adds.
            let s2r1: [u32; 4] = core::array::from_fn(|i| s1r1[i].wrapping_add(r1[i]).wrapping_add(msg[2 * i + 1]));
            write_words(&mut row.round[2].a_words, &s2r1);
            write_bits(&mut row.round[2].b_bits, &[0; 4]);
            write_words(&mut row.round[2].c_words, &s1r3);
            write_bits(&mut row.round[2].d_bits, &[0; 4]);

            // STATE3: solved backwards from the next row's input state (diagonal adds).
            let s3r1: [u32; 4] = core::array::from_fn(|i| s2r1[i].wrapping_add(msg[8 + 2 * i]));
            let (mut s3r2, mut s3r3, mut s3r4) = ([0u32; 4], [0u32; 4], [0u32; 4]);
            for i in 0..4 {
                let (b, c, d) = ((i + 1) % 4, (i + 2) % 4, (i + 3) % 4);
                s3r3[c] = nr3[c].wrapping_sub(nr4[d]);
                s3r4[d] = s3r3[c].wrapping_sub(s1r3[c]);
                s3r2[b] = nr1[i].wrapping_sub(s3r1[i]).wrapping_sub(msg[8 + 2 * i + 1]);
            }
            write_words(&mut row.round[3].a_words, &s3r1);
            write_bits(&mut row.round[3].b_bits, &s3r2);
            write_words(&mut row.round[3].c_words, &s3r3);
            write_bits(&mut row.round[3].d_bits, &s3r4);
        }

        // CV-routing multiplicities and the ROW_FLAGS_PACKED packing.
        for (r, row) in rows.iter_mut().enumerate() {
            let row: &mut Blake3ColumnsView<F> = row.borrow_mut();
            row.cv_out_freq = F::from_canonical_u64(freq[r]);
            row.row_flags_packed = pack_control(row);
        }

        // Public inputs.
        let bound_cv = |bind: PublicBinding| -> [u32; 8] {
            instrs
                .iter()
                .position(|i| i.bind == Some(bind))
                .map(|i| cvs_out[i])
                .unwrap_or([0; 8])
        };
        let mut pis = [F::ZERO; NUM_BLAKE3_PUBLIC_INPUTS];
        let mut set = |base: usize, words: &[u32; 8]| {
            for i in 0..8 {
                pis[base + i] = F::from_canonical_u32(words[i]);
            }
        };
        set(PI_KEY_A, &inputs.key_a);
        set(PI_KEY_B, &inputs.key_b);
        set(PI_JACKPOT_KEY, &inputs.jackpot_key);
        set(PI_HASH_A, &bound_cv(PublicBinding::HashA));
        set(PI_HASH_B, &bound_cv(PublicBinding::HashB));
        set(PI_HASH_ROUTING, &bound_cv(PublicBinding::HashRouting));
        set(PI_HASH_OFFSETS, &bound_cv(PublicBinding::HashOffsets));
        set(PI_HASH_JACKPOT, &bound_cv(PublicBinding::HashJackpot));

        (rows, pis)
    }

    /// Recomputes the leading schedule columns in trace order from public geometry.
    ///
    /// The batch verifier checks their trace openings against these values. This
    /// binds the compression schedule, CV-source selectors and public word pins
    /// without trusting the prover's choice of flags. No witness bytes are read.
    pub fn known_values<F: RichField>(&self, inputs: &Blake3KnownInputs) -> Vec<PolynomialValues<F>> {
        self.validate();
        let num_live = self.num_live_rows();
        let num_rows = self.num_rows();
        // The routing/offsets planes have no liveness length: their rows' verifier-known data
        // (selectors, pins, chain gates) is driven by the MoE schedule, never by a byte count.
        let raw_lens: [usize; NUM_PLANES] = [
            inputs.a_values_len,
            inputs.a_scales_len,
            inputs.b_values_len,
            inputs.b_scales_len,
            0,
            0,
        ];

        // `ROW_FLAGS_PACKED` bit packing (the `pack_control` weights).
        const MODE_BYTES_BITS: u64 = 1 << 11;
        const MODE_AUX_BITS: u64 = (1 << 12) | (1 << 13);
        const MODE_CV_BITS: u64 = 1 << 13;

        let mut row_flags_packed = vec![0u64; num_rows];
        let mut ctl_key_base = vec![F::ZERO; num_rows];
        let mut is_int8_message = vec![F::ZERO; num_rows];
        let mut is_scale_message = vec![F::ZERO; num_rows];
        let mut cv_route_key_or_tweak = vec![F::ZERO; num_rows];
        let mut moe_outer_indices_packed = vec![F::ZERO; num_rows];
        let mut word_pin_first = vec![F::ZERO; num_rows];
        let mut word_pin_second = vec![F::ZERO; num_rows];

        for (c, instr) in self.instructions.iter().enumerate() {
            let r0 = ROWS_PER_COMPRESSION * c;
            for j in 0..ROWS_PER_COMPRESSION {
                let r = r0 + j;
                let mut bits = 0u64;
                if j == 0 {
                    bits |= 1 << 9; // IS_NEW_BLAKE
                    match instr.cv {
                        CvSource::KeyA => bits |= 1 << 0,
                        CvSource::KeyB => bits |= 1 << 1,
                        CvSource::JackpotKey => bits |= 1 << 2,
                        CvSource::Iv => bits |= 1 << 3,
                        CvSource::Chain(_) => {} // the row-0 fetch below carries IS_CV_IN instead
                    }
                }
                if j == 1 {
                    cv_route_key_or_tweak[r] = F::from_canonical_u64(instr.tweak_packed());
                }
                if j == ROWS_PER_COMPRESSION - 1 {
                    bits |= 1 << 10; // IS_LAST_ROUND
                    match instr.bind {
                        Some(PublicBinding::HashA) => bits |= 1 << 4,
                        Some(PublicBinding::HashB) => bits |= 1 << 5,
                        Some(PublicBinding::HashRouting) => bits |= 1 << 6,
                        Some(PublicBinding::HashOffsets) => bits |= 1 << 16,
                        _ => {}
                    }
                }
                if instr.bind == Some(PublicBinding::HashJackpot) {
                    bits |= 1 << 7; // all 8 rows (the lottery CTL filter, see columns.rs)
                }

                match instr.msg {
                    MessageSource::PlaneBytes { plane, offset, ctl_base } => {
                        bits |= if plane == PlaneId::Routing || plane == PlaneId::Offsets {
                            MODE_AUX_BITS
                        } else {
                            MODE_BYTES_BITS
                        };
                        let byte_offset = offset + NUM_UINT8 * j;
                        let live = byte_offset + NUM_UINT8 <= raw_lens[plane.index()];
                        match plane {
                            PlaneId::AValues | PlaneId::BValues if live => {
                                is_int8_message[r] = F::ONE;
                                ctl_key_base[r] = F::from_canonical_u64(ctl_base + (NUM_UINT8 * j) as u64);
                            }
                            PlaneId::AScales | PlaneId::BScales if live => {
                                is_scale_message[r] = F::ONE;
                                ctl_key_base[r] = F::from_canonical_u64(ctl_base + (NUM_UINT8 * j / 2) as u64);
                            }
                            PlaneId::Routing => {
                                // A selector
                                // fires only where this row ingests a sampled entry, and the
                                // packed word carries only the pinned public values.
                                let base = offset / 4 + 2 * j;
                                let (w0, w1) = (self.routing_pin_at(base), self.routing_pin_at(base + 1));
                                if w0.is_some() {
                                    bits |= 1 << 14; // IS_FIRST_OUTER
                                }
                                if w1.is_some() {
                                    bits |= 1 << 15; // IS_SECOND_OUTER
                                }
                                moe_outer_indices_packed[r] = F::from_canonical_u64(w0.unwrap_or(0) + (w1.unwrap_or(0) << 26));
                                let s = self.moe.as_ref().expect("routing rows imply a MoE schedule");
                                bits |= 1 << 21; // IS_CHAIN_STRICT
                                bits |= 1 << 24; // IS_CHAIN_DATA
                                let g = (16 * s.routing_block_base() + base) as u64;
                                let (intra, inter, bound_first, bound_second) = s.routing_chain_at(g);
                                bits |= u64::from(intra) << 19;
                                bits |= u64::from(inter) << 20;
                                bits |= u64::from(bound_first) << 22;
                                bits |= u64::from(bound_second) << 23;
                                if bound_first {
                                    word_pin_first[r] = F::from_canonical_u32(s.m - 1);
                                }
                                if bound_second {
                                    word_pin_second[r] = F::from_canonical_u32(s.m - 1);
                                }
                            }
                            PlaneId::Offsets => {
                                let s = self.moe.as_ref().expect("offsets rows imply a MoE schedule");
                                bits |= 1 << 24; // IS_CHAIN_DATA
                                let p = offset / 4 + 2 * j;
                                if let Some(v) = s.offsets_pin_at(p) {
                                    bits |= 1 << 17; // IS_WORD_PIN_FIRST
                                    word_pin_first[r] = F::from_canonical_u64(v);
                                }
                                if let Some(v) = s.offsets_pin_at(p + 1) {
                                    bits |= 1 << 18; // IS_WORD_PIN_SECOND
                                    word_pin_second[r] = F::from_canonical_u64(v);
                                }
                                let (intra, inter) = s.offsets_chain_at(p);
                                bits |= u64::from(intra) << 19;
                                bits |= u64::from(inter) << 20;
                            }
                            _ => {} // Chunk-padding row: channel filters stay off.
                        }
                    }
                    MessageSource::PlaneBytesSplit {
                        plane,
                        ctl_base,
                        offset,
                        skip,
                        take,
                        ..
                    } => {
                        if (skip..skip + take).contains(&j) {
                            bits |= MODE_BYTES_BITS;
                            let byte_offset = offset + NUM_UINT8 * (j - skip);
                            let live = byte_offset + NUM_UINT8 <= raw_lens[plane.index()];
                            match plane {
                                PlaneId::AValues | PlaneId::BValues if live => {
                                    is_int8_message[r] = F::ONE;
                                    ctl_key_base[r] = F::from_canonical_u64(ctl_base + (NUM_UINT8 * (j - skip)) as u64);
                                }
                                PlaneId::AScales | PlaneId::BScales if live => {
                                    is_scale_message[r] = F::ONE;
                                    ctl_key_base[r] = F::from_canonical_u64(ctl_base + (NUM_UINT8 * (j - skip) / 2) as u64);
                                }
                                _ => {}
                            }
                        } else {
                            bits |= MODE_AUX_BITS;
                        }
                    }
                    MessageSource::AuxBytes { .. } | MessageSource::Lottery => bits |= MODE_AUX_BITS,
                    MessageSource::Parent { left, right } => {
                        let child = if j < ROWS_PER_COMPRESSION / 2 { left } else { right };
                        match child {
                            CvRef::Auxiliary(_) => bits |= MODE_AUX_BITS,
                            CvRef::Instruction(_) if j % 4 == 3 => bits |= MODE_CV_BITS,
                            CvRef::Instruction(_) => {}
                        }
                    }
                }
                row_flags_packed[r] = bits;
            }

            // CV-routing fetches: IS_CV_IN plus the source-row pointer (rows 0/3/7 — never the
            // tweak row 1).
            for (j, src) in instr.fetches() {
                row_flags_packed[r0 + j] |= 1 << 8;
                cv_route_key_or_tweak[r0 + j] = F::from_canonical_usize(ROWS_PER_COMPRESSION * src + (ROWS_PER_COMPRESSION - 1));
            }
        }

        // Padding rows carry IS_NEW_BLAKE alone.
        for bits in row_flags_packed.iter_mut().take(num_rows).skip(num_live) {
            *bits = 1 << 9;
        }

        let row_flags_packed: Vec<F> = row_flags_packed.into_iter().map(F::from_canonical_u64).collect();
        [
            row_flags_packed,
            ctl_key_base,
            is_int8_message,
            is_scale_message,
            cv_route_key_or_tweak,
            moe_outer_indices_packed,
            word_pin_first,
            word_pin_second,
        ]
        .into_iter()
        .map(PolynomialValues::new)
        .collect()
    }
}

/// Witness inputs of one job's hashing trace: the committed plane bytes (opened strips), the
/// auxiliary Merkle material (unopened blocks / sibling CVs), the lottery words and the keys.
#[derive(Clone, Debug)]
pub struct Blake3TraceInputs<'a> {
    /// A-side int8 values plane, raw bytes (the sampled strip rows, row-major).
    pub a_values: &'a [u8],
    /// A-side bf16 scales plane, raw bytes (LE bf16 code per 8-element block).
    pub a_scales: &'a [u8],
    /// B-side int8 values plane, raw bytes.
    pub b_values: &'a [u8],
    /// B-side bf16 scales plane, raw bytes.
    pub b_scales: &'a [u8],
    /// The opened routing hotspot blocks as u32 words, 16 per block, hotspot order. **Witness
    /// data**: only the words at [`Blake3Program::routing_pins`] positions are publicly
    /// pinned (and must equal the pinned values); the rest are free neighbors, bound only by
    /// the hash chain to `HASH_ROUTING`.
    pub routing_words: &'a [u32],
    /// The full offsets list `O` as u32 words (chunk padding appended by the trace generator).
    /// **Witness data** bound to `HASH_OFFSETS`; the [`MoeSchedule`] scalars are pinned
    /// in-circuit at their stream positions. Empty for dense jobs.
    pub offsets_words: &'a [u32],
    /// Auxiliary 64-byte messages (unopened blocks, plus the full block of every split leaf),
    /// indexed by [`MessageSource::AuxBytes`] / [`MessageSource::PlaneBytesSplit`].
    pub aux_msgs: &'a [[u8; 64]],
    /// Auxiliary sibling CVs (unopened subtree roots), indexed by [`CvRef::Auxiliary`].
    pub aux_cvs: &'a [[u8; 32]],
    /// The 16 folded lottery words (XorFoldStark's `FOLD_OUT`s).
    pub lottery_words: [u32; 16],
    /// `KEY_A` as 8 LE u32 limbs of the 32-byte A-side tree key.
    pub key_a: [u32; 8],
    /// `KEY_B` limbs (the B-side tree key).
    pub key_b: [u32; 8],
    /// `JACKPOT_KEY` limbs (the lottery key).
    pub jackpot_key: [u32; 8],
    /// Merkle leaf sizes from the job `HashId`s (routing is 1024 when dense).
    pub a_hash_id: HashId,
    pub b_hash_id: HashId,
    pub routing_hash_id: HashId,
    pub offsets_hash_id: HashId,
}

impl Blake3TraceInputs<'_> {
    /// The public projection of these inputs (what [`Blake3Program::known_values`] runs on):
    /// raw plane lengths only — no witness bytes. The MoE pin schedule is part of the
    /// [`Blake3Program`] itself ([`Blake3Program::routing_pins`]), not of these inputs.
    pub fn known_inputs(&self) -> Blake3KnownInputs {
        Blake3KnownInputs {
            a_values_len: self.a_values.len(),
            a_scales_len: self.a_scales.len(),
            b_values_len: self.b_values.len(),
            b_scales_len: self.b_scales.len(),
        }
    }
}

/// The public inputs of [`Blake3Program::known_values`]: the raw (pre-chunk-padding) plane
/// byte lengths — they gate which message rows are *live* for the CTL channels. All
/// verifier-side data; the MoE outer-index pins live on the [`Blake3Program`] itself.
#[derive(Clone, Copy, Debug)]
pub struct Blake3KnownInputs {
    /// `Blake3TraceInputs::a_values.len()` (raw A-side int8 plane bytes).
    pub a_values_len: usize,
    /// Raw A-side scales plane byte length.
    pub a_scales_len: usize,
    /// Raw B-side values plane byte length.
    pub b_values_len: usize,
    /// Raw B-side scales plane byte length.
    pub b_scales_len: usize,
}

/// Message modes for plane bytes, auxiliary bytes and CV windows; mode 010 is forbidden.
const MODE_BYTES: [bool; 3] = [true, false, false];
const MODE_AUX: [bool; 3] = [false, true, true];
const MODE_CV: [bool; 3] = [false, false, true];

fn set_msg_mode<F: RichField>(row: &mut Blake3ColumnsView<F>, mode: [bool; 3]) {
    for i in 0..3 {
        row.is_msg_bits[i] = F::from_bool(mode[i]);
    }
}

/// Fills each sampled routing slot's selector and 13-bit limbs.
/// Unsampled slots have zero selectors/limbs; their message words remain witness data.
fn set_outer<F: RichField>(row: &mut Blake3ColumnsView<F>, w0: Option<u64>, w1: Option<u64>) {
    if let Some(w0) = w0 {
        assert!(w0 < 1 << 26, "outer-index words must fit 26 bits");
        row.is_first_outer = F::ONE;
        row.outer_index_first = [F::from_canonical_u64(w0 & 0x1FFF), F::from_canonical_u64(w0 >> 13)];
    }
    if let Some(w1) = w1 {
        assert!(w1 < 1 << 26, "outer-index words must fit 26 bits");
        row.is_second_outer = F::ONE;
        row.outer_index_second = [F::from_canonical_u64(w1 & 0x1FFF), F::from_canonical_u64(w1 >> 13)];
    }
    row.moe_outer_indices_packed = F::from_canonical_u64(w0.unwrap_or(0) + (w1.unwrap_or(0) << 26));
}

/// Writes `hi - lo - strict` as the 16-bit limb pair of a gated order chain (the AIR proves
/// `lo + strict <= hi` by exhibiting the difference in `[0, 2^32)`).
fn set_moe_chain<F: RichField>(limbs: &mut [F; 2], hi: u32, lo: u32, strict: u32) {
    let d = u64::from(hi)
        .checked_sub(u64::from(lo) + u64::from(strict))
        .expect("MoE order violated in the trace inputs");
    limbs[0] = F::from_canonical_u64(d & 0xFFFF);
    limbs[1] = F::from_canonical_u64(d >> 16);
}

/// Packs the [`NUM_UNPACK_FLAGS`] unpack flags (already written on the row) into
/// `ROW_FLAGS_PACKED`, bit `j` = flag `j` in the [`super::columns::unpack_flag_cols`] order.
fn pack_control<F: RichField>(row: &Blake3ColumnsView<F>) -> F {
    let flags = [
        row.is_use_key_a,
        row.is_use_key_b,
        row.is_use_jackpot_key,
        row.is_use_iv,
        row.is_bind_hash_a,
        row.is_bind_hash_b,
        row.is_bind_routing_hash,
        row.is_bind_jackpot_hash,
        row.is_cv_in,
        row.is_new_blake,
        row.is_last_round,
        row.is_msg_bits[0],
        row.is_msg_bits[1],
        row.is_msg_bits[2],
        row.is_first_outer,
        row.is_second_outer,
        row.is_bind_offsets_hash,
        row.is_word_pin_first,
        row.is_word_pin_second,
        row.is_chain_intra,
        row.is_chain_inter,
        row.is_chain_strict,
        row.is_bound_first,
        row.is_bound_second,
        row.is_chain_data,
    ];
    debug_assert_eq!(flags.len(), NUM_UNPACK_FLAGS);
    let mut packed = 0u64;
    for (j, f) in flags.into_iter().enumerate() {
        debug_assert!(f == F::ZERO || f == F::ONE);
        packed |= ((f == F::ONE) as u64) << j;
    }
    F::from_canonical_u64(packed)
}

/// The compression's initial 16-word state: `cv | IV[0..4] | counter | block_len | flags`.
fn init_state(cv: &[u32; 8], instr: &Blake3Instruction) -> [u32; 16] {
    core::array::from_fn(|i| match i {
        0..=7 => cv[i],
        8..=11 => BLAKE3_IV[i - 8],
        12 => instr.counter as u32,
        13 => (instr.counter >> 32) as u32,
        14 => instr.block_len,
        _ => instr.flags,
    })
}

/// `next_buffer[0..14] = buffer[2..16]`, tail zeroed (the AIR's unconditional shift).
fn shift_buffer(buffer: &[u32; 16]) -> [u32; 16] {
    core::array::from_fn(|i| if i < 14 { buffer[i + 2] } else { 0 })
}

/// One half quarter-round (the native mirror of the AIR's `half_g`).
fn half_quarter_round(mut a: u32, mut b: u32, mut c: u32, mut d: u32, m: u32, second_half: bool) -> (u32, u32, u32, u32) {
    let (rot_1, rot_2) = if second_half { (8, 7) } else { (16, 12) };
    a = a.wrapping_add(b).wrapping_add(m);
    d = (d ^ a).rotate_right(rot_1);
    c = c.wrapping_add(d);
    b = (b ^ c).rotate_right(rot_2);
    (a, b, c, d)
}

/// One full BLAKE3 round; returns the 4 intermediate states (the 4th is the next row's input).
fn compute_blake3_round(state: &mut [u32; 16], msg: &[u32; 16]) -> [[u32; 16]; 4] {
    let mut states = [[0u32; 16]; 4];
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(state[i], state[4 + i], state[8 + i], state[12 + i], msg[2 * i], false);
        (state[i], state[4 + i], state[8 + i], state[12 + i]) = (a, b, c, d);
    }
    states[0] = *state;
    for i in 0..4 {
        let (a, b, c, d) = half_quarter_round(state[i], state[4 + i], state[8 + i], state[12 + i], msg[2 * i + 1], true);
        (state[i], state[4 + i], state[8 + i], state[12 + i]) = (a, b, c, d);
    }
    states[1] = *state;
    for i in 0..4 {
        let (b, c, d) = (4 + (i + 1) % 4, 8 + (i + 2) % 4, 12 + (i + 3) % 4);
        let (na, nb, nc, nd) = half_quarter_round(state[i], state[b], state[c], state[d], msg[8 + 2 * i], false);
        (state[i], state[b], state[c], state[d]) = (na, nb, nc, nd);
    }
    states[2] = *state;
    for i in 0..4 {
        let (b, c, d) = (4 + (i + 1) % 4, 8 + (i + 2) % 4, 12 + (i + 3) % 4);
        let (na, nb, nc, nd) = half_quarter_round(state[i], state[b], state[c], state[d], msg[8 + 2 * i + 1], true);
        (state[i], state[b], state[c], state[d]) = (na, nb, nc, nd);
    }
    states[3] = *state;
    states
}

/// The 16-word state after the 7 rounds (the finalization row's input; the CV output is
/// `state[i] ^ state[8 + i]`).
fn native_compress_state(cv: &[u32; 8], m: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    let mut state: [u32; 16] = core::array::from_fn(|i| match i {
        0..=7 => cv[i],
        8..=11 => BLAKE3_IV[i - 8],
        12 => counter as u32,
        13 => (counter >> 32) as u32,
        14 => block_len,
        _ => flags,
    });
    let mut msg = *m;
    for _ in 0..7 {
        compute_blake3_round(&mut state, &msg);
        blake3_permute(&mut msg);
    }
    state
}

fn write_state<F: RichField>(dst: &mut Blake3StateCols<F>, state: &[u32; 16]) {
    write_words(&mut dst.a_words, &core::array::from_fn(|i| state[i]));
    write_bits(&mut dst.b_bits, &core::array::from_fn(|i| state[4 + i]));
    write_words(&mut dst.c_words, &core::array::from_fn(|i| state[8 + i]));
    write_bits(&mut dst.d_bits, &core::array::from_fn(|i| state[12 + i]));
}

fn write_words<F: RichField>(dst: &mut [F; 4], words: &[u32; 4]) {
    for i in 0..4 {
        dst[i] = F::from_canonical_u32(words[i]);
    }
}

fn write_bits<F: RichField>(dst: &mut [[F; 32]; 4], words: &[u32; 4]) {
    for i in 0..4 {
        for b in 0..32 {
            dst[i][b] = F::from_canonical_u32((words[i] >> b) & 1);
        }
    }
}

fn read_words<F: RichField>(src: &[F; 4]) -> [u32; 4] {
    core::array::from_fn(|i| src[i].to_canonical_u64() as u32)
}

fn read_bits<F: RichField>(src: &[[F; 32]; 4]) -> [u32; 4] {
    core::array::from_fn(|i| (0..32).fold(0u32, |acc, b| acc | (((src[i][b] == F::ONE) as u32) << b)))
}

// Constraints, written once against the generic `Evaluator`

/// Evaluates every arithmetic constraint of Blake3Stark. The CV-routing lookup is
/// declared in [`Blake3Stark::lookups`] and evaluated by the framework; the BYTES2/RC16
/// instances and the CTL halves are declared in `super::ctl` and assembled by the batch
/// driver (module docs).
pub(crate) fn eval_blake3_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_BLAKE3_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &Blake3ColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_BLAKE3_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &Blake3ColumnsView<V> = nv.borrow();
    let pis = vars.get_public_inputs();
    let public8 = |eval: &mut E, base: usize| -> [V; 8] { core::array::from_fn(|i| eval.scalar(pis[base + i])) };
    let key_a = public8(eval, PI_KEY_A);
    let key_b = public8(eval, PI_KEY_B);
    let jackpot_key = public8(eval, PI_JACKPOT_KEY);
    let hash_a = public8(eval, PI_HASH_A);
    let hash_b = public8(eval, PI_HASH_B);
    let hash_routing = public8(eval, PI_HASH_ROUTING);
    let hash_offsets = public8(eval, PI_HASH_OFFSETS);
    let hash_jackpot = public8(eval, PI_HASH_JACKPOT);

    let one = eval.i32(1);
    let two = eval.i32(2);
    let c256 = eval.i32(256);

    // 2. Unpack/repack: every flag boolean, weighted sum = ROW_FLAGS_PACKED.
    let flags = [
        lv.is_use_key_a,
        lv.is_use_key_b,
        lv.is_use_jackpot_key,
        lv.is_use_iv,
        lv.is_bind_hash_a,
        lv.is_bind_hash_b,
        lv.is_bind_routing_hash,
        lv.is_bind_jackpot_hash,
        lv.is_cv_in,
        lv.is_new_blake,
        lv.is_last_round,
        lv.is_msg_bits[0],
        lv.is_msg_bits[1],
        lv.is_msg_bits[2],
        lv.is_first_outer,
        lv.is_second_outer,
        lv.is_bind_offsets_hash,
        lv.is_word_pin_first,
        lv.is_word_pin_second,
        lv.is_chain_intra,
        lv.is_chain_inter,
        lv.is_chain_strict,
        lv.is_bound_first,
        lv.is_bound_second,
        lv.is_chain_data,
    ];
    debug_assert_eq!(flags.len(), NUM_UNPACK_FLAGS);
    for f in flags {
        eval.constraint_bool(f);
    }
    let repacked = eval.polyval(&flags, two);
    eval.constraint_eq(repacked, lv.row_flags_packed);
    // Boolean unpacking uniquely recovers the verifier-known schedule flags.
    // That schedule makes CV-source selectors one-hot-or-zero and starts row 0 at a
    // compression boundary, as required by the mux and cyclic constraints.

    // 3. Select the CV source.
    for i in 0..8 {
        let iv = eval.u64(u64::from(BLAKE3_IV[i]));
        let mut acc = eval.mul(lv.is_use_key_a, key_a[i]);
        acc = eval.mad(lv.is_use_key_b, key_b[i], acc);
        acc = eval.mad(lv.is_use_jackpot_key, jackpot_key[i], acc);
        acc = eval.mad(lv.is_use_iv, iv, acc);
        acc = eval.mad(lv.is_cv_in, lv.cv_in[i], acc);
        eval.constraint_eq(lv.blake3_cv[i], acc);
    }

    // 4. Public bindings: the side digests, routing/offsets hashes, the lottery output.
    for (flag, target) in [
        (lv.is_bind_hash_a, hash_a),
        (lv.is_bind_hash_b, hash_b),
        (lv.is_bind_routing_hash, hash_routing),
        (lv.is_bind_offsets_hash, hash_offsets),
    ] {
        for i in 0..8 {
            let diff = eval.sub(lv.cv_out[i], target[i]);
            let c = eval.mul(flag, diff);
            eval.constraint(c);
        }
    }
    for i in 0..8 {
        // IS_BIND_JACKPOT_HASH spans all 8 lottery rows (the XorFold CTL filter needs it on the
        // load row), so the binding is anchored at the finalization row explicitly (degree 3).
        let diff = eval.sub(lv.cv_out[i], hash_jackpot[i]);
        let gate = eval.mul(lv.is_bind_jackpot_hash, lv.is_last_round);
        let c = eval.mul(gate, diff);
        eval.constraint(c);
    }

    // 5. Shift the message buffer by two words, including across the cyclic wrap.
    for i in 0..14 {
        eval.constraint_eq(nv.blake3_msg_buffer[i], lv.blake3_msg_buffer[i + 2]);
    }
    let (is_msg_jackpot, is_msg_uint8_data, is_msg_cv) = decode_is_msg_bits(eval, one, lv.is_msg_bits);
    // Mode 010 is forbidden: lottery bytes are loaded normally and bound by the XorFold CTL.
    eval.constraint(is_msg_jackpot);
    // Byte-ingestion rows (plane or auxiliary bytes) load the packed byte pairs into the tail.
    let word0 = eval.polyval(&lv.uint8_data[0..4], c256);
    let word1 = eval.polyval(&lv.uint8_data[4..8], c256);
    eval.constraint_eq_if(is_msg_uint8_data, lv.blake3_msg_buffer[14], word0);
    eval.constraint_eq_if(is_msg_uint8_data, lv.blake3_msg_buffer[15], word1);
    // CV-window rows (parent child fetches, rows 3 and 7): the fetched CV fills words
    // 0..8 / 8..16 through the shift.
    for i in 0..8 {
        eval.constraint_eq_if(is_msg_cv, lv.blake3_msg_buffer[8 + i], lv.cv_in[i]);
    }
    // Round-to-round message permutation, and the row-7 closure against the buffer
    // (permute^8 = id, so buffer(7) is the message the compression consumed).
    let mut permuted = lv.blake3_msg;
    blake3_permute(&mut permuted);
    let next_same_blake = eval.sub(one, nv.is_new_blake);
    for i in 0..16 {
        eval.constraint_eq_if(next_same_blake, permuted[i], nv.blake3_msg[i]);
        eval.constraint_eq_if(lv.is_last_round, permuted[i], lv.blake3_msg_buffer[i]);
    }

    // 1. Compression rounds and finalization.
    let states = [&lv.round[0], &lv.round[1], &lv.round[2], &lv.round[3], &nv.round[0]];
    verify_round(eval, &states, &lv.blake3_msg, next_same_blake);
    let blake3_output = finalize_blake(eval, states[0], states[1], nv.is_new_blake);
    for i in 0..8 {
        eval.constraint_eq(lv.cv_out[i], blake3_output[i]);
    }
    // The packed tweak rides the *next* row's CV_ROUTE_KEY_OR_TWEAK (row 1 of the compression).
    verify_init_state(eval, states[0], lv.is_new_blake, &lv.blake3_cv, nv.cv_route_key_or_tweak);

    // 6. Bind sampled routing words to the public indices. In order, let l_0..l_3
    // be the two limbs of outer_index_first followed by those of outer_index_second:
    //
    //   packed = l_0 + 2^13*l_1 + 2^26*l_2 + 2^39*l_3, with 0 <= l_i < 2^13.
    //
    // Thus packed < 2^52, below the field modulus, and its base-2^13 decomposition is unique.
    //
    // Limb bounds must apply even to unsampled slots: otherwise an unchecked limb
    // could offset a change in a sampled one while preserving packed. Unsampled slots
    // have zero limbs; their message words remain bound by the routing hash and order checks.
    let limb_base = eval.u64(1 << 13);
    let outer_first = eval.polyval(&lv.outer_index_first, limb_base);
    let outer_second = eval.polyval(&lv.outer_index_second, limb_base);
    let c2_26 = eval.u64(1 << 26);
    let moe_outer_indices_packed = eval.mad(outer_second, c2_26, outer_first);
    eval.constraint_eq(lv.moe_outer_indices_packed, moe_outer_indices_packed);
    eval.constraint_eq_if(lv.is_first_outer, word0, outer_first);
    eval.constraint_eq_if(lv.is_second_outer, word1, outer_second);

    // 7. MoE offsets/routing order machinery.
    // We need to check:
    // 1. `O_i` <= `O_{i+1}` for all i,
    // 2. `R[w][i]` < `R[w][i+1]` for all i (the winner slice is strictly increasing),
    // 3. the three public offsets `O_{w-1}`, `O_w` and `O_{e-1}` sit at their positions in
    //    the hashed offsets stream (bound to `HO`), and the words after `O_{e-1}` are zero,
    // 4. every value in the winner slice is < m.
    // All the gate flags and pins below are known columns, recomputed from public data.
    // Note that for `w = 0` there is no stream word to pin `O_{w-1}` to, and for `w = e-1`
    // the `O_w` and `O_{e-1}` pins coincide: `O_{w-1} = 0` and `O_w = O_{e-1}` are then
    // enforced natively (`MoEStatement::check`).

    // Check 3: `word_pin_first` and `word_pin_second` hold the public offset (or padding
    // zero) the row's words are pinned to.
    eval.constraint_eq_if(lv.is_word_pin_first, word0, lv.word_pin_first);
    eval.constraint_eq_if(lv.is_word_pin_second, word1, lv.word_pin_second);

    // Checks 1 and 2. If both words belong to the checked range, then:
    // word1 - word0 - is_chain_strict = chain_intra_limbs[0] + 2^16 * chain_intra_limbs[1]
    // (`is_chain_strict` is 1 exactly on routing rows). The limbs are RC16-bounded, so the
    // difference lies in [0, 2^32) and a wrapped negative can never satisfy the equality.
    let c2_16 = eval.u64(1 << 16);
    let intra = eval.polyval(&lv.chain_intra_limbs, c2_16);
    let inter = eval.polyval(&lv.chain_inter_limbs, c2_16);
    let inter_next = eval.polyval(&nv.chain_inter_limbs, c2_16);
    let d_intra = eval.sub(word1, word0);
    let d_intra = eval.sub(d_intra, lv.is_chain_strict);
    eval.constraint_eq_if(lv.is_chain_intra, d_intra, intra);
    // Then, we need to chain with the next leaf value:
    // word0 - chain_carry - is_chain_strict = chain_inter_limbs[0] + 2^16 * chain_inter_limbs[1]
    let d_inter = eval.sub(word0, lv.chain_carry);
    let d_inter = eval.sub(d_inter, lv.is_chain_strict);
    eval.constraint_eq_if(lv.is_chain_inter, d_inter, inter);
    // Check 4: the slice is strictly increasing, so bounding its last value by
    // `word_pin_* = m - 1` bounds the whole slice. The difference reuses a limb pair that
    // is idle on its row: this row's intra pair when the last value is `word0`, or the
    // next row's inter pair when it is `word1`.
    let d_bound_first = eval.sub(lv.word_pin_first, word0);
    eval.constraint_eq_if(lv.is_bound_first, d_bound_first, intra);
    let d_bound_second = eval.sub(lv.word_pin_second, word1);
    eval.constraint_eq_if(lv.is_bound_second, d_bound_second, inter_next);

    // We update `chain_carry`: on routing/offsets rows it becomes the row's second word;
    // on all other rows (the Merkle parents in between) it copies from the previous row,
    // so the inter check above always compares against the stream-previous word.
    let carry_step = eval.sub(word1, lv.chain_carry);
    let carry_next = eval.mad(lv.is_chain_data, carry_step, lv.chain_carry);
    eval.constraint_eq(nv.chain_carry, carry_next);

    // 8. Row counter: transition (the cyclic wrap cannot apply to a strict increment),
    // first-row anchor 0.
    let incremented = eval.add(lv.trace_row_index, one);
    let diff = eval.sub(nv.trace_row_index, incremented);
    eval.constraint_transition(diff);
    eval.constraint_first_row(lv.trace_row_index);
}

/// One half quarter-round:
/// `ea = a + packed(b) + m` (mod 2^32, unconditional), `ea = d ^ (ed <<< rot1)` (gated),
/// `ec = c + packed(ed)` (mod 2^32, unconditional), `ec = b ^ (eb <<< rot2)` (gated);
/// the produced bit columns `eb`/`ed` are boolean-checked unconditionally.
#[allow(clippy::too_many_arguments)]
fn half_g<V, S, E>(
    eval: &mut E,
    a: V,
    b: &[V; 32],
    c: V,
    d: &[V; 32],
    m: V,
    second_half: bool,
    expected_a: V,
    expected_b: &[V; 32],
    expected_c: V,
    expected_d: &[V; 32],
    is_activated: V,
) where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let (rot_1, rot_2) = if second_half { (8, 7) } else { (16, 12) };
    let two = eval.i32(2);
    let b_packed = eval.polyval(b, two);
    add3_unchecked(eval, expected_a, a, b_packed, m);
    xor_32_shift_if(eval, expected_a, d, expected_d, is_activated, rot_1);
    let expected_d_packed = eval.polyval(expected_d, two);
    add2_unchecked(eval, expected_c, c, expected_d_packed);
    xor_32_shift_if(eval, expected_c, b, expected_b, is_activated, rot_2);
}

/// One full round between `states[0]` and `states[4]` (= the next row's input state): two
/// column half-rounds then two diagonal half-rounds, message words in schedule order.
fn verify_round<V, S, E>(eval: &mut E, states: &[&Blake3StateCols<V>; 5], msg: &[V; 16], is_activated: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    for i in 0..4 {
        half_g(
            eval,
            states[0].a_words[i],
            &states[0].b_bits[i],
            states[0].c_words[i],
            &states[0].d_bits[i],
            msg[2 * i],
            false,
            states[1].a_words[i],
            &states[1].b_bits[i],
            states[1].c_words[i],
            &states[1].d_bits[i],
            is_activated,
        );
    }
    for i in 0..4 {
        half_g(
            eval,
            states[1].a_words[i],
            &states[1].b_bits[i],
            states[1].c_words[i],
            &states[1].d_bits[i],
            msg[2 * i + 1],
            true,
            states[2].a_words[i],
            &states[2].b_bits[i],
            states[2].c_words[i],
            &states[2].d_bits[i],
            is_activated,
        );
    }
    for i in 0..4 {
        half_g(
            eval,
            states[2].a_words[i],
            &states[2].b_bits[(i + 1) % 4],
            states[2].c_words[(i + 2) % 4],
            &states[2].d_bits[(i + 3) % 4],
            msg[8 + 2 * i],
            false,
            states[3].a_words[i],
            &states[3].b_bits[(i + 1) % 4],
            states[3].c_words[(i + 2) % 4],
            &states[3].d_bits[(i + 3) % 4],
            is_activated,
        );
    }
    for i in 0..4 {
        half_g(
            eval,
            states[3].a_words[i],
            &states[3].b_bits[(i + 1) % 4],
            states[3].c_words[(i + 2) % 4],
            &states[3].d_bits[(i + 3) % 4],
            msg[8 + 2 * i + 1],
            true,
            states[4].a_words[i],
            &states[4].b_bits[(i + 1) % 4],
            states[4].c_words[(i + 2) % 4],
            &states[4].d_bits[(i + 3) % 4],
            is_activated,
        );
    }
}

/// On active rows, state1's bits decompose state0's packed words.
/// Returns `v[i] ^ v[8+i]`; boolean-checked bits keep each output in u32 range.
fn finalize_blake<V, S, E>(eval: &mut E, state0: &Blake3StateCols<V>, state1: &Blake3StateCols<V>, is_activated: V) -> [V; 8]
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let two = eval.i32(2);
    for i in 0..4 {
        let row2_packed = eval.polyval(&state1.b_bits[i], two);
        eval.constraint_eq_if(is_activated, state0.a_words[i], row2_packed);
        let row4_packed = eval.polyval(&state1.d_bits[i], two);
        eval.constraint_eq_if(is_activated, state0.c_words[i], row4_packed);
    }
    core::array::from_fn(|i| {
        if i < 4 {
            xor_32(eval, &state1.b_bits[i], &state1.d_bits[i])
        } else {
            xor_32(eval, &state0.b_bits[i - 4], &state0.d_bits[i - 4])
        }
    })
}

/// On compression starts, bind words 0..8 to the selected CV, 8..12 to the IV,
/// and the final bit block to the packed counter/flags/block-length tweak.
fn verify_init_state<V, S, E>(eval: &mut E, init_state: &Blake3StateCols<V>, is_new_blake: V, cv: &[V; 8], blake3_tweak: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let two = eval.i32(2);
    for i in 0..4 {
        eval.constraint_eq_if(is_new_blake, init_state.a_words[i], cv[i]);
        let iv = eval.u64(BLAKE3_IV[i] as u64);
        eval.constraint_eq_if(is_new_blake, init_state.c_words[i], iv);
        let row2_packed = eval.polyval(&init_state.b_bits[i], two);
        eval.constraint_eq_if(is_new_blake, row2_packed, cv[i + 4]);
    }
    let active_bits: Vec<V> = init_state.d_bits[0]
        .iter()
        .chain(&init_state.d_bits[1][0..16])
        .chain(&init_state.d_bits[3][0..8])
        .chain(&init_state.d_bits[2][0..7])
        .copied()
        .collect();
    let packed = eval.polyval(&active_bits, two);
    eval.constraint_eq_if(is_new_blake, packed, blake3_tweak);
    let zero_bits: Vec<V> = init_state.d_bits[1][16..]
        .iter()
        .chain(&init_state.d_bits[2][7..])
        .chain(&init_state.d_bits[3][8..])
        .copied()
        .collect();
    for bit in zero_bits {
        let zeroed = eval.mul(is_new_blake, bit);
        eval.constraint(zeroed);
    }
}

/// `res = a + b + c (mod 2^32)`: `(diff)(diff - 2^32)(diff - 2^33) = 0` — unconditional
/// (finalization rows satisfy it via the postprocess fill). The mod-2^32 reading is sound only
/// because every operand is elsewhere bounded (packed bits, byte packings, or the routing
/// lookup's 2^34-injective keys).
fn add3_unchecked<V, S, E>(eval: &mut E, res: V, a: V, b: V, c: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let sum = eval.add(a, b);
    let sum = eval.add(sum, c);
    let c2_32 = eval.u64(1 << 32);
    let diff = eval.sub(sum, res);
    let diff_1 = eval.sub(diff, c2_32);
    let diff_2 = eval.sub(diff_1, c2_32);
    let poly = eval.mul(diff, diff_1);
    let poly = eval.mul(poly, diff_2);
    eval.constraint(poly);
}

/// `res = a + b (mod 2^32)`: `(diff)(diff - 2^32) = 0` — unconditional.
fn add2_unchecked<V, S, E>(eval: &mut E, res: V, a: V, b: V)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let sum = eval.add(a, b);
    let diff = eval.sub(sum, res);
    let c2_32 = eval.u64(1 << 32);
    let diff_1 = eval.sub(diff, c2_32);
    let c = eval.mul(diff, diff_1);
    eval.constraint(c);
}

/// If activated, `res = a ^ (b <<< shift)` over the bit columns (i.e. `b = (res ^ a) >>> shift`,
/// the rotation absorbed into the reindexing); `b`'s bits are boolean-checked unconditionally,
/// which is what makes every packed state word a range-checked u32.
fn xor_32_shift_if<V, S, E>(eval: &mut E, res: V, a: &[V; 32], b: &[V; 32], is_activated: V, shift: usize)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let two = eval.i32(2);
    for &bit in b.iter() {
        eval.constraint_bool(bit);
    }
    let xor_bits: [V; 32] = core::array::from_fn(|i| {
        let a_bit = a[i];
        let b_bit = b[(i + 32 - shift) % 32];
        eval.xor_bit(a_bit, b_bit)
    });
    let xor = eval.polyval(&xor_bits, two);
    eval.constraint_eq_if(is_activated, res, xor);
}

/// The packed XOR of two boolean-checked bit columns (no new constraints; degree 2).
fn xor_32<V, S, E>(eval: &mut E, a: &[V; 32], b: &[V; 32]) -> V
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let two = eval.i32(2);
    let xor_bits: [V; 32] = core::array::from_fn(|i| eval.xor_bit(a[i], b[i]));
    eval.polyval(&xor_bits, two)
}

// Stark impl

/// BLAKE3 AIR, proved through the batch driver with the channels in `super::ctl`.
#[derive(Clone, Debug)]
pub struct Blake3Stark<F: RichField + Extendable<D>, const D: usize> {
    pub program: Blake3Program,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Blake3Stark<F, D> {
    pub fn new(program: Blake3Program) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for Blake3Stark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_blake3_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_blake3_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the strip-bytes, block-scales and lottery-words channels, plus the committed
    // LUT channels (declared in `super::ctl`).
    fn requires_ctls(&self) -> bool {
        true
    }

    /// The CV-routing lookup: every row publishes `CV_OUT` at key
    /// `TRACE_ROW_INDEX` with witness multiplicity `CV_OUT_FREQ`; fetch rows (`IS_CV_IN`) consume
    /// `CV_IN` at key `CV_ROUTE_KEY_OR_TWEAK`. One [`Lookup`] per CV word, all sharing the
    /// multiplicity column (a consumer fetches the 8 words together under the same filter).
    /// Keys are `word + 2^34 * row` — injective because the round block bounds any accepted
    /// word below 2^34 (the `CV_ROUTING_KEY_FACTOR` bound) and `TRACE_ROW_INDEX` is a committed
    /// counter.
    fn lookups(&self) -> Vec<Lookup<F>> {
        let m = &BLAKE3_COL_MAP;
        let factor = F::from_canonical_u64(CV_ROUTING_KEY_FACTOR);
        (0..8)
            .map(|i| Lookup {
                columns: vec![Column::linear_combination([
                    (m.cv_in[i], F::ONE),
                    (m.cv_route_key_or_tweak, factor),
                ])],
                table_column: Column::linear_combination([(m.cv_out[i], F::ONE), (m.trace_row_index, factor)]),
                frequencies_column: Column::single(m.cv_out_freq),
                filter_columns: vec![Filter::from_column(Column::single(m.is_cv_in))],
            })
            .collect()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use pearl_blake3::{
        B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_PARENT, B3F_ROOT, BLAKE3_CHUNK_LEN, BLAKE3_MSG_LEN, Blake3Hasher,
        MerkleTree, blake3_digest, pad_to_chunk_boundary, padded_chunk_len,
    };
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::super::columns::NUM_BLAKE3_KNOWN_COLUMNS;
    use super::*;
    use crate::api::layout::{AxisPattern, DimType};
    use crate::api::proof_utils::operand_digest_fp10;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = Blake3Stark<F, D>;

    /// 64-byte blocks per 1024-byte chunk.
    const BLOCKS_PER_CHUNK: usize = BLAKE3_CHUNK_LEN / BLAKE3_MSG_LEN;

    /// Test-side program compiler: the keyed chunk tree over one chunk-padded plane
    /// (`pearl_blake3::MerkleTree` semantics — adjacent pairing, odd tail promoted, ROOT flag
    /// on the top merge / the single chunk's last block). With `subtree` the ROOT flag is
    /// withheld (the produced CV is an inner node of a larger tree — the sparse-opening case).
    /// The production schedule comes from the fp8 `BlakeProgram` compiler; this mirror
    /// only feeds the table's tests.
    fn compile_plane_tree(
        instrs: &mut Vec<Blake3Instruction>,
        plane: PlaneId,
        raw_len: usize,
        ctl_key_base: u64,
        subtree: bool,
        bind: Option<PublicBinding>,
    ) -> usize {
        let chunks = padded_chunk_len(raw_len) / BLAKE3_CHUNK_LEN;
        assert!(chunks >= 1, "empty plane {plane:?}");
        let mut cvs: Vec<usize> = Vec::with_capacity(chunks);
        for c in 0..chunks {
            for b in 0..BLOCKS_PER_CHUNK {
                let mut flags = B3F_KEYED_HASH as u32;
                if b == 0 {
                    flags |= B3F_CHUNK_START as u32;
                }
                let is_tree_root = chunks == 1 && b + 1 == BLOCKS_PER_CHUNK && !subtree;
                if b + 1 == BLOCKS_PER_CHUNK {
                    flags |= B3F_CHUNK_END as u32;
                    if is_tree_root {
                        flags |= B3F_ROOT as u32;
                    }
                }
                let offset = c * BLAKE3_CHUNK_LEN + b * BLAKE3_MSG_LEN;
                let ctl_base = match plane {
                    PlaneId::AValues | PlaneId::BValues => ctl_key_base + offset as u64,
                    PlaneId::AScales | PlaneId::BScales => ctl_key_base + (offset / 2) as u64,
                    PlaneId::Routing | PlaneId::Offsets => 0,
                };
                instrs.push(Blake3Instruction {
                    cv: if b == 0 {
                        CvSource::KeyA
                    } else {
                        CvSource::Chain(instrs.len() - 1)
                    },
                    msg: MessageSource::PlaneBytes { plane, offset, ctl_base },
                    counter: c as u64,
                    block_len: BLAKE3_MSG_LEN as u32,
                    flags,
                    bind: if is_tree_root { bind } else { None },
                });
            }
            cvs.push(instrs.len() - 1);
        }
        while cvs.len() > 1 {
            let is_root_layer = cvs.len() == 2 && !subtree;
            let mut next_layer = Vec::with_capacity(cvs.len().div_ceil(2));
            for pair in cvs.chunks(2) {
                if let [left, right] = *pair {
                    let mut flags = (B3F_KEYED_HASH | B3F_PARENT) as u32;
                    if is_root_layer {
                        flags |= B3F_ROOT as u32;
                    }
                    instrs.push(Blake3Instruction {
                        cv: CvSource::KeyA,
                        msg: MessageSource::Parent {
                            left: CvRef::Instruction(left),
                            right: CvRef::Instruction(right),
                        },
                        counter: 0,
                        block_len: BLAKE3_MSG_LEN as u32,
                        flags,
                        bind: if is_root_layer { bind } else { None },
                    });
                    next_layer.push(instrs.len() - 1);
                } else {
                    next_layer.push(pair[0]); // Odd tail: promoted unchanged.
                }
            }
            cvs = next_layer;
        }
        cvs[0]
    }

    /// The complete-forest schedule of one job whose planes are fully opened: four plane trees,
    /// the routing tree, the two commit folds, the lottery.
    fn compile_forest(h: usize, w: usize, k: usize, routing_blocks: usize) -> Blake3Program {
        let mut instrs = Vec::new();
        let a_values = compile_plane_tree(&mut instrs, PlaneId::AValues, h * k, 0, false, None);
        let a_scales = compile_plane_tree(&mut instrs, PlaneId::AScales, h * k / 4, 0, false, None);
        let b_values = compile_plane_tree(&mut instrs, PlaneId::BValues, w * k, (h * k) as u64, false, None);
        let b_scales = compile_plane_tree(&mut instrs, PlaneId::BScales, w * k / 4, (h * k / 8) as u64, false, None);
        if routing_blocks > 0 {
            compile_plane_tree(
                &mut instrs,
                PlaneId::Routing,
                routing_blocks * BLAKE3_MSG_LEN,
                0,
                false,
                Some(PublicBinding::HashRouting),
            );
            // A single expert owning the whole opened stream (`e = 1`, `w = 0`), the offsets
            // list [`O_0`] = the stream length; `m` past the 26-bit pin cap so oversized
            // *neighbor* words stay legal.
            compile_plane_tree(
                &mut instrs,
                PlaneId::Offsets,
                std::mem::size_of::<u32>(),
                0,
                false,
                Some(PublicBinding::HashOffsets),
            );
        }
        append_commit_fold(&mut instrs, a_values, a_scales, PublicBinding::HashA, CvSource::KeyA);
        append_commit_fold(&mut instrs, b_values, b_scales, PublicBinding::HashB, CvSource::KeyB);
        instrs.push(Blake3Instruction::lottery());
        Blake3Program {
            instructions: instrs,
            num_aux_msgs: 0,
            num_aux_cvs: 0,
            routing_pins: (0..routing_blocks).map(|b| (b * 16, test_routing_word(b * 16))).collect(),
            moe: (routing_blocks > 0).then(|| MoeSchedule {
                w: 0,
                experts: 1,
                o_w_prev: 0,
                o_w: 16 * routing_blocks as u32,
                o_last: 16 * routing_blocks as u32,
                m: 1 << 27,
            }),
        }
    }

    /// Owned test inputs (the `Blake3TraceInputs` borrows).
    struct TestData {
        a_values: Vec<u8>,
        a_scales: Vec<u8>,
        b_values: Vec<u8>,
        b_scales: Vec<u8>,
        routing_words: Vec<u32>,
        offsets_words: Vec<u32>,
        aux_msgs: Vec<[u8; 64]>,
        aux_cvs: Vec<[u8; 32]>,
        lottery_words: [u32; 16],
        key_a: [u32; 8],
        key_b: [u32; 8],
        jackpot_key: [u32; 8],
    }

    impl TestData {
        fn new(h: usize, w: usize, k: usize, routing_blocks: usize) -> Self {
            let bytes =
                |len: usize, salt: u64| -> Vec<u8> { (0..len).map(|i| ((i as u64).wrapping_mul(salt) >> 5) as u8).collect() };
            Self {
                a_values: bytes(h * k, 0x9E3779B97F4A7C15),
                a_scales: bytes(h * k / 4, 0xC2B2AE3D27D4EB4F),
                b_values: bytes(w * k, 0xD1B54A32D192ED03),
                b_scales: bytes(w * k / 4, 0xA0761D6478BD642F),
                routing_words: (0..routing_blocks * 16).map(test_routing_word).collect(),
                offsets_words: if routing_blocks > 0 {
                    vec![16 * routing_blocks as u32]
                } else {
                    Default::default()
                },
                aux_msgs: Vec::new(),
                aux_cvs: Vec::new(),
                lottery_words: core::array::from_fn(|i| (i as u32).wrapping_mul(0x85EBCA77) ^ 0x1234),
                key_a: core::array::from_fn(|i| 0x1000_0001u32.wrapping_mul(i as u32 + 1)),
                key_b: core::array::from_fn(|i| 0x4000_0005u32.wrapping_mul(i as u32 + 1)),
                jackpot_key: core::array::from_fn(|i| 0x3000_0007u32.wrapping_mul(i as u32 + 1)),
            }
        }

        fn inputs(&self) -> Blake3TraceInputs<'_> {
            Blake3TraceInputs {
                a_values: &self.a_values,
                a_scales: &self.a_scales,
                b_values: &self.b_values,
                b_scales: &self.b_scales,
                routing_words: &self.routing_words,
                offsets_words: &self.offsets_words,
                aux_msgs: &self.aux_msgs,
                aux_cvs: &self.aux_cvs,
                lottery_words: self.lottery_words,
                key_a: self.key_a,
                key_b: self.key_b,
                jackpot_key: self.jackpot_key,
                a_hash_id: HashId::Blake3Chunk1024,
                b_hash_id: HashId::Blake3Chunk1024,
                routing_hash_id: HashId::Blake3Chunk1024,
                offsets_hash_id: HashId::Blake3Chunk1024,
            }
        }
    }

    fn words_to_bytes(words: &[u32; 8]) -> [u8; 32] {
        core::array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4])
    }

    fn field_words_to_bytes(words: &[F; 8]) -> [u8; 32] {
        let w: [u32; 8] = core::array::from_fn(|i| words[i].to_canonical_u64() as u32);
        words_to_bytes(&w)
    }

    fn pi_bytes(pis: &[F; NUM_BLAKE3_PUBLIC_INPUTS], base: usize) -> [u8; 32] {
        let w: [F; 8] = core::array::from_fn(|i| pis[base + i]);
        field_words_to_bytes(&w)
    }

    /// Single-chunk plane trees.
    fn small_geometry() -> (usize, usize, usize, usize) {
        (2, 2, 64, 2)
    }

    /// A-values/B-values 3 chunks each (odd promote + parent cascade), routing 2 chunks;
    /// exercises chunk chaining across chunks, in-table parents, and multi-chunk roots.
    fn multi_chunk_geometry() -> (usize, usize, usize, usize) {
        (2, 2, 1536, 20)
    }

    fn generate(
        h: usize,
        w: usize,
        k: usize,
        routing_blocks: usize,
    ) -> (
        Blake3Program,
        TestData,
        Vec<[F; NUM_BLAKE3_COLUMNS]>,
        [F; NUM_BLAKE3_PUBLIC_INPUTS],
    ) {
        let program = compile_forest(h, w, k, routing_blocks);
        let data = TestData::new(h, w, k, routing_blocks);
        let (rows, pis) = program.generate_trace::<F>(&data.inputs());
        (program, data, rows, pis)
    }

    /// Runs every constraint over every row pair, *including* the last -> first wrap
    /// (`z_last` vanishes only on the last row, exactly as in the real vanishing polynomial).
    fn constraints_violated(stark: &S, rows: &[[F; NUM_BLAKE3_COLUMNS]], pis: &[F; NUM_BLAKE3_PUBLIC_INPUTS]) -> bool {
        let n = rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            if consumer.accumulators().into_iter().any(|acc| acc != F::ZERO) {
                return true;
            }
        }
        false
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, rows, pis) = generate(h, w, k, r);
        assert_eq!(rows.len(), 1024);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn honest_multi_chunk_trace_satisfies_all_constraints() {
        let (h, w, k, r) = multi_chunk_geometry();
        let (program, _, rows, pis) = generate(h, w, k, r);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn roots_are_bit_exact_vs_pearl_blake3() {
        for (h, w, k, r) in [small_geometry(), multi_chunk_geometry()] {
            let (_, data, _, pis) = generate(h, w, k, r);
            let key = words_to_bytes(&data.key_a);
            let key_b = words_to_bytes(&data.key_b);

            // HASH_A / HASH_B vs the native fold (`operand_digest_fp10`): the plane trees
            // are all keyed with `key_a`, but each side's fold keys under its own opening key.
            let root = |bytes: &[u8]| MerkleTree::new(&pad_to_chunk_boundary(bytes), key).root();
            let fold_a = |values: &[u8], scales: &[u8]| operand_digest_fp10(&root(values), &root(scales), &key);
            let fold_b = |values: &[u8], scales: &[u8]| operand_digest_fp10(&root(values), &root(scales), &key_b);
            assert_eq!(pi_bytes(&pis, PI_HASH_A), fold_a(&data.a_values, &data.a_scales));
            assert_eq!(pi_bytes(&pis, PI_HASH_B), fold_b(&data.b_values, &data.b_scales));

            // Routing tree root -> HASH_ROUTING, offsets tree root -> HASH_OFFSETS.
            let routing_bytes: Vec<u8> = data.routing_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let expected_routing = MerkleTree::new(&pad_to_chunk_boundary(&routing_bytes), key).root();
            assert_eq!(pi_bytes(&pis, PI_HASH_ROUTING), expected_routing);
            let offsets_bytes: Vec<u8> = data.offsets_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let expected_offsets = MerkleTree::new(&pad_to_chunk_boundary(&offsets_bytes), key).root();
            assert_eq!(pi_bytes(&pis, PI_HASH_OFFSETS), expected_offsets);

            // Lottery -> HASH_JACKPOT: keyed blake3 of the 16 folded words under JACKPOT_KEY.
            let fold_bytes: Vec<u8> = data.lottery_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let expected_jackpot = blake3_digest(&fold_bytes, Some(words_to_bytes(&data.jackpot_key)));
            assert_eq!(pi_bytes(&pis, PI_HASH_JACKPOT), expected_jackpot);
        }
    }

    /// The word `TestData::new` puts at routing stream position `i`: strictly increasing
    /// (the whole stream is the winner slice under `compile_forest`'s schedule, so the order
    /// chains apply) with gaps of 8, so a neighbor word can move without breaking the order;
    /// < 2^26 (the pin cap) and nonzero at position 0 (the pinned-value tamper tests need the
    /// packed prep column to move when a pin is cleared).
    fn test_routing_word(i: usize) -> u32 {
        (i as u32 + 1) * 8
    }

    /// Test sampled routing slots alongside unsampled neighbors, including one above
    /// 2^26. Selectors and limbs must match the public pins; neighbor words remain
    /// subject to hash/order checks. Recomputed schedule columns must match the trace.
    #[test]
    fn selective_pinning_matches_deployed_semantics() {
        let (h, w, k, routing_blocks) = small_geometry();
        let mut program = compile_forest(h, w, k, routing_blocks);
        // Words 1 (second slot of row 0), 4 and 5 (both slots of row 2), 17 (second slot of
        // block 1's row 0). Everything else — word 0 included — is a neighbor.
        let pinned: [usize; 4] = [1, 4, 5, 17];
        program.routing_pins = pinned.iter().map(|&i| (i, test_routing_word(i))).collect();

        let mut data = TestData::new(h, w, k, routing_blocks);
        // A neighbor word above the 26-bit pin cap: only *pinned* values are capped; an
        // unsampled slice entry is free witness under the hash chain and the order chain
        // (strictly increasing below `m = 2^27`), so the slice's last word can exceed 2^26.
        data.routing_words[31] = 0x0400_0001;
        let (rows, pis) = program.generate_trace::<F>(&data.inputs());

        // The routing instructions of the fixture forest: one full chunk (16 blocks), the
        // first two live. Walk their 8 message rows each and check the pin surface.
        let routing_row0: usize = ROWS_PER_COMPRESSION
            * program
                .instructions
                .iter()
                .position(|i| {
                    matches!(
                        i.msg,
                        MessageSource::PlaneBytes {
                            plane: PlaneId::Routing,
                            ..
                        }
                    )
                })
                .unwrap();
        for block in 0..BLOCKS_PER_CHUNK {
            for j in 0..ROWS_PER_COMPRESSION {
                let row: &Blake3ColumnsView<F> = rows[routing_row0 + ROWS_PER_COMPRESSION * block + j].borrow();
                let (w0, w1) = (16 * block + 2 * j, 16 * block + 2 * j + 1);
                let pin = |i: usize| pinned.contains(&i);
                assert_eq!(
                    row.is_first_outer,
                    F::from_bool(pin(w0)),
                    "block {block} row {j}: first selector"
                );
                assert_eq!(
                    row.is_second_outer,
                    F::from_bool(pin(w1)),
                    "block {block} row {j}: second selector"
                );
                let expect = |i: usize| if pin(i) { test_routing_word(i) as u64 } else { 0 };
                assert_eq!(
                    row.moe_outer_indices_packed,
                    F::from_canonical_u64(expect(w0) + (expect(w1) << 26)),
                    "block {block} row {j}: packed word carries pinned values only"
                );
                let limbs = |v: [F; 2]| v[0].to_canonical_u64() + (v[1].to_canonical_u64() << 13);
                assert_eq!(limbs(row.outer_index_first), expect(w0), "block {block} row {j}: first limbs");
                assert_eq!(
                    limbs(row.outer_index_second),
                    expect(w1),
                    "block {block} row {j}: second limbs"
                );
            }
        }

        assert!(!constraints_violated(&S::new(program.clone()), &rows, &pis));

        // The verifier-side recompute agrees with the trace on every known column.
        let known = program.known_values::<F>(&data.inputs().known_inputs());
        assert_eq!(known.len(), NUM_BLAKE3_KNOWN_COLUMNS);
        for (c, col) in known.iter().enumerate() {
            for (r, expected) in col.values.iter().enumerate() {
                let row: &Blake3ColumnsView<F> = rows[r].borrow();
                let got = [
                    row.row_flags_packed,
                    row.ctl_key_base,
                    row.is_int8_message,
                    row.is_scale_message,
                    row.cv_route_key_or_tweak,
                    row.moe_outer_indices_packed,
                    row.word_pin_first,
                    row.word_pin_second,
                ][c];
                assert_eq!(got, *expected, "known column {c} row {r} diverges from the trace");
            }
        }
    }

    /// A forged limb on a pinned slot breaks the unconditional packed decomposition
    /// (constraint 6): `MOE_OUTER_INDICES_PACKED` is verifier-known, so the limbs cannot move.
    #[test]
    fn tampered_pinned_limb_breaks_packed_decomposition() {
        let (h, w, k, routing_blocks) = small_geometry();
        let program = compile_forest(h, w, k, routing_blocks);
        let routing_row0: usize = ROWS_PER_COMPRESSION
            * program
                .instructions
                .iter()
                .position(|i| {
                    matches!(
                        i.msg,
                        MessageSource::PlaneBytes {
                            plane: PlaneId::Routing,
                            ..
                        }
                    )
                })
                .unwrap();
        let data = TestData::new(h, w, k, routing_blocks);
        let (mut rows, pis) = program.generate_trace::<F>(&data.inputs());
        {
            let row: &mut Blake3ColumnsView<F> = rows[routing_row0].borrow_mut();
            assert_eq!(row.is_first_outer, F::ONE, "the fixture pins the block's first word");
            row.outer_index_first[0] += F::ONE;
        }
        assert!(
            constraints_violated(&S::new(program), &rows, &pis),
            "a forged pinned limb must break the packed decomposition"
        );
    }

    /// A forged message word on a pinned slot breaks the selector-gated word pin: the limbs
    /// carry the public outer index, the ingested word must equal it.
    #[test]
    fn tampered_pinned_message_word_breaks_word_pin() {
        let (h, w, k, routing_blocks) = small_geometry();
        let program = compile_forest(h, w, k, routing_blocks);
        let routing_row0: usize = ROWS_PER_COMPRESSION
            * program
                .instructions
                .iter()
                .position(|i| {
                    matches!(
                        i.msg,
                        MessageSource::PlaneBytes {
                            plane: PlaneId::Routing,
                            ..
                        }
                    )
                })
                .unwrap();
        let data = TestData::new(h, w, k, routing_blocks);
        let (mut rows, pis) = program.generate_trace::<F>(&data.inputs());
        {
            let row: &mut Blake3ColumnsView<F> = rows[routing_row0].borrow_mut();
            assert_eq!(row.is_first_outer, F::ONE);
            // Move the ingested word *and* its buffer copy consistently: only the gated
            // equality against the pinned limbs is left to object.
            row.uint8_data[0] += F::ONE;
            row.blake3_msg_buffer[14] += F::ONE;
        }
        assert!(
            constraints_violated(&S::new(program), &rows, &pis),
            "a forged pinned message word must break the gated word pin"
        );
    }

    /// A forged fetched root CV on the commit-fold row breaks the CV window: the fold's
    /// message words *are* the tree-root CVs, so a prover cannot fold a different values
    /// root into `HASH_A`.
    #[test]
    fn tampered_fold_root_cv_breaks_commit_fold() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let fold_row0 = ROWS_PER_COMPRESSION
            * program
                .instructions
                .iter()
                .position(|i| i.bind == Some(PublicBinding::HashA))
                .unwrap();
        {
            // Row 3 fetches the values root through the CV window: move the fetched CV (and
            // its mux copy) consistently — the sliding-buffer equality is left to object.
            let row: &mut Blake3ColumnsView<F> = rows[fold_row0 + 3].borrow_mut();
            assert_eq!(row.is_cv_in, F::ONE);
            row.cv_in[0] += F::ONE;
            row.blake3_cv[0] += F::ONE;
        }
        assert!(
            constraints_violated(&S::new(program), &rows, &pis),
            "a forged fold input must break the CV window"
        );
    }

    /// The slice bound's idle-limb reuse, second-word case (even `o_w`: the slice-last entry
    /// is its row's second word, so the bound decomposition lives in the *next* row's inter
    /// slot — idle there, the next pair starts past the slice). That slot feeds no other
    /// gate, so tampering it isolates the bound equality.
    #[test]
    fn tampered_bound_limbs_break_the_slice_bound() {
        let (h, w, k, routing_blocks) = small_geometry();
        let program = compile_forest(h, w, k, routing_blocks);
        let data = TestData::new(h, w, k, routing_blocks);
        let (mut rows, pis) = program.generate_trace::<F>(&data.inputs());
        let r = (0..rows.len())
            .find(|&r| {
                let row: &Blake3ColumnsView<F> = rows[r].borrow();
                row.is_bound_second == F::ONE
            })
            .expect("the fixture slice ends on a second word");
        {
            let row: &mut Blake3ColumnsView<F> = rows[r + 1].borrow_mut();
            row.chain_inter_limbs[0] += F::ONE;
        }
        assert!(
            constraints_violated(&S::new(program), &rows, &pis),
            "a forged bound decomposition must break the slice bound"
        );
    }

    /// As above, first-word case (odd `o_w`: the slice-last entry is its row's first word,
    /// so the bound rides the row's own idle intra slot).
    #[test]
    fn bound_first_rides_the_rows_idle_intra_limbs() {
        let (h, w, k, routing_blocks) = small_geometry();
        let mut program = compile_forest(h, w, k, routing_blocks);
        let o_w = 16 * routing_blocks as u32 - 1;
        let s = program.moe.as_mut().unwrap();
        s.o_w = o_w;
        s.o_last = o_w;
        let mut data = TestData::new(h, w, k, routing_blocks);
        data.offsets_words = vec![o_w];
        let (mut rows, pis) = program.generate_trace::<F>(&data.inputs());
        let r = (0..rows.len())
            .find(|&r| {
                let row: &Blake3ColumnsView<F> = rows[r].borrow();
                row.is_bound_first == F::ONE
            })
            .expect("an odd o_w puts the slice's last entry on a first word");
        let stark = S::new(program);
        assert!(
            !constraints_violated(&stark, &rows, &pis),
            "the tweaked fixture must stay honest"
        );
        {
            let row: &mut Blake3ColumnsView<F> = rows[r].borrow_mut();
            row.chain_intra_limbs[0] += F::ONE;
        }
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "a forged bound decomposition must break the slice bound"
        );
    }

    /// Clearing a selector and consistently repacking its flags can satisfy the AIR.
    /// The verifier-known schedule check must reject the changed flags.
    #[test]
    fn cleared_outer_selector_is_caught_by_the_known_column_recompute() {
        let (h, w, k, routing_blocks) = small_geometry();
        let program = compile_forest(h, w, k, routing_blocks);
        let routing_row0: usize = ROWS_PER_COMPRESSION
            * program
                .instructions
                .iter()
                .position(|i| {
                    matches!(
                        i.msg,
                        MessageSource::PlaneBytes {
                            plane: PlaneId::Routing,
                            ..
                        }
                    )
                })
                .unwrap();
        let data = TestData::new(h, w, k, routing_blocks);
        let (mut rows, pis) = program.generate_trace::<F>(&data.inputs());
        {
            // Freeing the word also requires zeroing the pinned limbs (the packed word is
            // verifier-known too and must keep decomposing); zero limbs match packed = 0 only if
            // the packed column is also moved — which the known-column check pins. Here we
            // only clear the selector and zero the limbs+packed consistently.
            let row: &mut Blake3ColumnsView<F> = rows[routing_row0].borrow_mut();
            assert_eq!(row.is_first_outer, F::ONE);
            row.is_first_outer = F::ZERO;
            row.row_flags_packed = pack_control(row);
            row.outer_index_first = [F::ZERO; 2];
            row.moe_outer_indices_packed = F::ZERO;
        }
        assert!(
            !constraints_violated(&S::new(program.clone()), &rows, &pis),
            "the AIR alone accepts a self-consistent selector clear"
        );
        // ... but the verifier-known recompute does not: ROW_FLAGS_PACKED and the packed word both
        // diverge from the verifier's own values on that row.
        let known = program.known_values::<F>(&data.inputs().known_inputs());
        let row: &Blake3ColumnsView<F> = rows[routing_row0].borrow();
        assert_ne!(
            known[0].values[routing_row0], row.row_flags_packed,
            "ROW_FLAGS_PACKED must diverge"
        );
        assert_ne!(
            known[5].values[routing_row0], row.moe_outer_indices_packed,
            "MOE_OUTER_INDICES_PACKED must diverge"
        );
    }

    /// A forged *neighbor* word (unsampled entry of a hotspot block) is a legitimate free
    /// witness for the AIR — but it changes the routing tree, so the bound `HASH_ROUTING`
    /// public input moves and the statement's expected-public-input pinning rejects it.
    #[test]
    fn forged_neighbor_word_moves_hash_routing() {
        let (h, w, k, routing_blocks) = small_geometry();
        let program = compile_forest(h, w, k, routing_blocks);
        let mut data = TestData::new(h, w, k, routing_blocks);
        let (_, honest_pis) = program.generate_trace::<F>(&data.inputs());

        data.routing_words[3] += 1; // Unpinned (pins sit at 0 and 16); the gap keeps the order.
        let (forged_rows, forged_pis) = program.generate_trace::<F>(&data.inputs());
        assert!(
            !constraints_violated(&S::new(program), &forged_rows, &forged_pis),
            "a neighbor word is free witness — the AIR must accept its own trace"
        );
        let pi_range = PI_HASH_ROUTING..PI_HASH_ROUTING + 8;
        assert_ne!(
            &forged_pis[pi_range.clone()],
            &honest_pis[pi_range],
            "the routing root must move with the neighbor word — the public-input pinning rejects the forgery"
        );
    }

    #[test]
    #[should_panic(expected = "Blake3 program must contain exactly one lottery compression")]
    fn validation_requires_a_lottery_compression() {
        let (h, w, k, _) = small_geometry();
        let mut program = compile_forest(h, w, k, 0);
        assert_eq!(program.instructions.pop().unwrap().bind, Some(PublicBinding::HashJackpot));
        program.validate();
    }

    /// A four-chunk values tree with only chunks 0/1 opened. An auxiliary CV supplies
    /// the unopened right subtree; the commitment fold must bind the full matrix root.
    fn sparse_opening_setup() -> (Blake3Program, TestData, Vec<u8>) {
        let full: Vec<u8> = (0..4 * BLAKE3_CHUNK_LEN)
            .map(|i| ((i as u64).wrapping_mul(0xA24BAED4963EE407) >> 7) as u8)
            .collect();
        let opened = full[..2 * BLAKE3_CHUNK_LEN].to_vec();

        let mut instrs = Vec::new();
        // Chunks 0 and 1 in-table (the a_values "strip" plane), then parent(c0, c1), then the
        // root parent against the auxiliary CV of the unopened parent(c2, c3).
        let c0c1 = compile_plane_tree(&mut instrs, PlaneId::AValues, opened.len(), 0, true, None);
        instrs.push(Blake3Instruction {
            cv: CvSource::KeyA,
            msg: MessageSource::Parent {
                left: CvRef::Instruction(c0c1),
                right: CvRef::Auxiliary(0),
            },
            counter: 0,
            block_len: BLAKE3_MSG_LEN as u32,
            flags: (B3F_KEYED_HASH | B3F_PARENT | B3F_ROOT) as u32,
            bind: None,
        });
        let a_values = instrs.len() - 1;
        let a_scales = compile_plane_tree(&mut instrs, PlaneId::AScales, 32, 0, false, None);
        append_commit_fold(&mut instrs, a_values, a_scales, PublicBinding::HashA, CvSource::KeyA);
        instrs.push(Blake3Instruction::lottery());
        let program = Blake3Program {
            instructions: instrs,
            num_aux_msgs: 0,
            num_aux_cvs: 1,
            routing_pins: vec![],
            moe: None,
        };

        // Minimal other planes (only a_values carries the membership statement).
        let mut data = TestData::new(2, 2, 64, 0);
        data.a_values = opened;
        data.a_scales = vec![0; 32];
        data.b_values = vec![0; 128];
        data.b_scales = vec![0; 32];
        // The witness sibling: parent CV of the unopened chunks 2 and 3.
        let hasher = Blake3Hasher::with_key(words_to_bytes(&data.key_a));
        let c2 = hasher.chunk_cv(&full[2 * BLAKE3_CHUNK_LEN..3 * BLAKE3_CHUNK_LEN], 2);
        let c3 = hasher.chunk_cv(&full[3 * BLAKE3_CHUNK_LEN..], 3);
        data.aux_cvs = vec![hasher.parent_cv(&c2, &c3)];
        (program, data, full)
    }

    #[test]
    fn sparse_opening_root_proves_membership_in_full_tree() {
        let (program, data, full) = sparse_opening_setup();
        // Only the strip's tree instructions exist: far fewer than the full tree's 64 blocks.
        assert_eq!(program.instructions.len(), 2 * 16 + 1 + 1 + 16 + 1 + 1);
        let (rows, pis) = program.generate_trace::<F>(&data.inputs());

        let key = words_to_bytes(&data.key_a);
        let expected_hash_a = operand_digest_fp10(
            &MerkleTree::new(&full, key).root(),
            &MerkleTree::new(&pad_to_chunk_boundary(&data.a_scales), key).root(),
            &key,
        );
        assert_eq!(
            pi_bytes(&pis, PI_HASH_A),
            expected_hash_a,
            "in-table fold != full-matrix fold"
        );
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn tampered_aux_cv_breaks_root_binding() {
        let (program, mut data, _) = sparse_opening_setup();
        let (_, honest_pis) = program.generate_trace::<F>(&data.inputs());
        // A forged sibling produces a consistent trace whose fold no longer matches the
        // public one — the IS_BIND_HASH_A binding must catch it.
        data.aux_cvs[0][0] ^= 1;
        let (forged_rows, _) = program.generate_trace::<F>(&data.inputs());
        assert!(constraints_violated(&S::new(program), &forged_rows, &honest_pis));
    }

    /// The schedule has exactly `num_aux_cvs` sibling slots, so a padded witness cannot
    /// even reach constraint evaluation — there is no separate ZK membership error.
    #[test]
    #[should_panic(expected = "aux CVs must match the compiled schedule")]
    fn extra_aux_cv_cannot_pad_the_witness() {
        let (program, mut data, _) = sparse_opening_setup();
        data.aux_cvs.push([0x11; 32]);
        let _ = program.generate_trace::<F>(&data.inputs());
    }

    #[test]
    #[should_panic(expected = "aux CVs must match the compiled schedule")]
    fn missing_aux_cv_is_rejected() {
        let (program, mut data, _) = sparse_opening_setup();
        data.aux_cvs.pop();
        let _ = program.generate_trace::<F>(&data.inputs());
    }

    #[test]
    fn tampered_state_word_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let stark = S::new(program);
        // A state word mid-compression (row 2 of compression 0, STATE1's packed half).
        let row: &mut Blake3ColumnsView<F> = rows[2].borrow_mut();
        row.round[1].a_words[0] += F::ONE;
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "corrupt state word must break add3"
        );
    }

    #[test]
    fn tampered_message_byte_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let stark = S::new(program);
        // A message byte on a values-plane byte row (compression 0, row 0).
        {
            let row: &mut Blake3ColumnsView<F> = rows[0].borrow_mut();
            assert_eq!(row.is_int8_message, F::ONE);
            row.uint8_data[3] += F::ONE;
        }
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "corrupt byte must break the tail load"
        );
    }

    #[test]
    fn tampered_cv_mux_selector_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        // Clear IS_USE_JACKPOT_KEY on the lottery's row 0 *and* repack ROW_FLAGS_PACKED so only the
        // mux itself can catch it: the muxed CV becomes 0 while the committed CV still carries
        // JACKPOT_KEY.
        let lottery = program
            .instructions
            .iter()
            .position(|i| i.cv == CvSource::JackpotKey)
            .unwrap();
        {
            let row: &mut Blake3ColumnsView<F> = rows[ROWS_PER_COMPRESSION * lottery].borrow_mut();
            assert_eq!(row.is_use_jackpot_key, F::ONE);
            row.is_use_jackpot_key = F::ZERO;
            row.row_flags_packed = pack_control(row);
        }
        let stark = S::new(program);
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "corrupt CV selector must break the mux"
        );
    }

    #[test]
    fn tampered_rotation_bit_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let stark = S::new(program);
        // Flip one bit of a rotated/xored state word (STATE1's b_bits bits feed the first
        // half-round's xor recomposition and the next add3).
        {
            let row: &mut Blake3ColumnsView<F> = rows[1].borrow_mut();
            let bit = &mut row.round[1].b_bits[0][5];
            *bit = F::ONE - *bit;
        }
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "flipped rotation bit must break the xor"
        );
    }

    #[test]
    fn tampered_public_binding_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let lottery = program
            .instructions
            .iter()
            .position(|i| i.bind == Some(PublicBinding::HashJackpot))
            .unwrap();
        {
            let row: &mut Blake3ColumnsView<F> = rows[ROWS_PER_COMPRESSION * lottery + 7].borrow_mut();
            assert_eq!(row.is_bind_jackpot_hash, F::ONE);
            row.cv_out[0] += F::ONE;
        }
        let stark = S::new(program);
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "corrupt lottery output must break the HASH_JACKPOT binding"
        );
    }

    #[test]
    fn tampered_row_counter_breaks_constraints() {
        let (h, w, k, r) = small_geometry();
        let (program, _, mut rows, pis) = generate(h, w, k, r);
        let stark = S::new(program);
        {
            let row: &mut Blake3ColumnsView<F> = rows[5].borrow_mut();
            row.trace_row_index += F::ONE;
        }
        assert!(
            constraints_violated(&stark, &rows, &pis),
            "corrupt row counter must break the increment"
        );
    }

    #[test]
    fn degree_is_at_most_three() {
        let (h, w, k, r) = small_geometry();
        test_stark_low_degree::<F, S, D>(S::new(compile_forest(h, w, k, r))).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        let (h, w, k, r) = small_geometry();
        test_stark_circuit_constraints::<F, C, S, D>(S::new(compile_forest(h, w, k, r))).unwrap();
    }

    /// Parse a prequant proof, adapt its public hash schedule and generate a trace from
    /// its extracted witness. Check constraints and side digests against the full matrix
    /// commitments, including unopened blocks and auxiliary subtrees.
    #[test]
    fn adapter_consumes_deployed_blake_program() {
        use crate::api::fp8::plain_proof::PlainProofV4;
        use crate::api::fp8::public_params::{CommonParams, Device, HashId, JobParams, OperandParams, Quant};
        use crate::api::layout::{AxisPattern, DimType};
        use crate::api::primitives::{IncompleteBlockHeader, Sides};
        use crate::ffi::plain_proof::MatrixMerkleProof;
        // k = 2048, 4×64 tile, 16 Blake lanes. m > h and n > w so every
        // plane tree still mixes opened blocks, auxiliary blocks (unopened rows sharing an
        // opened chunk) and auxiliary CVs (whole unopened subtrees).
        let (m, n, k) = (8usize, 128usize, 2048usize);
        let n_blocks = k / BLOCK_SIZE;
        let rows_pattern = AxisPattern::new(&[(4, DimType::Blake)]).unwrap();
        let cols_pattern = AxisPattern::new(&[(4, DimType::Blake), (16, DimType::Fold)]).unwrap();
        let a_rows: Vec<usize> = rows_pattern.tile_offsets().iter().map(|&o| o as usize).collect();
        let b_rows: Vec<usize> = cols_pattern.tile_offsets().iter().map(|&o| o as usize).collect();
        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let (key_a, key_b) = crate::circuit::fp8::consistency::fixture_tree_keys(&header);
        let key_a_words = core::array::from_fn(|i| u32::from_le_bytes(key_a[4 * i..4 * i + 4].try_into().unwrap()));
        let key_b_words = core::array::from_fn(|i| u32::from_le_bytes(key_b[4 * i..4 * i + 4].try_into().unwrap()));

        let value_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| (0..k).map(|j| ((seed + i * k + j) % 251) as u8).collect())
                .collect()
        };
        let scale_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| {
                    (0..n_blocks)
                        .flat_map(|b| (0x3f80u16 + ((seed + i + b) % 16) as u16).to_le_bytes())
                        .collect()
                })
                .collect()
        };

        let keyed_proof = |rows_bytes: &[Vec<u8>], row_indices: &[usize], key: [u8; 32]| -> MatrixMerkleProof {
            let flat: Vec<u8> = rows_bytes.iter().flatten().copied().collect();
            let tree = MerkleTree::new(&pad_to_chunk_boundary(&flat), key);
            let leaves = MerkleTree::compute_leaf_indices_from_rows(
                row_indices,
                (rows_bytes.len(), rows_bytes[0].len()),
                BLAKE3_CHUNK_LEN,
            )
            .unwrap();
            MatrixMerkleProof {
                proof: tree.get_multileaf_proof(&leaves),
                row_indices: row_indices.to_vec(),
            }
        };
        let proof = PlainProofV4 {
            job: JobParams {
                // σ̂ and σ_Δ coincide in this fixture (the ancestor is the proposed header).
                ancestor_header: header,
                common: CommonParams {
                    k: k as u32,
                    r: 32,
                    quant: Quant::Fp8E4M3Prequant,
                    device: Device::B200,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: m as u32,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: rows_pattern,
                    },
                    b: OperandParams {
                        num_rows: n as u32,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: cols_pattern,
                    },
                },
                moe: None,
            },
            values: Sides {
                a: keyed_proof(&value_tree(m, 0), &a_rows, key_a),
                b: keyed_proof(&value_tree(n, 7), &b_rows, key_b),
            },
            scales: Sides {
                a: keyed_proof(&scale_tree(m, 1), &a_rows, key_a),
                b: keyed_proof(&scale_tree(n, 5), &b_rows, key_b),
            },
            moe_witness: None,
        };

        // The real verifier: membership-checks everything and extracts the witness.
        let (private, public) = proof.parse_proof(&header).expect("fixture must parse");
        let (compiled, _, _) = public.compile(&header).expect("fixture must compile");
        let program = Blake3Program::from_blake_program(&compiled.blake_proof, k, vec![], None);

        // The verifier's witness, adapted: concatenated opened strips per plane, the
        // auxiliary material verbatim, arbitrary lottery/POW inputs.
        let data = TestData {
            a_values: private.operands.a.values.as_bytes().to_vec(),
            a_scales: private.operands.a.scales.as_bytes().to_vec(),
            b_values: private.operands.b.values.as_bytes().to_vec(),
            b_scales: private.operands.b.scales.as_bytes().to_vec(),
            routing_words: vec![],
            offsets_words: vec![],
            aux_msgs: private.external_msgs.clone(),
            aux_cvs: private.external_cvs.clone(),
            lottery_words: core::array::from_fn(|i| (i as u32).wrapping_mul(0x85EBCA77) ^ 0x1234),
            key_a: key_a_words,
            key_b: key_b_words,
            jackpot_key: core::array::from_fn(|i| 0x3000_0007u32.wrapping_mul(i as u32 + 1)),
        };
        let (rows, pis) = program.generate_trace::<F>(&data.inputs());

        // The bound digests fold the committed full-tree roots — membership of the strips —
        // and equal the statement's `hash_a`/`hash_b` (what `parse_proof` computes).
        assert_eq!(pi_bytes(&pis, PI_HASH_A), public.hash_a());
        assert_eq!(pi_bytes(&pis, PI_HASH_B), public.hash_b());

        // The CTL surface covers exactly the opened strips: h*k/8 A-values message rows (8
        // int8 elements each), h*k/32 A-scales rows (4 scales each, 2 bytes per scale).
        let (h, w) = (a_rows.len(), b_rows.len());
        let count = |get: fn(&Blake3ColumnsView<F>) -> F| {
            rows.iter()
                .filter(|r| {
                    let v: &Blake3ColumnsView<F> = (*r).borrow();
                    get(v) == F::ONE
                })
                .count()
        };
        assert_eq!(count(|v| v.is_int8_message), (h + w) * k / 8);
        assert_eq!(count(|v| v.is_scale_message), (h + w) * 2 * n_blocks / 8);

        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    /// Exercise k = 2080: values/scales rows cross 64-byte block boundaries.
    /// The compiler emits cross-strip leaves when both sides are opened and split leaves
    /// when one side is unopened or padding. Check digest binding, exact CTL key coverage,
    /// known columns and constraints. Return split-kind counts for each fixture's assertions.
    fn k_mod_32_adapter_fixture(
        m: usize,
        n: usize,
        k: usize,
        a_rows: &[usize],
        a_dims: &[(u32, DimType)],
        b_rows: &[usize],
        b_dims: &[(u32, DimType)],
    ) -> (usize, usize, usize) {
        use crate::api::fp8::plain_proof::PlainProofV4;
        use crate::api::fp8::public_params::{CommonParams, Device, HashId, JobParams, OperandParams, Quant};
        use crate::api::primitives::{IncompleteBlockHeader, Sides};
        use crate::ffi::plain_proof::MatrixMerkleProof;

        assert_eq!(k % 32, 0);
        assert_ne!(k % 64, 0, "the fixture exists to exercise straddling rows");
        let n_blocks = k / BLOCK_SIZE;
        let rows_pattern = AxisPattern::new(a_dims).unwrap();
        let cols_pattern = AxisPattern::new(b_dims).unwrap();
        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let (key_a, key_b) = crate::circuit::fp8::consistency::fixture_tree_keys(&header);
        let key_a_words = core::array::from_fn(|i| u32::from_le_bytes(key_a[4 * i..4 * i + 4].try_into().unwrap()));
        let key_b_words = core::array::from_fn(|i| u32::from_le_bytes(key_b[4 * i..4 * i + 4].try_into().unwrap()));

        let value_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| (0..k).map(|j| ((seed + i * k + j) % 251) as u8).collect())
                .collect()
        };
        let scale_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| {
                    (0..n_blocks)
                        .flat_map(|b| (0x3f80u16 + ((seed + i + b) % 16) as u16).to_le_bytes())
                        .collect()
                })
                .collect()
        };
        let keyed_proof = |rows_bytes: &[Vec<u8>], row_indices: &[usize], key: [u8; 32]| -> MatrixMerkleProof {
            let flat: Vec<u8> = rows_bytes.iter().flatten().copied().collect();
            let tree = MerkleTree::new(&pad_to_chunk_boundary(&flat), key);
            let leaves = MerkleTree::compute_leaf_indices_from_rows(
                row_indices,
                (rows_bytes.len(), rows_bytes[0].len()),
                BLAKE3_CHUNK_LEN,
            )
            .unwrap();
            MatrixMerkleProof {
                proof: tree.get_multileaf_proof(&leaves),
                row_indices: row_indices.to_vec(),
            }
        };
        let proof = PlainProofV4 {
            job: JobParams {
                // σ̂ and σ_Δ coincide in this fixture (the ancestor is the proposed header).
                ancestor_header: header,
                common: CommonParams {
                    k: k as u32,
                    r: 32,
                    quant: Quant::Fp8E4M3Prequant,
                    device: Device::B200,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: m as u32,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: rows_pattern,
                    },
                    b: OperandParams {
                        num_rows: n as u32,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: cols_pattern,
                    },
                },
                moe: None,
            },
            values: Sides {
                a: keyed_proof(&value_tree(m, 0), a_rows, key_a),
                b: keyed_proof(&value_tree(n, 7), b_rows, key_b),
            },
            scales: Sides {
                a: keyed_proof(&scale_tree(m, 1), a_rows, key_a),
                b: keyed_proof(&scale_tree(n, 5), b_rows, key_b),
            },
            moe_witness: None,
        };

        // The real verifier: the wire layer now admits k % 32 == 0, membership-checks the
        // trees, and reruns the (split-aware) compiled program natively against the roots.
        let (private, public) = proof.parse_proof(&header).expect("k % 32 fixture must parse");
        let (compiled, _, _) = public.compile(&header).expect("fixture must compile");
        let program = Blake3Program::from_blake_program(&compiled.blake_proof, k, vec![], None);

        // Tally the straddle kinds the schedule produced (returned for per-geometry pins).
        let (mut cross_strip, mut split_opened_first, mut split_unopened_first) = (0usize, 0usize, 0usize);
        for instr in &program.instructions {
            match instr.msg {
                MessageSource::PlaneBytes { plane, offset, .. }
                    if plane != PlaneId::Routing && {
                        let row_bytes = match plane {
                            PlaneId::AValues | PlaneId::BValues => k,
                            _ => k / 4,
                        };
                        offset / row_bytes != (offset + BLAKE3_MSG_LEN - 1) / row_bytes
                    } =>
                {
                    cross_strip += 1;
                }
                MessageSource::PlaneBytesSplit { skip: 0, .. } => split_opened_first += 1,
                MessageSource::PlaneBytesSplit { .. } => split_unopened_first += 1,
                _ => {}
            }
        }

        let data = TestData {
            a_values: private.operands.a.values.as_bytes().to_vec(),
            a_scales: private.operands.a.scales.as_bytes().to_vec(),
            b_values: private.operands.b.values.as_bytes().to_vec(),
            b_scales: private.operands.b.scales.as_bytes().to_vec(),
            routing_words: vec![],
            offsets_words: vec![],
            aux_msgs: private.external_msgs.clone(),
            aux_cvs: private.external_cvs.clone(),
            lottery_words: core::array::from_fn(|i| (i as u32).wrapping_mul(0x85EBCA77) ^ 0x1234),
            key_a: key_a_words,
            key_b: key_b_words,
            jackpot_key: core::array::from_fn(|i| 0x3000_0007u32.wrapping_mul(i as u32 + 1)),
        };
        let (rows, pis) = program.generate_trace::<F>(&data.inputs());

        // Committed-root binding through the straddling schedule: the bound digests fold
        // the committed roots and equal the statement's `hash_a`/`hash_b`.
        assert_eq!(pi_bytes(&pis, PI_HASH_A), public.hash_a());
        assert_eq!(pi_bytes(&pis, PI_HASH_B), public.hash_b());

        // The CTL surface covers exactly the opened strips, split dwords included: every
        // element / scale crosses exactly once, none of the unopened-neighbor bytes do.
        let (h, w) = (a_rows.len(), b_rows.len());
        let count = |get: fn(&Blake3ColumnsView<F>) -> F| {
            rows.iter()
                .filter(|r| {
                    let v: &Blake3ColumnsView<F> = (*r).borrow();
                    get(v) == F::ONE
                })
                .count()
        };
        assert_eq!(count(|v| v.is_int8_message), (h + w) * k / 8);
        assert_eq!(count(|v| v.is_scale_message), (h + w) * 2 * n_blocks / 8);
        // ... at pairwise-distinct keys: with per-key uniqueness, the counts above pin the
        // exposed key sets to exactly [0, h*k) + [h*k, (h+w)*k) (values) and the block range
        // (scales), matching InputQuant's demand multiset.
        let mut value_keys: Vec<u64> = Vec::new();
        let mut scale_keys: Vec<u64> = Vec::new();
        for r in rows.iter() {
            let v: &Blake3ColumnsView<F> = r.borrow();
            if v.is_int8_message == F::ONE {
                value_keys.push(v.ctl_key_base.to_canonical_u64());
            }
            if v.is_scale_message == F::ONE {
                scale_keys.push(v.ctl_key_base.to_canonical_u64());
            }
        }
        value_keys.sort_unstable();
        scale_keys.sort_unstable();
        assert!(value_keys.iter().enumerate().all(|(i, &b)| b == 8 * i as u64));
        assert!(scale_keys.iter().enumerate().all(|(i, &b)| b == 4 * i as u64));

        // The verifier-known recompute is bit-exact with the trace, split rows included.
        let known = program.known_values::<F>(&data.inputs().known_inputs());
        assert_eq!(known.len(), NUM_BLAKE3_KNOWN_COLUMNS);
        for (c, column) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(column.values[r], row[c], "known column {c} differs from trace at row {r}");
            }
        }

        assert!(!constraints_violated(&S::new(program), &rows, &pis));
        (cross_strip, split_opened_first, split_unopened_first)
    }

    /// `k = 2080`, 8×32 tile: A opens the contiguous band `{0..7}` (cross-strip
    /// blocks at even|odd row boundaries) and B the isolated odd rows `{1,3,…,63}` (fold
    /// `{0,2,4,6}` (+) blake `{0,8,…,56}`, based at 1) of `n = 64`. Odd B rows start
    /// mid-block against unopened even neighbors (`skip > 0` splits); the 520-byte scales
    /// rows add opened-first splits (`skip = 0`).
    #[test]
    fn adapter_handles_row_straddling_blocks_at_k_mod_32() {
        let a_rows: Vec<usize> = (0..8).collect();
        let b_rows: Vec<usize> = (0..32).map(|i| 1 + 2 * i).collect();
        let (cross, opened_first, unopened_first) = k_mod_32_adapter_fixture(
            8,
            64,
            2080,
            &a_rows,
            &[(4, DimType::Fold), (2, DimType::Blake)],
            &b_rows,
            &[(2, DimType::Null), (4, DimType::Fold), (8, DimType::Blake)],
        );
        assert!(cross > 0, "adjacent opened rows must yield cross-strip blocks");
        assert!(opened_first > 0, "opened-tail blocks must yield skip = 0 splits");
        assert!(unopened_first > 0, "opened-head blocks must yield skip > 0 splits");
    }

    /// `m = 17` (odd) with every A row opened: the raw A plane ends mid-block
    /// (`17 * 2080 ≡ 32 (mod 64)`), so row 16's tail block straddles into the chunk padding —
    /// an opened-first split whose auxiliary half carries the padding zeros. B keeps the
    /// isolated-odd 32-col tile so values-plane splits against unopened neighbors still appear.
    #[test]
    fn adapter_handles_straddle_into_chunk_padding() {
        let a_rows: Vec<usize> = (0..17).collect();
        let b_rows: Vec<usize> = (0..32).map(|i| 1 + 2 * i).collect();
        let (_, opened_first, unopened_first) = k_mod_32_adapter_fixture(
            17,
            64,
            2080,
            &a_rows,
            &[(17, DimType::Fold)],
            &b_rows,
            &[(2, DimType::Null), (2, DimType::Fold), (16, DimType::Blake)],
        );
        assert!(opened_first > 0, "the raw-tail block must split into the chunk padding");
        assert!(unopened_first > 0);
    }

    /// Reject an Int7 schedule, which lacks the four values/scales planes FP8 requires.
    #[test]
    #[should_panic(expected = "not a prequant four-plane program")]
    fn adapter_rejects_legacy_int7_programs() {
        let h = 16;
        let w = 16;
        let k = 2048;
        let blake_proof = chip_program::BlakeProgram {
            num_a_rows: h,
            num_b_cols: w,
            strip_length: k,
            num_routing_strips: 0,
            num_offsets_strips: 0,
            num_auxiliary_msgs: 0,
            num_auxiliary_cvs: 0,
            instructions: vec![],
        };
        let _ = Blake3Program::from_blake_program(&blake_proof, k, vec![], None);
    }
}
