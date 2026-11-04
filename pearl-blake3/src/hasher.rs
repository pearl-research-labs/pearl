//! `Blake3Hasher` is the single implementation for all BLAKE3 operations in pearl-blake3.
//! `MerkleTree` and `MerkleProof` delegate to it for cryptographic operations.

use blake3::hazmat::{merge_subtrees_non_root, merge_subtrees_root, HasherExt, Mode};
use blake3::platform::Platform;
use blake3::{Hasher, IncrementCounter, CHUNK_LEN, OUT_LEN};
use rayon::prelude::*;

pub(crate) type Digest = [u8; OUT_LEN];
type Key = [u8; OUT_LEN];

/// BLAKE3 domain separation flags.
/// The `blake3` crate does not export these; defined here once, imported by consumers.
pub const B3F_CHUNK_START: u8 = 1;
pub const B3F_CHUNK_END: u8 = 2;
pub const B3F_PARENT: u8 = 4;
pub const B3F_ROOT: u8 = 8;
pub const B3F_KEYED_HASH: u8 = 16;

const BYTES_PER_WORD: usize = std::mem::size_of::<u32>();
const KEY_WORD_COUNT: usize = OUT_LEN / BYTES_PER_WORD;

/// BLAKE3 IV (SHA-256 fractional parts of square roots of first 8 primes).
const IV: [u32; KEY_WORD_COUNT] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Segment size for parallel hashing (each chunk = 1KB).
const CHUNKS_PER_SEGMENT: usize = 64;
const BYTES_PER_SEGMENT: usize = CHUNKS_PER_SEGMENT * CHUNK_LEN;

fn key_to_words(key: &Key) -> [u32; KEY_WORD_COUNT] {
    std::array::from_fn(|i| {
        u32::from_le_bytes(
            key[i * BYTES_PER_WORD..][..BYTES_PER_WORD]
                .try_into()
                .unwrap(),
        )
    })
}

/// SIMD-optimized BLAKE3 hasher with optional keyed mode.
/// Parallelism is controlled globally via rayon's thread pool (RAYON_NUM_THREADS).
pub struct Blake3Hasher {
    key: Option<Key>,
    key_words: [u32; KEY_WORD_COUNT],
    base_flags: u8,
    platform: Platform,
}

impl Blake3Hasher {
    /// Create an unkeyed hasher.
    pub fn new() -> Self {
        Self {
            key: None,
            key_words: IV,
            base_flags: 0,
            platform: Platform::detect(),
        }
    }

    /// Create a keyed hasher.
    pub fn with_key(key: Key) -> Self {
        Self {
            key: Some(key),
            key_words: key_to_words(&key),
            base_flags: B3F_KEYED_HASH,
            platform: Platform::detect(),
        }
    }

    /// Compute BLAKE3 hash of data.
    pub fn hash(&self, data: &[u8]) -> Digest {
        let mut hasher = self.create_hasher();
        hasher.update_rayon(data);
        *hasher.finalize().as_bytes()
    }

    /// Compute chunk chaining value (non-root).
    pub fn chunk_cv(&self, data: &[u8], chunk_index: u64) -> Digest {
        let mut hasher = self.create_hasher();
        hasher
            .set_input_offset(chunk_index * CHUNK_LEN as u64)
            .update(data)
            .finalize_non_root()
    }

    /// Combine two child CVs into a parent CV (non-root).
    pub fn parent_cv(&self, left: &Digest, right: &Digest) -> Digest {
        merge_subtrees_non_root(left, right, self.mode())
    }

    /// Combine two child CVs into the root hash.
    pub fn root_cv(&self, left: &Digest, right: &Digest) -> Digest {
        *merge_subtrees_root(left, right, self.mode()).as_bytes()
    }

    fn mode(&self) -> Mode<'_> {
        match &self.key {
            Some(k) => Mode::KeyedHash(k),
            None => Mode::Hash,
        }
    }

    fn create_hasher(&self) -> Hasher {
        match &self.key {
            Some(k) => Hasher::new_keyed(k),
            None => Hasher::new(),
        }
    }

    /// Hash all chunks in data, returning one CV per chunk. Uses SIMD batching.
    pub(crate) fn hash_chunks(&self, data: &[u8]) -> Vec<Digest> {
        if data.is_empty() {
            return vec![];
        }

        let num_chunks = data.len().div_ceil(CHUNK_LEN);
        let mut all_cvs: Vec<Digest> = vec![[0u8; OUT_LEN]; num_chunks];

        all_cvs
            .par_chunks_mut(CHUNKS_PER_SEGMENT)
            .enumerate()
            .for_each(|(seg_idx, cv_slice)| self.process_segment(data, seg_idx, cv_slice));

        all_cvs
    }

    /// Combine pairs of CVs into parent CVs for one tree layer.
    pub(crate) fn combine_layer(&self, prev: &[Digest]) -> Vec<Digest> {
        let mode = self.mode();
        let pairs: Vec<_> = prev.chunks(2).collect();
        let combine = |pair: &[Digest]| {
            if pair.len() == 2 {
                merge_subtrees_non_root(&pair[0], &pair[1], mode)
            } else {
                pair[0]
            }
        };

        pairs.par_iter().map(|p| combine(p)).collect()
    }

    fn process_segment(&self, data: &[u8], seg_idx: usize, cv_slice: &mut [Digest]) {
        let start_chunk = seg_idx * CHUNKS_PER_SEGMENT;
        let start_byte = start_chunk * CHUNK_LEN;
        let end_byte = ((seg_idx + 1) * BYTES_PER_SEGMENT).min(data.len());
        let segment_data = &data[start_byte..end_byte];

        let (full_chunks, last_chunk) = segment_data.as_chunks::<CHUNK_LEN>();

        let num_full = full_chunks.len();
        let full_chunks: Vec<&[u8; CHUNK_LEN]> = full_chunks.iter().collect();

        if !full_chunks.is_empty() {
            let mut simd_out = vec![0u8; num_full * OUT_LEN];
            self.platform.hash_many(
                &full_chunks,
                &self.key_words,
                start_chunk as u64,
                IncrementCounter::Yes,
                self.base_flags,
                B3F_CHUNK_START,
                B3F_CHUNK_END,
                &mut simd_out,
            );
            for (i, chunk_out) in simd_out.as_chunks::<OUT_LEN>().0.iter().enumerate() {
                cv_slice[i].copy_from_slice(chunk_out);
            }
        }

        if !last_chunk.is_empty() {
            let mut hasher = self.create_hasher();
            cv_slice[num_full] = hasher
                .set_input_offset((start_chunk + num_full) as u64 * CHUNK_LEN as u64)
                .update(last_chunk)
                .finalize_non_root();
        }
    }
}

/// One-shot BLAKE3 hash returning raw `[u8; 32]`. Wraps `blake3::hash` / `blake3::keyed_hash`.
pub fn blake3_digest(data: &[u8], key: Option<[u8; 32]>) -> [u8; 32] {
    match key {
        Some(k) => *blake3::keyed_hash(&k, data).as_bytes(),
        None => *blake3::hash(data).as_bytes(),
    }
}

impl Default for Blake3Hasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_hash_empty() {
        let result = Blake3Hasher::new().hash(&[]);
        assert_eq!(result, blake3_digest(&[], None));
    }

    #[test]
    fn test_hash_single_block() {
        let data = test_data(64);
        assert_eq!(Blake3Hasher::new().hash(&data), blake3_digest(&data, None));
    }

    #[test]
    fn test_hash_single_chunk() {
        let data = test_data(CHUNK_LEN);
        assert_eq!(Blake3Hasher::new().hash(&data), blake3_digest(&data, None));
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let data = test_data(2048);
        assert_eq!(Blake3Hasher::new().hash(&data), blake3_digest(&data, None));
    }

    #[test]
    fn test_hash_with_key() {
        let key = [42u8; 32];
        let data = test_data(3072);
        let hasher = Blake3Hasher::with_key(key);
        assert_eq!(hasher.hash(&data), blake3_digest(&data, Some(key)));
    }

    #[test]
    fn test_hash_arbitrary_lengths() {
        let lengths = [
            1, 7, 31, 63, 64, 65, 127, 512, 1023, 1024, 1025, 2047, 2048, 4097,
        ];
        for len in lengths {
            let data = test_data(len);
            assert_eq!(
                Blake3Hasher::new().hash(&data),
                blake3_digest(&data, None),
                "Failed for length {len}"
            );

            let key = [42u8; 32];
            assert_eq!(
                Blake3Hasher::with_key(key).hash(&data),
                blake3_digest(&data, Some(key)),
                "Failed for keyed hash with length {len}"
            );
        }
    }

    #[test]
    fn test_hash_large_data() {
        for size in [10240, 102400, 7168] {
            let data = test_data(size);
            assert_eq!(
                Blake3Hasher::new().hash(&data),
                blake3_digest(&data, None),
                "Failed for {size} bytes"
            );
        }
    }

    #[test]
    fn test_chunk_cv() {
        let key = [42u8; 32];
        let data = test_data(CHUNK_LEN);
        let hasher = Blake3Hasher::with_key(key);
        let cv = hasher.chunk_cv(&data, 0);

        let expected = Hasher::new_keyed(&key)
            .set_input_offset(0)
            .update(&data)
            .finalize_non_root();
        assert_eq!(cv, expected);
    }

    #[test]
    fn test_parent_cv_and_root_cv() {
        let key = [42u8; 32];
        let hasher = Blake3Hasher::with_key(key);
        let left = [1u8; OUT_LEN];
        let right = [2u8; OUT_LEN];

        let parent = hasher.parent_cv(&left, &right);
        let root = hasher.root_cv(&left, &right);

        let mode = Mode::KeyedHash(&key);
        assert_eq!(parent, merge_subtrees_non_root(&left, &right, mode));
        assert_eq!(root, *merge_subtrees_root(&left, &right, mode).as_bytes());
        assert_ne!(parent, root);
    }

    #[test]
    fn test_hash_chunks_matches_hash() {
        let key = [42u8; 32];
        for num_chunks in [1, 2, 3, 5, 7, 10] {
            let data = test_data(num_chunks * CHUNK_LEN);
            let hasher = Blake3Hasher::with_key(key);
            let cvs = hasher.hash_chunks(&data);
            assert_eq!(
                cvs.len(),
                num_chunks,
                "Wrong CV count for {num_chunks} chunks"
            );

            for (i, cv) in cvs.iter().enumerate() {
                let chunk = &data[i * CHUNK_LEN..(i + 1) * CHUNK_LEN];
                assert_eq!(*cv, hasher.chunk_cv(chunk, i as u64));
            }
        }
    }
}
