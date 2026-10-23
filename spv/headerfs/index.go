package headerfs

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

var (
	// indexBucket is the main top-level bucket for the header index.
	// Nothing is stored in this bucket other than the sub-buckets which
	// contains the indexes for the various header types.
	indexBucket = []byte("header-index")

	// blockTip is the database key that stores the hash of the latest
	// block header we have synced. Used by the Block header type.
	blockTip = []byte("block")

	// regFilterTip is the database key that stores the hash of the latest
	// regular compact filter header we have synced. Used by the
	// RegularFilter header type.
	regFilterTip = []byte("regular")
)

var (
	// ErrHeightNotFound is returned when a specified height isn't found in
	// a target index.
	ErrHeightNotFound = fmt.Errorf("target height not found in index")

	// ErrHashNotFound is returned when a specified block hash isn't found
	// in a target index.
	ErrHashNotFound = fmt.Errorf("target hash not found in index")

	// ErrHeaderStoreCorrupt is returned when a header file and its index can't be reconciled. The headers have to be
	// discarded and synced again.
	ErrHeaderStoreCorrupt = errors.New("header store is corrupt")
)

func corruptErrf(format string, args ...interface{}) error {
	return fmt.Errorf("%w: %w", ErrHeaderStoreCorrupt, fmt.Errorf(format, args...))
}

// HeaderType is an enum-like type which defines the various header types that
// are stored within the index.
type HeaderType uint8

const (
	// Block is the header type that represents regular block
	// headers.
	Block HeaderType = 0

	// RegularFilter is a header type that represents the basic filter
	// header type for the filter header chain.
	RegularFilter HeaderType = 1

	// UnknownHeader represents an unrecognized header type.
	UnknownHeader HeaderType = 255
)

const (
	// BlockHeaderSize is the size in bytes of the Block header type.
	BlockHeaderSize = wire.MaxBlockHeaderPayload

	// RegularFilterHeaderSize is the size in bytes of the RegularFilter
	// header type.
	RegularFilterHeaderSize = 32

	// UnknownHeaderSize is the size for the unknown header type.
	UnknownHeaderSize = 0

	// numSubBucketBytes is the number of bytes of a hash that's used as a
	// sub bucket to store the index keys (hash->height) in. Storing a large
	// number of keys in the same bucket has a large impact on memory usage
	// in bbolt if small-ish batch sizes are used (the b+ tree needs to be
	// copied with every resize operation). Using sub buckets is a
	// compromise between memory usage and access time. 2 bytes (=max 65535
	// sub buckets) seems to be the sweet spot (-50% memory usage,
	// +30% access time). We take the bytes from the beginning of the byte-
	// serialized hash since all block hashes are reverse-serialized when
	// displayed as strings. That means the leading zeroes of a block hash
	// are actually at the end of the byte slice.
	numSubBucketBytes = 2
)

// String returns the string representation of the HeaderType.
func (h HeaderType) String() string {
	switch h {
	case Block:
		return "BlockHeader"
	case RegularFilter:
		return "RegularFilterHeader"
	default:
		return fmt.Sprintf("UnknownHeaderType(%d)", h)
	}
}

// Size returns the size in bytes for a given header type.
func (h HeaderType) Size() (int, error) {
	switch h {
	case Block:
		return BlockHeaderSize, nil
	case RegularFilter:
		return RegularFilterHeaderSize, nil
	default:
		return UnknownHeaderSize, fmt.Errorf("unknown header type: "+
			"%d", h)
	}
}

// TipKey returns the current tip key for the given header type.
// Returns an error if the header type is unknown.
func (h HeaderType) TipKey() ([]byte, error) {
	switch h {
	case Block:
		return blockTip, nil
	case RegularFilter:
		return regFilterTip, nil
	default:
		return nil, fmt.Errorf("unknown header type: %d", h)
	}
}

// headerIndex is an index stored within the database that allows for random
// access into the on-disk header file. This, in conjunction with a flat file
// of headers consists of header database. The keys have been specifically
// crafted in order to ensure maximum write performance during IBD, and also to
// provide the necessary indexing properties required.
type headerIndex struct {
	db walletdb.DB

	indexType HeaderType
}

// newHeaderIndex creates a new headerIndex given an already open database, and
// a particular header type.
func newHeaderIndex(db walletdb.DB, indexType HeaderType) (*headerIndex, error) {
	// As an initially step, we'll attempt to create all the buckets
	// necessary for functioning of the index. If these buckets has already
	// been created, then we can exit early.
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		_, err := tx.CreateTopLevelBucket(indexBucket)
		return err
	})
	if err != nil && err != walletdb.ErrBucketExists {
		return nil, err
	}

	return &headerIndex{
		db:        db,
		indexType: indexType,
	}, nil
}

// headerEntry is an internal type that's used to quickly map a (height, hash)
// pair into the proper key that'll be stored within the database.
type headerEntry struct {
	hash   chainhash.Hash
	height uint32
}

// headerBatch is a batch of header entries to be written to disk.
//
// NOTE: The entries within a batch SHOULD be properly sorted by hash in
// order to ensure the batch is written in a sequential write.
type headerBatch []headerEntry

// Len returns the number of routes in the collection.
//
// NOTE: This is part of the sort.Interface implementation.
func (h headerBatch) Len() int {
	return len(h)
}

// Less reports where the entry with index i should sort before the entry with
// index j. As we want to ensure the items are written in sequential order,
// items with the "first" hash.
//
// NOTE: This is part of the sort.Interface implementation.
func (h headerBatch) Less(i, j int) bool {
	return bytes.Compare(h[i].hash[:], h[j].hash[:]) < 0
}

// Swap swaps the elements with indexes i and j.
//
// NOTE: This is part of the sort.Interface implementation.
func (h headerBatch) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

// addHeaders writes a batch of header entries in a single atomic batch.
func (h *headerIndex) addHeaders(batch headerBatch) error {
	// If we're writing a 0-length batch, make no changes and return.
	if len(batch) == 0 {
		return nil
	}

	// In order to ensure optimal write performance, we'll ensure that the
	// items are sorted by their hash before insertion into the database.
	sort.Sort(batch)

	return walletdb.Update(h.db, func(tx walletdb.ReadWriteTx) error {
		rootBucket := tx.ReadWriteBucket(indexBucket)

		var tipKey []byte

		// Based on the specified index type of this instance of the
		// index, we'll grab the key that tracks the tip of the chain
		// so we can update the index once all the header entries have
		// been updated.
		// TODO(roasbeef): only need block tip?
		tipKey, err := h.indexType.TipKey()
		if err != nil {
			return err
		}

		var (
			chainTipHash   chainhash.Hash
			chainTipHeight uint32
		)

		for _, header := range batch {
			if err := putHeaderEntry(rootBucket, header); err != nil {
				return err
			}

			// TODO(roasbeef): need to remedy if side-chain
			// tracking added
			if header.height >= chainTipHeight {
				chainTipHash = header.hash
				chainTipHeight = header.height
			}
		}

		return rootBucket.Put(tipKey, chainTipHash[:])
	})
}

// heightFromHash returns the height of the entry that matches the specified
// height. With this height, the caller is then able to seek to the appropriate
// spot in the flat files in order to extract the true header.
func (h *headerIndex) heightFromHash(hash *chainhash.Hash) (uint32, error) {
	var height uint32
	err := walletdb.View(h.db, func(tx walletdb.ReadTx) error {
		var err error
		height, err = h.heightFromHashWithTx(tx, hash)
		return err
	})
	if err != nil {
		return 0, err
	}

	return height, nil
}

// heightFromHashWithTx returns the height of the entry that matches the
// specified hash by using the given DB transaction. With this height, the
// caller is then able to seek to the appropriate spot in the flat files in
// order to extract the true header.
func (h *headerIndex) heightFromHashWithTx(tx walletdb.ReadTx,
	hash *chainhash.Hash) (uint32, error) {

	rootBucket := tx.ReadBucket(indexBucket)

	return getHeaderEntry(rootBucket, hash[:])
}

// chainTip returns the best hash and height that the index knows of.
func (h *headerIndex) chainTip() (*chainhash.Hash, uint32, error) {
	var (
		tipHeight uint32
		tipHash   *chainhash.Hash
	)

	err := walletdb.View(h.db, func(tx walletdb.ReadTx) error {
		var err error
		tipHash, tipHeight, err = h.chainTipWithTx(tx)
		return err
	})
	if err != nil {
		return nil, 0, err
	}

	return tipHash, tipHeight, nil
}

// chainTipWithTx returns the best hash and height that the index knows of by
// using the given DB transaction.
func (h *headerIndex) chainTipWithTx(tx walletdb.ReadTx) (*chainhash.Hash,
	uint32, error) {

	rootBucket := tx.ReadBucket(indexBucket)

	// Based on the specified index type of this instance of the index,
	// we'll grab the particular key that tracks the chain tip.
	tipKey, err := h.indexType.TipKey()
	if err != nil {
		return nil, 0, err
	}

	// Now that we have the particular tip key for this header type, we'll
	// fetch the hash for this tip, then using that we'll fetch the height
	// that corresponds to that hash.
	tipHashBytes := rootBucket.Get(tipKey)
	if tipHashBytes == nil {
		return nil, 0, fmt.Errorf(
			"the key %s does not exist in bucket %s",
			tipKey, indexBucket,
		)
	}
	tipHeight, err := getHeaderEntry(rootBucket, tipHashBytes)
	if err != nil {
		return nil, 0, ErrHeightNotFound
	}

	// With the height fetched, we can now populate our return
	// parameters.
	tipHash, err := chainhash.NewHash(tipHashBytes)
	if err != nil {
		return nil, 0, err
	}

	return tipHash, tipHeight, nil
}

// truncateIndices truncates the index for a particular header type by removing
// a set of header entries. The passed newTip pointer should point to the hash
// of the new chain tip. The blockHeadersToTruncate contains the hashes of all
// block headers that should be removed from the index, which are the block
// headers after the new tip. Optionally, if the entries are to be deleted as
// well, then the remove flag should be set to true.
func (h *headerIndex) truncateIndices(newTip *chainhash.Hash,
	blockHeadersToTruncate []*chainhash.Hash, remove bool) error {

	if remove && len(blockHeadersToTruncate) == 0 {
		return errors.New("remove flag set but headers to truncate " +
			"beyond new tip not provided")
	}

	if !remove && len(blockHeadersToTruncate) != 0 {
		return errors.New("headers to truncate beyond new tip " +
			"provided but remove flag not set")
	}

	return walletdb.Update(h.db, func(tx walletdb.ReadWriteTx) error {
		rootBucket := tx.ReadWriteBucket(indexBucket)

		var tipKey []byte

		// Based on the specified index type of this instance of the
		// index, we'll grab the key that tracks the tip of the chain
		// we need to update.
		tipKey, err := h.indexType.TipKey()
		if err != nil {
			return err
		}

		// If the remove flag is set, then we'll also delete those
		// entries from the database as the primary index
		// (block headers) is being rolled back.
		if remove {
			if err := deleteHeaderEntries(
				rootBucket, blockHeadersToTruncate,
			); err != nil {
				return err
			}
		}

		// With the now stale entry deleted, we'll update the chain tip
		// to point to the new hash.
		return rootBucket.Put(tipKey, newTip[:])
	})
}

// putHeaderEntry stores a headerEntry into the bbolt database. The entry is
// always placed below a sub bucket with the first few bytes of the hash as its
// name to improve write performance.
func putHeaderEntry(rootBucket walletdb.ReadWriteBucket,
	header headerEntry) error {

	// Place key in a sub bucket to improve bbolt memory behavior at the
	// expense of a slight increase in access latency.
	subBucket, err := rootBucket.CreateBucketIfNotExists(
		header.hash[0:numSubBucketBytes],
	)
	if err != nil {
		return err
	}

	var heightBytes [4]byte
	binary.BigEndian.PutUint32(heightBytes[:], header.height)
	return subBucket.Put(header.hash[:], heightBytes[:])
}

// getHeaderEntry tries to look up the height of a header by its hash, in the sub bucket named after the first bytes of
// the hash. If no key for the given hash is found, ErrHashNotFound is returned.
func getHeaderEntry(rootBucket walletdb.ReadBucket, hashBytes []byte) (uint32,
	error) {

	subBucket := rootBucket.NestedReadBucket(hashBytes[0:numSubBucketBytes])
	if subBucket == nil {
		return 0, ErrHashNotFound
	}

	heightBytes := subBucket.Get(hashBytes)
	if heightBytes == nil {
		return 0, ErrHashNotFound
	}

	return binary.BigEndian.Uint32(heightBytes), nil
}

// deleteHeaderEntries tries to remove multiple header entries from the sub buckets of the bbolt database.
func deleteHeaderEntries(rootBucket walletdb.ReadWriteBucket,
	headerHashes []*chainhash.Hash) error {

	if len(headerHashes) == 0 {
		return nil
	}

	// Group hashes by their sub-bucket for more efficient deletion.
	bySubBucket := make(map[string][]*chainhash.Hash)
	for _, hash := range headerHashes {
		prefix := string(hash[0:numSubBucketBytes])
		bySubBucket[prefix] = append(bySubBucket[prefix], hash)
	}

	// Delete enties from sub-buckets.
	for prefix, hashes := range bySubBucket {
		// Try to get the sub-bucket for this prefix. If it doesn't
		// exist, something is wrong and we want to return an error.
		subBucket := rootBucket.NestedReadWriteBucket([]byte(prefix))
		if subBucket == nil {
			return fmt.Errorf("%w: sub-bucket for prefix %x not "+
				"found", ErrHashNotFound, prefix)
		}

		for _, hash := range hashes {
			hashBytes := hash.CloneBytes()
			if err := subBucket.Delete(hashBytes); err != nil {
				return err
			}
		}
	}

	return nil
}

// forEachEntry calls fn with every hash and height in the index's sub buckets. It runs while recovering from a crash,
// so entries that don't hold a height are skipped rather than read.
func forEachEntry(rootBucket walletdb.ReadBucket, fn func(hash chainhash.Hash, height uint32) error) error {
	return rootBucket.ForEach(func(k, _ []byte) error {
		subBucket := rootBucket.NestedReadBucket(k)
		if subBucket == nil {
			return nil
		}

		return subBucket.ForEach(func(k, v []byte) error {
			if len(k) != chainhash.HashSize || len(v) != 4 {
				return nil
			}

			var hash chainhash.Hash
			copy(hash[:], k)

			return fn(hash, binary.BigEndian.Uint32(v))
		})
	})
}

// dropEntriesAbove deletes every entry above height.
func dropEntriesAbove(rootBucket walletdb.ReadWriteBucket, height uint32) error {
	var stale []*chainhash.Hash
	err := forEachEntry(rootBucket, func(hash chainhash.Hash, at uint32) error {
		if at > height {
			stale = append(stale, &hash)
		}

		return nil
	})
	if err != nil {
		return err
	}

	return deleteHeaderEntries(rootBucket, stale)
}

// rollbackIndex makes tip, which must be indexed at height, the tip of the index and drops every entry above it.
func (h *headerIndex) rollbackIndex(tip *chainhash.Hash, height uint32) error {
	tipKey, err := h.indexType.TipKey()
	if err != nil {
		return err
	}

	return walletdb.Update(h.db, func(tx walletdb.ReadWriteTx) error {
		rootBucket := tx.ReadWriteBucket(indexBucket)

		indexed, err := getHeaderEntry(rootBucket, tip[:])
		if err != nil || indexed != height {
			return corruptErrf("header %v is not indexed at height %d", tip, height)
		}
		if err := dropEntriesAbove(rootBucket, height); err != nil {
			return err
		}

		return rootBucket.Put(tipKey, tip[:])
	})
}

// setTipAtHeight points the tip at the header indexed at height. There must be exactly one, as there is no telling
// which of several is right.
func (h *headerIndex) setTipAtHeight(height uint32) error {
	tipKey, err := h.indexType.TipKey()
	if err != nil {
		return err
	}

	return walletdb.Update(h.db, func(tx walletdb.ReadWriteTx) error {
		rootBucket := tx.ReadWriteBucket(indexBucket)

		var tip *chainhash.Hash
		err := forEachEntry(rootBucket, func(hash chainhash.Hash, at uint32) error {
			switch {
			case at != height:
			case tip != nil:
				return fmt.Errorf("several headers at height %d", height)
			default:
				tip = &hash
			}

			return nil
		})
		if err == nil && tip == nil {
			err = ErrHeightNotFound
		}
		if err != nil {
			return corruptErrf("%v tip at height %d: %w", h.indexType, height, err)
		}

		return rootBucket.Put(tipKey, tip[:])
	})
}
