package headerfs

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// TestHeightFromHashShortStoredValue ensures that a truncated height
// value in the header index (e.g. from a corrupt or partially written
// database) returns ErrHashNotFound instead of panicking inside
// binary.BigEndian.Uint32, on both the sub-bucket path and the legacy
// root-bucket fallback path.
func TestHeightFromHashShortStoredValue(t *testing.T) {
	cleanUp, hIndex, err := createTestIndex(t)
	if err != nil {
		t.Fatalf("unable to create test db: %v", err)
	}
	defer cleanUp()

	var subHash, rootHash chainhash.Hash
	copy(subHash[:], []byte("sub-bucket-short-value-hash-0001"))
	copy(rootHash[:], []byte("root-bucket-short-value-hash-001"))

	err = walletdb.Update(hIndex.db, func(tx walletdb.ReadWriteTx) error {
		rootBucket := tx.ReadWriteBucket(indexBucket)

		// Main path: entry inside the hash-prefix sub-bucket with a
		// 3-byte value (a valid height is exactly 4 bytes).
		subBucket, err := rootBucket.CreateBucketIfNotExists(
			subHash[0:numSubBucketBytes],
		)
		if err != nil {
			return err
		}
		if err := subBucket.Put(subHash[:], []byte{0x01, 0x02, 0x03}); err != nil {
			return err
		}

		// Fallback path: entry directly in the root bucket with a
		// 1-byte value.
		return rootBucket.Put(rootHash[:], []byte{0x07})
	})
	if err != nil {
		t.Fatalf("unable to seed corrupt index values: %v", err)
	}

	for _, hash := range []*chainhash.Hash{&subHash, &rootHash} {
		_, err := hIndex.heightFromHash(hash)
		if err != ErrHashNotFound {
			t.Fatalf("heightFromHash(%x): expected ErrHashNotFound, "+
				"got %v", hash[:8], err)
		}
	}
}
