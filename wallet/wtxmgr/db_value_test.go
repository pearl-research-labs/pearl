// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wtxmgr

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// putRawBucketValue writes v under k in the named nested bucket of the
// txstore namespace, creating the bucket if needed.
func putRawBucketValue(t *testing.T, db walletdb.DB, bucket, k, v []byte) {
	t.Helper()

	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(namespaceKey)
		b, err := ns.CreateBucketIfNotExists(bucket)
		if err != nil {
			return err
		}
		return b.Put(k, v)
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestIsLockedOutputTruncatedValue ensures a locked-output record shorter
// than the serialized LockID+expiry (40 bytes) does not panic coin
// selection; the output is reported as not locked.
func TestIsLockedOutputTruncatedValue(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	op := wire.OutPoint{Hash: chainhash.Hash{0x01}, Index: 0}
	k := canonicalOutPoint(&op.Hash, op.Index)

	for _, v := range [][]byte{
		{},
		{0x01, 0x02, 0x03},
		make([]byte, 32), // full LockID, missing expiry
		make([]byte, 39), // one byte short
	} {
		putRawBucketValue(t, db, bucketLockedOutputs, k, v)

		err := walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(namespaceKey)
			_, _, locked := isLockedOutput(ns, op, time.Now())
			if locked {
				t.Errorf("value len %d: reported locked", len(v))
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

// TestLockedOutputRoundTrip pins the well-formed contract: a value written
// by serializeLockedOutput is read back as locked with the same ID/expiry.
func TestLockedOutputRoundTrip(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	var id LockID
	copy(id[:], []byte("a-lock-id-for-testing-purposes!!"))
	expiry := time.Now().Add(time.Hour).Truncate(time.Second)

	op := wire.OutPoint{Hash: chainhash.Hash{0x02}, Index: 1}
	k := canonicalOutPoint(&op.Hash, op.Index)
	putRawBucketValue(t, db, bucketLockedOutputs, k,
		serializeLockedOutput(id, expiry))

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(namespaceKey)

		gotID, gotExpiry, locked := isLockedOutput(ns, op, time.Now())
		if !locked {
			t.Fatal("well-formed lock not reported as locked")
		}
		if gotID != id {
			t.Errorf("lock ID mismatch: %x != %x", gotID, id)
		}
		if !gotExpiry.Equal(expiry) {
			t.Errorf("expiry mismatch: %v != %v", gotExpiry, expiry)
		}

		seen := 0
		err := forEachLockedOutput(ns, func(wire.OutPoint, LockID,
			time.Time) {
			seen++
		})
		if err != nil {
			return err
		}
		if seen != 1 {
			t.Errorf("forEachLockedOutput visited %d, want 1", seen)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestForEachLockedOutputTruncatedValue ensures a truncated locked-output
// record is skipped by the listing path instead of panicking it.
func TestForEachLockedOutputTruncatedValue(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	op := wire.OutPoint{Hash: chainhash.Hash{0x03}, Index: 0}
	k := canonicalOutPoint(&op.Hash, op.Index)
	putRawBucketValue(t, db, bucketLockedOutputs, k, []byte{0x01, 0x02, 0x03})

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(namespaceKey)
		seen := 0
		err := forEachLockedOutput(ns, func(wire.OutPoint, LockID,
			time.Time) {
			seen++
		})
		if err != nil {
			return err
		}
		if seen != 0 {
			t.Errorf("forEachLockedOutput visited %d, want 0", seen)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestFetchUnminedInputSpendTxHashesTruncated ensures a stored spender
// list whose length is not a multiple of 32 bytes does not panic; only the
// complete hashes are returned.
func TestFetchUnminedInputSpendTxHashesTruncated(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	k := []byte("outpoint-key")
	hashA := chainhash.Hash{0x0a}
	hashB := chainhash.Hash{0x0b}

	tests := []struct {
		name string
		v    []byte
		want int
	}{
		{"partial only", make([]byte, 10), 0},
		{"one hash plus trailing bytes", append(
			hashA[:], 0x01, 0x02, 0x03, 0x04, 0x05,
		), 1},
		{"two full hashes", append(hashA[:], hashB[:]...), 2},
	}

	for _, test := range tests {
		putRawBucketValue(t, db, bucketUnminedInputs, k, test.v)

		err := walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(namespaceKey)
			hashes := fetchUnminedInputSpendTxHashes(ns, k)
			if len(hashes) != test.want {
				t.Errorf("%s: got %d hashes, want %d",
					test.name, len(hashes), test.want)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}
