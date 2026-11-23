// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wtxmgr

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// putRawStoredValue writes v under k in an existing nested bucket of the
// txstore namespace.
func putRawStoredValue(t *testing.T, db walletdb.DB, bucket, k, v []byte) {
	t.Helper()

	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(namespaceKey)
		return ns.NestedReadWriteBucket(bucket).Put(k, v)
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestUnspendRawCreditTruncatedValue ensures a credit record shorter than
// the serialized amount+flags (9 bytes) returns an error from
// unspendRawCredit instead of panicking, while a missing credit still
// returns (0, nil) and a well-formed spent credit round-trips.
func TestUnspendRawCreditTruncatedValue(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	k := []byte("credit-key")

	for _, v := range [][]byte{
		{0x01},
		{0x01, 0x02, 0x03, 0x04, 0x05},
		make([]byte, 8), // full amount, missing flags byte
	} {
		putRawStoredValue(t, db, bucketCredits, k, v)

		err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
			ns := tx.ReadWriteBucket(namespaceKey)
			_, err := unspendRawCredit(ns, k)
			return err
		})
		if err == nil {
			t.Errorf("value len %d: expected error, got nil", len(v))
		}
	}

	// Missing credit: (0, nil), per the documented contract.
	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(namespaceKey)
		amt, err := unspendRawCredit(ns, []byte("no-such-key"))
		if err != nil {
			return err
		}
		if amt != 0 {
			t.Errorf("missing credit: amount = %v, want 0", amt)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Well-formed spent credit (81-byte value): amount is returned and
	// the stored record is rewritten unspent with the spent bit clear.
	spent := make([]byte, 81)
	byteOrder.PutUint64(spent, 12345)
	spent[8] = 1 << 0
	putRawStoredValue(t, db, bucketCredits, k, spent)

	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(namespaceKey)
		amt, err := unspendRawCredit(ns, k)
		if err != nil {
			return err
		}
		if amt != btcutil.Amount(12345) {
			t.Errorf("amount = %v, want 12345", amt)
		}
		got := ns.NestedReadBucket(bucketCredits).Get(k)
		if len(got) != 9 {
			t.Fatalf("stored value len = %d, want 9", len(got))
		}
		if got[8]&(1<<0) != 0 {
			t.Error("spent bit still set after unspend")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestDeserializeLabelMalformed ensures stored label values that are
// shorter than the 2-byte length prefix, or whose declared length does
// not match the stored bytes, return errors instead of panicking or
// silently returning a partial label.
func TestDeserializeLabelMalformed(t *testing.T) {
	for _, v := range [][]byte{
		{},
		{0x00},
		{0x00, 0x05, 'a', 'b'}, // declares 5, stores 2
		{0x00, 0x01, 'a', 'b'}, // declares 1, stores 2
	} {
		if _, err := DeserializeLabel(v); err == nil {
			t.Errorf("value %x: expected error, got nil", v)
		}
	}

	label, err := DeserializeLabel([]byte{0x00, 0x03, 'a', 'b', 'c'})
	if err != nil {
		t.Fatal(err)
	}
	if label != "abc" {
		t.Errorf("label = %q, want %q", label, "abc")
	}

	if _, err := DeserializeLabel([]byte{0x00, 0x00}); err != ErrEmptyLabel {
		t.Errorf("empty label err = %v, want ErrEmptyLabel", err)
	}
}

// TestExistsRawUnspentShortValue pins the existing contract the audit
// verified: an unspent record whose stored value is shorter than the
// serialized block reference (36 bytes) is treated as absent, and a
// well-formed value yields a full 72-byte credit key. This guards the
// length check in existsRawUnspent that extractRawCreditTxRecordKey's
// callers rely on.
func TestExistsRawUnspentShortValue(t *testing.T) {
	_, db, err := testStore(t)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	prevOut := wire.OutPoint{Hash: chainhash.Hash{0x07}, Index: 1}
	opKey := canonicalOutPoint(&prevOut.Hash, prevOut.Index)

	for _, v := range [][]byte{
		{0x01},
		make([]byte, 35), // one byte short of a block reference
	} {
		putRawStoredValue(t, db, bucketUnspent, opKey, v)

		err := walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(namespaceKey)
			if credKey := existsRawUnspent(ns, opKey); credKey != nil {
				t.Errorf("value len %d: credKey = %x, want nil",
					len(v), credKey)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	putRawStoredValue(t, db, bucketUnspent, opKey, make([]byte, 36))
	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(namespaceKey)
		credKey := existsRawUnspent(ns, opKey)
		if len(credKey) != 72 {
			t.Errorf("credKey len = %d, want 72", len(credKey))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
