// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package waddrmgr

import (
	"crypto/sha256"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// TestFetchAddrAccountTruncated stores address-account index values shorter
// than the 4 bytes the format requires and ensures fetchAddrAccount reports
// the corruption as an error instead of panicking. The neighbouring index
// fetches in this file length-check their values (fetchAccountByName,
// fetchLastAccount); fetchAddrAccount only nil-checked.
func TestFetchAddrAccountTruncated(t *testing.T) {
	t.Parallel()

	teardown, db, _ := setupManager(t)
	defer teardown()

	scope := KeyScopeBIP0086
	addressID := []byte("pearl-test-address-id")
	addrHash := sha256.Sum256(addressID)

	for _, size := range []int{1, 2, 3} {
		size := size
		err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
			ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
			scopedBucket, err := fetchWriteScopeBucket(ns, &scope)
			if err != nil {
				return err
			}
			bucket := scopedBucket.NestedReadWriteBucket(
				addrAcctIdxBucketName,
			)
			return bucket.Put(addrHash[:], make([]byte, size))
		})
		if err != nil {
			t.Fatalf("unable to store %d-byte index value: %v",
				size, err)
		}

		err = walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(waddrmgrNamespaceKey)
			if _, err := fetchAddrAccount(
				ns, &scope, addressID,
			); err == nil {
				return fmt.Errorf("expected error for %d-byte "+
					"index value, got nil", size)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	// A well-formed index entry must still round-trip.
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		return putAddrAccountIndex(ns, &scope, 7, addrHash[:])
	})
	if err != nil {
		t.Fatalf("unable to store well-formed index entry: %v", err)
	}
	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		account, err := fetchAddrAccount(ns, &scope, addressID)
		if err != nil {
			return fmt.Errorf("unexpected error for well-formed "+
				"index entry: %v", err)
		}
		if account != 7 {
			return fmt.Errorf("account mismatch: got %d, want 7",
				account)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// A missing entry must still report ErrAddressNotFound, not the
	// malformed error.
	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		_, err := fetchAddrAccount(ns, &scope, []byte("unknown-address"))
		if err == nil {
			return fmt.Errorf("expected error for missing entry")
		}
		if !IsError(err, ErrAddressNotFound) {
			return fmt.Errorf("expected ErrAddressNotFound, got %v",
				err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestForEachAccountShortKey stores an account-bucket entry whose key is
// shorter than the 4 bytes an account number occupies and ensures
// forEachAccount reports the corruption as an error instead of panicking.
// The key-iterating neighbour forEachKeyScope guards its key length before
// decoding; forEachAccount decoded unconditionally.
func TestForEachAccountShortKey(t *testing.T) {
	t.Parallel()

	teardown, db, _ := setupManager(t)
	defer teardown()

	scope := KeyScopeBIP0086

	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		scopedBucket, err := fetchWriteScopeBucket(ns, &scope)
		if err != nil {
			return err
		}
		bucket := scopedBucket.NestedReadWriteBucket(acctBucketName)
		return bucket.Put([]byte{0xff, 0xff}, []byte{0x01})
	})
	if err != nil {
		t.Fatalf("unable to store short account key: %v", err)
	}

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		return forEachAccount(ns, &scope, func(account uint32) error {
			return nil
		})
	})
	if err == nil {
		t.Fatal("expected error for short account key, got nil")
	}
	if !IsError(err, ErrDatabase) {
		t.Fatalf("expected ErrDatabase ManagerError, got %v", err)
	}
}
