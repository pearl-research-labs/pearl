// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package waddrmgr

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// TestFetchManagerVersionTruncated stores version values shorter than the
// 4 bytes the format requires and ensures fetchManagerVersion reports the
// corruption as an error instead of panicking. Every neighbouring scalar
// fetch in this file (watching-only flag, last account, birthday, synced-to)
// already length-checks its value; the version fetch only nil-checked.
func TestFetchManagerVersionTruncated(t *testing.T) {
	t.Parallel()

	teardown, db, _ := setupManager(t)
	defer teardown()

	for _, size := range []int{1, 2, 3} {
		size := size
		err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
			ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
			mainBucket := ns.NestedReadWriteBucket(mainBucketName)
			return mainBucket.Put(mgrVersionName, make([]byte, size))
		})
		if err != nil {
			t.Fatalf("unable to store %d-byte version value: %v",
				size, err)
		}

		err = walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(waddrmgrNamespaceKey)
			if _, err := fetchManagerVersion(ns); err == nil {
				return fmt.Errorf("expected error for %d-byte "+
					"version value, got nil", size)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	// A well-formed version must still round-trip.
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		return putManagerVersion(ns, 7)
	})
	if err != nil {
		t.Fatalf("unable to store well-formed version: %v", err)
	}
	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		version, err := fetchManagerVersion(ns)
		if err != nil {
			return fmt.Errorf("unexpected error for well-formed "+
				"version: %v", err)
		}
		if version != 7 {
			return fmt.Errorf("version mismatch: got %d, want 7",
				version)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestFetchBirthdayBlockVerificationTruncated stores a 1-byte verification
// value — too short for the stored uint16 — and ensures the fetch reports
// "not verified" instead of panicking. A missing value already means not
// verified; a malformed one must fail safe the same way, never crash wallet
// startup and never claim the birthday block was verified.
func TestFetchBirthdayBlockVerificationTruncated(t *testing.T) {
	t.Parallel()

	teardown, db, _ := setupManager(t)
	defer teardown()

	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		syncBucket := ns.NestedReadWriteBucket(syncBucketName)
		return syncBucket.Put(birthdayBlockVerifiedName, []byte{0x01})
	})
	if err != nil {
		t.Fatalf("unable to store truncated verification value: %v", err)
	}

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		if fetchBirthdayBlockVerification(ns) {
			return fmt.Errorf("expected truncated verification " +
				"value to report not verified")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Well-formed values must still round-trip both ways.
	for _, want := range []bool{true, false} {
		want := want
		err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
			ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
			return putBirthdayBlockVerification(ns, want)
		})
		if err != nil {
			t.Fatalf("unable to store verification=%v: %v", want, err)
		}
		err = walletdb.View(db, func(tx walletdb.ReadTx) error {
			ns := tx.ReadBucket(waddrmgrNamespaceKey)
			if got := fetchBirthdayBlockVerification(ns); got != want {
				return fmt.Errorf("verification mismatch: got "+
					"%v, want %v", got, want)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}
