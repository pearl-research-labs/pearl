// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package waddrmgr

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// TestDeserializeAccountRowTruncated ensures that an account row whose
// declared raw-data length exceeds the bytes actually stored is rejected
// with an error instead of panicking.
func TestDeserializeAccountRowTruncated(t *testing.T) {
	t.Parallel()

	// acctType(1) + rdlen(4) + only 3 bytes of raw data, while rdlen
	// claims 10.
	serialized := make([]byte, 5+3)
	serialized[0] = byte(accountDefault)
	binary.LittleEndian.PutUint32(serialized[1:5], 10)

	if _, err := deserializeAccountRow(uint32ToBytes(0), serialized); err == nil {
		t.Fatal("expected error for truncated account row, got nil")
	}

	// A well-formed row must still round-trip.
	want := &dbAccountRow{acctType: accountDefault, rawData: []byte{1, 2, 3}}
	row, err := deserializeAccountRow(
		uint32ToBytes(0), serializeAccountRow(want),
	)
	if err != nil {
		t.Fatalf("unexpected error for well-formed account row: %v", err)
	}
	if row.acctType != want.acctType || !bytes.Equal(row.rawData, want.rawData) {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", row, want)
	}
}

// TestDeserializeDefaultAccountRowTruncated ensures that a default account
// row whose inner length prefixes (pubkey, privkey, name) exceed the bytes
// actually stored is rejected with an error instead of panicking.
func TestDeserializeDefaultAccountRowTruncated(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		rawData []byte
	}{
		{
			// pubLen claims 100 bytes; only 16 bytes follow.
			name: "pubkey length lies",
			rawData: func() []byte {
				b := make([]byte, 20)
				binary.LittleEndian.PutUint32(b[0:4], 100)
				return b
			}(),
		},
		{
			// Empty pubkey, then privLen claims 100 bytes with
			// nothing following.
			name: "privkey length lies",
			rawData: func() []byte {
				b := make([]byte, 20)
				binary.LittleEndian.PutUint32(b[4:8], 100)
				return b
			}(),
		},
		{
			// Empty keys and zero indexes, then nameLen claims 50
			// bytes with only 2 present.
			name: "name length lies",
			rawData: func() []byte {
				b := make([]byte, 20+2)
				binary.LittleEndian.PutUint32(b[16:20], 50)
				return b
			}(),
		},
		{
			// Exactly the 20-byte minimum, but pubLen consumes all
			// of it so no room remains for the privkey length.
			name: "pubkey consumes the rest",
			rawData: func() []byte {
				b := make([]byte, 20)
				binary.LittleEndian.PutUint32(b[0:4], 16)
				return b
			}(),
		},
	}

	for _, test := range tests {
		row := &dbAccountRow{acctType: accountDefault, rawData: test.rawData}
		if _, err := deserializeDefaultAccountRow(
			uint32ToBytes(0), row,
		); err == nil {
			t.Errorf("%s: expected error, got nil", test.name)
		}
	}

	// A well-formed row must still round-trip.
	raw := serializeDefaultAccountRow(
		[]byte{1, 2, 3}, []byte{4, 5}, 7, 9, "savings",
	)
	row, err := deserializeDefaultAccountRow(
		uint32ToBytes(0), &dbAccountRow{rawData: raw},
	)
	if err != nil {
		t.Fatalf("unexpected error for well-formed default account: %v", err)
	}
	if !bytes.Equal(row.pubKeyEncrypted, []byte{1, 2, 3}) ||
		!bytes.Equal(row.privKeyEncrypted, []byte{4, 5}) ||
		row.nextExternalIndex != 7 || row.nextInternalIndex != 9 ||
		row.name != "savings" {
		t.Fatalf("round-trip mismatch: %+v", row)
	}
}

// TestDeserializeAddressRowTruncated ensures that an address row whose
// declared raw-data length exceeds the bytes actually stored is rejected
// with an error instead of panicking.
func TestDeserializeAddressRowTruncated(t *testing.T) {
	t.Parallel()

	// Header(18) + only 4 bytes of raw data, while rdlen claims 64.
	serialized := make([]byte, 18+4)
	serialized[0] = byte(adtChain)
	binary.LittleEndian.PutUint32(serialized[14:18], 64)

	if _, err := deserializeAddressRow(serialized); err == nil {
		t.Fatal("expected error for truncated address row, got nil")
	}

	// A well-formed row must still round-trip.
	want := &dbAddressRow{
		addrType: adtChain, account: 3, addTime: 42,
		syncStatus: ssFull, rawData: []byte{9, 8, 7},
	}
	row, err := deserializeAddressRow(serializeAddressRow(want))
	if err != nil {
		t.Fatalf("unexpected error for well-formed address row: %v", err)
	}
	if row.addrType != want.addrType || row.account != want.account ||
		row.addTime != want.addTime ||
		!bytes.Equal(row.rawData, want.rawData) {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", row, want)
	}
}

// TestDeserializeImportedAddressTruncated ensures that an imported address
// row whose inner length prefixes exceed the bytes actually stored is
// rejected with an error instead of panicking.
func TestDeserializeImportedAddressTruncated(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		rawData []byte
	}{
		{
			// pubLen claims 100 bytes; only 4 bytes follow.
			name: "pubkey length lies",
			rawData: func() []byte {
				b := make([]byte, 8)
				binary.LittleEndian.PutUint32(b[0:4], 100)
				return b
			}(),
		},
		{
			// Empty pubkey, then privLen claims 100 bytes with
			// nothing following.
			name: "privkey length lies",
			rawData: func() []byte {
				b := make([]byte, 8)
				binary.LittleEndian.PutUint32(b[4:8], 100)
				return b
			}(),
		},
	}

	for _, test := range tests {
		row := &dbAddressRow{addrType: adtImport, rawData: test.rawData}
		if _, err := deserializeImportedAddress(row); err == nil {
			t.Errorf("%s: expected error, got nil", test.name)
		}
	}

	// A well-formed row must still round-trip.
	raw := serializeImportedAddress([]byte{1, 2}, []byte{3, 4, 5})
	row, err := deserializeImportedAddress(&dbAddressRow{rawData: raw})
	if err != nil {
		t.Fatalf("unexpected error for well-formed imported address: %v", err)
	}
	if !bytes.Equal(row.encryptedPubKey, []byte{1, 2}) ||
		!bytes.Equal(row.encryptedPrivKey, []byte{3, 4, 5}) {
		t.Fatalf("round-trip mismatch: %+v", row)
	}
}

// TestDeserializeWitnessScriptAddressTruncated ensures that a witness
// script address row whose inner length prefixes exceed the bytes actually
// stored is rejected with an error instead of panicking.
func TestDeserializeWitnessScriptAddressTruncated(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		rawData []byte
	}{
		{
			// hashLen claims 100 bytes; only the 4-byte scriptLen
			// field follows.
			name: "hash length lies",
			rawData: func() []byte {
				b := make([]byte, 10)
				binary.LittleEndian.PutUint32(b[2:6], 100)
				return b
			}(),
		},
		{
			// Empty hash, then scriptLen claims 100 bytes with
			// nothing following.
			name: "script length lies",
			rawData: func() []byte {
				b := make([]byte, 10)
				binary.LittleEndian.PutUint32(b[6:10], 100)
				return b
			}(),
		},
	}

	for _, test := range tests {
		row := &dbAddressRow{
			addrType: adtTaprootScript, rawData: test.rawData,
		}
		if _, err := deserializeWitnessScriptAddress(row); err == nil {
			t.Errorf("%s: expected error, got nil", test.name)
		}
	}

	// A well-formed row must still round-trip.
	raw := serializeWitnessScriptAddress(1, true, []byte{1, 2}, []byte{3})
	row, err := deserializeWitnessScriptAddress(&dbAddressRow{rawData: raw})
	if err != nil {
		t.Fatalf("unexpected error for well-formed witness script "+
			"address: %v", err)
	}
	if row.witnessVersion != 1 || !row.isSecretScript ||
		!bytes.Equal(row.encryptedHash, []byte{1, 2}) ||
		!bytes.Equal(row.encryptedScript, []byte{3}) {
		t.Fatalf("round-trip mismatch: %+v", row)
	}
}

// TestFetchAccountNameIndexCorruptValues stores corrupt (truncated) values
// in the account name/id index buckets and ensures the fetch helpers return
// errors instead of panicking.
func TestFetchAccountNameIndexCorruptValues(t *testing.T) {
	t.Parallel()

	teardown, db, _ := setupManager(t)
	defer teardown()

	scope := KeyScopeBIP0086

	// Store a 2-byte value in both index buckets: too short to hold the
	// length prefix (name index) or the account number (id index).
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		scopedBucket, err := fetchWriteScopeBucket(ns, &scope)
		if err != nil {
			return err
		}

		idIdx := scopedBucket.NestedReadWriteBucket(acctIDIdxBucketName)
		if err := idIdx.Put(uint32ToBytes(77), []byte{0x01, 0x02}); err != nil {
			return err
		}

		nameIdx := scopedBucket.NestedReadWriteBucket(acctNameIdxBucketName)
		return nameIdx.Put(stringToBytes("corrupt"), []byte{0x01, 0x02})
	})
	if err != nil {
		t.Fatalf("unable to store corrupt index values: %v", err)
	}

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)

		if _, err := fetchAccountName(ns, &scope, 77); err == nil {
			return fmt.Errorf("fetchAccountName: expected error " +
				"for corrupt index value, got nil")
		}
		if _, err := fetchAccountByName(ns, &scope, "corrupt"); err == nil {
			return fmt.Errorf("fetchAccountByName: expected error " +
				"for corrupt index value, got nil")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
