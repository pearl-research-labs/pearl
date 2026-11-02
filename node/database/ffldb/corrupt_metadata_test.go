// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package ffldb

import (
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/syndtr/goleveldb/leveldb"
)

// TestDeserializeWriteRowShort ensures a stored write-cursor row that is
// shorter than the serialized 12-byte form is reported as corruption
// instead of panicking the deserializer (which runs on every database
// open via reconcileDB).
func TestDeserializeWriteRowShort(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 11} {
		// Copy into an exact-length slice: values returned by the
		// metadata store have cap == len, unlike a sub-slice of the
		// serializer's array, whose spare capacity would mask the
		// out-of-range reads.
		row := make([]byte, n)
		copy(row, serializeWriteRow(3, 42))
		_, _, err := deserializeWriteRow(row)
		if err == nil {
			t.Errorf("len %d: expected corruption error, got nil", n)
			continue
		}
		dbErr, ok := err.(database.Error)
		if !ok || dbErr.ErrorCode != database.ErrCorruption {
			t.Errorf("len %d: expected ErrCorruption, got %v", n, err)
		}
	}

	// A well-formed row still round-trips, and a full-length row with a
	// bad checksum still reports corruption.
	fileNum, offset, err := deserializeWriteRow(serializeWriteRow(3, 42))
	if err != nil || fileNum != 3 || offset != 42 {
		t.Errorf("valid row: got (%d, %d, %v)", fileNum, offset, err)
	}
	bad := serializeWriteRow(3, 42)
	bad[0] ^= 0xff
	if _, _, err := deserializeWriteRow(bad); err == nil {
		t.Error("bad checksum: expected corruption error, got nil")
	}
}

// TestCreateBucketCorruptBucketID corrupts the stored current-bucket-ID
// value in the metadata database and ensures creating a bucket reports
// corruption instead of panicking in nextBucketID.
func TestCreateBucketCorruptBucketID(t *testing.T) {
	dbPath := t.TempDir()
	idb, err := openDB(dbPath, blockDataNet, true, defaultFlushSecs)
	if err != nil {
		t.Fatalf("openDB create: %v", err)
	}
	if err := idb.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Truncate the stored bucket ID to two bytes, simulating a torn or
	// corrupt metadata write.
	ldb, err := leveldb.OpenFile(filepath.Join(dbPath, metadataDbName), nil)
	if err != nil {
		t.Fatalf("open metadata: %v", err)
	}
	if err := ldb.Put(curBucketIDKeyName, []byte{0x00, 0x02}, nil); err != nil {
		t.Fatalf("corrupt bucket id: %v", err)
	}
	if err := ldb.Close(); err != nil {
		t.Fatalf("close metadata: %v", err)
	}

	idb, err = openDB(dbPath, blockDataNet, false, defaultFlushSecs)
	if err != nil {
		t.Fatalf("openDB reopen: %v", err)
	}
	defer idb.Close()

	err = idb.Update(func(tx database.Tx) error {
		_, err := tx.Metadata().CreateBucket([]byte("corrupt-id-test"))
		return err
	})
	if err == nil {
		t.Fatal("expected corruption error, got nil")
	}
	dbErr, ok := err.(database.Error)
	if !ok || dbErr.ErrorCode != database.ErrCorruption {
		t.Fatalf("expected ErrCorruption, got %v", err)
	}
}
