// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package indexers

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// bucketBase exists so errorBucket can embed the interface and still define its own Bucket method.
type bucketBase struct {
	database.Bucket
}

// errorBucket wraps a database bucket and injects errors into the operations dropIndex uses while cataloging and
// deleting index buckets.
type errorBucket struct {
	bucketBase
	forEachBucketErr error
	deleteBucketErr  error
}

func (b *errorBucket) Bucket(key []byte) database.Bucket {
	bucket := b.bucketBase.Bucket.Bucket(key)
	if bucket == nil {
		return nil
	}

	return &errorBucket{
		bucketBase:       bucketBase{Bucket: bucket},
		forEachBucketErr: b.forEachBucketErr,
		deleteBucketErr:  b.deleteBucketErr,
	}
}

func (b *errorBucket) ForEachBucket(fn func(k []byte) error) error {
	if b.forEachBucketErr != nil {
		return b.forEachBucketErr
	}

	return b.bucketBase.Bucket.ForEachBucket(fn)
}

func (b *errorBucket) DeleteBucket(key []byte) error {
	if b.deleteBucketErr != nil {
		return b.deleteBucketErr
	}

	return b.bucketBase.Bucket.DeleteBucket(key)
}

type errorTx struct {
	database.Tx
	forEachBucketErr error
	deleteBucketErr  error
}

func (tx *errorTx) Metadata() database.Bucket {
	return &errorBucket{
		bucketBase:       bucketBase{Bucket: tx.Tx.Metadata()},
		forEachBucketErr: tx.forEachBucketErr,
		deleteBucketErr:  tx.deleteBucketErr,
	}
}

// errorDB wraps every managed transaction with the configured error injector.
type errorDB struct {
	database.DB
	forEachBucketErr error
	deleteBucketErr  error
}

func (db *errorDB) wrap(tx database.Tx) database.Tx {
	return &errorTx{
		Tx:               tx,
		forEachBucketErr: db.forEachBucketErr,
		deleteBucketErr:  db.deleteBucketErr,
	}
}

func (db *errorDB) View(fn func(database.Tx) error) error {
	return db.DB.View(func(tx database.Tx) error { return fn(db.wrap(tx)) })
}

func (db *errorDB) Update(fn func(database.Tx) error) error {
	return db.DB.Update(func(tx database.Tx) error { return fn(db.wrap(tx)) })
}

// createDropIndexTestDB creates a database holding the tip entry and bucket dropIndex expects for idxKey.
func createDropIndexTestDB(t *testing.T, idxKey []byte) database.DB {
	t.Helper()

	db, err := database.Create("ffldb", filepath.Join(t.TempDir(), "blocks_ffldb"), wire.MainNet)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })

	err = db.Update(func(tx database.Tx) error {
		meta := tx.Metadata()
		indexesBucket, err := meta.CreateBucket(indexTipsBucketName)
		if err != nil {
			return err
		}
		if err := indexesBucket.Put(idxKey, []byte{0x01}); err != nil {
			return err
		}

		_, err = meta.CreateBucket(idxKey)
		return err
	})
	require.NoError(t, err)

	return db
}

// TestDropIndexPropagatesErrors ensures failures while cataloging or deleting index buckets reach the caller
// instead of being reported as a clean drop.
func TestDropIndexPropagatesErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		build func(database.DB, error) *errorDB
	}{
		{
			name: "catalog error",
			build: func(db database.DB, err error) *errorDB {
				return &errorDB{DB: db, forEachBucketErr: err}
			},
		},
		{
			name: "delete bucket error",
			build: func(db database.DB, err error) *errorDB {
				return &errorDB{DB: db, deleteBucketErr: err}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			idxKey := []byte("testidx")
			wantErr := errors.New(tt.name)
			db := tt.build(createDropIndexTestDB(t, idxKey), wantErr)

			err := dropIndex(db, idxKey, "test index", nil)
			require.ErrorIs(t, err, wantErr)
		})
	}
}
