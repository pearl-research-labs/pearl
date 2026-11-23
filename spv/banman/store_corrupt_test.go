package banman_test

import (
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/spv/banman"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	_ "github.com/pearl-research-labs/pearl/wallet/walletdb/bdb"
	"github.com/stretchr/testify/require"
)

// createCorruptibleBanStore creates a test Store and also returns the raw
// database handle so a test can damage the stored values the way disk
// corruption, a partial restore, or a foreign writer could.
func createCorruptibleBanStore(t *testing.T) (banman.Store, walletdb.DB,
	func()) {

	t.Helper()

	dbDir, err := ioutil.TempDir("", "")
	require.NoError(t, err)

	dbPath := filepath.Join(dbDir, "test.db")
	db, err := walletdb.Create(
		"bdb", dbPath, true, time.Second*10, false,
	)
	if err != nil {
		os.RemoveAll(dbDir)
		t.Fatalf("unable to create db: %v", err)
	}

	cleanUp := func() {
		db.Close()
		os.RemoveAll(dbDir)
	}

	banStore, err := banman.NewStore(db)
	if err != nil {
		cleanUp()
		t.Fatalf("unable to create ban store: %v", err)
	}

	return banStore, db, cleanUp
}

// tamperBanStore rewrites the raw ban-store buckets for a banned IP network:
// mutate receives the ban-index and reason-index buckets and the encoded key.
func tamperBanStore(t *testing.T, db walletdb.DB, ipNet *net.IPNet,
	mutate func(banIndex, reasonIndex walletdb.ReadWriteBucket,
		key []byte)) {

	t.Helper()

	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		banStore := tx.ReadWriteBucket([]byte("ban-store"))
		require.NotNil(t, banStore)
		banIndex := banStore.NestedReadWriteBucket([]byte("ban-index"))
		require.NotNil(t, banIndex)
		reasonIndex := banStore.NestedReadWriteBucket(
			[]byte("reason-index"),
		)
		require.NotNil(t, reasonIndex)

		// The store holds exactly one ban entry in these tests; take
		// its key from the bucket itself rather than re-implementing
		// the (unexported) IPNet key encoding.
		var key []byte
		err := banIndex.ForEach(func(k, _ []byte) error {
			key = append([]byte(nil), k...)
			return nil
		})
		require.NoError(t, err)
		require.NotEmpty(t, key)

		mutate(banIndex, reasonIndex, key)
		return nil
	})
	require.NoError(t, err)
}

// TestBanStoreCorruptValues ensures Status never panics on damaged stored
// values: a truncated expiration in the ban index and a missing entry in the
// reason index. Status is consulted on peer paths (ChainService.IsBanned),
// so a panic here takes down the SPV service for a store only it wrote.
func TestBanStoreCorruptValues(t *testing.T) {
	t.Parallel()

	ipNet, err := banman.ParseIPNet("127.0.0.1:8333", nil)
	require.NoError(t, err)

	// A ban-index value shorter than the 8-byte expiration must not
	// panic. The entry cannot be trusted, so Status reports the network
	// as not banned and drops the corrupt entry: a second lookup must
	// behave the same way.
	t.Run("truncated expiration", func(t *testing.T) {
		banStore, db, cleanUp := createCorruptibleBanStore(t)
		defer cleanUp()

		require.NoError(t, banStore.BanIPNet(
			ipNet, banman.NoCompactFilters, time.Hour,
		))

		tamperBanStore(t, db, ipNet,
			func(banIndex, _ walletdb.ReadWriteBucket, key []byte) {
				require.NoError(t, banIndex.Put(key, []byte{1, 2, 3}))
			})

		status, err := banStore.Status(ipNet)
		require.NoError(t, err)
		require.False(t, status.Banned)

		status, err = banStore.Status(ipNet)
		require.NoError(t, err)
		require.False(t, status.Banned)
	})

	// A missing reason entry must not panic either. The expiration is
	// still readable, so the ban itself still applies; only the reason
	// degrades to its zero value ("unknown reason").
	t.Run("missing reason", func(t *testing.T) {
		banStore, db, cleanUp := createCorruptibleBanStore(t)
		defer cleanUp()

		require.NoError(t, banStore.BanIPNet(
			ipNet, banman.NoCompactFilters, time.Hour,
		))

		tamperBanStore(t, db, ipNet,
			func(_, reasonIndex walletdb.ReadWriteBucket, key []byte) {
				require.NoError(t, reasonIndex.Delete(key))
			})

		status, err := banStore.Status(ipNet)
		require.NoError(t, err)
		require.True(t, status.Banned)
		require.Equal(t, banman.Reason(0), status.Reason)
		require.True(t, time.Now().Before(status.Expiration))
	})
}
