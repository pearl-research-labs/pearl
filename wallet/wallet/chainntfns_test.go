package wallet

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	_ "github.com/pearl-research-labs/pearl/wallet/walletdb/bdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/require"
)

const (
	// defaultBlockInterval is the default time interval between any two
	// blocks in a mocked chain.
	defaultBlockInterval = 10 * time.Minute
)

var (
	// chainParams are the chain parameters used throughout the wallet
	// tests.
	chainParams = chaincfg.MainNetParams
)

// mockChainConn is a mock in-memory implementation of the chainConn interface
// that will be used for the birthday block sanity check tests. The struct is
// capable of being backed by a chain in order to reproduce real-world
// scenarios.
type mockChainConn struct {
	chainTip    uint32
	blockHashes map[uint32]chainhash.Hash
	blocks      map[chainhash.Hash]*wire.MsgBlock
}

var _ chainConn = (*mockChainConn)(nil)

// createMockChainConn creates a new mock chain connection backed by a chain
// with N blocks. Each block has a timestamp that is exactly blockInterval after
// the previous block's timestamp.
func createMockChainConn(genesis *wire.MsgBlock, n uint32,
	blockInterval time.Duration) *mockChainConn {

	c := &mockChainConn{
		chainTip:    n,
		blockHashes: make(map[uint32]chainhash.Hash),
		blocks:      make(map[chainhash.Hash]*wire.MsgBlock),
	}

	genesisHash := genesis.BlockHash()
	c.blockHashes[0] = genesisHash
	c.blocks[genesisHash] = genesis

	for i := uint32(1); i <= n; i++ {
		prevTimestamp := c.blocks[c.blockHashes[i-1]].BlockHeader().Timestamp
		block := &wire.MsgBlock{
			MsgHeader: wire.MsgHeader{BlockHeader: wire.BlockHeader{
				Timestamp: prevTimestamp.Add(blockInterval),
			}},
		}

		blockHash := block.BlockHash()
		c.blockHashes[i] = blockHash
		c.blocks[blockHash] = block
	}

	return c
}

// GetBestBlock returns the hash and height of the best block known to the
// backend.
func (c *mockChainConn) GetBestBlock() (*chainhash.Hash, int32, error) {
	bestHash, ok := c.blockHashes[c.chainTip]
	if !ok {
		return nil, 0, fmt.Errorf("block with height %d not found",
			c.chainTip)
	}

	return &bestHash, int32(c.chainTip), nil
}

// GetBlockHash returns the hash of the block with the given height.
func (c *mockChainConn) GetBlockHash(height int64) (*chainhash.Hash, error) {
	hash, ok := c.blockHashes[uint32(height)]
	if !ok {
		return nil, fmt.Errorf("block with height %d not found", height)
	}

	return &hash, nil
}

// GetBlockHeader returns the header for the block with the given hash.
func (c *mockChainConn) GetBlockHeader(hash *chainhash.Hash) (*wire.BlockHeader, error) {
	block, ok := c.blocks[*hash]
	if !ok {
		return nil, fmt.Errorf("header for block %v not found", hash)
	}

	return block.BlockHeader(), nil
}

func addMaturingOutputLock(t *testing.T, w *Wallet, spendHeight int32) {
	t.Helper()

	creditTx := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{{Value: 1000}},
	}

	creditRec, err := wtxmgr.NewTxRecordFromMsgTx(creditTx, time.Now())
	require.NoError(t, err)

	creditBlock := &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{
			Hash:   chainhash.Hash{1},
			Height: spendHeight - 1,
		},
		Time: time.Now(),
	}
	creditOutpoint := wire.OutPoint{
		Hash:  creditRec.Hash,
		Index: 0,
	}

	spendTx := &wire.MsgTx{
		TxIn: []*wire.TxIn{{
			PreviousOutPoint: creditOutpoint,
		}},
		TxOut: []*wire.TxOut{{Value: 900}},
	}

	spendRec, err := wtxmgr.NewTxRecordFromMsgTx(spendTx, time.Now())
	require.NoError(t, err)

	spendBlock := &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{
			Hash:   chainhash.Hash{2},
			Height: spendHeight,
		},
		Time: time.Now(),
	}

	err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(wtxmgrNamespaceKey)

		err := w.TxStore.InsertTx(ns, creditRec, creditBlock)
		if err != nil {
			return err
		}

		err = w.TxStore.AddCredit(ns, creditRec, creditBlock, 0, false)
		if err != nil {
			return err
		}

		_, err = w.TxStore.LockOutput(ns, wtxmgr.LockID{1}, creditOutpoint, time.Hour, wtxmgr.WithReleaseAfterSpend(3))
		if err != nil {
			return err
		}

		return w.TxStore.InsertTx(ns, spendRec, spendBlock)
	})
	require.NoError(t, err)
}

// TestCatchUpHashesMaturesOutputLocks verifies that offline rescan progress releases retained output leases at the
// validated catch-up tip.
func TestCatchUpHashesMaturesOutputLocks(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	const (
		spendHeight   = 2
		catchUpHeight = 4
	)
	addMaturingOutputLock(t, w, spendHeight)

	chainConn := createMockChainConn(w.chainParams.GenesisBlock, catchUpHeight, defaultBlockInterval)
	err := w.catchUpHashes(chainConn, catchUpHeight)
	require.NoError(t, err)

	var locks []*wtxmgr.LockedOutput

	err = walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)

		var err error

		locks, err = w.TxStore.ListLockedOutputs(ns)

		return err
	})
	require.NoError(t, err)
	require.Empty(t, locks)
	require.Equal(t, int32(catchUpHeight), w.Manager.SyncedTo().Height)
}

// mockBirthdayStore is a mock in-memory implementation of the birthdayStore interface
// that will be used for the birthday block sanity check tests.
type mockBirthdayStore struct {
	birthday              time.Time
	birthdayBlock         *waddrmgr.BlockStamp
	birthdayBlockVerified bool
	syncedTo              waddrmgr.BlockStamp
}

var _ birthdayStore = (*mockBirthdayStore)(nil)

// Birthday returns the birthday timestamp of the wallet.
func (s *mockBirthdayStore) Birthday() time.Time {
	return s.birthday
}

// BirthdayBlock returns the birthday block of the wallet.
func (s *mockBirthdayStore) BirthdayBlock() (waddrmgr.BlockStamp, bool, error) {
	if s.birthdayBlock == nil {
		err := waddrmgr.ManagerError{
			ErrorCode: waddrmgr.ErrBirthdayBlockNotSet,
		}
		return waddrmgr.BlockStamp{}, false, err
	}

	return *s.birthdayBlock, s.birthdayBlockVerified, nil
}

// SetBirthdayBlock updates the birthday block of the wallet to the given block.
// The boolean can be used to signal whether this block should be sanity checked
// the next time the wallet starts.
func (s *mockBirthdayStore) SetBirthdayBlock(block waddrmgr.BlockStamp) error {
	s.birthdayBlock = &block
	s.birthdayBlockVerified = true
	s.syncedTo = block
	return nil
}

// TestBirthdaySanityCheckEmptyBirthdayBlock ensures that a sanity check is not
// done if the birthday block does not exist in the first place.
func TestBirthdaySanityCheckEmptyBirthdayBlock(t *testing.T) {
	t.Parallel()

	chainConn := &mockChainConn{}

	// Our birthday store will reflect that we don't have a birthday block
	// set, so we should not attempt a sanity check.
	birthdayStore := &mockBirthdayStore{}

	birthdayBlock, err := birthdaySanityCheck(chainConn, birthdayStore)
	if !waddrmgr.IsError(err, waddrmgr.ErrBirthdayBlockNotSet) {
		t.Fatalf("expected ErrBirthdayBlockNotSet, got %v", err)
	}

	if birthdayBlock != nil {
		t.Fatalf("expected birthday block to be nil due to not being "+
			"set, got %v", *birthdayBlock)
	}
}

// TestBirthdaySanityCheckVerifiedBirthdayBlock ensures that a sanity check is
// not performed if the birthday block has already been verified.
func TestBirthdaySanityCheckVerifiedBirthdayBlock(t *testing.T) {
	t.Parallel()

	const chainTip = 5000
	chainConn := createMockChainConn(
		chainParams.GenesisBlock, chainTip, defaultBlockInterval,
	)
	expectedBirthdayBlock := waddrmgr.BlockStamp{Height: 1337}

	// Our birthday store reflects that our birthday block has already been
	// verified and should not require a sanity check.
	birthdayStore := &mockBirthdayStore{
		birthdayBlock:         &expectedBirthdayBlock,
		birthdayBlockVerified: true,
		syncedTo: waddrmgr.BlockStamp{
			Height: chainTip,
		},
	}

	// Now, we'll run the sanity check. We should see that the birthday
	// block hasn't changed.
	birthdayBlock, err := birthdaySanityCheck(chainConn, birthdayStore)
	if err != nil {
		t.Fatalf("unable to sanity check birthday block: %v", err)
	}
	if !reflect.DeepEqual(*birthdayBlock, expectedBirthdayBlock) {
		t.Fatalf("expected birthday block %v, got %v",
			expectedBirthdayBlock, birthdayBlock)
	}

	// To ensure the sanity check didn't proceed, we'll check our synced to
	// height, as this value should have been modified if a new candidate
	// was found.
	if birthdayStore.syncedTo.Height != chainTip {
		t.Fatalf("expected synced height remain the same (%d), got %d",
			chainTip, birthdayStore.syncedTo.Height)
	}
}

// TestBirthdaySanityCheckLowerEstimate ensures that we can properly locate a
// better birthday block candidate if our estimate happens to be too far back in
// the chain.
func TestBirthdaySanityCheckLowerEstimate(t *testing.T) {
	t.Parallel()

	// We'll start by defining our birthday timestamp to be around the
	// timestamp of the 1337th block.
	genesisTimestamp := chainParams.GenesisBlock.BlockHeader().Timestamp
	birthday := genesisTimestamp.Add(1337 * defaultBlockInterval)

	// We'll establish a connection to a mock chain of 5000 blocks.
	chainConn := createMockChainConn(
		chainParams.GenesisBlock, 5000, defaultBlockInterval,
	)

	// Our birthday store will reflect that our birthday block is currently
	// set as the genesis block. This value is too low and should be
	// adjusted by the sanity check.
	birthdayStore := &mockBirthdayStore{
		birthday: birthday,
		birthdayBlock: &waddrmgr.BlockStamp{
			Hash:      *chainParams.GenesisHash,
			Height:    0,
			Timestamp: genesisTimestamp,
		},
		birthdayBlockVerified: false,
		syncedTo: waddrmgr.BlockStamp{
			Height: 5000,
		},
	}

	// We'll perform the sanity check and determine whether we were able to
	// find a better birthday block candidate.
	birthdayBlock, err := birthdaySanityCheck(chainConn, birthdayStore)
	if err != nil {
		t.Fatalf("unable to sanity check birthday block: %v", err)
	}
	if birthday.Sub(birthdayBlock.Timestamp) >= birthdayBlockDelta {
		t.Fatalf("expected birthday block timestamp=%v to be within "+
			"%v of birthday timestamp=%v", birthdayBlock.Timestamp,
			birthdayBlockDelta, birthday)
	}

	// Finally, our synced to height should now reflect our new birthday
	// block to ensure the wallet doesn't miss any events from this point
	// forward.
	if !reflect.DeepEqual(birthdayStore.syncedTo, *birthdayBlock) {
		t.Fatalf("expected syncedTo and birthday block to match: "+
			"%v vs %v", birthdayStore.syncedTo, birthdayBlock)
	}
}

// TestBirthdaySanityCheckHigherEstimate ensures that we can properly locate a
// better birthday block candidate if our estimate happens to be too far in the
// chain.
func TestBirthdaySanityCheckHigherEstimate(t *testing.T) {
	t.Parallel()

	// We'll start by defining our birthday timestamp to be around the
	// timestamp of the 1337th block.
	genesisTimestamp := chainParams.GenesisBlock.BlockHeader().Timestamp
	birthday := genesisTimestamp.Add(1337 * defaultBlockInterval)

	// We'll establish a connection to a mock chain of 5000 blocks.
	chainConn := createMockChainConn(
		chainParams.GenesisBlock, 5000, defaultBlockInterval,
	)

	// Our birthday store will reflect that our birthday block is currently
	// set as the chain tip. This value is too high and should be adjusted
	// by the sanity check.
	bestBlock := chainConn.blocks[chainConn.blockHashes[5000]]
	birthdayStore := &mockBirthdayStore{
		birthday: birthday,
		birthdayBlock: &waddrmgr.BlockStamp{
			Hash:      bestBlock.BlockHash(),
			Height:    5000,
			Timestamp: bestBlock.BlockHeader().Timestamp,
		},
		birthdayBlockVerified: false,
		syncedTo: waddrmgr.BlockStamp{
			Height: 5000,
		},
	}

	// We'll perform the sanity check and determine whether we were able to
	// find a better birthday block candidate.
	birthdayBlock, err := birthdaySanityCheck(chainConn, birthdayStore)
	if err != nil {
		t.Fatalf("unable to sanity check birthday block: %v", err)
	}
	if birthday.Sub(birthdayBlock.Timestamp) >= birthdayBlockDelta {
		t.Fatalf("expected birthday block timestamp=%v to be within "+
			"%v of birthday timestamp=%v", birthdayBlock.Timestamp,
			birthdayBlockDelta, birthday)
	}

	// Finally, our synced to height should now reflect our new birthday
	// block to ensure the wallet doesn't miss any events from this point
	// forward.
	if !reflect.DeepEqual(birthdayStore.syncedTo, *birthdayBlock) {
		t.Fatalf("expected syncedTo and birthday block to match: "+
			"%v vs %v", birthdayStore.syncedTo, birthdayBlock)
	}
}

// reorgRecord creates a synthetic transaction for notification-handler tests.
func reorgRecord(t *testing.T, n byte) *wtxmgr.TxRecord {
	t.Helper()

	msg := wire.NewMsgTx(2)
	msg.AddTxIn(wire.NewTxIn(&wire.OutPoint{Hash: chainhash.Hash{n}}, nil, nil))
	msg.AddTxOut(wire.NewTxOut(1000, []byte{0x51}))

	rec, err := wtxmgr.NewTxRecordFromMsgTx(msg, time.Unix(1, 0))
	require.NoError(t, err)

	return rec
}

// TestDisconnectBlockDuringSync verifies disconnects remove matching history with a matching, lagging, sparse, or
// already replaced address-manager tip.
func TestDisconnectBlockDuringSync(t *testing.T) {
	t.Parallel()

	for _, synced := range []bool{false, true} {
		for _, tip := range []struct {
			name       string
			height     int32
			hash       chainhash.Hash
			wantHeight int32
			wantHash   chainhash.Hash
		}{
			{"behind", 0, *chaincfg.TestNetParams.GenesisHash, 0, *chaincfg.TestNetParams.GenesisHash},
			{"matching", 1, chainhash.Hash{1}, 0, *chaincfg.TestNetParams.GenesisHash},
			{"replacement", 1, chainhash.Hash{2}, 1, chainhash.Hash{2}},
			{"sparse", 2, chainhash.Hash{5}, 2, chainhash.Hash{5}},
		} {
			t.Run(fmt.Sprintf("synced=%v/tip=%s", synced, tip.name), func(t *testing.T) {
				t.Parallel()

				w, cleanup := testWalletWatchingOnly(t)
				t.Cleanup(cleanup)
				t.Cleanup(func() {
					require.NoError(t, w.db.Close())
				})
				w.SetChainSynced(synced)
				w.chainClient = &mockChainClient{
					getBlockHeader: &wire.BlockHeader{
						Timestamp: time.Unix(1, 0),
					},
				}

				old := &wtxmgr.BlockMeta{
					Block: wtxmgr.Block{
						Height: 1,
						Hash:   chainhash.Hash{1},
					},
					Time: time.Unix(1, 0),
				}
				replacement := *old
				replacement.Hash = chainhash.Hash{2}
				a, b := reorgRecord(t, 3), reorgRecord(t, 4)

				err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					require.NoError(t, w.addRelevantTx(tx, a, old))

					return w.connectBlock(tx, wtxmgr.BlockMeta{
						Block: wtxmgr.Block{
							Height: tip.height,
							Hash:   tip.hash,
						},
						Time: time.Unix(1, 0),
					})
				})
				require.NoError(t, err)

				err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.disconnectBlock(tx, *old)
				})
				require.NoError(t, err)

				require.Equal(t, tip.wantHeight, w.Manager.SyncedTo().Height)
				require.Equal(t, tip.wantHash, w.Manager.SyncedTo().Hash)

				err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.addRelevantTx(tx, b, &replacement)
				})
				require.NoError(t, err)

				// A replayed disconnect must not remove the replacement.
				err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.disconnectBlock(tx, *old)
				})
				require.NoError(t, err)

				// A delayed connection from the other notification producer can leave the manager on the old block.
				err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.connectBlock(tx, *old)
				})
				require.NoError(t, err)

				notifications := w.NtfnServer.TransactionNotifications()
				t.Cleanup(notifications.Done)

				err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.disconnectBlock(tx, *old)
				})
				require.NoError(t, err)
				require.Equal(t, waddrmgr.BlockStamp{
					Height:    0,
					Hash:      *chaincfg.TestNetParams.GenesisHash,
					Timestamp: time.Unix(1, 0),
				}, w.Manager.SyncedTo())
				require.NotNil(t, w.NtfnServer.currentTxNtfn)
				require.Equal(t, []*chainhash.Hash{&old.Hash}, w.NtfnServer.currentTxNtfn.DetachedBlocks)

				result, err := w.GetTransactions(nil, nil, "", nil)
				require.NoError(t, err)
				require.Len(t, result.MinedTransactions, 1)
				require.Len(t, result.UnminedTransactions, 1)

				oldTx, err := w.GetTransaction(a.Hash)
				require.NoError(t, err)
				require.EqualValues(t, -1, oldTx.Height)
				require.Nil(t, oldTx.BlockHash)

				newTx, err := w.GetTransaction(b.Hash)
				require.NoError(t, err)
				require.Equal(t, replacement.Height, newTx.Height)
				require.Equal(t, &replacement.Hash, newTx.BlockHash)
			})
		}
	}
}

var errReorgBackend = errors.New("reorg backend unavailable")

// reorgChainClient supplies canonical block hashes, including lookup failures.
type reorgChainClient struct {
	*mockChainClient

	blockHashes map[int64]chainhash.Hash
	bestErr     error
}

func (c *reorgChainClient) GetBestBlock() (*chainhash.Hash, int32, error) {
	if c.bestErr != nil {
		return nil, 0, c.bestErr
	}

	return c.mockChainClient.GetBestBlock()
}

func (c *reorgChainClient) GetBlockHash(height int64) (*chainhash.Hash, error) {
	hash, ok := c.blockHashes[height]
	if !ok {
		return nil, fmt.Errorf("block hash at height %d: %w", height, errReorgBackend)
	}

	return &hash, nil
}

type mixedHistoryCase struct {
	name    string
	mask    uint8
	tip     int32
	missing int64
	bestErr bool
}

// TestDisconnectMixedHistory covers every canonical/orphan arrangement across three stored heights, with and without a
// record at the disconnected height. The all-orphan case also covers disconnects of empty blocks.
func TestDisconnectMixedHistory(t *testing.T) {
	t.Parallel()

	tests := []mixedHistoryCase{
		{"all orphaned", 0, 4, 0, false},
		{"canonical prefix", 1, 4, 0, false},
		{"canonical middle", 2, 4, 0, false},
		{"orphan suffix", 3, 4, 0, false},
		{"orphan prefix", 4, 4, 0, false},
		{"orphan middle", 5, 4, 0, false},
		{"canonical suffix", 6, 4, 0, false},
		{"all canonical", 7, 4, 0, false},
		{"above backend tip", 7, 3, 0, false},
		{"early lookup failure", 7, 4, 2, false},
		{"late lookup failure", 0, 4, 4, false},
		{"best block failure", 0, 4, 0, true},
	}
	for _, stored := range []bool{false, true} {
		for _, test := range tests {
			t.Run(fmt.Sprintf("stored=%v/%s", stored, test.name), func(t *testing.T) {
				t.Parallel()

				w, cleanup := testWalletWatchingOnly(t)
				t.Cleanup(cleanup)
				t.Cleanup(func() { require.NoError(t, w.db.Close()) })

				client := &reorgChainClient{
					mockChainClient: &mockChainClient{
						getBestBlockHeight: test.tip,
						getBlockHeader: &wire.BlockHeader{
							Timestamp: time.Unix(1, 0),
						},
					},
					blockHashes: map[int64]chainhash.Hash{1: {9}},
				}
				if test.bestErr {
					client.bestErr = errReorgBackend
				}

				w.chainClient = client

				old := reorgBlock(1)
				replacement := old
				replacement.Hash = chainhash.Hash{9}
				records, heights, wantHeights := mixedHistoryRecords(t, stored, test, client)

				delete(client.blockHashes, test.missing)

				wantErr := test.bestErr || test.missing != 0
				if wantErr {
					wantHeights = heights
				}

				require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					require.NoError(t, w.connectBlock(tx, replacement))
					// Higher records arrive before the old block.
					for i := len(records) - 1; i >= 0; i-- {
						block := reorgBlock(heights[i])
						addReorgCredit(t, w, tx, records[i], &block, i%2 != 0)
					}

					return w.connectBlock(tx, old)
				}))

				err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
					return w.disconnectBlock(tx, old)
				})
				if wantErr {
					require.ErrorIs(t, err, errReorgBackend)
					require.Equal(t, old.Hash, w.Manager.SyncedTo().Hash)
				} else {
					require.NoError(t, err)
					require.Zero(t, w.Manager.SyncedTo().Height)
				}

				assertReorgHistory(t, w, records, wantHeights)

				if !wantErr {
					// A duplicate detach must not change the result.
					require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
						return w.disconnectBlock(tx, old)
					}))
					assertReorgHistory(t, w, records, wantHeights)
				}
			})
		}
	}
}

func mixedHistoryRecords(t *testing.T, stored bool, test mixedHistoryCase, client *reorgChainClient) (
	[]*wtxmgr.TxRecord, []int32, []int32) {

	t.Helper()

	records := make([]*wtxmgr.TxRecord, 0, 4)
	heights := make([]int32, 0, 4)
	wantHeights := make([]int32, 0, 4)

	if stored {
		records = append(records, reorgRecord(t, 10))
		heights = append(heights, 1)
		wantHeights = append(wantHeights, -1)
	}

	for i := range 3 {
		height := int32(i + 2)
		records = append(records, reorgRecord(t, byte(i+11)))
		heights = append(heights, height)
		want := int32(-1)

		hash := chainhash.Hash{byte(height + 10)}
		if test.mask&(1<<i) != 0 {
			hash = chainhash.Hash{byte(height)}
			if height <= test.tip {
				want = height
			}
		}

		client.blockHashes[int64(height)] = hash

		wantHeights = append(wantHeights, want)
	}

	return records, heights, wantHeights
}

func reorgBlock(height int32) wtxmgr.BlockMeta {
	return wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Height: height, Hash: chainhash.Hash{byte(height)}},
		Time:  time.Unix(int64(height), 0),
	}
}

func addReorgCredit(t *testing.T, w *Wallet, tx walletdb.ReadWriteTx, record *wtxmgr.TxRecord, block *wtxmgr.BlockMeta,
	change bool) {

	t.Helper()
	require.NoError(t, w.addRelevantTx(tx, record, block))
	require.NoError(t, w.TxStore.AddCredit(tx.ReadWriteBucket(wtxmgrNamespaceKey), record, block, 0, change))
}

func assertReorgHistory(t *testing.T, w *Wallet, records []*wtxmgr.TxRecord, heights []int32) {
	t.Helper()
	require.NoError(t, walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)

		var wantConfirmed int64
		for i, record := range records {
			detail, err := w.TxStore.TxDetails(ns, &record.Hash)
			require.NoError(t, err)
			require.NotNil(t, detail)
			require.Equal(t, heights[i], detail.Block.Height)
			require.Equal(t, record.Received, detail.Received)
			require.Len(t, detail.Credits, 1)
			require.EqualValues(t, 1000, detail.Credits[0].Amount)
			require.Equal(t, i%2 != 0, detail.Credits[0].Change)

			if heights[i] >= 0 {
				block := reorgBlock(heights[i])
				require.Equal(t, block, detail.Block)

				wantConfirmed += 1000
			}
		}

		confirmed, err := w.TxStore.Balance(ns, 1, 4)
		require.NoError(t, err)
		require.EqualValues(t, wantConfirmed, confirmed)

		total, err := w.TxStore.Balance(ns, 0, 4)
		require.NoError(t, err)
		require.EqualValues(t, len(records)*1000, total)

		return nil
	}))
}

func reorgSpend(t *testing.T, parent *wtxmgr.TxRecord, amount int64) *wtxmgr.TxRecord {
	t.Helper()

	msg := wire.NewMsgTx(2)
	msg.AddTxIn(wire.NewTxIn(&wire.OutPoint{Hash: parent.Hash}, nil, nil))
	msg.AddTxOut(wire.NewTxOut(amount, []byte{0x51}))
	rec, err := wtxmgr.NewTxRecordFromMsgTx(msg, time.Unix(1, 0))
	require.NoError(t, err)

	return rec
}

func reorgCoinbase(t *testing.T, tag byte) *wtxmgr.TxRecord {
	t.Helper()

	msg := wire.NewMsgTx(2)
	msg.AddTxIn(wire.NewTxIn(&wire.OutPoint{Index: ^uint32(0)}, []byte{tag}, nil))
	msg.AddTxOut(wire.NewTxOut(1000, []byte{0x51}))
	rec, err := wtxmgr.NewTxRecordFromMsgTx(msg, time.Unix(1, 0))
	require.NoError(t, err)

	return rec
}

// TestDisconnectCoinbaseSpends covers canonical coinbases, same-block debit order, unmined descendants, orphaned
// coinbase cleanup, labels and leases.
func TestDisconnectCoinbaseSpends(t *testing.T) {
	t.Parallel()

	w, cleanup := testWalletWatchingOnly(t)
	t.Cleanup(cleanup)
	t.Cleanup(func() { require.NoError(t, w.db.Close()) })

	w.chainClient = &reorgChainClient{
		mockChainClient: &mockChainClient{
			getBestBlockHeight: 200,
			getBlockHeader:     &wire.BlockHeader{Timestamp: time.Unix(1, 0)},
		},
		blockHashes: map[int64]chainhash.Hash{
			1: {9}, 2: {2}, 3: {9}, 102: {102},
		},
	}
	old := reorgRecord(t, 60)
	coinbaseMsg := reorgCoinbase(t, 61).MsgTx
	// Output 0 stays spent, output 1 becomes unspent when the mixed descendant is removed, and output 2 stays unspent
	// throughout.
	coinbaseMsg.AddTxOut(wire.NewTxOut(1000, []byte{0x51}))
	coinbaseMsg.AddTxOut(wire.NewTxOut(1000, []byte{0x51}))
	coinbase, err := wtxmgr.NewTxRecordFromMsgTx(&coinbaseMsg, time.Unix(1, 0))
	require.NoError(t, err)

	parent := reorgSpend(t, coinbase, 900)
	child := reorgSpend(t, parent, 800)
	pending := reorgSpend(t, child, 700)
	pendingChild := reorgSpend(t, pending, 600)
	orphanCoinbase := reorgCoinbase(t, 62)
	orphanPending := reorgSpend(t, orphanCoinbase, 900)
	orphanPendingChild := reorgSpend(t, orphanPending, 800)
	// A transaction with both valid and orphaned coinbase ancestry must not be resurrected when the valid ancestry is
	// restored.
	mixedMsg := wire.NewMsgTx(2)
	for _, rec := range []*wtxmgr.TxRecord{pendingChild, orphanPendingChild} {
		mixedMsg.AddTxIn(wire.NewTxIn(&wire.OutPoint{Hash: rec.Hash}, nil, nil))
	}

	mixedMsg.AddTxIn(wire.NewTxIn(&wire.OutPoint{Hash: coinbase.Hash, Index: 1}, nil, nil))
	mixedMsg.AddTxOut(wire.NewTxOut(2300, []byte{0x51}))
	mixed, err := wtxmgr.NewTxRecordFromMsgTx(mixedMsg, time.Unix(1, 0))
	require.NoError(t, err)

	oldBlock, coinbaseBlock := reorgBlock(1), reorgBlock(2)
	orphanBlock, spendBlock := reorgBlock(3), reorgBlock(102)
	leaseID := wtxmgr.LockID{1}
	outpoint := wire.OutPoint{Hash: coinbase.Hash}

	updateErr := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(wtxmgrNamespaceKey)
		addReorgCredit(t, w, tx, old, &oldBlock, false)
		addReorgCredit(t, w, tx, coinbase, &coinbaseBlock, false)

		for _, index := range []uint32{1, 2} {
			require.NoError(t, w.TxStore.AddCredit(ns, coinbase, &coinbaseBlock, index, false))
		}

		_, err := w.TxStore.LockOutput(ns, leaseID, outpoint, time.Hour, wtxmgr.WithReleaseAfterSpend(200))
		require.NoError(t, err)
		addReorgCredit(t, w, tx, parent, &spendBlock, true)
		addReorgCredit(t, w, tx, child, &spendBlock, false)
		addReorgCredit(t, w, tx, pending, nil, true)
		addReorgCredit(t, w, tx, pendingChild, nil, false)
		addReorgCredit(t, w, tx, orphanCoinbase, &orphanBlock, false)
		addReorgCredit(t, w, tx, orphanPending, nil, false)
		addReorgCredit(t, w, tx, orphanPendingChild, nil, false)
		addReorgCredit(t, w, tx, mixed, nil, false)
		require.NoError(t, w.TxStore.PutTxLabel(ns, parent.Hash, "retained"))

		return w.connectBlock(tx, oldBlock)
	})
	require.NoError(t, updateErr)

	// Record the callback that forwards unspent events to public clients.
	var unspent []wire.OutPoint

	notify := w.TxStore.NotifyUnspent
	w.TxStore.NotifyUnspent = func(hash *chainhash.Hash, index uint32) {
		unspent = append(unspent, wire.OutPoint{Hash: *hash, Index: index})
		notify(hash, index)
	}

	// An error after successful replay must discard queued notifications along with the database changes. Retry against
	// the same history.
	abortErr := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(wtxmgrNamespaceKey)
		require.NoError(t, w.rollbackBlockHistory(ns, oldBlock.Height))
		require.Empty(t, unspent)

		return errReorgWrite
	})
	require.ErrorIs(t, abortErr, errReorgWrite)
	require.Empty(t, unspent)
	require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		err := w.disconnectBlock(tx, oldBlock)

		require.Empty(t, unspent)

		return err
	}))
	// Only removing the mixed-ancestry spend changes spentness. Unchanged spent and unspent credits must not produce
	// replay notifications.
	require.ElementsMatch(t, []wire.OutPoint{
		{Hash: pendingChild.Hash}, {Hash: coinbase.Hash, Index: 1},
	}, unspent)
	require.NoError(t, walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)
		for _, test := range []struct {
			rec           *wtxmgr.TxRecord
			height        int32
			spent, change bool
			debit         int64
		}{
			{old, -1, false, false, 0},
			{parent, 102, true, true, 1000},
			{child, 102, true, false, 900},
			{pending, -1, true, true, 800},
			{pendingChild, -1, false, false, 700},
		} {
			detail, err := w.TxStore.TxDetails(ns, &test.rec.Hash)
			require.NoError(t, err)
			require.NotNil(t, detail)
			require.Equal(t, test.height, detail.Block.Height)
			require.Equal(t, test.rec.Received, detail.Received)
			require.Len(t, detail.Credits, 1)
			require.Equal(t, test.spent, detail.Credits[0].Spent)
			require.Equal(t, test.change, detail.Credits[0].Change)

			if test.debit > 0 {
				require.Len(t, detail.Debits, 1)
				require.EqualValues(t, test.debit, detail.Debits[0].Amount)
			}
		}

		for _, rec := range []*wtxmgr.TxRecord{
			orphanCoinbase, orphanPending, orphanPendingChild, mixed,
		} {
			detail, err := w.TxStore.TxDetails(ns, &rec.Hash)
			require.NoError(t, err)
			require.Nil(t, detail)
		}

		cbDetail, err := w.TxStore.TxDetails(ns, &coinbase.Hash)
		require.NoError(t, err)
		require.NotNil(t, cbDetail)
		require.Equal(t, coinbaseBlock, cbDetail.Block)
		require.Equal(t, coinbase.Received, cbDetail.Received)
		require.Equal(t, []wtxmgr.CreditRecord{
			{Amount: 1000, Index: 0, Spent: true},
			{Amount: 1000, Index: 1},
			{Amount: 1000, Index: 2},
		}, cbDetail.Credits)

		label, err := w.TxStore.TxLabel(ns, parent.Hash)
		require.NoError(t, err)
		require.Equal(t, "retained", label)

		locks, err := w.TxStore.ListLockedOutputs(ns)
		require.NoError(t, err)
		require.Len(t, locks, 1)
		require.Equal(t, leaseID, locks[0].LockID)
		require.EqualValues(t, 102, locks[0].ConfirmedSpendHeight)

		confirmed, err := w.TxStore.Balance(ns, 1, 200)
		require.NoError(t, err)
		require.EqualValues(t, 2000, confirmed)

		total, err := w.TxStore.Balance(ns, 0, 200)
		require.NoError(t, err)
		require.EqualValues(t, 3600, total)

		return nil
	}))
}

// TestDisconnectConflictingSpends verifies replay removes orphaned double spends and their unmined children while
// rebuilding the canonical debit.
func TestDisconnectConflictingSpends(t *testing.T) {
	t.Parallel()

	w, cleanup := testWalletWatchingOnly(t)
	t.Cleanup(cleanup)
	t.Cleanup(func() { require.NoError(t, w.db.Close()) })

	w.chainClient = &reorgChainClient{
		mockChainClient: &mockChainClient{
			getBestBlockHeight: 2,
			getBlockHeader:     &wire.BlockHeader{Timestamp: time.Unix(1, 0)},
		},
		blockHashes: map[int64]chainhash.Hash{1: {9}, 2: {2}},
	}
	funding := reorgRecord(t, 70)
	orphan := reorgSpend(t, funding, 900)
	orphanChild := reorgSpend(t, orphan, 800)
	canonical := reorgSpend(t, funding, 700)
	fundingBlock := reorgBlock(0)
	oldBlock, newBlock := reorgBlock(1), reorgBlock(2)
	require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		addReorgCredit(t, w, tx, funding, &fundingBlock, false)
		addReorgCredit(t, w, tx, orphan, &oldBlock, false)
		addReorgCredit(t, w, tx, orphanChild, nil, false)
		addReorgCredit(t, w, tx, canonical, &newBlock, true)

		return w.connectBlock(tx, oldBlock)
	}))
	require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		return w.disconnectBlock(tx, oldBlock)
	}))
	require.NoError(t, walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)
		for _, rec := range []*wtxmgr.TxRecord{orphan, orphanChild} {
			detail, err := w.TxStore.TxDetails(ns, &rec.Hash)
			require.NoError(t, err)
			require.Nil(t, detail)
		}

		detail, err := w.TxStore.TxDetails(ns, &canonical.Hash)
		require.NoError(t, err)
		require.EqualValues(t, 2, detail.Block.Height)
		require.Len(t, detail.Debits, 1)
		require.EqualValues(t, 1000, detail.Debits[0].Amount)

		for _, minConf := range []int32{0, 1} {
			balance, err := w.TxStore.Balance(ns, minConf, 2)
			require.NoError(t, err)
			require.EqualValues(t, 700, balance)
		}

		return nil
	}))
}

var errReorgWrite = errors.New("reorg replay write failed")

// failingReorgBucket rejects writes of one saved block hash. This forces an error during replay, after the original
// records have been rolled back.
type failingReorgBucket struct {
	walletdb.ReadWriteBucket

	failHash chainhash.Hash
}

//nolint:ireturn // Implements the walletdb bucket interface.
func (b failingReorgBucket) NestedReadWriteBucket(key []byte) walletdb.ReadWriteBucket {
	bucket := b.ReadWriteBucket.NestedReadWriteBucket(key)
	if bucket == nil {
		return nil
	}

	return failingReorgBucket{ReadWriteBucket: bucket, failHash: b.failHash}
}

func (b failingReorgBucket) Put(key, value []byte) error {
	// A block record starts with its hash. Rollback writes no block records.
	if len(key) == 4 && len(value) >= 32 && bytes.Equal(value[:32], b.failHash[:]) {
		return fmt.Errorf("restore block: %w", errReorgWrite)
	}

	return b.ReadWriteBucket.Put(key, value)
}

func TestDisconnectReplayFailure(t *testing.T) {
	t.Parallel()

	w, cleanup := testWalletWatchingOnly(t)
	t.Cleanup(cleanup)
	t.Cleanup(func() { require.NoError(t, w.db.Close()) })

	w.chainClient = &reorgChainClient{
		mockChainClient: &mockChainClient{getBestBlockHeight: 3},
		blockHashes:     map[int64]chainhash.Hash{1: {9}, 2: {2}, 3: {3}},
	}
	records := []*wtxmgr.TxRecord{
		reorgRecord(t, 80), reorgRecord(t, 81), reorgRecord(t, 82),
	}
	heights := []int32{1, 2, 3}
	require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		for i, rec := range records {
			block := reorgBlock(heights[i])
			addReorgCredit(t, w, tx, rec, &block, i%2 != 0)
		}

		return nil
	}))
	err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := failingReorgBucket{
			ReadWriteBucket: tx.ReadWriteBucket(wtxmgrNamespaceKey),
			failHash:        chainhash.Hash{3},
		}

		return w.rollbackBlockHistory(ns, 1)
	})
	require.ErrorIs(t, err, errReorgWrite)
	assertReorgHistory(t, w, records, heights)
}

// TestDisconnectOrphanCoinbaseSpend checks that a normal transaction detached from an orphan block survives as unmined
// when its coinbase ancestor is valid.
func TestDisconnectOrphanCoinbaseSpend(t *testing.T) {
	t.Parallel()

	w, cleanup := testWalletWatchingOnly(t)
	t.Cleanup(cleanup)
	t.Cleanup(func() { require.NoError(t, w.db.Close()) })

	w.chainClient = &reorgChainClient{
		mockChainClient: &mockChainClient{
			getBestBlockHeight: 102,
			getBlockHeader:     &wire.BlockHeader{Timestamp: time.Unix(1, 0)},
		},
		blockHashes: map[int64]chainhash.Hash{1: {9}, 2: {2}, 102: {9}},
	}
	old := reorgRecord(t, 90)
	coinbase := reorgCoinbase(t, 91)
	spend := reorgSpend(t, coinbase, 900)
	child := reorgSpend(t, spend, 800)
	oldBlock, cbBlock := reorgBlock(1), reorgBlock(2)
	spendBlock := reorgBlock(102)
	err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		addReorgCredit(t, w, tx, old, &oldBlock, false)
		addReorgCredit(t, w, tx, coinbase, &cbBlock, false)
		addReorgCredit(t, w, tx, spend, &spendBlock, true)
		addReorgCredit(t, w, tx, child, nil, false)

		return w.connectBlock(tx, oldBlock)
	})
	require.NoError(t, err)
	err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		return w.disconnectBlock(tx, oldBlock)
	})
	require.NoError(t, err)
	require.NoError(t, walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)
		for _, rec := range []*wtxmgr.TxRecord{spend, child} {
			detail, err := w.TxStore.TxDetails(ns, &rec.Hash)
			require.NoError(t, err)
			require.NotNil(t, detail)
			require.EqualValues(t, -1, detail.Block.Height)
			require.Len(t, detail.Credits, 1)
		}

		detail, err := w.TxStore.TxDetails(ns, &coinbase.Hash)
		require.NoError(t, err)
		require.EqualValues(t, 2, detail.Block.Height)
		require.True(t, detail.Credits[0].Spent)

		balance, err := w.TxStore.Balance(ns, 0, 102)
		require.NoError(t, err)
		require.EqualValues(t, 1800, balance)

		return nil
	}))
}
