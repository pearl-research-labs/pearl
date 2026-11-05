package wallet

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
)

var (
	TstSerializedTx, _ = hex.DecodeString("010000000114d9ff358894c486b4ae11c2a8cf7851b1df64c53d2e511278eff17c22fb7373000000008c493046022100995447baec31ee9f6d4ec0e05cb2a44f6b817a99d5f6de167d1c75354a946410022100c9ffc23b64d770b0e01e7ff4d25fbc2f1ca8091053078a247905c39fce3760b601410458b8e267add3c1e374cf40f1de02b59213a82e1d84c2b94096e22e2f09387009c96debe1d0bcb2356ffdcf65d2a83d4b34e72c62eccd8490dbf2110167783b2bffffffff0280969800000000001976a914479ed307831d0ac19ebc5f63de7d5f1a430ddb9d88ac38bfaa00000000001976a914dadf9e3484f28b385ddeaa6c575c0c0d18e9788a88ac00000000")
	TstTx, _           = btcutil.NewTxFromBytes(TstSerializedTx)
	TstTxHash          = TstTx.Hash()

	TstMinedTxBlockHeight        = int32(279143)
	TstMinedSignedTxBlockDetails = &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{
			Hash:   *TstTxHash,
			Height: TstMinedTxBlockHeight,
		},
		Time: time.Now(),
	}
)

// TestLocateBirthdayBlock ensures we can properly map a block in the chain to a
// timestamp.
func TestLocateBirthdayBlock(t *testing.T) {
	t.Parallel()

	// We'll use test chains of 30 blocks with a duration between two
	// consecutive blocks being slightly greater than the largest margin
	// allowed by locateBirthdayBlock. Doing so lets us test the method more
	// effectively as there is only one block within the chain that can map
	// to a timestamp (this does not apply to the first and last blocks,
	// which can map to many timestamps beyond either end of chain).
	const (
		numBlocks     = 30
		blockInterval = birthdayBlockDelta + 1
	)

	genesisTimestamp := chainParams.GenesisBlock.BlockHeader().Timestamp

	testCases := []struct {
		name           string
		birthday       time.Time
		birthdayHeight int32
	}{
		{
			name:           "left-right-left-left",
			birthday:       genesisTimestamp.Add(8 * blockInterval),
			birthdayHeight: 8,
		},
		{
			name:           "right-right-right-left",
			birthday:       genesisTimestamp.Add(27 * blockInterval),
			birthdayHeight: 27,
		},
		{
			name:           "before start height",
			birthday:       genesisTimestamp.Add(-blockInterval),
			birthdayHeight: 0,
		},
		{
			name:           "start height",
			birthday:       genesisTimestamp,
			birthdayHeight: 0,
		},
		{
			name:           "end height",
			birthday:       genesisTimestamp.Add(numBlocks * blockInterval),
			birthdayHeight: numBlocks - 1,
		},
		{
			name:           "after end height",
			birthday:       genesisTimestamp.Add(2 * numBlocks * blockInterval),
			birthdayHeight: numBlocks - 1,
		},
	}

	for _, testCase := range testCases {
		testCase := testCase
		success := t.Run(testCase.name, func(t *testing.T) {
			chainConn := createMockChainConn(
				chainParams.GenesisBlock, numBlocks, blockInterval,
			)
			birthdayBlock, err := locateBirthdayBlock(
				chainConn, testCase.birthday,
			)
			if err != nil {
				t.Fatalf("unable to locate birthday block: %v",
					err)
			}
			if birthdayBlock.Height != testCase.birthdayHeight {
				t.Fatalf("expected birthday block with height "+
					"%d, got %d", testCase.birthdayHeight,
					birthdayBlock.Height)
			}
		})
		if !success {
			break
		}
	}
}

// TestLabelTransaction tests labelling of transactions with invalid labels,
// and failure to label a transaction when it already has a label.
func TestLabelTransaction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string

		// Whether the transaction should be known to the wallet.
		txKnown bool

		// Whether the test should write an existing label to disk.
		existingLabel bool

		// The overwrite parameter to call label transaction with.
		overwrite bool

		// The error we expect to be returned.
		expectedErr error
	}{
		{
			name:          "existing label, not overwrite",
			txKnown:       true,
			existingLabel: true,
			overwrite:     false,
			expectedErr:   ErrTxLabelExists,
		},
		{
			name:          "existing label, overwritten",
			txKnown:       true,
			existingLabel: true,
			overwrite:     true,
			expectedErr:   nil,
		},
		{
			name:          "no prexisting label, ok",
			txKnown:       true,
			existingLabel: false,
			overwrite:     false,
			expectedErr:   nil,
		},
		{
			name:          "transaction unknown",
			txKnown:       false,
			existingLabel: false,
			overwrite:     false,
			expectedErr:   ErrUnknownTransaction,
		},
	}

	for _, test := range tests {
		test := test

		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			w, cleanup := testWallet(t)
			defer cleanup()

			// If the transaction should be known to the store, we
			// write txdetail to disk.
			if test.txKnown {
				rec, err := wtxmgr.NewTxRecord(
					TstSerializedTx, time.Now(),
				)
				if err != nil {
					t.Fatal(err)
				}

				err = walletdb.Update(w.db,
					func(tx walletdb.ReadWriteTx) error {

						ns := tx.ReadWriteBucket(
							wtxmgrNamespaceKey,
						)

						return w.TxStore.InsertTx(
							ns, rec, nil,
						)
					})
				if err != nil {
					t.Fatalf("could not insert tx: %v", err)
				}
			}

			// If we want to setup an existing label for the purpose
			// of the test, write one to disk.
			if test.existingLabel {
				err := w.LabelTransaction(
					*TstTxHash, "existing label", false,
				)
				if err != nil {
					t.Fatalf("could not write label: %v",
						err)
				}
			}

			newLabel := "new label"
			err := w.LabelTransaction(
				*TstTxHash, newLabel, test.overwrite,
			)
			if err != test.expectedErr {
				t.Fatalf("expected: %v, got: %v",
					test.expectedErr, err)
			}
		})
	}
}

// TestGetTransaction tests if we can fetch a mined, an existing
// and a non-existing transaction from the wallet like we expect.
func TestGetTransaction(t *testing.T) {
	t.Parallel()
	rec, err := wtxmgr.NewTxRecord(TstSerializedTx, time.Now())
	require.NoError(t, err)

	tests := []struct {
		name string

		// Transaction id.
		txid chainhash.Hash

		// Expected height.
		expectedHeight int32

		// Store function.
		f func(*wtxmgr.Store, walletdb.ReadWriteBucket) (*wtxmgr.Store, error)

		// The error we expect to be returned.
		expectedErr error
	}{
		{
			name:           "existing unmined transaction",
			txid:           *TstTxHash,
			expectedHeight: -1,
			// We write txdetail for the tx to disk.
			f: func(s *wtxmgr.Store, ns walletdb.ReadWriteBucket) (
				*wtxmgr.Store, error) {

				err = s.InsertTx(ns, rec, nil)
				return s, err
			},
			expectedErr: nil,
		},
		{
			name: "existing mined transaction",
			txid: *TstTxHash,
			// We write txdetail for the tx to disk.
			f: func(s *wtxmgr.Store, ns walletdb.ReadWriteBucket) (
				*wtxmgr.Store, error) {

				err = s.InsertTx(ns, rec, TstMinedSignedTxBlockDetails)
				return s, err
			},
			expectedHeight: TstMinedTxBlockHeight,
			expectedErr:    nil,
		},
		{
			name: "non-existing transaction",
			txid: *TstTxHash,
			// Write no txdetail to disk.
			f: func(s *wtxmgr.Store, ns walletdb.ReadWriteBucket) (
				*wtxmgr.Store, error) {

				return s, nil
			},
			expectedErr: ErrNoTx,
		},
	}
	for _, test := range tests {
		test := test

		t.Run(test.name, func(t *testing.T) {
			w, cleanup := testWallet(t)
			defer cleanup()

			err := walletdb.Update(w.db, func(rw walletdb.ReadWriteTx) error {
				ns := rw.ReadWriteBucket(wtxmgrNamespaceKey)
				_, err := test.f(w.TxStore, ns)
				return err
			})
			require.NoError(t, err)
			tx, err := w.GetTransaction(test.txid)
			require.ErrorIs(t, err, test.expectedErr)

			// Discontinue if no transaction were found.
			if err != nil {
				return
			}

			// Check if we get the expected hash.
			require.Equal(t, &test.txid, tx.Summary.Hash)

			// Check the block height.
			require.Equal(t, test.expectedHeight, tx.Height)
		})
	}
}

// TestGetTransactionConfirmations tests that GetTransaction correctly
// calculates confirmations for both confirmed and unconfirmed transactions.
// This is a regression test for a bug where confirmations were set to the
// block height instead of being calculated as currentHeight - blockHeight + 1.
//
// The bug had several negative impacts:
//   - Unconfirmed transactions showed -1 confirmations instead of 0, breaking
//     zero-conf (accepting transactions before block inclusion)
//   - Confirmed transactions showed block height instead of actual confirmation
//     count
//   - LND and other consumers would make incorrect decisions based on wrong
//     counts
func TestGetTransactionConfirmations(t *testing.T) {
	t.Parallel()

	rec, err := wtxmgr.NewTxRecord(TstSerializedTx, time.Now())
	require.NoError(t, err)

	tests := []struct {
		name string

		// Block height where transaction is mined (-1 for unmined).
		txBlockHeight int32

		// Current wallet sync height.
		currentHeight int32

		// Expected confirmations.
		expectedConfirmations int32

		// Expected height in result.
		expectedHeight int32

		// Whether to check for non-zero timestamp.
		expectTimestamp bool
	}{
		{
			name:                  "unconfirmed tx",
			txBlockHeight:         -1,
			currentHeight:         100,
			expectedConfirmations: 0,
			expectedHeight:        -1,
			expectTimestamp:       false,
		},
		{
			name:                  "tx with 1 confirmation",
			txBlockHeight:         100,
			currentHeight:         100,
			expectedConfirmations: 1,
			expectedHeight:        100,
			expectTimestamp:       true,
		},
		{
			name:                  "tx with 3 confirmations",
			txBlockHeight:         8,
			currentHeight:         10,
			expectedConfirmations: 3,
			expectedHeight:        8,
			expectTimestamp:       true,
		},
		{
			name:                  "old tx with many confirmations",
			txBlockHeight:         1,
			currentHeight:         1000,
			expectedConfirmations: 1000,
			expectedHeight:        1,
			expectTimestamp:       true,
		},
		{
			name:                  "tx in future block",
			txBlockHeight:         105,
			currentHeight:         100,
			expectedConfirmations: 0,
			expectedHeight:        105,
			expectTimestamp:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			w, cleanup := testWallet(t)
			t.Cleanup(cleanup)

			// Set the wallet's synced height.
			err := walletdb.Update(
				w.db, func(tx walletdb.ReadWriteTx) error {
					addrmgrNs := tx.ReadWriteBucket(
						waddrmgrNamespaceKey,
					)
					bs := &waddrmgr.BlockStamp{
						Height: tt.currentHeight,
						Hash:   chainhash.Hash{},
					}

					return w.Manager.SetSyncedTo(
						addrmgrNs, bs,
					)
				},
			)
			require.NoError(t, err)

			// Insert transaction into wallet.
			err = walletdb.Update(
				w.db, func(tx walletdb.ReadWriteTx) error {
					ns := tx.ReadWriteBucket(
						wtxmgrNamespaceKey,
					)

					// Create block metadata if transaction
					// is mined.
					var blockMeta *wtxmgr.BlockMeta
					if tt.txBlockHeight != -1 {
						hash := chainhash.Hash{}
						height := tt.txBlockHeight
						block := wtxmgr.Block{
							Hash:   hash,
							Height: height,
						}
						blockMeta = &wtxmgr.BlockMeta{
							Block: block,
							Time:  time.Now(),
						}
					}

					return w.TxStore.InsertTx(
						ns, rec, blockMeta,
					)
				},
			)
			require.NoError(t, err)

			result, err := w.GetTransaction(*TstTxHash)
			require.NoError(t, err)

			require.Equal(
				t, tt.expectedConfirmations,
				result.Confirmations,
			)

			require.Equal(t, tt.expectedHeight, result.Height)

			if tt.expectTimestamp {
				require.NotZero(t, result.Timestamp)
			} else {
				require.Zero(t, result.Timestamp)
			}

			// Additional checks for unconfirmed transactions.
			if tt.txBlockHeight == -1 {
				require.Nil(t, result.BlockHash)
				require.Equal(t, int32(0), result.Confirmations)
			} else {
				require.NotNil(t, result.BlockHash)
				// Only expect positive confirmations when tx is
				// not in a future block.
				if tt.txBlockHeight <= tt.currentHeight {
					require.Positive(
						t, result.Confirmations,
					)
				} else {
					// Confirmed txns in future blocks for
					// example due to reorg should be
					// treated as unconfirmed and have 0
					// confirmations.
					require.Equal(
						t, int32(0),
						result.Confirmations,
					)
				}
			}
		})
	}
}

// TestDuplicateAddressDerivation tests that duplicate addresses are not
// derived when multiple goroutines are concurrently requesting new addresses.
func TestDuplicateAddressDerivation(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	var (
		m           sync.Mutex
		globalAddrs = make(map[string]btcutil.Address)
	)

	for o := 0; o < 10; o++ {
		var eg errgroup.Group

		for n := 0; n < 10; n++ {
			eg.Go(func() error {
				addrs := make([]btcutil.Address, 10)
				for i := 0; i < 10; i++ {
					addr, err := w.NewAddress(
						0, waddrmgr.KeyScopeBIP0086,
					)
					if err != nil {
						return err
					}

					addrs[i] = addr
				}

				m.Lock()
				defer m.Unlock()

				for idx := range addrs {
					addrStr := addrs[idx].String()
					if a, ok := globalAddrs[addrStr]; ok {
						return fmt.Errorf("duplicate "+
							"address! already "+
							"have %v, want to "+
							"add %v", a, addrs[idx])
					}

					globalAddrs[addrStr] = addrs[idx]
				}

				return nil
			})
		}

		require.NoError(t, eg.Wait())
	}
}

func TestEndRecovery(t *testing.T) {
	// This is an unconventional unit test, but I'm trying to keep things as
	// succint as possible so that this test is readable without having to mock
	// up literally everything.
	// The unmonitored goroutine we're looking at is pretty deep:
	// SynchronizeRPC -> handleChainNotifications -> syncWithChain -> recovery
	// The "deadlock" we're addressing isn't actually a deadlock, but the wallet
	// will hang on Stop() -> WaitForShutdown() until (*Wallet).recovery gets
	// every single block, which could be hours depending on hardware and
	// network factors. The WaitGroup is incremented in SynchronizeRPC, and
	// WaitForShutdown will not return until handleChainNotifications returns,
	// which is blocked by a running (*Wallet).recovery loop.
	// It is noted that the conditions for long recovery are difficult to hit
	// when using Oyster with a fresh seed, because it requires an early
	// birthday to be set or established.

	w, cleanup := testWallet(t)

	blockHashCalled := make(chan struct{})

	chainClient := &mockChainClient{
		// Force the loop to iterate about forever.
		getBestBlockHeight: math.MaxInt32,
		// Get control of when the loop iterates.
		getBlockHashFunc: func() (*chainhash.Hash, error) {
			blockHashCalled <- struct{}{}
			return &chainhash.Hash{}, nil
		},
		// Avoid a panic.
		getBlockHeader: &wire.BlockHeader{},
	}

	recoveryDone := make(chan struct{})
	go func() {
		defer close(recoveryDone)
		w.recovery(chainClient, &waddrmgr.BlockStamp{})
	}()

	getBlockHashCalls := func(expCalls int) {
		var i int
		for {
			select {
			case <-blockHashCalled:
				i++
			case <-time.After(time.Second):
				t.Fatal("expected BlockHash to be called")
			}
			if i == expCalls {
				break
			}
		}
	}

	// Recovery is running.
	getBlockHashCalls(3)

	// Closing the quit channel, e.g. Stop() without endRecovery, alone will not
	// end the recovery loop.
	w.quitMu.Lock()
	close(w.quit)
	w.quitMu.Unlock()
	// Continues scanning.
	getBlockHashCalls(3)

	// We're done with this one
	atomic.StoreUint32(&w.recovering.Load().(*recoverySyncer).quit, 1)
	select {
	case <-blockHashCalled:
	case <-recoveryDone:
	}
	cleanup()

	// Try again.
	w, cleanup = testWallet(t)
	defer cleanup()

	// We'll catch the error to make sure we're hitting our desired path. The
	// WaitGroup isn't required for the test, but does show how it completes
	// shutdown at a higher level.
	var err error
	w.wg.Add(1)
	recoveryDone = make(chan struct{})
	go func() {
		defer w.wg.Done()
		defer close(recoveryDone)
		err = w.recovery(chainClient, &waddrmgr.BlockStamp{})
	}()

	waitedForShutdown := make(chan struct{})
	go func() {
		w.WaitForShutdown()
		close(waitedForShutdown)
	}()

	// Recovery is running.
	getBlockHashCalls(3)

	// endRecovery is required to exit the unmonitored goroutine.
	end := w.endRecovery()
	select {
	case <-blockHashCalled:
	case <-recoveryDone:
	}
	<-end

	// testWallet starts a couple of other unrelated goroutines that need to be
	// killed, so we still need to close the quit channel.
	w.quitMu.Lock()
	close(w.quit)
	w.quitMu.Unlock()

	select {
	case <-waitedForShutdown:
	case <-time.After(time.Second):
		t.Fatal("WaitForShutdown never returned")
	}

	if !strings.EqualFold(err.Error(), "recovery: forced shutdown") {
		t.Fatal("wrong error")
	}
}

// newRecoveryTestState creates the recovery state and scoped-manager map used by a production recovery run, including
// state already persisted in walletdb.
func newRecoveryTestState(t *testing.T, w *Wallet) (
	*RecoveryManager, map[waddrmgr.KeyScope]*waddrmgr.ScopedKeyManager) {

	t.Helper()

	recoveryMgr := NewRecoveryManager(w.recoveryWindow, recoveryBatchSize, w.chainParams)

	scopedMgrs := make(map[waddrmgr.KeyScope]*waddrmgr.ScopedKeyManager)
	for _, scopedMgr := range w.Manager.ActiveScopedKeyManagers() {
		scopedMgrs[scopedMgr.Scope()] = scopedMgr
	}

	err := walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		txMgrNS := tx.ReadBucket(wtxmgrNamespaceKey)

		credits, err := w.TxStore.UnspentOutputs(txMgrNS)
		if err != nil {
			return err
		}

		addrMgrNS := tx.ReadBucket(waddrmgrNamespaceKey)

		return recoveryMgr.Resurrect(addrMgrNS, scopedMgrs, credits)
	})
	require.NoError(t, err)

	return recoveryMgr, scopedMgrs
}

// newRecoveryTestBlock returns matching filter and sync records for a block.
func newRecoveryTestBlock(height int32) (wtxmgr.BlockMeta, *waddrmgr.BlockStamp) {
	var hash chainhash.Hash

	hash[0] = byte(height)

	timestamp := time.Unix(int64(height), 0)
	blockMeta := wtxmgr.BlockMeta{
		Block: wtxmgr.Block{
			Hash:   hash,
			Height: height,
		},
		Time: timestamp,
	}
	blockStamp := &waddrmgr.BlockStamp{
		Hash:      hash,
		Height:    height,
		Timestamp: timestamp,
	}

	return blockMeta, blockStamp
}

// newRecoveryTestResponse builds a filter response for the highest external address in a scope and a transaction that
// pays to that address.
//
//nolint:ireturn
func newRecoveryTestResponse(req *chain.FilterBlocksRequest, scope waddrmgr.KeyScope, value int64) (
	*chain.FilterBlocksResponse, btcutil.Address, uint32, *wire.MsgTx, error) {

	var (
		recoveredAddr  btcutil.Address
		recoveredIndex uint32
	)
	for scopedIndex, candidate := range req.ExternalAddrs {
		if scopedIndex.Scope != scope {
			continue
		}

		if recoveredAddr != nil && scopedIndex.Index < recoveredIndex {
			continue
		}

		recoveredAddr = candidate
		recoveredIndex = scopedIndex.Index
	}

	if recoveredAddr == nil {
		return nil, nil, 0, nil, errors.New("missing recovery address")
	}

	pkScript, err := txscript.PayToAddrScript(recoveredAddr)
	if err != nil {
		return nil, nil, 0, nil, err
	}

	relevantTx := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{wire.NewTxOut(value, pkScript)},
	}
	response := &chain.FilterBlocksResponse{
		BatchIndex: 0,
		BlockMeta:  req.Blocks[0],
		FoundExternalAddrs: map[waddrmgr.KeyScope]map[uint32]struct{}{
			scope: {recoveredIndex: {}},
		},
		RelevantTxns: []*wire.MsgTx{relevantTx},
	}

	return response, recoveredAddr, recoveredIndex, relevantTx, nil
}

// TestRecoveryFilterBlocksDoesNotHoldWriter verifies that a chain backend can obtain a wallet database write
// transaction while FilterBlocks is running. A network-backed backend may block for an unbounded amount of time, so
// holding the writer across this call would also block every unrelated wallet operation that needs to persist state.
func TestRecoveryFilterBlocksDoesNotHoldWriter(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	recoveryMgr, scopedMgrs := newRecoveryTestState(t, w)

	height := w.Manager.SyncedTo().Height + 1
	blockMeta, blockStamp := newRecoveryTestBlock(height)
	batch := []wtxmgr.BlockMeta{blockMeta}
	blocks := []*waddrmgr.BlockStamp{blockStamp}

	errWriterHeld := errors.New("wallet writer held during FilterBlocks")
	filterCalls := 0
	chainClient := &mockChainClient{}
	chainClient.filterBlocksFunc = func(*chain.FilterBlocksRequest) (*chain.FilterBlocksResponse, error) {
		filterCalls++

		writerDone := make(chan error, 1)
		go func() {
			writerDone <- walletdb.Update(w.db, func(walletdb.ReadWriteTx) error {
				return nil
			},
			)
		}()

		select {
		case err := <-writerDone:
			return nil, err

		case <-time.After(time.Second):
			return nil, errWriterHeld
		}
	}

	err := w.recoverBatch(chainClient, batch, blocks, recoveryMgr.State(), scopedMgrs)
	require.NoError(t, err)
	require.Equal(t, 1, filterCalls)
}

// TestRecoveryFilterBlocksSerializesAddressCreation verifies that a recovery request's address horizon stays fixed
// until the batch is committed. Address creation may continue as soon as the backend call returns.
func TestRecoveryFilterBlocksSerializesAddressCreation(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	recoveryMgr, scopedMgrs := newRecoveryTestState(t, w)

	height := w.Manager.SyncedTo().Height + 1
	blockMeta, blockStamp := newRecoveryTestBlock(height)
	batch := []wtxmgr.BlockMeta{blockMeta}
	blocks := []*waddrmgr.BlockStamp{blockStamp}

	errAddressNotSerialized := errors.New("address creation completed before recovery commit")
	addressDone := make(chan error, 1)
	filterCalls := 0
	expectedSync := *blockStamp

	chainClient := &mockChainClient{}
	chainClient.notifyReceivedFunc = func([]btcutil.Address) error {
		if w.Manager.SyncedTo() != expectedSync {
			return errAddressNotSerialized
		}

		return nil
	}
	chainClient.filterBlocksFunc = func(*chain.FilterBlocksRequest) (*chain.FilterBlocksResponse, error) {
		filterCalls++

		addressGoroutineStarted := make(chan struct{})
		go func() {
			close(addressGoroutineStarted)

			_, err := w.NewAddress(waddrmgr.DefaultAccountNum, waddrmgr.KeyScopeBIP0086)
			addressDone <- err
		}()

		<-addressGoroutineStarted

		select {
		case err := <-addressDone:
			if err != nil {
				return nil, err
			}

			return nil, errAddressNotSerialized

		case <-time.After(100 * time.Millisecond):
		}

		return nil, nil //nolint:nilnil
	}
	w.chainClient = chainClient

	err := w.recoverBatch(chainClient, batch, blocks, recoveryMgr.State(), scopedMgrs)
	require.NoError(t, err)
	require.Equal(t, 1, filterCalls)
	require.Equal(t, expectedSync, w.Manager.SyncedTo())

	select {
	case err := <-addressDone:
		require.NoError(t, err)

	case <-time.After(time.Second):
		t.Fatal("address creation remained blocked after recovery commit")
	}
}

// TestRecoveryBatchCommitsResults verifies that discoveries from multiple backend calls and the synced-to marker are
// committed together after the backend finishes filtering.
func TestRecoveryBatchCommitsResults(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	recoveryMgr, scopedMgrs := newRecoveryTestState(t, w)

	height := w.Manager.SyncedTo().Height + 1
	firstMeta, firstStamp := newRecoveryTestBlock(height)
	secondMeta, secondStamp := newRecoveryTestBlock(height + 1)
	thirdMeta, thirdStamp := newRecoveryTestBlock(height + 2)
	batch := []wtxmgr.BlockMeta{firstMeta, secondMeta, thirdMeta}
	blocks := []*waddrmgr.BlockStamp{firstStamp, secondStamp, thirdStamp}

	scope := waddrmgr.KeyScopeBIP0086

	var (
		filterCalls      int
		recoveredAddrs   []btcutil.Address
		recoveredIndexes []uint32
		relevantTxns     []*wire.MsgTx
	)

	chainClient := &mockChainClient{}
	chainClient.filterBlocksFunc = func(req *chain.FilterBlocksRequest) (*chain.FilterBlocksResponse, error) {
		filterCalls++
		if filterCalls == 3 {
			// A nil response means the backend scanned the rest of the batch without another match.
			return nil, nil //nolint:nilnil
		}

		response, recoveredAddr, recoveredIndex, relevantTx, err := newRecoveryTestResponse(req, scope,
			int64(1000+filterCalls))
		if err != nil {
			return nil, err
		}

		recoveredAddrs = append(recoveredAddrs, recoveredAddr)
		recoveredIndexes = append(recoveredIndexes, recoveredIndex)
		relevantTxns = append(relevantTxns, relevantTx)

		return response, nil
	}

	err := w.recoverBatch(chainClient, batch, blocks, recoveryMgr.State(), scopedMgrs)
	require.NoError(t, err)
	require.Equal(t, 3, filterCalls)
	require.Equal(t, *thirdStamp, w.Manager.SyncedTo())

	var (
		accountProps *waddrmgr.AccountProperties

		addressesUsed []bool
		txDetails     []*wtxmgr.TxDetails
	)

	err = walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		addrMgrNS := tx.ReadBucket(waddrmgrNamespaceKey)
		scopedMgr := scopedMgrs[scope]

		var err error

		accountProps, err = scopedMgr.AccountProperties(addrMgrNS, waddrmgr.DefaultAccountNum)
		if err != nil {
			return err
		}

		for _, recoveredAddr := range recoveredAddrs {
			managedAddr, err := scopedMgr.Address(addrMgrNS, recoveredAddr)
			if err != nil {
				return err
			}

			addressesUsed = append(addressesUsed, managedAddr.Used(addrMgrNS))
		}

		txMgrNS := tx.ReadBucket(wtxmgrNamespaceKey)
		for _, relevantTx := range relevantTxns {
			txHash := relevantTx.TxHash()

			details, err := w.TxStore.TxDetails(txMgrNS, &txHash)
			if err != nil {
				return err
			}

			txDetails = append(txDetails, details)
		}

		return nil
	})
	require.NoError(t, err)
	require.Equal(t, recoveredIndexes[len(recoveredIndexes)-1]+1, accountProps.ExternalKeyCount)
	require.Equal(t, []bool{true, true}, addressesUsed)
	require.Len(t, txDetails, 2)
	require.NotNil(t, txDetails[0])
	require.NotNil(t, txDetails[1])
}

// TestRecoveryBatchFilterErrorRollsBack verifies that an error from a later backend request cannot leave earlier
// discoveries or a newer synced-to marker in walletdb.
func TestRecoveryBatchFilterErrorRollsBack(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	recoveryMgr, scopedMgrs := newRecoveryTestState(t, w)

	initialSync := w.Manager.SyncedTo()
	firstMeta, firstStamp := newRecoveryTestBlock(initialSync.Height + 1)
	secondMeta, secondStamp := newRecoveryTestBlock(initialSync.Height + 2)
	batch := []wtxmgr.BlockMeta{firstMeta, secondMeta}
	blocks := []*waddrmgr.BlockStamp{firstStamp, secondStamp}

	scope := waddrmgr.KeyScopeBIP0086

	var (
		filterCalls  int
		relevantTx   *wire.MsgTx
		initialProps *waddrmgr.AccountProperties
	)

	err := walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		addrMgrNS := tx.ReadBucket(waddrmgrNamespaceKey)

		var err error

		initialProps, err = scopedMgrs[scope].AccountProperties(addrMgrNS, waddrmgr.DefaultAccountNum)

		return err
	})
	require.NoError(t, err)

	errLaterFilter := errors.New("later FilterBlocks call failed")
	chainClient := &mockChainClient{}
	chainClient.filterBlocksFunc = func(req *chain.FilterBlocksRequest) (*chain.FilterBlocksResponse, error) {
		filterCalls++
		if filterCalls > 1 {
			return nil, errLaterFilter
		}

		response, _, _, transaction, err := newRecoveryTestResponse(req, scope, 1000)
		if err != nil {
			return nil, err
		}

		relevantTx = transaction

		return response, nil
	}

	err = w.recoverBatch(chainClient, batch, blocks, recoveryMgr.State(), scopedMgrs)
	require.ErrorIs(t, err, errLaterFilter)
	require.Equal(t, 2, filterCalls)
	require.Equal(t, initialSync, w.Manager.SyncedTo())

	var (
		finalProps *waddrmgr.AccountProperties
		txDetails  *wtxmgr.TxDetails
	)

	err = walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		addrMgrNS := tx.ReadBucket(waddrmgrNamespaceKey)

		var err error

		finalProps, err = scopedMgrs[scope].AccountProperties(addrMgrNS, waddrmgr.DefaultAccountNum)
		if err != nil {
			return err
		}

		txMgrNS := tx.ReadBucket(wtxmgrNamespaceKey)
		txHash := relevantTx.TxHash()
		txDetails, err = w.TxStore.TxDetails(txMgrNS, &txHash)

		return err
	})
	require.NoError(t, err)
	require.Equal(t, initialProps.ExternalKeyCount, finalProps.ExternalKeyCount)
	require.Nil(t, txDetails)
}

// TestListTransactions pins the listing rules and paging. Callers page by the transactions they were shown, so a
// transaction listing several entries or none must not shift what from and count select.
func TestListTransactions(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)

	var receives []string
	for i := range 3 {
		receives = append(receives, fundWallet(t, w, 100_000+int64(i)).Hash.String())
	}

	changeAddr, err := w.NewChangeAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	changeScript, err := txscript.PayToAddrScript(changeAddr)
	require.NoError(t, err)
	changePayment := &wire.MsgTx{TxIn: []*wire.TxIn{{}}, TxOut: []*wire.TxOut{wire.NewTxOut(50_000, changeScript)}}
	rec, err := wtxmgr.NewTxRecordFromMsgTx(changePayment, time.Now())
	require.NoError(t, err)
	block := &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Hash: *testBlockHash, Height: testBlockHeight},
		Time:  time.Unix(1387737310, 0),
	}
	require.NoError(t, walletdb.Update(w.db, func(dbtx walletdb.ReadWriteTx) error {
		return w.addRelevantTx(dbtx, rec, block)
	}))

	twoOutputs, err := w.SendOutputs(
		[]*wire.TxOut{externalTaprootOutput(t, 20_000), externalTaprootOutput(t, 10_000)}, nil, 0, 1, 1000,
		CoinSelectionLargest, "",
	)
	require.NoError(t, err)
	oneOutput := sendTo(t, w, 30_000, 1)
	consolidation, err := w.SendOutputs(
		[]*wire.TxOut{wire.NewTxOut(20_000, changeScript)}, nil, 0, 1, 1000, CoinSelectionLargest, "",
	)
	require.NoError(t, err)

	all, err := w.ListTransactions(0, 100)
	require.NoError(t, err)
	txids := listedTxids(all)
	require.Len(t, txids, 6)
	assert.ElementsMatch(t, []string{twoOutputs.TxHash().String(), oneOutput.TxHash().String()}, txids[:2],
		"unmined first")
	assert.Equal(t, []string{changePayment.TxHash().String(), receives[2], receives[1], receives[0]}, txids[2:],
		"then mined, newest first")

	t.Run("entries", func(t *testing.T) {
		categories := make(map[string][]string)
		for _, entry := range all {
			categories[entry.TxID] = append(categories[entry.TxID], entry.Category)
		}
		assert.Equal(t, []string{"receive"}, categories[changePayment.TxHash().String()])
		assert.Equal(t, []string{"send", "send"}, categories[twoOutputs.TxHash().String()])
		assert.Equal(t, []string{"send"}, categories[oneOutput.TxHash().String()])
		assert.NotContains(t, categories, consolidation.TxHash().String())
		for _, txid := range receives {
			assert.Equal(t, []string{"receive"}, categories[txid], "spent receive %s", txid)
		}
	})

	t.Run("count", func(t *testing.T) {
		for count := 1; count <= len(txids)+1; count++ {
			page, err := w.ListTransactions(0, count)
			require.NoError(t, err)
			assert.Len(t, listedTxids(page), min(count, len(txids)), "count %d", count)
		}
	})

	t.Run("paging", func(t *testing.T) {
		for size := 1; size <= len(txids); size++ {
			var paged []string
			for from := 0; from < len(txids); from += size {
				page, err := w.ListTransactions(from, size)
				require.NoError(t, err)
				paged = append(paged, listedTxids(page)...)
			}
			assert.Equal(t, txids, paged, "pages of %d", size)
		}
	})
}

// listedTxids compacts consecutive repeats only, since a transaction's entries arrive together.
func listedTxids(entries []btcjson.ListTransactionsResult) []string {
	var txids []string
	for _, entry := range entries {
		txids = append(txids, entry.TxID)
	}
	return slices.Compact(txids)
}
