package wallet

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/btcsuite/btclog"
	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	neutrino "github.com/pearl-research-labs/pearl/spv"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedChainClient is the plain mock backend with a scripted broadcast verdict and backend name.
type scriptedChainClient struct {
	mockChainClient

	send     func(*wire.MsgTx) error
	notify   func([]btcutil.Address) error
	progress func() (*chain.SyncProgress, error)
	backEnd  string
}

var _ chain.Interface = (*scriptedChainClient)(nil)

func (c *scriptedChainClient) SendRawTransaction(tx *wire.MsgTx, _ bool) (*chainhash.Hash, error) {
	if c.send != nil {
		if err := c.send(tx); err != nil {
			return nil, err
		}
	}
	hash := tx.TxHash()
	return &hash, nil
}

func (c *scriptedChainClient) BackEnd() string {
	if c.backEnd != "" {
		return c.backEnd
	}
	return c.mockChainClient.BackEnd()
}

func (c *scriptedChainClient) NotifyReceived(addrs []btcutil.Address) error {
	if c.notify != nil {
		return c.notify(addrs)
	}
	return c.mockChainClient.NotifyReceived(addrs)
}

func (c *scriptedChainClient) SyncProgress() (*chain.SyncProgress, error) {
	if c.progress != nil {
		return c.progress()
	}
	return &chain.SyncProgress{Connections: 1}, nil
}

// neutrinoSendClient routes broadcast and sync progress through a real Neutrino client. The other backend calls stay
// mocked so that NotifyReceived never starts a rescan, which would wait on peers this harness does not have.
type neutrinoSendClient struct {
	mockChainClient

	neutrino *chain.NeutrinoClient
}

var _ chain.Interface = (*neutrinoSendClient)(nil)

func (c *neutrinoSendClient) SendRawTransaction(tx *wire.MsgTx, allowHighFees bool) (*chainhash.Hash, error) {
	return c.neutrino.SendRawTransaction(tx, allowHighFees)
}

func (c *neutrinoSendClient) BackEnd() string {
	return "neutrino"
}

func (c *neutrinoSendClient) SyncProgress() (*chain.SyncProgress, error) {
	return c.neutrino.SyncProgress()
}

// fundWallet credits the wallet with one confirmed taproot output and returns its outpoint.
func fundWallet(t *testing.T, w *Wallet, value int64) wire.OutPoint {
	t.Helper()

	addr, err := w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	pkScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	fundingTx := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{wire.NewTxOut(value, pkScript)},
	}
	addUtxo(t, w, fundingTx)

	return wire.OutPoint{Hash: fundingTx.TxHash(), Index: 0}
}

// externalTaprootOutput pays a key the wallet does not own.
func externalTaprootOutput(t *testing.T, value int64) *wire.TxOut {
	t.Helper()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	pkScript, err := txscript.PayToTaprootScript(txscript.ComputeTaprootKeyNoScript(privKey.PubKey()))
	require.NoError(t, err)

	return wire.NewTxOut(value, pkScript)
}

// sendTo spends from the wallet to an external output. minconf 0 lets a second call chain onto the first one's
// unconfirmed change.
func sendTo(t *testing.T, w *Wallet, value int64, minconf int32) *wire.MsgTx {
	t.Helper()

	tx, err := w.SendOutputs(
		[]*wire.TxOut{externalTaprootOutput(t, value)}, nil, 0, minconf, 1000, CoinSelectionLargest, "",
	)
	require.NoError(t, err)
	return tx
}

// walletTxState reads the unmined transactions and unspent outputs straight from the transaction store.
func walletTxState(t *testing.T, w *Wallet) ([]*wire.MsgTx, []wtxmgr.Credit) {
	t.Helper()

	var (
		unmined []*wire.MsgTx
		unspent []wtxmgr.Credit
	)
	err := walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)
		var err error
		unmined, err = w.TxStore.UnminedTxs(ns)
		if err != nil {
			return err
		}
		unspent, err = w.TxStore.UnspentOutputs(ns)
		return err
	})
	require.NoError(t, err)

	return unmined, unspent
}

func hasOutPoint(credits []wtxmgr.Credit, op wire.OutPoint) bool {
	return slices.ContainsFunc(credits, func(credit wtxmgr.Credit) bool {
		return credit.OutPoint == op
	})
}

// TestGhostPendingSendRegression covers the Desktop "ghost pending" send: a peerless SPV backend must fail before
// signing and leave no pending record.
func TestGhostPendingSendRegression(t *testing.T) {
	w, cleanup := testWalletWithParams(t, &chaincfg.SimNetParams)
	t.Cleanup(cleanup)

	// Simnet is a dev network, so the chain service never DNS-seeds and stays peerless.
	dir := t.TempDir()
	db, err := walletdb.Create("bdb", filepath.Join(dir, "neutrino.db"), true, defaultDBTimeout, false)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, db.Close()) })

	cs, err := neutrino.NewChainService(neutrino.Config{
		DataDir: dir, Database: db, ChainParams: chaincfg.SimNetParams,
	})
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { assert.NoError(t, cs.Stop()) })

	w.chainClient = &neutrinoSendClient{neutrino: chain.NewNeutrinoClient(&chaincfg.SimNetParams, cs)}
	funding := fundWallet(t, w, 100_000)

	tx, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil,
		0, 1, 1000, CoinSelectionLargest, "")
	require.ErrorIs(t, err, chain.ErrTxNotRelayed)
	assert.ErrorContains(t, err, "no connected peers")
	assert.Nil(t, tx)

	unmined, unspent := walletTxState(t, w)
	assert.Empty(t, unmined)
	require.Len(t, unspent, 1)
	assert.Equal(t, funding, unspent[0].OutPoint)
}

func TestSPVSendPreflight(t *testing.T) {
	progressErr := errors.New("SPV sync state unavailable")
	for _, test := range []struct {
		name    string
		err     error
		wantErr error
	}{
		{"offline", nil, chain.ErrTxNotRelayed},
		{"unavailable", progressErr, progressErr},
	} {
		t.Run(test.name, func(t *testing.T) {
			w, cleanup := testWallet(t)
			t.Cleanup(cleanup)
			funding := fundWallet(t, w, 100_000)
			w.chainClient = &scriptedChainClient{
				backEnd: "neutrino",
				progress: func() (*chain.SyncProgress, error) {
					return &chain.SyncProgress{}, test.err
				},
				notify: func([]btcutil.Address) error {
					t.Error("preflight failure must not reach change registration after signing")
					return nil
				},
				send: func(*wire.MsgTx) error {
					t.Error("preflight failure must not reach broadcast")
					return nil
				},
			}
			tx, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil,
				0, 1, 1000, CoinSelectionLargest, "")
			require.ErrorIs(t, err, test.wantErr)
			assert.Nil(t, tx)
			unmined, unspent := walletTxState(t, w)
			assert.Empty(t, unmined)
			require.Len(t, unspent, 1)
			assert.Equal(t, funding, unspent[0].OutPoint)
		})
	}
}

// TestResendAfterRescanBackendGate checks that the post-rescan resend reaches a full-node backend and never an SPV one.
func TestResendAfterRescanBackendGate(t *testing.T) {
	tests := []struct {
		backEnd     string
		wantResends int
	}{
		{backEnd: "pearld", wantResends: 1},
		{backEnd: "neutrino", wantResends: 0},
	}
	for _, tt := range tests {
		t.Run(tt.backEnd, func(t *testing.T) {
			w, cleanup := testWallet(t)
			t.Cleanup(cleanup)
			fundWallet(t, w, 100_000)
			sendTo(t, w, 50_000, 1)

			var resends int
			w.chainClient = &scriptedChainClient{
				backEnd: tt.backEnd,
				send: func(*wire.MsgTx) error {
					resends++
					return nil
				},
			}
			w.resendUnminedTxsAfterRescan()

			assert.Equal(t, tt.wantResends, resends)
		})
	}
}

// A peer's rejection or claimed confirmation cannot erase a signed payment or return a send failure that encourages the
// caller to create a second payment.
func TestSPVSendRetainsPayment(t *testing.T) {
	// The package logger is shared, so this test must remain serial.
	var logOutput bytes.Buffer
	originalLog := log
	log = btclog.NewBackend(&logOutput).Logger("WLLT")
	log.SetLevel(btclog.LevelWarn)
	t.Cleanup(func() { log = originalLog })

	verdicts := []error{
		chain.ErrTxNotRelayed,
		chain.ErrMissingInputs,
		chain.ErrInsufficientFee,
		chain.ErrTxAlreadyKnown,
		chain.ErrTxAlreadyConfirmed,
		chain.ErrTxAlreadyInMempool,
		errors.New("unknown peer rejection\nforged log\r\x1b[31m"),
	}
	for _, verdict := range verdicts {
		t.Run(verdict.Error(), func(t *testing.T) {
			logOutput.Reset()
			w, cleanup := testWallet(t)
			t.Cleanup(func() {
				w.Stop()
				w.WaitForShutdown()
				cleanup()
			})
			funding := fundWallet(t, w, 300_000_000)
			fundWallet(t, w, 200_000_000)
			var published *wire.MsgTx
			w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(tx *wire.MsgTx) error {
				published = tx.Copy()
				return verdict
			}}

			tx, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 111_111_100)}, nil,
				0, 1, 1000, CoinSelectionLargest, "")
			require.NoError(t, err,
				"an uncertain broadcast must return its txid instead of a fresh-send failure")
			require.NotNil(t, tx)
			output := logOutput.String()
			assert.Equal(t, 1, strings.Count(output, "\n"),
				"a peer rejection must produce only one physical log line")
			assert.Contains(t, output, "[WRN] WLLT: Retaining pending transaction "+tx.TxHash().String())
			assert.Contains(t, output, strconv.Quote(verdict.Error()))
			assert.NotContains(t, output, "\r")
			assert.NotContains(t, output, "\x1b")
			require.NotNil(t, published)
			assert.Equal(t, published.TxHash(), tx.TxHash())
			assert.Equal(t, funding, tx.TxIn[0].PreviousOutPoint)
			assert.NotEmpty(t, tx.TxIn[0].Witness)

			unmined, unspent := walletTxState(t, w)
			require.Len(t, unmined, 1)
			assert.Equal(t, tx.TxHash(), unmined[0].TxHash())
			assert.False(t, hasOutPoint(unspent, funding))

			// The next queue item cannot consume the first payment's input.
			w.chainClient = &scriptedChainClient{backEnd: "neutrino"}
			next := sendTo(t, w, 77_777_700, 1)
			assert.NotEqual(t, funding, next.TxIn[0].PreviousOutPoint)
			var rebroadcast []*wire.MsgTx
			w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(tx *wire.MsgTx) error {
				rebroadcast = append(rebroadcast, tx.Copy())
				return nil
			}}
			_, err = w.RebroadcastTransaction(tx.TxHash())
			require.NoError(t, err)
			require.Len(t, rebroadcast, 1)
			assert.Equal(t, tx, rebroadcast[0], "retry by txid must reuse the original signed transaction")
		})
	}
}

func TestSPVRepublishKeepsPendingDescendants(t *testing.T) {
	for _, verdict := range []error{chain.ErrTxNotRelayed, chain.ErrMissingInputs, chain.ErrTxAlreadyConfirmed} {
		t.Run(verdict.Error(), func(t *testing.T) {
			pc := newPendingChain(t)
			pc.client.backEnd = "neutrino"
			pc.client.send = func(*wire.MsgTx) error { return verdict }

			require.NoError(t, pc.w.PublishTransaction(pc.parent, ""))
			unmined, unspent := walletTxState(t, pc.w)
			require.Len(t, unmined, 2)
			assert.False(t, hasOutPoint(unspent, pc.funding))
			assert.False(t, hasOutPoint(unspent, pc.child.TxIn[0].PreviousOutPoint))
		})
	}
}

func TestSPVPublishExternalTransactionNotRelayed(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	created, err := w.CreateSimpleTx(nil, 0, []*wire.TxOut{externalTaprootOutput(t, 50_000)},
		1, 1000, CoinSelectionLargest, false)
	require.NoError(t, err)

	// The raw caller already has the signed bytes and may have relayed them elsewhere, even though this wallet has
	// no transaction record yet.
	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrTxNotRelayed
	}}
	require.NoError(t, w.PublishTransaction(created.Tx, ""))
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 1)
	assert.Equal(t, created.Tx.TxHash(), unmined[0].TxHash())
	assert.False(t, hasOutPoint(unspent, funding))
}

func TestFullNodeSendRejection(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	w.chainClient = &scriptedChainClient{backEnd: "pearld", send: func(*wire.MsgTx) error {
		return chain.ErrMissingInputs
	}}
	tx, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil,
		0, 1, 1000, CoinSelectionLargest, "")
	require.ErrorIs(t, err, chain.ErrMissingInputs)
	assert.Nil(t, tx)
	unmined, unspent := walletTxState(t, w)
	assert.Empty(t, unmined)
	assert.True(t, hasOutPoint(unspent, funding))
}

func TestSPVPendingPaymentSurvivesReopen(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrMissingInputs
	}}
	sent := sendTo(t, w, 50_000, 1)

	// Reopen a committed database snapshot with fresh wallet and store objects.
	path := filepath.Join(t.TempDir(), "wallet.db")
	file, err := os.Create(path)
	require.NoError(t, err)
	require.NoError(t, w.db.Copy(file))
	require.NoError(t, file.Close())
	db, err := walletdb.Open("bdb", path, true, defaultDBTimeout, false)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, db.Close()) })
	reopened, err := Open(db, []byte("hello"), nil, w.chainParams, 0)
	require.NoError(t, err)
	t.Cleanup(reopened.Manager.Close)
	reopened.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrTxNotRelayed
	}}
	require.NoError(t, reopened.PublishTransaction(sent, ""))
	unmined, unspent := walletTxState(t, reopened)
	require.Len(t, unmined, 1)
	var originalBytes, restoredBytes bytes.Buffer
	require.NoError(t, sent.Serialize(&originalBytes))
	require.NoError(t, unmined[0].Serialize(&restoredBytes))
	assert.Equal(t, originalBytes.Bytes(), restoredBytes.Bytes())
	assert.False(t, hasOutPoint(unspent, funding))
}

func TestSPVRebroadcastDuringInitialSend(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	initialTx := make(chan *wire.MsgTx, 1)
	release := make(chan struct{})
	var releaseOnce sync.Once
	finishInitial := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(finishInitial)
	var broadcasts atomic.Int32
	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(tx *wire.MsgTx) error {
		if broadcasts.Add(1) == 1 {
			initialTx <- tx.Copy()
			<-release
			return chain.ErrTxNotRelayed
		}
		return nil
	}}
	output := externalTaprootOutput(t, 50_000)
	sendResult := make(chan error, 1)
	go func() {
		_, err := w.SendOutputs([]*wire.TxOut{output}, nil, 0, 1, 1000, CoinSelectionLargest, "")
		sendResult <- err
	}()
	var tx *wire.MsgTx
	select {
	case tx = <-initialTx:
	case <-time.After(5 * time.Second):
		t.Fatal("initial send did not reach backend")
	}
	rebroadcastResult := make(chan error, 1)
	go func() {
		_, err := w.RebroadcastTransaction(tx.TxHash())
		rebroadcastResult <- err
	}()
	// A concurrent rebroadcast hands the bytes to a peer before the initial attempt reports no relay. Neither
	// attempt can erase the payment.
	select {
	case err := <-rebroadcastResult:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("rebroadcast did not finish")
	}
	finishInitial()
	require.NoError(t, <-sendResult)
	assert.EqualValues(t, 2, broadcasts.Load())
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 1)
	assert.Equal(t, tx.TxHash(), unmined[0].TxHash())
	assert.False(t, hasOutPoint(unspent, funding))
}

func TestSPVNotificationBeforeRelayResult(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrTxNotRelayed
	}}
	client := w.NtfnServer.TransactionNotifications()
	t.Cleanup(client.Done)
	output := externalTaprootOutput(t, 50_000)
	result := make(chan error, 1)
	go func() {
		_, err := w.SendOutputs([]*wire.TxOut{output}, nil, 0, 1, 1000, CoinSelectionLargest, "")
		result <- err
	}()
	var notification *TransactionNotifications
	select {
	case notification = <-client.C:
	case <-time.After(5 * time.Second):
		t.Fatal("pending payment was not notified")
	}
	require.Len(t, notification.UnminedTransactions, 1)
	var exposed wire.MsgTx
	require.NoError(t, exposed.Deserialize(bytes.NewReader(notification.UnminedTransactions[0].Transaction)))
	require.NotEmpty(t, exposed.TxIn[0].Witness, "a subscriber already has signed bytes it can relay")
	require.NoError(t, <-result)
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 1)
	assert.Equal(t, exposed.TxHash(), unmined[0].TxHash())
	assert.False(t, hasOutPoint(unspent, funding))
}

func TestSPVNotificationRegistrationFailure(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	var notified, broadcasts int
	w.chainClient = &scriptedChainClient{
		backEnd: "neutrino",
		notify: func([]btcutil.Address) error {
			notified++
			if notified == 1 {
				return nil // Register change during transaction creation.
			}
			// Losing the active backend after storing the transaction must not turn a retained payment into
			// an ordinary send failure.
			w.chainClientLock.Lock()
			w.chainClient = nil
			w.chainClientLock.Unlock()
			return errors.New("rescan stopped after transaction was recorded")
		},
		send: func(*wire.MsgTx) error {
			broadcasts++
			return chain.ErrTxNotRelayed
		},
	}
	sent, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil,
		0, 1, 1000, CoinSelectionLargest, "")
	require.NoError(t, err)
	require.NotNil(t, sent)
	assert.Equal(t, 2, notified)
	assert.Equal(t, 1, broadcasts, "use the captured SPV backend for the retained payment")
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 1)
	assert.Equal(t, sent.TxHash(), unmined[0].TxHash())
	assert.False(t, hasOutPoint(unspent, funding))
}
