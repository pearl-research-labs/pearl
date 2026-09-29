package wallet

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
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

	send    func(*wire.MsgTx) error
	backEnd string
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

// neutrinoSendClient routes only SendRawTransaction through a real Neutrino client. The other backend calls stay mocked
// so that NotifyReceived never starts a rescan, which would wait on peers this harness does not have.
type neutrinoSendClient struct {
	mockChainClient

	neutrino *chain.NeutrinoClient
}

var _ chain.Interface = (*neutrinoSendClient)(nil)

func (c *neutrinoSendClient) SendRawTransaction(tx *wire.MsgTx, allowHighFees bool) (*chainhash.Hash, error) {
	return c.neutrino.SendRawTransaction(tx, allowHighFees)
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

// TestGhostPendingSendRegression covers the Desktop "ghost pending" send: a spend published while the SPV backend has
// no peer that requests the transaction must fail and leave no local record, instead of being reported as sent with
// the coins locked.
func TestGhostPendingSendRegression(t *testing.T) {
	w, cleanup := testWalletWithParams(t, &chaincfg.SimNetParams)
	t.Cleanup(cleanup)

	// Simnet is a dev network, so the chain service never DNS-seeds and stays peerless.
	dir := t.TempDir()
	db, err := walletdb.Create("bdb", filepath.Join(dir, "neutrino.db"), true, defaultDBTimeout, false)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, db.Close()) })

	cs, err := neutrino.NewChainService(neutrino.Config{DataDir: dir, Database: db, ChainParams: chaincfg.SimNetParams})
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { assert.NoError(t, cs.Stop()) })

	w.chainClient = &neutrinoSendClient{neutrino: chain.NewNeutrinoClient(&chaincfg.SimNetParams, cs)}
	funding := fundWallet(t, w, 100_000)

	tx, err := w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil, 0, 1, 1000, CoinSelectionLargest, "")
	require.ErrorIs(t, err, chain.ErrTxNotRelayed)
	assert.ErrorContains(t, err, "no connected peers")
	assert.Nil(t, tx)

	unmined, unspent := walletTxState(t, w)
	assert.Empty(t, unmined)
	require.Len(t, unspent, 1)
	assert.Equal(t, funding, unspent[0].OutPoint)
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
