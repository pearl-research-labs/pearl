package wallet

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	neutrino "github.com/pearl-research-labs/pearl/spv"
	"github.com/pearl-research-labs/pearl/spv/pushtx"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/require"
)

// neutrinoSendClient routes only SendRawTransaction through a real Neutrino
// client. The other backend calls stay mocked so that NotifyReceived never
// starts a rescan, which would wait on peers this harness does not have.
type neutrinoSendClient struct {
	*mockChainClient
	neutrino *chain.NeutrinoClient
}

var _ chain.Interface = (*neutrinoSendClient)(nil)

func (c *neutrinoSendClient) SendRawTransaction(tx *wire.MsgTx,
	allowHighFees bool) (*chainhash.Hash, error) {

	return c.neutrino.SendRawTransaction(tx, allowHighFees)
}

// newPeerlessSPVWallet returns a funded simnet wallet whose broadcasts go
// through a running Neutrino ChainService that has no peers, plus the funding
// outpoint. Simnet is a dev network, so the service never DNS-seeds and stays
// peerless for the whole test.
func newPeerlessSPVWallet(t *testing.T) (*Wallet, wire.OutPoint) {
	t.Helper()

	w, cleanup := testWalletWithParams(t, &chaincfg.SimNetParams)
	t.Cleanup(cleanup)

	dir := t.TempDir()
	db, err := walletdb.Create(
		"bdb", filepath.Join(dir, "neutrino.db"), true,
		defaultDBTimeout, false,
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })

	cs, err := neutrino.NewChainService(neutrino.Config{
		DataDir:     dir,
		Database:    db,
		ChainParams: chaincfg.SimNetParams,
	})
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { require.NoError(t, cs.Stop()) })
	require.Zero(t, cs.ConnectedCount())

	w.chainClient = &neutrinoSendClient{
		mockChainClient: &mockChainClient{},
		neutrino: chain.NewNeutrinoClient(
			&chaincfg.SimNetParams, cs,
		),
	}

	return w, fundWallet(t, w, 100_000)
}

// fundWallet credits the wallet with one confirmed taproot output and returns
// its outpoint.
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
	pkScript, err := txscript.PayToTaprootScript(
		txscript.ComputeTaprootKeyNoScript(privKey.PubKey()),
	)
	require.NoError(t, err)

	return wire.NewTxOut(value, pkScript)
}

// walletTxState reads the unmined transactions and unspent outputs straight
// from the transaction store.
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

// TestGhostPendingSendRegression covers the Desktop "ghost pending" send: a
// spend published while the SPV backend has no peer that requests the
// transaction. Nothing left the machine, so the send must fail and leave no
// local record instead of being reported as sent with the coins locked.
func TestGhostPendingSendRegression(t *testing.T) {
	w, fundingOut := newPeerlessSPVWallet(t)

	start := time.Now()
	tx, err := w.SendOutputs(
		[]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil, 0, 1, 1000,
		CoinSelectionLargest, "",
	)
	elapsed := time.Since(start)

	unmined, unspent := walletTxState(t, w)
	state := fmt.Sprintf("err=%v, unmined=%d, funding output unspent=%v",
		err, len(unmined), hasOutPoint(unspent, fundingOut))

	require.ErrorIs(t, err, chain.ErrTxNotRelayed, "ghost pending: %s", state)
	require.ErrorContains(t, err, "no connected peers")
	require.Nil(t, tx)
	require.Less(t, elapsed, pushtx.DefaultBroadcastTimeout)

	require.Empty(t, unmined, state)
	require.Len(t, unspent, 1, state)
	require.Equal(t, fundingOut, unspent[0].OutPoint, state)
}

func hasOutPoint(credits []wtxmgr.Credit, op wire.OutPoint) bool {
	return slices.ContainsFunc(credits, func(credit wtxmgr.Credit) bool {
		return credit.OutPoint == op
	})
}
