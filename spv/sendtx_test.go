package neutrino

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/pushtx"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	_ "github.com/pearl-research-labs/pearl/wallet/walletdb/bdb"
	"github.com/stretchr/testify/require"
)

// startPeerlessChainService runs a ChainService on simnet with no peers.
// Simnet is a dev network, so nothing DNS-seeds and the service stays
// peerless for the whole test.
func startPeerlessChainService(t *testing.T) *ChainService {
	t.Helper()

	dir := t.TempDir()
	db, err := walletdb.Create(
		"bdb", filepath.Join(dir, "neutrino.db"), true, 10*time.Second,
		false,
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })

	cs, err := NewChainService(Config{
		DataDir:     dir,
		Database:    db,
		ChainParams: chaincfg.SimNetParams,
	})
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { require.NoError(t, cs.Stop()) })

	return cs
}

func testTx() *wire.MsgTx {
	tx := wire.NewMsgTx(wire.TxVersion)
	tx.AddTxIn(&wire.TxIn{PreviousOutPoint: wire.OutPoint{Index: 0}})
	tx.AddTxOut(&wire.TxOut{Value: 1000, PkScript: []byte{0x51}})

	return tx
}

// TestSendTransactionNoPeers pins the broadcast contract when no peer can be
// asked for the transaction: the caller must learn that nothing was relayed
// instead of being told the broadcast succeeded.
func TestSendTransactionNoPeers(t *testing.T) {
	cs := startPeerlessChainService(t)
	require.Zero(t, cs.ConnectedCount())

	start := time.Now()
	err := cs.SendTransaction(testTx())
	elapsed := time.Since(start)

	require.Truef(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed),
		"SendTransaction with zero peers returned err=%v", err)
	require.ErrorContains(t, err, "no connected peers")
	require.Less(t, elapsed, time.Second)
}

// TestTransactionInvAnnouncesByTxid ensures a broadcast is announced with
// MSG_TX. BIP-144 allows the witness inventory types only in getdata, and
// peers that follow it never request a transaction announced otherwise.
func TestTransactionInvAnnouncesByTxid(t *testing.T) {
	t.Parallel()

	tx := testTx()
	inv := newTransactionInv(tx)

	require.Len(t, inv.InvList, 1)
	require.Equal(t, wire.InvTypeTx, inv.InvList[0].Type)
	require.Equal(t, tx.TxHash(), inv.InvList[0].Hash)
}

// TestNotRelayedErrorReason ensures the reason distinguishes having no peers
// from having peers that did not ask, since operators triage them differently.
func TestNotRelayedErrorReason(t *testing.T) {
	t.Parallel()

	txHash := testTx().TxHash()

	err := notRelayedError(txHash, 0)
	require.True(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed))
	require.ErrorContains(t, err, "no connected peers")
	require.ErrorContains(t, err, txHash.String())

	err = notRelayedError(txHash, 3)
	require.True(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed))
	require.ErrorContains(t, err, "none of 3 connected peers")
	require.ErrorContains(t, err, txHash.String())
}
