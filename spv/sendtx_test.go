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

	tx := testTx()
	start := time.Now()
	err := cs.SendTransaction(tx)
	elapsed := time.Since(start)

	require.Truef(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed),
		"SendTransaction with zero peers returned err=%v", err)
	require.ErrorContains(t, err, "no connected peers")
	require.Less(t, elapsed, time.Second)

	_, ok := cs.LastRelayed(tx.TxHash())
	require.False(t, ok, "an unrelayed announcement must leave no evidence")
}
