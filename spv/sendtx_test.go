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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSendTransactionNoPeers pins the broadcast contract when no peer can be asked for the transaction: the caller must
// learn at once that nothing was relayed instead of being told the broadcast succeeded.
func TestSendTransactionNoPeers(t *testing.T) {
	// Simnet is a dev network, so nothing DNS-seeds and the service stays peerless.
	dir := t.TempDir()
	db, err := walletdb.Create("bdb", filepath.Join(dir, "neutrino.db"), true, 10*time.Second, false)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, db.Close()) })

	cs, err := NewChainService(Config{DataDir: dir, Database: db, ChainParams: chaincfg.SimNetParams})
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { assert.NoError(t, cs.Stop()) })

	start := time.Now()
	err = cs.SendTransaction(wire.NewMsgTx(wire.TxVersion))

	assert.Less(t, time.Since(start), time.Second)
	assert.Truef(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed), "got %v", err)
	assert.ErrorContains(t, err, "no connected peers")
}
