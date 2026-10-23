package neutrino

import (
	"context"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/pushtx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSendTransactionNoPeers pins the broadcast contract when no peer can be asked for the transaction: the caller must
// learn at once that nothing was relayed instead of being told the broadcast succeeded.
func TestSendTransactionNoPeers(t *testing.T) {
	// Simnet is a dev network, so nothing DNS-seeds and the service stays peerless.
	cs, err := NewChainService(testChainServiceConfig(t))
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { assert.NoError(t, cs.Stop()) })

	start := time.Now()
	err = cs.SendTransaction(wire.NewMsgTx(wire.TxVersion))

	assert.Less(t, time.Since(start), time.Second)
	assert.Truef(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed), "got %v", err)
	assert.ErrorContains(t, err, "no connected peers")
}
