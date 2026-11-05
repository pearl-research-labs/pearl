package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/stretchr/testify/require"
)

// A chain created without Config.Metrics must reorganize and flush without panicking.
func TestNilMetrics(t *testing.T) {
	chain, funding := inputOrderTestChain(t)
	_, _, err := addBlock(chain, funding, nil)
	require.NoError(t, err)

	// The spend makes the side block differ from the main one at the same height.
	spend := testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0)
	branch, _, err := addBlock(chain, funding, []*testhelper.SpendableOut{&spend})
	require.NoError(t, err)
	newTip, _, err := addBlock(chain, branch, nil)
	require.NoError(t, err)
	require.Equal(t, *newTip.Hash(), chain.BestSnapshot().Hash, "the reorg happened")

	require.NoError(t, chain.FlushUtxoCache(FlushRequired))
}
