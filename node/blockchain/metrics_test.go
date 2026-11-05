package blockchain

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/stretchr/testify/require"
)

// utxoCacheReports keeps the last UTXO cache occupancy reported.
type utxoCacheReports struct {
	noopMetrics
	count    int
	entries  int
	maxBytes uint64
}

func (r *utxoCacheReports) UtxoCacheChanged(entries int, _, maxMemoryBytes uint64) {
	r.count++
	r.entries = entries
	r.maxBytes = maxMemoryBytes
}

// The UTXO cache gauges come only from these reports, so a block connected without a flush must still
// produce one.
func TestUtxoCacheReportedOnConnect(t *testing.T) {
	chain, funding := inputOrderTestChain(t)
	reports := &utxoCacheReports{}
	chain.metrics = reports

	_, _, err := addBlock(chain, funding, nil)
	require.NoError(t, err)

	require.Positive(t, reports.count)
	require.Positive(t, reports.entries, "the connected block's outputs are cached")
	require.Equal(t, chain.utxoCache.maxTotalMemoryUsage, reports.maxBytes)
}

// IsCurrent and IsCurrentSnapshot read the tip through different paths, so they must reach the same
// verdict.
func TestIsCurrentSnapshot(t *testing.T) {
	chain, _ := inputOrderTestChain(t)
	require.Equal(t, chain.IsCurrent(), chain.IsCurrentSnapshot())

	now := chain.timeSource.AdjustedTime()
	require.True(t, chain.isCurrentAt(0, now.Unix()))
	require.False(t, chain.isCurrentAt(0, now.Add(-25*time.Hour).Unix()))
}

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
