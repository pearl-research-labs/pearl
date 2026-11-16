//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cpuminer

import (
	"math"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/mining"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

type emptyTxSource struct{}

func (emptyTxSource) LastUpdated() time.Time               { return time.Time{} }
func (emptyTxSource) MiningDescs() []*mining.TxDesc        { return nil }
func (emptyTxSource) HaveTransaction(*chainhash.Hash) bool { return false }

// newTestMiner builds a CPU miner on a fresh chain holding only the genesis block. Solved blocks go through the chain's
// full ProcessBlock validation.
func newTestMiner(t *testing.T, params *chaincfg.Params) (*CPUMiner, *blockchain.BlockChain) {
	t.Helper()

	db, err := database.Create("ffldb", t.TempDir(), params.Net)
	require.NoError(t, err)
	t.Cleanup(func() { db.Close() })

	timeSource := blockchain.NewMedianTime()
	sigCache := txscript.NewSigCache(100)
	hashCache := txscript.NewHashCache(100)
	chain, err := blockchain.New(&blockchain.Config{
		DB:          db,
		ChainParams: params,
		TimeSource:  timeSource,
		SigCache:    sigCache,
		HashCache:   hashCache,
	})
	require.NoError(t, err)

	addr, err := btcutil.NewAddressTaproot(make([]byte, 32), params)
	require.NoError(t, err)
	miner := New(&Config{
		ChainParams: params,
		BlockTemplateGenerator: mining.NewBlkTmplGenerator(
			&mining.Policy{BlockMaxVsize: blockchain.MaxBlockVsize},
			params, emptyTxSource{}, chain, timeSource, sigCache, hashCache,
		),
		MiningAddrs: []btcutil.Address{addr},
		ProcessBlock: func(block *btcutil.Block, flags blockchain.BehaviorFlags) (bool, error) {
			_, isOrphan, err := chain.ProcessBlock(block, flags)
			return isOrphan, err
		},
		ConnectedCount: func() int32 { return 1 },
		IsCurrent:      func() bool { return true },
	})
	t.Cleanup(miner.Stop)
	return miner, chain
}

func TestGenerateNBlocks(t *testing.T) {
	regTest := chaincfg.RegressionNetParams

	// Every height requires V1, and only V4 certificates can be mined.
	noForks := chaincfg.RegressionNetParams
	noForks.MoEForkHeight = 0
	noForks.RankPenaltyForkHeight = 0
	noForks.SaltedSeedForkHeight = 0
	noForks.Fp8ForkHeight = 0

	for _, test := range []struct {
		name      string
		params    *chaincfg.Params
		numBlocks uint32
		wantErr   string
	}{
		{
			name:      "regtest mines V4 blocks",
			params:    &regTest,
			numBlocks: 2,
		},
		{
			name:      "certificate cannot be produced",
			params:    &noForks,
			numBlocks: 1,
			wantErr:   "requires certificate version 1",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			miner, chain := newTestMiner(t, test.params)

			hashes, err := miner.GenerateNBlocks(test.numBlocks)
			require.False(t, miner.IsMining())
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				require.Zero(t, chain.BestSnapshot().Height)
				return
			}
			require.NoError(t, err)
			require.Len(t, hashes, int(test.numBlocks))
			require.Equal(t, int32(len(hashes)), chain.BestSnapshot().Height)
			for i, hash := range hashes {
				block, err := chain.BlockByHeight(int32(i + 1))
				require.NoError(t, err)
				require.Equal(t, hash, block.Hash())
				require.Equal(t, wire.CertificateVersionV4, block.MsgBlock().BlockCertificate().Version())
				require.NoError(t, blockchain.CheckProofOfWork(block, test.params.PowLimit),
					"the stored certificate must still verify")
			}
		})
	}
}

// TestCPUMinerStop checks that Stop ends both mining modes and leaves the miner ready for a new run.
func TestCPUMinerStop(t *testing.T) {
	params := chaincfg.SimNetParams
	for _, test := range []struct {
		name string
		// start begins mining and returns the channel that delivers the mining call's result, if it blocks.
		start func(*CPUMiner) <-chan error
	}{
		{
			name: "generate",
			start: func(m *CPUMiner) <-chan error {
				errc := make(chan error, 1)
				go func() {
					_, err := m.GenerateNBlocks(math.MaxUint32)
					errc <- err
				}()
				return errc
			},
		},
		{
			name: "setgenerate",
			start: func(m *CPUMiner) <-chan error {
				m.Start()
				return nil
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			miner, chain := newTestMiner(t, &params)

			errc := test.start(miner)
			require.Eventually(t, func() bool { return chain.BestSnapshot().Height > 0 },
				time.Minute, 10*time.Millisecond)
			miner.Stop()
			require.False(t, miner.IsMining())
			if errc != nil {
				require.ErrorIs(t, <-errc, errMinerStopped)
			}

			height := chain.BestSnapshot().Height
			_, err := miner.GenerateNBlocks(1)
			require.NoError(t, err)
			require.Equal(t, height+1, chain.BestSnapshot().Height)
		})
	}
}
