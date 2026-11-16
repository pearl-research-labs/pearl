//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cpuminer

import (
	"math"
	"sync/atomic"
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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type emptyTxSource struct{}

func (emptyTxSource) LastUpdated() time.Time               { return time.Time{} }
func (emptyTxSource) MiningDescs() []*mining.TxDesc        { return nil }
func (emptyTxSource) HaveTransaction(*chainhash.Hash) bool { return false }

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
	simNet := chaincfg.SimNetParams

	noForks := chaincfg.RegressionNetParams
	noForks.MoEForkHeight = 0
	noForks.RankPenaltyForkHeight = 0
	noForks.SaltedSeedForkHeight = 0
	noForks.Fp8ForkHeight = 0

	tests := []struct {
		name             string
		params           *chaincfg.Params
		numBlocks        uint32
		checkPoW         bool
		invalidateAndGen bool
		wantErr          string
	}{
		{
			name:      "regtest mines V4 blocks",
			params:    &regTest,
			numBlocks: 2,
			checkPoW:  true,
		},
		{
			name:             "replaces invalidated tip ahead of wall clock",
			params:           &simNet,
			numBlocks:        10,
			invalidateAndGen: true,
		},
		{
			name:      "certificate cannot be produced",
			params:    &noForks,
			numBlocks: 1,
			wantErr:   "requires certificate version 1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			miner, chain := newTestMiner(t, tt.params)

			hashes, err := miner.GenerateNBlocks(tt.numBlocks)
			assert.False(t, miner.IsMining())
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				assert.Zero(t, chain.BestSnapshot().Height)
				return
			}
			require.NoError(t, err)
			require.Len(t, hashes, int(tt.numBlocks))
			assert.Equal(t, int32(len(hashes)), chain.BestSnapshot().Height)
			for i, hash := range hashes {
				block, err := chain.BlockByHeight(int32(i + 1))
				require.NoError(t, err)
				assert.Equal(t, hash, block.Hash())
				assert.Equal(t, wire.CertificateVersionV4, block.MsgBlock().BlockCertificate().Version())
				if tt.checkPoW {
					assert.NoError(t, blockchain.CheckProofOfWork(block, tt.params.PowLimit))
				}
			}

			if tt.invalidateAndGen {
				invalidated := hashes[len(hashes)-1]
				require.NoError(t, chain.InvalidateBlock(invalidated))
				replacement, err := miner.GenerateNBlocks(1)
				require.NoError(t, err)
				require.Len(t, replacement, 1)
				assert.NotEqual(t, invalidated, replacement[0])
				assert.Equal(t, int32(tt.numBlocks), chain.BestSnapshot().Height)
			}
		})
	}
}

type gatedSolver struct {
	calls   atomic.Int32
	entered chan struct{}
	release chan struct{}
}

func (s *gatedSolver) solve(header, parent *wire.BlockHeader, params *chaincfg.Params,
	height int32) (*wire.CertificateV4, error) {

	if s.calls.Add(1) == 1 {
		close(s.entered)
		<-s.release
	}
	return blockchain.SolveBlock(header, parent, params, height)
}

func TestCPUMinerStop(t *testing.T) {
	params := chaincfg.SimNetParams
	tests := []struct {
		name  string
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			miner, chain := newTestMiner(t, &params)
			solver := &gatedSolver{entered: make(chan struct{}), release: make(chan struct{})}
			miner.solve = solver.solve
			miner.SetNumWorkers(4)

			errc := tt.start(miner)
			<-solver.entered
			miner.SetNumWorkers(3)
			miner.SetNumWorkers(4)
			// Give the remaining workers time to queue behind solveLock while the first proof blocks.
			time.Sleep(50 * time.Millisecond)
			r := miner.currentRun()
			require.NotNil(t, r)

			stopped := make(chan struct{})
			go func() {
				miner.Stop()
				close(stopped)
			}()
			<-r.quit
			assert.True(t, miner.IsMining(), "IsMining must not block while Stop waits on a proof")
			miner.SetNumWorkers(2)
			close(solver.release)
			<-stopped

			assert.False(t, miner.IsMining())
			assert.Equal(t, int32(1), solver.calls.Load(), "only the proof in progress may run after Stop")
			if errc != nil {
				assert.ErrorIs(t, <-errc, errMinerStopped)
			}

			height := chain.BestSnapshot().Height
			_, err := miner.GenerateNBlocks(1)
			require.NoError(t, err)
			assert.Equal(t, height+1, chain.BestSnapshot().Height)
		})
	}
}

func TestCPUMinerQueuedWorkerSeesNewTip(t *testing.T) {
	params := chaincfg.SimNetParams
	miner, chain := newTestMiner(t, &params)
	var supersededProofs atomic.Int32
	miner.solve = func(header, parent *wire.BlockHeader, p *chaincfg.Params,
		height int32) (*wire.CertificateV4, error) {

		time.Sleep(10 * time.Millisecond)
		if header.PrevBlock != chain.BestSnapshot().Hash {
			supersededProofs.Add(1)
		}
		return blockchain.SolveBlock(header, parent, p, height)
	}
	miner.SetNumWorkers(4)

	miner.Start()
	require.Eventually(t, func() bool { return chain.BestSnapshot().Height >= 5 }, time.Minute, 10*time.Millisecond)
	miner.Stop()
	assert.Zero(t, supersededProofs.Load(), "proofs whose tip changed while they ran")
}
