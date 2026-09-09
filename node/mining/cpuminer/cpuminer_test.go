// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cpuminer

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/btcsuite/btclog"
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

type testTxSource struct {
	calls   atomic.Int32
	onDescs func()
}

func (*testTxSource) LastUpdated() time.Time               { return time.Time{} }
func (*testTxSource) HaveTransaction(*chainhash.Hash) bool { return false }
func (s *testTxSource) MiningDescs() []*mining.TxDesc {
	s.calls.Add(1)
	if s.onDescs != nil {
		s.onDescs()
	}
	return nil
}

func newTestMiner(t *testing.T, params *chaincfg.Params, source *testTxSource) (*CPUMiner, *blockchain.BlockChain) {
	t.Helper()
	db, err := database.Create("ffldb", t.TempDir(), params.Net)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	timeSource := blockchain.NewMedianTime()
	sigCache := txscript.NewSigCache(100)
	hashCache := txscript.NewHashCache(100)
	chain, err := blockchain.New(&blockchain.Config{
		DB: db, ChainParams: params, TimeSource: timeSource,
		SigCache: sigCache, HashCache: hashCache,
	})
	require.NoError(t, err)
	generator := mining.NewBlkTmplGenerator(
		&mining.Policy{BlockMaxVsize: blockchain.MaxBlockVsize},
		params, source, chain, timeSource, sigCache, hashCache,
	)
	address, err := btcutil.NewAddressTaproot(make([]byte, 32), params)
	require.NoError(t, err)
	miner := New(&Config{
		ChainParams: params, BlockTemplateGenerator: generator,
		MiningAddrs: []btcutil.Address{address},
		ProcessBlock: func(block *btcutil.Block, flags blockchain.BehaviorFlags) (bool, error) {
			_, orphan, err := chain.ProcessBlock(block, flags)
			return orphan, err
		},
		ConnectedCount: func() int32 { return 1 },
		IsCurrent:      func() bool { return true },
	})
	return miner, chain
}

func TestRejectUnsupportedCPUMining(t *testing.T) {
	params := chaincfg.RegressionNetParams
	source := &testTxSource{}
	miner, _ := newTestMiner(t, &params, source)
	miner.SetNumWorkers(4)
	for range 2 {
		require.ErrorIs(t, miner.Start(), blockchain.ErrCPUMiningUnsupported)
		require.ErrorIs(t, miner.StartWithNumWorkers(1), blockchain.ErrCPUMiningUnsupported)
		require.Equal(t, int32(4), miner.NumWorkers())
		hashes, err := miner.GenerateNBlocks(1)
		require.ErrorIs(t, err, blockchain.ErrCPUMiningUnsupported)
		require.Nil(t, hashes)
		require.False(t, miner.IsMining())
		require.False(t, miner.discreteMining)
		require.Nil(t, miner.quit)
	}
	require.Zero(t, source.calls.Load(), "unsupported mining must not build a template")
	require.NoError(t, miner.StartWithNumWorkers(0))
	require.Zero(t, miner.NumWorkers())
}

func TestGenerateNBlocksSimNet(t *testing.T) {
	params := chaincfg.SimNetParams
	params.ReduceMinDifficulty = false
	miner, chain := newTestMiner(t, &params, &testTxSource{})
	hashes, err := miner.GenerateNBlocks(2)
	require.NoError(t, err)
	require.Len(t, hashes, 2)
	require.Equal(t, int32(2), chain.BestSnapshot().Height)
	require.False(t, miner.IsMining())
	require.False(t, miner.discreteMining)
}

func TestStartAndUpdateWorkers(t *testing.T) {
	params := chaincfg.SimNetParams
	miner, _ := newTestMiner(t, &params, &testTxSource{})
	miner.cfg.ConnectedCount = func() int32 { return 0 }
	miner.SetNumWorkers(1)
	require.NoError(t, miner.Start())
	t.Cleanup(miner.Stop)
	for _, workers := range []int32{2, 3, 1, 4, 2} {
		require.NoError(t, miner.StartWithNumWorkers(workers))
		require.Equal(t, workers, miner.NumWorkers())
	}
	// Exercise zero counts in the controller as well as the public stop path.
	miner.updateNumWorkers <- 0
	miner.SetNumWorkers(0)
	require.False(t, miner.IsMining())
	require.Zero(t, miner.NumWorkers())
}

func TestStartWithRequestedWorkerCount(t *testing.T) {
	params := chaincfg.SimNetParams
	source := &testTxSource{}
	miner, _ := newTestMiner(t, &params, source)
	miner.SetNumWorkers(4)
	entered := make(chan struct{}, 4)
	release := make(chan struct{})
	miner.cfg.ConnectedCount = func() int32 {
		entered <- struct{}{}
		<-release
		return 0
	}
	var releaseOnce sync.Once
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
		miner.Stop()
	})
	require.NoError(t, miner.StartWithNumWorkers(1))
	require.Equal(t, int32(1), miner.NumWorkers())
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("requested worker did not start")
	}
	// Hold workers before template creation so even transient over-launch
	// is observable without allowing an unwanted mining attempt.
	select {
	case <-entered:
		t.Fatal("started more than the requested one worker")
	case <-time.After(100 * time.Millisecond):
	}
	require.Zero(t, source.calls.Load())
	releaseOnce.Do(func() { close(release) })
	require.NoError(t, miner.StartWithNumWorkers(0))
	require.False(t, miner.IsMining())
	require.Zero(t, miner.NumWorkers())
}

func TestGenerateNBlocksSolverErrorRestoresState(t *testing.T) {
	params := chaincfg.RegressionNetParams
	params.Fp8ForkHeight = 0
	source := &testTxSource{}
	miner, _ := newTestMiner(t, &params, source)
	v4Params := chaincfg.RegressionNetParams
	source.onDescs = func() {
		// Exercise an unsupported solver after the initial support check.
		// Updating the requested worker count must not block discrete mining.
		require.NoError(t, miner.StartWithNumWorkers(2))
		miner.cfg.ChainParams = &v4Params
	}
	hashes, err := miner.GenerateNBlocks(1)
	require.ErrorIs(t, err, blockchain.ErrCPUMiningUnsupported)
	require.Nil(t, hashes)
	require.Equal(t, int32(1), source.calls.Load())
	require.Equal(t, int32(2), miner.NumWorkers())
	require.False(t, miner.IsMining())
	require.False(t, miner.discreteMining)
	_, err = miner.GenerateNBlocks(1)
	require.ErrorIs(t, err, blockchain.ErrCPUMiningUnsupported)
}

func TestSolveBlockStale(t *testing.T) {
	params := chaincfg.RegressionNetParams
	miner, _ := newTestMiner(t, &params, &testTxSource{})
	solved, err := miner.solveBlock(&wire.MsgBlock{}, 1)
	require.NoError(t, err, "a stale template is retryable, even when its version is unsupported")
	require.False(t, solved)
}

type unsupportedLogWriter struct {
	seen chan struct{}
	once sync.Once
}

func (w *unsupportedLogWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), blockchain.ErrCPUMiningUnsupported.Error()) {
		w.once.Do(func() { close(w.seen) })
	}
	return len(p), nil
}

func TestUnsupportedWorkerWaitsForStop(t *testing.T) {
	params := chaincfg.RegressionNetParams
	params.Fp8ForkHeight = 0
	source := &testTxSource{}
	miner, _ := newTestMiner(t, &params, source)
	v4Params := chaincfg.RegressionNetParams
	source.onDescs = func() { miner.cfg.ChainParams = &v4Params }
	writer := &unsupportedLogWriter{seen: make(chan struct{})}
	logger := btclog.NewBackend(writer).Logger("TEST")
	logger.SetLevel(btclog.LevelError)
	UseLogger(logger)
	t.Cleanup(DisableLog)
	miner.SetNumWorkers(1)
	require.NoError(t, miner.Start())
	t.Cleanup(miner.Stop)
	select {
	case <-writer.seen:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not report unsupported mining")
	}
	require.NoError(t, miner.Start(), "Start remains a no-op when already enabled")
	require.ErrorIs(t, miner.StartWithNumWorkers(2), blockchain.ErrCPUMiningUnsupported)
	require.Equal(t, int32(1), miner.NumWorkers())
	// The controller still owns the parked worker and can shut it down.
	done := make(chan struct{})
	go func() {
		miner.Stop()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("stopping an unsupported worker blocked")
	}
	require.False(t, miner.IsMining())
	require.Equal(t, int32(1), source.calls.Load(), "unsupported mining must not retry")
}
