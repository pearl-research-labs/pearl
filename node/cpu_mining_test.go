// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/btcsuite/btclog"
	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func newCPUMiningTestServer(t *testing.T, params *chaincfg.Params, generate bool) (*server, error) {
	t.Helper()
	// These constructor tests do not initialize the application's log rotator.
	for _, logger := range subsystemLoggers {
		previousLevel := logger.Level()
		logger.SetLevel(btclog.LevelOff)
		t.Cleanup(func() { logger.SetLevel(previousLevel) })
	}
	address, err := btcutil.NewAddressTaproot(make([]byte, 32), params)
	require.NoError(t, err)
	previousConfig := cfg
	cfg = &config{
		DataDir: t.TempDir(), DisableListen: true, DisableRPC: true,
		NoCFilters: true, MaxPeers: 8, SigCacheMaxSize: 100,
		BlockMaxVsize: blockchain.MaxBlockVsize,
		SimNet:        params.Net == wire.SimNet, Generate: generate,
		miningAddrs: []btcutil.Address{address},
	}
	t.Cleanup(func() { cfg = previousConfig })
	db, err := database.Create("ffldb", t.TempDir(), params.Net)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	return newServer(nil, nil, nil, db, params, nil)
}

func TestCPUMiningRPCRejectsUnsupported(t *testing.T) {
	params := chaincfg.RegressionNetParams
	srv, err := newCPUMiningTestServer(t, &params, false)
	require.NoError(t, err)
	rpc := &rpcServer{cfg: rpcserverConfig{
		ChainParams: &params, Chain: srv.chain, CPUMiner: srv.cpuMiner,
	}}
	workers := srv.cpuMiner.NumWorkers()
	limit := 2
	for _, test := range []struct {
		name    string
		handler commandHandler
		command interface{}
	}{
		{"generate", handleGenerate, btcjson.NewGenerateCmd(1)},
		{"setgenerate", handleSetGenerate, btcjson.NewSetGenerateCmd(true, &limit)},
	} {
		t.Run(test.name, func(t *testing.T) {
			for range 2 {
				result, err := test.handler(rpc, test.command, nil)
				var rpcErr *btcjson.RPCError
				require.ErrorAs(t, err, &rpcErr)
				require.Equal(t, btcjson.ErrRPCDifficulty, rpcErr.Code)
				require.Contains(t, rpcErr.Message, "FP8 (V4)")
				require.Nil(t, result)
				require.False(t, srv.cpuMiner.IsMining())
				require.Equal(t, workers, srv.cpuMiner.NumWorkers())
			}
		})
	}
	_, err = handleSetGenerate(rpc, btcjson.NewSetGenerateCmd(false, nil), nil)
	require.NoError(t, err, "disabling generation remains available")
}

func TestServerGenerateSupport(t *testing.T) {
	t.Run("regtest", func(t *testing.T) {
		params := chaincfg.RegressionNetParams
		srv, err := newCPUMiningTestServer(t, &params, true)
		require.ErrorIs(t, err, blockchain.ErrCPUMiningUnsupported)
		require.Contains(t, err.Error(), "--generate")
		require.Nil(t, srv)
	})
	t.Run("simnet", func(t *testing.T) {
		params := chaincfg.SimNetParams
		srv, err := newCPUMiningTestServer(t, &params, true)
		require.NoError(t, err)
		require.NotNil(t, srv)
	})
}
