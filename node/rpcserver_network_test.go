// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"testing"

	"github.com/pearl-research-labs/pearl/node/blockchain/indexers"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var otherNetworks = []*chaincfg.Params{&chaincfg.TestNetParams, &chaincfg.RegressionNetParams}

func addressFor(t *testing.T, params *chaincfg.Params) string {
	t.Helper()

	addr, err := btcutil.NewAddressTaproot(bytes.Repeat([]byte{1}, 32), params)
	require.NoError(t, err)

	return addr.EncodeAddress()
}

func TestHandleCreateRawTransactionNetworkCheck(t *testing.T) {
	t.Parallel()

	s := &rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams}}
	create := func(addr string) (any, error) {
		cmd := btcjson.NewCreateRawTransactionCmd(nil, map[string]float64{addr: 1}, nil)

		return handleCreateRawTransaction(s, cmd, make(chan struct{}))
	}

	result, err := create(addressFor(t, &chaincfg.MainNetParams))
	require.NoError(t, err)
	assert.NotEmpty(t, result)

	for _, params := range otherNetworks {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()

			addr := addressFor(t, params)
			result, err := create(addr)
			requireRPCErrorCode(t, err, btcjson.ErrRPCInvalidAddressOrKey)
			require.ErrorContains(t, err, addr+" is for the wrong network")
			assert.Nil(t, result)
		})
	}
}

func TestHandleValidateAddressNetworkCheck(t *testing.T) {
	t.Parallel()

	s := &rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams}}
	validate := func(t *testing.T, addr string) btcjson.ValidateAddressChainResult {
		t.Helper()

		result, err := handleValidateAddress(s, btcjson.NewValidateAddressCmd(addr), make(chan struct{}))
		require.NoError(t, err)

		got, ok := result.(btcjson.ValidateAddressChainResult)
		require.True(t, ok)

		return got
	}

	addr := addressFor(t, &chaincfg.MainNetParams)
	got := validate(t, addr)
	assert.True(t, got.IsValid)
	assert.Equal(t, addr, got.Address)

	for _, params := range otherNetworks {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, btcjson.ValidateAddressChainResult{}, validate(t, addressFor(t, params)))
		})
	}
}

func TestHandleSearchRawTransactionsNetworkCheck(t *testing.T) {
	t.Parallel()

	// The address is rejected before the index is consulted, so an empty one is enough.
	s := &rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams, AddrIndex: &indexers.AddrIndex{}}}

	for _, params := range otherNetworks {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()
			defer requireNoHandlerPanic(t, "address index")

			addr := addressFor(t, params)
			cmd := btcjson.NewSearchRawTransactionsCmd(addr, nil, nil, nil, nil, nil, nil)
			result, err := handleSearchRawTransactions(s, cmd, make(chan struct{}))
			requireRPCErrorCode(t, err, btcjson.ErrRPCInvalidAddressOrKey)
			require.ErrorContains(t, err, addr+" is for the wrong network")
			assert.Nil(t, result)
		})
	}
}
