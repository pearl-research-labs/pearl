// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newMainNetWSClient() *wsClient {
	return &wsClient{server: &rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams}}}
}

func TestCheckAddressValidityNetworkCheck(t *testing.T) {
	t.Parallel()

	mainnet := addressFor(t, &chaincfg.MainNetParams)
	require.NoError(t, checkAddressValidity([]string{mainnet}, &chaincfg.MainNetParams))

	for _, params := range otherNetworks {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()

			addr := addressFor(t, params)
			err := checkAddressValidity([]string{mainnet, addr}, &chaincfg.MainNetParams)
			requireRPCErrorCode(t, err, btcjson.ErrRPCInvalidAddressOrKey)
			require.ErrorContains(t, err, addr)
		})
	}
}

func TestNotifyReceivedHandlersNetworkCheck(t *testing.T) {
	t.Parallel()

	for _, params := range otherNetworks {
		t.Run(params.Name, func(t *testing.T) {
			t.Parallel()
			defer requireNoHandlerPanic(t, "notification manager")

			addrs := []string{addressFor(t, params)}

			result, err := handleNotifyReceived(newMainNetWSClient(), &btcjson.NotifyReceivedCmd{Addresses: addrs})
			requireRPCErrorCode(t, err, btcjson.ErrRPCInvalidAddressOrKey)
			assert.Nil(t, result)

			result, err = handleStopNotifyReceived(
				newMainNetWSClient(), &btcjson.StopNotifyReceivedCmd{Addresses: addrs},
			)
			requireRPCErrorCode(t, err, btcjson.ErrRPCInvalidAddressOrKey)
			assert.Nil(t, result)
		})
	}
}

func TestHandleLoadTxFilterNetworkCheck(t *testing.T) {
	t.Parallel()

	mainnet := addressFor(t, &chaincfg.MainNetParams)
	mainnetAddr, err := btcutil.DecodeAddress(mainnet, &chaincfg.MainNetParams)
	require.NoError(t, err)

	addrs := []string{mainnet}
	for _, params := range otherNetworks {
		addrs = append(addrs, addressFor(t, params))
	}

	t.Run("reload", func(t *testing.T) {
		t.Parallel()

		wsc := newMainNetWSClient()
		_, err := handleLoadTxFilter(wsc, btcjson.NewLoadTxFilterCmd(true, addrs, nil))
		require.NoError(t, err)

		assert.True(t, wsc.filterData.existsAddress(mainnetAddr))
		assert.Len(t, wsc.filterData.otherAddresses, 1)
	})

	t.Run("add to existing filter", func(t *testing.T) {
		t.Parallel()

		wsc := newMainNetWSClient()
		wsc.filterData = newWSClientFilter(nil, nil, &chaincfg.MainNetParams)
		_, err := handleLoadTxFilter(wsc, btcjson.NewLoadTxFilterCmd(false, addrs, nil))
		require.NoError(t, err)

		assert.True(t, wsc.filterData.existsAddress(mainnetAddr))
		assert.Len(t, wsc.filterData.otherAddresses, 1)
	})
}
