// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/big"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestRPCTransactionAmountPrecision(t *testing.T) {
	t.Parallel()

	const reported int64 = 6710886400000002
	cases := []struct {
		name  string
		grain int64
	}{
		{"zero", 0},
		{"one_grain", 1},
		{"reported_minus_one", reported - 1},
		{"reported", reported},
		{"reported_plus_one", reported + 1},
		{"above_integer_float_limit", 9007199254740993},
		{"consensus_max_minus_one", btcutil.MaxGrain - 1},
		{"consensus_max", btcutil.MaxGrain},
		// decoderawtransaction parses signed wire amounts without applying
		// consensus checks, so formatting must also handle these edge cases.
		{"negative_one_grain", -1},
		{"negative_fraction", -100000001},
		{"negative_reported", -reported},
		{"int64_min", math.MinInt64},
		{"int64_max", math.MaxInt64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			tx := wire.NewMsgTx(wire.TxVersion)
			outpoint := wire.OutPoint{Hash: chainhash.Hash{1}, Index: 0}
			tx.AddTxIn(wire.NewTxIn(&outpoint, nil, nil))
			script := append([]byte{0x51, 0x20}, bytes.Repeat([]byte{0x01}, 32)...)
			tx.AddTxOut(wire.NewTxOut(tc.grain, script))

			var serialized bytes.Buffer
			require.NoError(t, tx.Serialize(&serialized))
			rawHex := hex.EncodeToString(serialized.Bytes())
			var decoded wire.MsgTx
			require.NoError(t, decoded.Deserialize(bytes.NewReader(serialized.Bytes())))
			require.Equal(t, tc.grain, decoded.TxOut[0].Value)

			t.Run("decoderawtransaction", func(t *testing.T) {
				requestJSON, err := btcjson.MarshalCmd(btcjson.RpcVersion1, 1,
					btcjson.NewDecodeRawTransactionCmd(rawHex))
				require.NoError(t, err)
				var request btcjson.Request
				require.NoError(t, json.Unmarshal(requestJSON, &request))
				server := rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams}}
				response := server.processRequest(&request, true, make(chan struct{}))
				requireExactRPCOutput(t, rpcAmountResponseResult(t, response), tc.grain)
			})

			// The shared constructor supplies verbose getrawtransaction and
			// getblock results, and verbose transaction notifications.
			result, err := createTxRawResult(&chaincfg.MainNetParams, tx,
				tx.TxHash().String(), nil, "", 0, 0)
			require.NoError(t, err)
			require.Equal(t, rawHex, result.Hex)

			t.Run("getrawtransaction", func(t *testing.T) {
				encoded, err := btcjson.MarshalResponse(btcjson.RpcVersion1, 1, result, nil)
				require.NoError(t, err)
				requireExactRPCOutput(t, rpcAmountResponseResult(t, encoded), tc.grain)
			})

			t.Run("verbose_block_envelope", func(t *testing.T) {
				block := btcjson.GetBlockVerboseResult{RawTx: []btcjson.TxRawResult{*result}}
				encoded, err := btcjson.MarshalResponse(btcjson.RpcVersion1, 1, block, nil)
				require.NoError(t, err)
				var response struct {
					RawTx []json.RawMessage `json:"rawtx"`
				}
				require.NoError(t, json.Unmarshal(rpcAmountResponseResult(t, encoded), &response))
				require.Len(t, response.RawTx, 1)
				requireExactRPCOutput(t, response.RawTx[0], tc.grain)
			})

			t.Run("search_transactions_envelope", func(t *testing.T) {
				search := []btcjson.SearchRawTransactionsResult{{Vout: result.Vout}}
				encoded, err := btcjson.MarshalResponse(btcjson.RpcVersion1, 1, search, nil)
				require.NoError(t, err)
				var response []json.RawMessage
				require.NoError(t, json.Unmarshal(rpcAmountResponseResult(t, encoded), &response))
				require.Len(t, response, 1)
				requireExactRPCOutput(t, response[0], tc.grain)
			})

			t.Run("verbose_websocket_notification", func(t *testing.T) {
				manager := wsNotificationManager{
					server: &rpcServer{cfg: rpcserverConfig{ChainParams: &chaincfg.MainNetParams}},
				}
				clients := make(map[chan struct{}]*wsClient)
				for i := 0; i < 2; i++ {
					quit := make(chan struct{})
					clients[quit] = &wsClient{
						verboseTxUpdates: true,
						ntfnChan:         make(chan []byte, 1),
						quit:             quit,
					}
				}
				manager.notifyForNewTx(clients, btcutil.NewTx(tx))
				for _, client := range clients {
					select {
					case encoded := <-client.ntfnChan:
						// Inspect the numeric token without decoding through float64.
						var notification struct {
							Method string            `json:"method"`
							Params []json.RawMessage `json:"params"`
						}
						require.NoError(t, json.Unmarshal(encoded, &notification))
						require.Equal(t, "txacceptedverbose", notification.Method)
						require.Len(t, notification.Params, 1)
						requireExactRPCOutput(t, notification.Params[0], tc.grain)
					default:
						require.FailNow(t, "verbose transaction notification was not queued")
					}
				}
			})
		})
	}
}

// requireExactRPCOutput checks the JSON number itself, without first decoding it
// into a float. Exact decimal clients must recover every grain from this token.
func requireExactRPCOutput(t *testing.T, encoded json.RawMessage, grain int64) {
	t.Helper()

	var tx struct {
		Vout []struct {
			Value json.RawMessage `json:"value"`
		} `json:"vout"`
	}
	require.NoError(t, json.Unmarshal(encoded, &tx))
	require.Len(t, tx.Vout, 1)
	token := string(tx.Vout[0].Value)
	recovered, ok := new(big.Rat).SetString(token)
	require.Truef(t, ok, "value must be a JSON number, got %q", token)
	recovered.Mul(recovered, big.NewRat(btcutil.GrainPerPearl, 1))
	require.Equalf(t, 0, recovered.Cmp(new(big.Rat).SetInt64(grain)),
		"JSON %s recovers %s grain, want %d", token, recovered.RatString(), grain)

	var decoded struct {
		Vout []btcjson.Vout `json:"vout"`
	}
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	require.Equal(t, btcjson.Amount(grain), decoded.Vout[0].Value)
}

func rpcAmountResponseResult(t *testing.T, encoded []byte) json.RawMessage {
	t.Helper()

	var response btcjson.Response
	require.NoError(t, json.Unmarshal(encoded, &response))
	require.Nil(t, response.Error)
	return response.Result
}
