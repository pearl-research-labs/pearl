// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
)

// rpcAmount encodes an integer grain amount as an exact PRL JSON number.
type rpcAmount int64

func (a rpcAmount) MarshalJSON() ([]byte, error) {
	whole := int64(a) / btcutil.GrainPerPearl
	fraction := int64(a) % btcutil.GrainPerPearl
	sign := ""
	if a < 0 {
		sign = "-"
		// Negating the quotient and remainder separately also handles MinInt64.
		whole = -whole
		fraction = -fraction
	}
	return []byte(fmt.Sprintf("%s%d.%08d", sign, whole, fraction)), nil
}

// These server-only results retain integer amounts until JSON encoding while
// preserving the public btcjson types used by existing Go RPC clients.
type rpcVout struct {
	btcjson.Vout
	Value rpcAmount `json:"value"`
}

type rpcTxRawResult struct {
	btcjson.TxRawResult
	Vout []rpcVout `json:"vout"`
}

type rpcTxRawDecodeResult struct {
	btcjson.TxRawDecodeResult
	Vout []rpcVout `json:"vout"`
}

type rpcGetBlockVerboseResult struct {
	btcjson.GetBlockVerboseResult
	RawTx []rpcTxRawResult `json:"rawtx,omitempty"`
}

type rpcSearchRawTransactionsResult struct {
	btcjson.SearchRawTransactionsResult
	Vin  []rpcVinPrevOut `json:"vin"`
	Vout []rpcVout       `json:"vout"`
}

type rpcGetTxOutResult struct {
	btcjson.GetTxOutResult
	Value rpcAmount `json:"value"`
}

type rpcPrevOut struct {
	btcjson.PrevOut
	Value rpcAmount `json:"value"`
}

// rpcVinPrevOut preserves btcjson.VinPrevOut's conditional fields with an exact
// previous-output amount. Embedding that type would inherit its JSON marshaler.
type rpcVinPrevOut struct {
	Coinbase  string             `json:"coinbase,omitempty"`
	Txid      string             `json:"txid"`
	Vout      uint32             `json:"vout"`
	ScriptSig *btcjson.ScriptSig `json:"scriptSig"`
	Witness   []string           `json:"txinwitness,omitempty"`
	PrevOut   *rpcPrevOut        `json:"prevOut,omitempty"`
	Sequence  uint32             `json:"sequence"`
}

func (v rpcVinPrevOut) MarshalJSON() ([]byte, error) {
	if len(v.Coinbase) > 0 {
		return json.Marshal(struct {
			Coinbase string `json:"coinbase"`
			Sequence uint32 `json:"sequence"`
		}{
			Coinbase: v.Coinbase,
			Sequence: v.Sequence,
		})
	}

	type plainVin rpcVinPrevOut
	return json.Marshal(plainVin(v))
}
