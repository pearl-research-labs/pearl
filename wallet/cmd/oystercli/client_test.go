// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsRPCErrorCode(t *testing.T) {
	unlock := &btcjson.RPCError{Code: btcjson.ErrRPCWalletUnlockNeeded, Message: "locked"}

	// The send flow relies on this: a locked-wallet error (-13) is what
	// triggers withAutoUnlock's passphrase prompt.
	assert.True(t, isRPCErrorCode(unlock, btcjson.ErrRPCWalletUnlockNeeded))
	assert.True(t, isRPCErrorCode(fmt.Errorf("wrap: %w", unlock), btcjson.ErrRPCWalletUnlockNeeded))

	assert.False(t, isRPCErrorCode(unlock, btcjson.ErrRPCWalletPassphraseIncorrect))
	assert.False(t, isRPCErrorCode(errors.New("plain"), btcjson.ErrRPCWalletUnlockNeeded))
	assert.False(t, isRPCErrorCode(nil, btcjson.ErrRPCWalletUnlockNeeded))
}

func TestIsNotRelayedError(t *testing.T) {
	// The daemon reports this verdict as a generic internal error, so the message is the only handle the CLI has.
	notRelayed := &btcjson.RPCError{
		Code:    btcjson.ErrRPCInternal.Code,
		Message: "transaction not relayed to any peer: no peer requested it",
	}
	other := &btcjson.RPCError{Code: btcjson.ErrRPCInternal.Code, Message: "db closed"}

	assert.True(t, isNotRelayedError(notRelayed))
	assert.True(t, isNotRelayedError(fmt.Errorf("wrap: %w", notRelayed)))
	assert.False(t, isNotRelayedError(other))
	assert.False(t, isNotRelayedError(errors.New("not relayed")))
	assert.False(t, isNotRelayedError(nil))
}

// rpcRequest is what fakeRPCClient recorded about the last call it served.
type rpcRequest struct {
	Method string        `json:"method"`
	Params []interface{} `json:"params"`
}

// fakeRPCClient dials a client against an in-process JSON-RPC server that
// answers every call with respond(request) and records the last request.
func fakeRPCClient(t *testing.T, respond func(rpcRequest) interface{}) (*client, *rpcRequest) {
	t.Helper()

	var last rpcRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			rpcRequest
			ID interface{} `json:"id"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		last = req.rpcRequest
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"jsonrpc": "1.0", "id": req.ID, "result": respond(req.rpcRequest), "error": nil,
		})
	}))
	t.Cleanup(srv.Close)

	cfg := &config{Connect: strings.TrimPrefix(srv.URL, "http://"), RPCUser: "u", RPCPass: "p", NoTLS: true}
	cfg.activeNet = mainNetForTest()
	c, err := dialClient(cfg)
	require.NoError(t, err)
	t.Cleanup(c.shutdown)
	return c, &last
}

// TestPendingTxCalls checks the two pending-transaction commands go out with
// the txid as their single positional parameter and decode their results.
func TestPendingTxCalls(t *testing.T) {
	txid := "aa11bc0de2331fd6bb381f5bdc37a20c1c92cd6b71dc7f7f7ea9c1f0b4a1c2d3"
	parent := "bb11bc0de2331fd6bb381f5bdc37a20c1c92cd6b71dc7f7f7ea9c1f0b4a1c2d3"

	c, last := fakeRPCClient(t, func(req rpcRequest) interface{} {
		switch req.Method {
		case "removetransaction":
			return map[string][]string{"removed": {txid}}
		case "rebroadcasttransaction":
			return map[string][]string{"announced": {parent, txid}}
		}
		return nil
	})

	removed, err := c.removeTransaction(txid)
	require.NoError(t, err)
	assert.Equal(t, rpcRequest{"removetransaction", []interface{}{txid}}, *last)
	assert.Equal(t, []string{txid}, removed)

	announced, err := c.rebroadcastTransaction(txid)
	require.NoError(t, err)
	assert.Equal(t, rpcRequest{"rebroadcasttransaction", []interface{}{txid}}, *last)
	assert.Equal(t, []string{parent, txid}, announced)
}

func TestSendUsesSendmany(t *testing.T) {
	// sendfrom is gated on an RPC-typed chain client and fails in SPV mode,
	// so the send must go out as sendmany (which broadcasts over P2P too).
	txid := "aa11bc0de2331fd6bb381f5bdc37a20c1c92cd6b71dc7f7f7ea9c1f0b4a1c2d3"
	c, last := fakeRPCClient(t, func(rpcRequest) interface{} { return txid })

	addr := "prl1ptt05u0gzvrhtzxjygk0tnzy29pmgylr6nccsl349verhcs5hzqqs26rg9s"
	hash, err := c.send("default", addr, 100000000, 0, 1)
	require.NoError(t, err)
	assert.Equal(t, "sendmany", last.Method)
	assert.Equal(t, txid, hash.String())
}
