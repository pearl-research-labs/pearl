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

func TestSendUsesSendmany(t *testing.T) {
	// sendfrom is gated on an RPC-typed chain client and fails in SPV mode,
	// so the send must go out as sendmany (which broadcasts over P2P too).
	txid := "aa11bc0de2331fd6bb381f5bdc37a20c1c92cd6b71dc7f7f7ea9c1f0b4a1c2d3"
	var gotMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID     interface{} `json:"id"`
			Method string      `json:"method"`
		}
		_ = json.Unmarshal(body, &req)
		gotMethod = req.Method
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"jsonrpc": "1.0", "id": req.ID, "result": txid, "error": nil,
		})
	}))
	defer srv.Close()

	cfg := &config{Connect: strings.TrimPrefix(srv.URL, "http://"), RPCUser: "u", RPCPass: "p", NoTLS: true}
	cfg.activeNet = mainNetForTest()
	c, err := dialClient(cfg)
	require.NoError(t, err)
	defer c.shutdown()

	addr := "prl1ptt05u0gzvrhtzxjygk0tnzy29pmgylr6nccsl349verhcs5hzqqs26rg9s"
	hash, err := c.send("default", addr, 100000000, 0, 1)
	require.NoError(t, err)
	assert.Equal(t, "sendmany", gotMethod)
	assert.Equal(t, txid, hash.String())
}
