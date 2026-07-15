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

func TestStopDaemon(t *testing.T) {
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
			"jsonrpc": "1.0", "id": req.ID, "result": "oyster stopping", "error": nil,
		})
	}))
	defer srv.Close()

	cfg := &config{
		Connect: strings.TrimPrefix(srv.URL, "http://"),
		RPCUser: "u",
		RPCPass: "p",
		NoTLS:   true,
	}
	cfg.activeNet = mainNetForTest()
	c, err := dialClient(cfg)
	require.NoError(t, err)
	defer c.shutdown()

	c.unlockedByUs = true
	require.NoError(t, c.stopDaemon())

	// It sent the authenticated "stop" method, and cleared the re-lock
	// flag so exit does not try to talk to the now-stopping daemon.
	assert.Equal(t, "stop", gotMethod)
	assert.False(t, c.unlockedByUs)
}
