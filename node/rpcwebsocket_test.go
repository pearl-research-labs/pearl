// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/sha256"
	"encoding/json"
	"testing"

	"github.com/btcsuite/btclog"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/require"
)

func init() {
	// The subsystem loggers write through a log rotator that is only
	// initialized by the running daemon, so emitting a log in tests would
	// dereference a nil rotator.  authorizeRequest logs warnings on the auth
	// failure paths exercised here, so silence the RPC subsystem logger.
	rpcsLog.SetLevel(btclog.LevelOff)
}

// TestAuthorizeRequest covers request validation, the in-band authenticate
// state machine, and the limited-user method gate.  It guards against
// re-introducing a fail-open authenticate bypass: an unauthenticated client
// whose first message is not a valid authenticate command must be
// disconnected, and only correct credentials may authenticate.
func TestAuthorizeRequest(t *testing.T) {
	t.Parallel()

	s := &rpcServer{
		adminCredHash: sha256.Sum256([]byte("admin:adminpass")),
		limitCredHash: sha256.Sum256([]byte("limit:limitpass")),
	}
	mustRequest := func(raw string) btcjson.Request {
		t.Helper()

		var req btcjson.Request
		require.NoError(t, json.Unmarshal([]byte(raw), &req))
		return req
	}

	t.Run("malformed returns reply", func(t *testing.T) {
		c := &wsClient{server: s}
		req := btcjson.Request{Jsonrpc: btcjson.RpcVersion1, ID: 1}
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.NotNil(t, outcome.reply)
		require.Nil(t, outcome.cmd)
	})

	t.Run("unauthenticated notification disconnects", func(t *testing.T) {
		c := &wsClient{server: s}
		req := mustRequest(`{"jsonrpc":"1.0","method":"getinfo","params":[]}`)
		require.True(t, c.authorizeRequest(&req).disconnect)
	})

	t.Run("authenticated notification is skipped", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true}
		req := mustRequest(`{"jsonrpc":"1.0","method":"getinfo","params":[]}`)
		require.Equal(t, requestOutcome{}, c.authorizeRequest(&req))
	})

	t.Run("parse error disconnects unauthenticated client", func(t *testing.T) {
		c := &wsClient{server: s}
		req := mustRequest(`{"jsonrpc":"1.0","method":"bogusmethod","params":[],"id":1}`)
		require.True(t, c.authorizeRequest(&req).disconnect)
	})

	t.Run("parse error returns reply when authenticated", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true}
		req := mustRequest(`{"jsonrpc":"1.0","method":"bogusmethod","params":[],"id":1}`)
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.NotNil(t, outcome.reply)
	})

	t.Run("first message not authenticate disconnects", func(t *testing.T) {
		c := &wsClient{server: s}
		req := mustRequest(`{"jsonrpc":"1.0","method":"getinfo","params":[],"id":1}`)
		require.True(t, c.authorizeRequest(&req).disconnect)
		require.False(t, c.authenticated)
	})

	t.Run("wrong credentials disconnect", func(t *testing.T) {
		c := &wsClient{server: s}
		req := mustRequest(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","wrong"],"id":1}`)
		require.True(t, c.authorizeRequest(&req).disconnect)
		require.False(t, c.authenticated)
	})

	t.Run("admin authenticates", func(t *testing.T) {
		c := &wsClient{server: s}
		req := mustRequest(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","adminpass"],"id":1}`)
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.NotNil(t, outcome.reply)
		require.True(t, c.authenticated)
		require.True(t, c.isAdmin)
	})

	t.Run("authenticate while authenticated disconnects", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true, isAdmin: true}
		req := mustRequest(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","adminpass"],"id":1}`)
		require.True(t, c.authorizeRequest(&req).disconnect)
	})

	t.Run("limited user denied disallowed method", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true}
		// "stop" is registered but admin-only (absent from rpcLimited).
		req := mustRequest(`{"jsonrpc":"1.0","method":"stop","params":[],"id":1}`)
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.NotNil(t, outcome.reply)
		require.Nil(t, outcome.cmd)
	})

	t.Run("limited user allowed method proceeds", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true}
		req := mustRequest(`{"jsonrpc":"1.0","method":"getbestblockhash","params":[],"id":1}`)
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.Nil(t, outcome.reply)
		require.NotNil(t, outcome.cmd)
	})

	t.Run("admin proceeds for any method", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true, isAdmin: true}
		req := mustRequest(`{"jsonrpc":"1.0","method":"stop","params":[],"id":1}`)
		outcome := c.authorizeRequest(&req)
		require.False(t, outcome.disconnect)
		require.Nil(t, outcome.reply)
		require.NotNil(t, outcome.cmd)
	})
}

// TestAuthorizeRequestBatchSequence locks in the sequential auth behavior that
// inHandler relies on for batched requests: a failed authenticate in an
// unauthenticated batch must not let a following privileged command run, while
// a valid authenticate lets the following command proceed.
func TestAuthorizeRequestBatchSequence(t *testing.T) {
	t.Parallel()

	s := &rpcServer{
		adminCredHash: sha256.Sum256([]byte("admin:adminpass")),
		limitCredHash: sha256.Sum256([]byte("limit:limitpass")),
	}
	mustRequest := func(raw string) btcjson.Request {
		t.Helper()

		var req btcjson.Request
		require.NoError(t, json.Unmarshal([]byte(raw), &req))
		return req
	}

	t.Run("failed authenticate blocks following command", func(t *testing.T) {
		c := &wsClient{server: s}
		bad := mustRequest(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","wrong"],"id":1}`)
		require.True(t, c.authorizeRequest(&bad).disconnect)
		require.False(t, c.authenticated)

		// The still-unauthenticated client's privileged command is also
		// rejected (inHandler stops on the first disconnect outcome).
		stop := mustRequest(`{"jsonrpc":"1.0","method":"stop","params":[],"id":2}`)
		require.True(t, c.authorizeRequest(&stop).disconnect)
	})

	t.Run("valid authenticate lets following command proceed", func(t *testing.T) {
		c := &wsClient{server: s}
		auth := mustRequest(`{"jsonrpc":"1.0","method":"authenticate","params":["admin","adminpass"],"id":1}`)
		first := c.authorizeRequest(&auth)
		require.False(t, first.disconnect)
		require.NotNil(t, first.reply)
		require.True(t, c.authenticated)

		stop := mustRequest(`{"jsonrpc":"1.0","method":"stop","params":[],"id":2}`)
		second := c.authorizeRequest(&stop)
		require.False(t, second.disconnect)
		require.NotNil(t, second.cmd)
	})
}

// TestRunCommand verifies that an authorized command is executed and its
// response marshalled - the dispatch step that authorizeRequest leaves to the
// caller.
func TestRunCommand(t *testing.T) {
	t.Parallel()

	c := &wsClient{server: &rpcServer{}, sessionID: 7}

	var req btcjson.Request
	require.NoError(t, json.Unmarshal(
		[]byte(`{"jsonrpc":"1.0","method":"session","params":[],"id":1}`), &req))
	cmd := parseCmd(&req)
	require.Nil(t, cmd.err)

	require.NotNil(t, c.runCommand(cmd))
}
