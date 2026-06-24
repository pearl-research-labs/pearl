// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/sha256"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/btcsuite/btclog"
	"github.com/btcsuite/websocket"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/wire"
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

	t.Run("malformed disconnects unauthenticated client", func(t *testing.T) {
		// An unauthenticated client whose first message is malformed must
		// be disconnected, not told what it did wrong.
		c := &wsClient{server: s}
		req := btcjson.Request{Jsonrpc: btcjson.RpcVersion1, ID: 1}
		require.True(t, c.authorizeRequest(&req).disconnect)
	})

	t.Run("malformed returns reply to authenticated client", func(t *testing.T) {
		c := &wsClient{server: s, authenticated: true}
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

// startWSClient drives a real websocket connection through the node's
// inHandler/notificationQueueHandler/outHandler goroutines against a minimal
// rpcServer. It returns the client-side connection and a channel that is
// closed once the server-side client has fully shut down. The wsClient is
// constructed directly (rather than via newWebsocketClient/WebsocketHandler)
// to avoid the global daemon config those paths require.
func startWSClient(t *testing.T, s *rpcServer, authenticated, isAdmin bool) (*websocket.Conn, <-chan struct{}) {
	t.Helper()

	done := make(chan struct{})
	upgrader := websocket.Upgrader{
		CheckOrigin: func(*http.Request) bool { return true },
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		c := &wsClient{
			conn:              conn,
			addr:              r.RemoteAddr,
			authenticated:     authenticated,
			isAdmin:           isAdmin,
			sessionID:         1,
			server:            s,
			addrRequests:      make(map[string]struct{}),
			spentRequests:     make(map[wire.OutPoint]struct{}),
			serviceRequestSem: makeSemaphore(10),
			ntfnChan:          make(chan []byte, 1),
			sendChan:          make(chan wsResponse, websocketSendBufferSize),
			quit:              make(chan struct{}),
		}
		c.wg.Add(3)
		go c.inHandler()
		go c.notificationQueueHandler()
		go c.outHandler()
		c.WaitForShutdown()
		close(done)
	}))
	t.Cleanup(srv.Close)

	dialer := websocket.Dialer{HandshakeTimeout: 5 * time.Second}
	conn, resp, err := dialer.Dial("ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	require.NoError(t, err)
	if resp != nil {
		_ = resp.Body.Close()
	}
	t.Cleanup(func() { _ = conn.Close() })

	return conn, done
}

// TestWebsocketBatchHandling exercises the real websocket batch path:
// notification-only batches emit no frame, mixed batches reply only to the
// requests, and a failed authenticate in an unauthenticated batch disconnects
// before any following command runs.
func TestWebsocketBatchHandling(t *testing.T) {
	t.Parallel()

	s := &rpcServer{
		adminCredHash: sha256.Sum256([]byte("admin:adminpass")),
		limitCredHash: sha256.Sum256([]byte("limit:limitpass")),
	}

	t.Run("notification-only batch produces no frame", func(t *testing.T) {
		conn, _ := startWSClient(t, s, true, true)

		// A batch of only a notification (no id) must produce no frame.
		// Follow it with a real request; the first frame received must be
		// that request's reply, proving the notification batch sent nothing.
		require.NoError(t, conn.WriteMessage(websocket.TextMessage,
			[]byte(`[{"jsonrpc":"1.0","method":"session","params":[]}]`)))
		require.NoError(t, conn.WriteMessage(websocket.TextMessage,
			[]byte(`{"jsonrpc":"1.0","method":"session","params":[],"id":42}`)))

		require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
		_, msg, err := conn.ReadMessage()
		require.NoError(t, err)

		var reply struct {
			Result *btcjson.SessionResult `json:"result"`
			ID     json.RawMessage        `json:"id"`
		}
		require.NoError(t, json.Unmarshal(msg, &reply))
		require.Equal(t, "42", string(reply.ID))
		require.NotNil(t, reply.Result)
	})

	t.Run("mixed batch replies only to requests", func(t *testing.T) {
		conn, _ := startWSClient(t, s, true, true)

		require.NoError(t, conn.WriteMessage(websocket.TextMessage,
			[]byte(`[{"jsonrpc":"1.0","method":"session","params":[]},`+
				`{"jsonrpc":"1.0","method":"session","params":[],"id":7}]`)))

		require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
		_, msg, err := conn.ReadMessage()
		require.NoError(t, err)

		var batch []struct {
			ID json.RawMessage `json:"id"`
		}
		require.NoError(t, json.Unmarshal(msg, &batch))
		require.Len(t, batch, 1, "only the request should get a reply")
		require.Equal(t, "7", string(batch[0].ID))
	})

	t.Run("failed unauthenticated batch disconnects", func(t *testing.T) {
		conn, done := startWSClient(t, s, false, false)

		// The first element is an invalid authenticate; the client must be
		// disconnected before the second (privileged) command is run.
		require.NoError(t, conn.WriteMessage(websocket.TextMessage,
			[]byte(`[{"jsonrpc":"1.0","method":"authenticate","params":["admin","wrong"],"id":1},`+
				`{"jsonrpc":"1.0","method":"session","params":[],"id":2}]`)))

		require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
		_, _, err := conn.ReadMessage()
		require.Error(t, err, "server should disconnect rather than reply")

		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("server did not shut the client down after failed auth")
		}
	})
}

// TestWebsocketClientCloseUnblocksShutdown verifies that a client-side close
// unblocks the server's per-client goroutines (WaitForShutdown returns).
func TestWebsocketClientCloseUnblocksShutdown(t *testing.T) {
	t.Parallel()

	conn, done := startWSClient(t, &rpcServer{}, true, true)
	require.NoError(t, conn.Close())

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("server client did not shut down after the client closed")
	}
}

// TestQueueNotificationAfterDisconnect verifies that queuing a notification on
// a disconnected client returns ErrClientQuit immediately instead of blocking.
func TestQueueNotificationAfterDisconnect(t *testing.T) {
	t.Parallel()

	c := &wsClient{disconnected: true}

	result := make(chan error, 1)
	go func() { result <- c.QueueNotification([]byte(`{"method":"x"}`)) }()

	select {
	case err := <-result:
		require.ErrorIs(t, err, ErrClientQuit)
	case <-time.After(5 * time.Second):
		t.Fatal("QueueNotification blocked after disconnect")
	}
}
