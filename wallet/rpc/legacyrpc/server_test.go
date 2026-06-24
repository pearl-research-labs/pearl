// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package legacyrpc

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/btcsuite/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCheckCredentials(t *testing.T) {
	t.Parallel()

	s := &Server{credHash: sha256.Sum256([]byte("user:pass"))}

	cases := []struct {
		name       string
		user, pass string
		want       bool
	}{
		{"correct", "user", "pass", true},
		{"wrong pass", "user", "nope", false},
		{"wrong user", "nope", "pass", false},
		{"empty", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := s.checkCredentials(tc.user, tc.pass)
			require.Equal(t, tc.want, got)
		})
	}

}

// TestWebsocketHandshakeAuth is the regression test for the former
// authentication bypass: a websocket client with missing or incorrect
// credentials must be rejected during the HTTP handshake (HTTP 401), before
// the connection is ever upgraded.  Only correct credentials succeed.
func TestWebsocketHandshakeAuth(t *testing.T) {
	t.Parallel()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	opts := &Options{
		Username:            "user",
		Password:            "pass",
		MaxPOSTClients:      10,
		MaxWebsocketClients: 10,
	}
	srv := NewServer(opts, nil, []net.Listener{lis})
	srv.Start()
	defer srv.Stop()

	url := "ws://" + lis.Addr().String() + "/ws"
	dialer := websocket.Dialer{HandshakeTimeout: 5 * time.Second}

	dial := func(user, pass string, withAuth bool) (int, error) {
		h := http.Header{}
		if withAuth {
			cred := base64.StdEncoding.EncodeToString([]byte(user + ":" + pass))
			h.Set("Authorization", "Basic "+cred)
		}
		conn, resp, err := dialer.Dial(url, h)
		if conn != nil {
			conn.Close()
		}
		code := 0
		if resp != nil {
			code = resp.StatusCode
			resp.Body.Close()
		}
		return code, err
	}

	// Unauthenticated handshake should fail with 401 Unauthorized.
	code, err := dial("", "", false)
	require.Error(t, err)
	require.Equal(t, http.StatusUnauthorized, code)

	// Wrong password handshake should fail with 401 Unauthorized.
	code, err = dial("user", "wrong", true)
	require.Error(t, err)
	require.Equal(t, http.StatusUnauthorized, code)

	// Correct credentials handshake should succeed.
	_, err = dial("user", "pass", true)
	require.NoError(t, err)
}

// TestWebsocketAuthenticateHandledLocally is the regression test for the
// authenticate fallthrough: a redundant in-band authenticate from an
// already-authenticated websocket client must be answered locally with
// success and never proxied to the chain server.  With no chain client
// configured, a passthrough would instead report "Chain RPC is inactive".
func TestWebsocketAuthenticateHandledLocally(t *testing.T) {
	t.Parallel()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	opts := &Options{
		Username:            "user",
		Password:            "pass",
		MaxPOSTClients:      10,
		MaxWebsocketClients: 10,
	}
	srv := NewServer(opts, nil, []net.Listener{lis})
	srv.Start()
	defer srv.Stop()

	h := http.Header{}
	cred := base64.StdEncoding.EncodeToString([]byte("user:pass"))
	h.Set("Authorization", "Basic "+cred)

	dialer := websocket.Dialer{HandshakeTimeout: 5 * time.Second}
	conn, resp, err := dialer.Dial("ws://"+lis.Addr().String()+"/ws", h)
	require.NoError(t, err)
	if resp != nil {
		resp.Body.Close()
	}
	defer conn.Close()

	req := `{"jsonrpc":"1.0","id":7,"method":"authenticate","params":["user","pass"]}`
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte(req)))

	require.NoError(t, conn.SetReadDeadline(time.Now().Add(5*time.Second)))
	_, msg, err := conn.ReadMessage()
	require.NoError(t, err)

	var got struct {
		Result json.RawMessage `json:"result"`
		Error  json.RawMessage `json:"error"`
		ID     json.RawMessage `json:"id"`
	}
	require.NoError(t, json.Unmarshal(msg, &got))
	assert.Equal(t, "null", string(got.Error),
		"authenticate must be answered locally, not proxied to the chain server")
	assert.Equal(t, "7", string(got.ID))
}

// TestPostHandshakeAuth checks route-level HTTP Basic auth on the POST
// endpoint: missing or wrong credentials are rejected with 401 before the
// request is processed, and correct credentials are accepted.
func TestPostHandshakeAuth(t *testing.T) {
	t.Parallel()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	opts := &Options{
		Username:            "user",
		Password:            "pass",
		MaxPOSTClients:      10,
		MaxWebsocketClients: 10,
	}
	srv := NewServer(opts, nil, []net.Listener{lis})
	srv.Start()
	defer srv.Stop()

	endpoint := "http://" + lis.Addr().String() + "/"
	body := `{"jsonrpc":"1.0","id":1,"method":"getinfo","params":[]}`

	post := func(user, pass string, withAuth bool) int {
		req, err := http.NewRequest(http.MethodPost, endpoint,
			strings.NewReader(body))
		require.NoError(t, err)
		if withAuth {
			req.SetBasicAuth(user, pass)
		}
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		resp.Body.Close()
		return resp.StatusCode
	}

	assert.Equal(t, http.StatusUnauthorized, post("", "", false))
	assert.Equal(t, http.StatusUnauthorized, post("user", "wrong", true))
	assert.Equal(t, http.StatusOK, post("user", "pass", true))
}

func TestThrottle(t *testing.T) {
	const threshold = 1
	busy := make(chan struct{})

	srv := httptest.NewServer(throttledFn(threshold,
		func(w http.ResponseWriter, r *http.Request) {
			<-busy
		}),
	)

	codes := make(chan int, 2)
	for i := 0; i < cap(codes); i++ {
		go func() {
			res, err := http.Get(srv.URL)
			if !assert.NoError(t, err) {
				return
			}
			codes <- res.StatusCode
			_ = res.Body.Close()
		}()
	}

	got := make(map[int]int, cap(codes))
	for i := 0; i < cap(codes); i++ {
		got[<-codes]++

		if i == 0 {
			close(busy)
		}
	}

	require.Equal(t, map[int]int{200: 1, 429: 1}, got)
}
