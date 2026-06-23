// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package legacyrpc

import (
	"crypto/sha256"
	"encoding/base64"
	"net"
	"net/http"
	"net/http/httptest"
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
		// The credential is hashed as user + ":" + pass, so a colon
		// shifted between the fields must not collide.
		{"colon shift", "use", "r:pass", false},
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
