// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpcclient

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/btcsuite/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testRPCUser     = "testuser"
	testRPCPass     = "testpass"
	testCallerAuth  = "Bearer test-api-key"
	testExtraHeader = "X-Test-API-Key"
	testExtraValue  = "test-api-key"
)

// disableAuthTestCase describes one authentication header configuration that must behave the same for HTTP POST
// and WebSocket transports.
type disableAuthTestCase struct {
	name              string
	configure         func(*ConnConfig)
	wantAuthorization string
}

func disableAuthTestCases(missingCookie string) []disableAuthTestCase {
	basicAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte(testRPCUser+":"+testRPCPass))

	return []disableAuthTestCase{
		{
			name: "disabled omits generated authorization",
			configure: func(config *ConnConfig) {
				config.User = ""
				config.Pass = ""
				config.CookiePath = missingCookie
				config.DisableAuth = true
			},
		},
		{
			name: "disabled preserves caller authorization",
			configure: func(config *ConnConfig) {
				config.User = ""
				config.Pass = ""
				config.CookiePath = missingCookie
				config.DisableAuth = true
				config.ExtraHeaders["Authorization"] = testCallerAuth
			},
			wantAuthorization: testCallerAuth,
		},
		{
			name: "explicit false includes basic authorization",
			configure: func(config *ConnConfig) {
				config.DisableAuth = false
			},
			wantAuthorization: basicAuth,
		},
		{
			name:              "zero value includes basic authorization",
			configure:         func(*ConnConfig) {},
			wantAuthorization: basicAuth,
		},
	}
}

func newDisableAuthConfig() *ConnConfig {
	return &ConnConfig{
		User:         testRPCUser,
		Pass:         testRPCPass,
		ExtraHeaders: map[string]string{testExtraHeader: testExtraValue},
	}
}

func assertAuthHeaders(t *testing.T, header http.Header, wantAuthorization string) {
	t.Helper()

	assert.Equal(t, wantAuthorization, header.Get("Authorization"))
	assert.Equal(t, testExtraValue, header.Get(testExtraHeader))
}

func TestDisableAuthHTTPPost(t *testing.T) {
	t.Parallel()

	missingCookie := filepath.Join(t.TempDir(), "missing-cookie")

	for _, tt := range disableAuthTestCases(missingCookie) {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			requestHeader := make(chan http.Header, 1)
			client := newPostModeTestClient(postRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				requestHeader <- req.Header.Clone()
				return okJSONResponse(`{"result":1,"error":null,"id":1}`), nil
			}))
			client.config = newDisableAuthConfig()
			client.config.Host = "127.0.0.1:8332"
			client.config.DisableTLS = true
			client.config.HTTPPostMode = true
			tt.configure(client.config)

			result, err := sendPostRequestWithRetry(
				context.Background(), newPostTestRequest(), 1, client.httpClient, client.config, client.httpURL, false,
			)
			require.NoError(t, err)
			assert.Equal(t, []byte("1"), result)

			select {
			case header := <-requestHeader:
				assertAuthHeaders(t, header, tt.wantAuthorization)

			case <-time.After(time.Second):
				require.Fail(t, "timed out waiting for HTTP POST request")
			}
		})
	}
}

// newWebsocketAuthServer creates a server that records the WebSocket handshake headers before upgrading the
// connection.
func newWebsocketAuthServer(t *testing.T) (string, <-chan http.Header) {
	t.Helper()

	requestHeader := make(chan http.Header, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		requestHeader <- req.Header.Clone()

		conn, err := upgrader.Upgrade(w, req, nil)
		if err != nil {
			return
		}
		_ = conn.Close()
	}))
	t.Cleanup(server.Close)

	return strings.TrimPrefix(server.URL, "http://"), requestHeader
}

func TestDisableAuthWebsocket(t *testing.T) {
	t.Parallel()

	missingCookie := filepath.Join(t.TempDir(), "missing-cookie")

	for _, tt := range disableAuthTestCases(missingCookie) {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			host, requestHeader := newWebsocketAuthServer(t)
			config := newDisableAuthConfig()
			config.Host = host
			config.DisableTLS = true
			tt.configure(config)

			conn, err := dial(config)
			require.NoError(t, err)
			t.Cleanup(func() { _ = conn.Close() })

			select {
			case header := <-requestHeader:
				assertAuthHeaders(t, header, tt.wantAuthorization)

			case <-time.After(time.Second):
				require.Fail(t, "timed out waiting for WebSocket handshake")
			}
		})
	}
}
