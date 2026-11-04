// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/wallet/internal/cfgutil"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "oyster-rpc-logs")
	if err != nil {
		fmt.Fprintf(os.Stderr, "temp log dir: %v\n", err)
		os.Exit(1)
	}
	initLogRotator(filepath.Join(dir, "oyster.log"), 1024)
	os.Exit(m.Run())
}

func useTestConfig(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	cfg = &config{
		RPCKey:           cfgutil.NewExplicitString(filepath.Join(dir, "rpc.key")),
		RPCCert:          cfgutil.NewExplicitString(filepath.Join(dir, "rpc.cert")),
		Username:         "rpcuser",
		Password:         "rpcpass",
		RPCListeners:     []string{"127.0.0.1:0"},
		RPCMaxClients:    defaultRPCMaxClients,
		RPCMaxWebsockets: defaultRPCMaxWebsockets,
	}
}

func testLoader(t *testing.T) *wallet.Loader {
	t.Helper()
	return wallet.NewLoader(&chaincfg.SimNetParams, t.TempDir(), true, time.Minute, 250)
}

func serveTestRPC(t *testing.T) string {
	t.Helper()
	listeners, err := openRPCListeners()
	require.NoError(t, err)
	require.NotEmpty(t, listeners)

	server := serveRPC(testLoader(t), listeners)
	t.Cleanup(func() { server.Stop() })
	return listeners[0].Addr().String()
}

func postRPC(t *testing.T, url, user, pass string, tlsCfg *tls.Config) *http.Response {
	t.Helper()
	const body = `{"jsonrpc":"1.0","id":"1","method":"help","params":[]}`
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	require.NoError(t, err)
	if user != "" || pass != "" {
		req.SetBasicAuth(user, pass)
	}
	client := &http.Client{Timeout: 5 * time.Second}
	if tlsCfg != nil {
		client.Transport = &http.Transport{TLSClientConfig: tlsCfg}
	}
	resp, err := client.Do(req)
	require.NoError(t, err)
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func assertJSONRPCReply(t *testing.T, resp *http.Response) {
	t.Helper()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Contains(t, resp.Header.Get("Content-Type"), "application/json")

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var payload struct {
		ID    json.RawMessage `json:"id"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Result json.RawMessage `json:"result"`
	}
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.JSONEq(t, `"1"`, string(payload.ID))
	if payload.Error == nil {
		assert.NotEmpty(t, payload.Result)
		return
	}
	assert.NotEmpty(t, payload.Error.Message)
}

func TestStartRPCServerRequiresCredentials(t *testing.T) {
	tests := []struct {
		name string
		user string
		pass string
	}{
		{name: "missing username", pass: "rpcpass"},
		{name: "missing password", user: "rpcuser"},
		{name: "missing both"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			useTestConfig(t)
			cfg.Username = tt.user
			cfg.Password = tt.pass

			server, err := startRPCServer(testLoader(t))
			require.Error(t, err)
			assert.Contains(t, err.Error(), "username and password")
			assert.Nil(t, server)
		})
	}
}

func TestOpenRPCKeyPairLifecycle(t *testing.T) {
	t.Run("generates cert and key", func(t *testing.T) {
		useTestConfig(t)

		_, err := openRPCKeyPair()
		require.NoError(t, err)
		assert.FileExists(t, cfg.RPCCert.Value)
		assert.FileExists(t, cfg.RPCKey.Value)
	})

	t.Run("loads an existing pair", func(t *testing.T) {
		useTestConfig(t)

		first, err := openRPCKeyPair()
		require.NoError(t, err)
		certBefore, err := os.ReadFile(cfg.RPCCert.Value)
		require.NoError(t, err)
		keyBefore, err := os.ReadFile(cfg.RPCKey.Value)
		require.NoError(t, err)

		second, err := openRPCKeyPair()
		require.NoError(t, err)
		certAfter, err := os.ReadFile(cfg.RPCCert.Value)
		require.NoError(t, err)
		keyAfter, err := os.ReadFile(cfg.RPCKey.Value)
		require.NoError(t, err)

		assert.Equal(t, certBefore, certAfter)
		assert.Equal(t, keyBefore, keyAfter)
		assert.Equal(t, first.Certificate, second.Certificate)
	})

	t.Run("one time key writes only the cert", func(t *testing.T) {
		useTestConfig(t)
		cfg.OneTimeTLSKey = true

		_, err := openRPCKeyPair()
		require.NoError(t, err)
		assert.FileExists(t, cfg.RPCCert.Value)
		_, err = os.Stat(cfg.RPCKey.Value)
		assert.ErrorIs(t, err, os.ErrNotExist)
	})

	t.Run("one time key rejects an existing key file", func(t *testing.T) {
		useTestConfig(t)
		require.NoError(t, os.WriteFile(cfg.RPCKey.Value, []byte("key"), 0o600))
		cfg.OneTimeTLSKey = true

		_, err := openRPCKeyPair()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "one time TLS keys")
	})
}

func TestRPCServerTLS(t *testing.T) {
	useTestConfig(t)
	addr := serveTestRPC(t)

	tlsCfg := &tls.Config{
		InsecureSkipVerify: true,
		MinVersion:         tls.VersionTLS12,
		NextProtos:         []string{"h2", "http/1.1"},
	}
	resp := postRPC(t, "https://"+addr+"/", cfg.Username, cfg.Password, tlsCfg)
	require.NotNil(t, resp.TLS)
	assert.Equal(t, uint16(tls.VersionTLS13), resp.TLS.Version)
	assert.Equal(t, "http/1.1", resp.TLS.NegotiatedProtocol)
	assertJSONRPCReply(t, resp)

	conn, err := tls.Dial("tcp", addr, &tls.Config{
		InsecureSkipVerify: true,
		MinVersion:         tls.VersionTLS12,
		MaxVersion:         tls.VersionTLS12,
	})
	if conn != nil {
		conn.Close()
	}
	require.Error(t, err)
}

func TestRPCServerWithoutTLS(t *testing.T) {
	useTestConfig(t)
	cfg.DisableServerTLS = true
	addr := serveTestRPC(t)

	resp := postRPC(t, "http://"+addr+"/", cfg.Username, cfg.Password, nil)
	assert.Nil(t, resp.TLS)
	assertJSONRPCReply(t, resp)
}

func TestRPCServerAuth(t *testing.T) {
	useTestConfig(t)
	cfg.DisableServerTLS = true
	addr := serveTestRPC(t)
	url := "http://" + addr + "/"

	t.Run("accepts configured credentials", func(t *testing.T) {
		resp := postRPC(t, url, cfg.Username, cfg.Password, nil)
		assertJSONRPCReply(t, resp)
	})

	t.Run("rejects a wrong password", func(t *testing.T) {
		resp := postRPC(t, url, cfg.Username, "wrong", nil)
		assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)
		assert.Contains(t, resp.Header.Get("WWW-Authenticate"), "Basic")
	})

	t.Run("rejects a missing password", func(t *testing.T) {
		resp := postRPC(t, url, "", "", nil)
		assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)
	})
}
