// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestScrapeOysterConf(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    oysterConfValues
	}{
		{
			name:    "oyster style options",
			content: "[Application Options]\nusername=alice\npassword=hunter2\n",
			want:    oysterConfValues{username: "alice", password: "hunter2"},
		},
		{
			name:    "pearld style aliases",
			content: "rpcuser=bob\nrpcpass=secret\n",
			want:    oysterConfValues{username: "bob", password: "secret"},
		},
		{
			name:    "commented options ignored",
			content: "; username=nope\n;password=nope\nusername=real\npassword=pw\n",
			want:    oysterConfValues{username: "real", password: "pw"},
		},
		{
			name:    "noservertls enabled",
			content: "username=u\npassword=p\nnoservertls=1\n",
			want:    oysterConfValues{username: "u", password: "p", noServerTLS: true},
		},
		{
			name:    "noservertls disabled",
			content: "noservertls=0\n",
			want:    oysterConfValues{},
		},
		{
			name:    "leading whitespace",
			content: "  username=indented\n\tpassword=tabbed\n",
			want:    oysterConfValues{username: "indented", password: "tabbed"},
		},
		{
			name:    "empty file",
			content: "",
			want:    oysterConfValues{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "oyster.conf")
			require.NoError(t, os.WriteFile(path, []byte(tt.content), 0o600))
			assert.Equal(t, tt.want, scrapeOysterConf(path))
		})
	}
}

func TestScrapeOysterConfMissingFile(t *testing.T) {
	got := scrapeOysterConf(filepath.Join(t.TempDir(), "does-not-exist.conf"))
	assert.Equal(t, oysterConfValues{}, got)
}

func TestNormalizeAddress(t *testing.T) {
	tests := []struct {
		name string
		addr string
		want string
	}{
		{"bare host", "localhost", "localhost:44207"},
		{"host with port", "localhost:1234", "localhost:1234"},
		{"empty defaults to localhost", "", "localhost:44207"},
		{"ipv4", "10.0.0.5", "10.0.0.5:44207"},
		{"ipv6 with port", "[::1]:9999", "[::1]:9999"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeAddress(tt.addr, "44207")
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCleanAndExpandPath(t *testing.T) {
	t.Setenv("OYSTERCLI_TEST_DIR", "/tmp/oystercli")

	home, err := os.UserHomeDir()
	require.NoError(t, err)

	tests := []struct {
		name string
		path string
		want string
	}{
		{"plain path", "/var/log/oyster.log", "/var/log/oyster.log"},
		{"env expansion", "$OYSTERCLI_TEST_DIR/logs", "/tmp/oystercli/logs"},
		{"tilde expansion", "~/wallet", filepath.Join(home, "wallet")},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, cleanAndExpandPath(tt.path))
		})
	}
}

func TestRescrapeConf(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	require.NoError(t, os.WriteFile(cfg.oysterConfPath(),
		[]byte("username=fresh\npassword=secret\nnoservertls=1\n"), 0o600))

	cfg.rescrapeConf()

	assert.Equal(t, "fresh", cfg.RPCUser)
	assert.Equal(t, "secret", cfg.RPCPass)
	assert.True(t, cfg.NoTLS)
	assert.Equal(t, "found", cfg.src.conf)
	assert.Equal(t, "oyster.conf (auto-provisioned)", cfg.src.creds)
}

func TestWalletDBPaths(t *testing.T) {
	dir := t.TempDir()
	cfg := &config{AppData: dir}
	cfg.activeNet = mainNetForTest()

	assert.Equal(t, filepath.Join(dir, "mainnet", "wallet.db"), cfg.walletDBPath())
	assert.Equal(t, filepath.Join(dir, "logs", "mainnet", "oyster.log"), cfg.logFilePath())
	assert.False(t, cfg.walletDBExists())

	require.NoError(t, os.MkdirAll(filepath.Join(dir, "mainnet"), 0o700))
	require.NoError(t, os.WriteFile(cfg.walletDBPath(), []byte("db"), 0o600))
	assert.True(t, cfg.walletDBExists())
}
