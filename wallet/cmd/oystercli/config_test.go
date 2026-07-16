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

// The connect target must follow an explicit rpclisten in oyster.conf: the
// daemon listens exactly where the conf says, whatever the active network's
// default port is. Wildcard listeners are dialed via loopback.
func TestDialTarget(t *testing.T) {
	tests := []struct {
		listen string
		want   string
	}{
		{"127.0.0.1:44209", "127.0.0.1:44209"},
		{"127.0.0.1", "127.0.0.1:44207"},
		{"[::1]:44209", "[::1]:44209"},
		{"0.0.0.0:44209", "localhost:44209"},
		{":44209", "localhost:44209"},
		{"localhost:44209", "localhost:44209"},
	}
	for _, tt := range tests {
		t.Run(tt.listen, func(t *testing.T) {
			assert.Equal(t, tt.want, dialTarget(tt.listen, "44207"))
		})
	}
}

// Local bootstrapping (provisioning, wallet creation, daemon spawning) must
// be suppressed for targets on other machines.
func TestRemoteTarget(t *testing.T) {
	tests := []struct {
		connect string
		want    bool
	}{
		{"localhost:44207", false},
		{"127.0.0.1:44207", false},
		{"[::1]:44207", false},
		{"127.0.0.2:44207", false},
		{"192.168.1.50:44207", true},
		{"wallet.example.com:44207", true},
		{"[2001:db8::1]:44207", true},
	}
	for _, tt := range tests {
		t.Run(tt.connect, func(t *testing.T) {
			cfg := &config{Connect: tt.connect}
			assert.Equal(t, tt.want, cfg.remoteTarget())
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
