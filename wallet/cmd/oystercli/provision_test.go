// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAutoProvisionWritesSecureConfWhenAbsent(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()

	require.NoError(t, autoProvision(cfg))

	// Credentials are now resolved on the config.
	assert.NotEmpty(t, cfg.RPCUser)
	assert.NotEmpty(t, cfg.RPCPass)

	fi, err := os.Stat(cfg.oysterConfPath())
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), fi.Mode().Perm())

	content, err := os.ReadFile(cfg.oysterConfPath())
	require.NoError(t, err)
	text := string(content)
	// Secure defaults: SPV on, TLS untouched, and no pinned listener —
	// oyster defaults to loopback on the active network's port, so the
	// same conf must serve mainnet and testnet.
	assert.Contains(t, text, "usespv=1")
	assert.NotContains(t, text, "rpclisten=")
	assert.NotContains(t, text, "noservertls")
}

func TestAutoProvisionRespectsExistingCredentials(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	original := "[Application Options]\nusername=existing\npassword=secret\nnoservertls=1\nlogdir=/custom\n"
	require.NoError(t, os.WriteFile(cfg.oysterConfPath(), []byte(original), 0o600))

	require.NoError(t, autoProvision(cfg))

	// Nothing was rewritten — the file is byte-identical.
	content, err := os.ReadFile(cfg.oysterConfPath())
	require.NoError(t, err)
	assert.Equal(t, original, string(content))
	assert.Equal(t, "existing", cfg.RPCUser)
}

func TestAutoProvisionAppendsOnlyMissingCreds(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	// A conf that configures the backend but has no credentials (the
	// "ran with flags, lost the creds" case).
	original := "[Application Options]\nusespv=1\naddpeer=seed.example.com:44108\n"
	require.NoError(t, os.WriteFile(cfg.oysterConfPath(), []byte(original), 0o600))

	require.NoError(t, autoProvision(cfg))

	content, err := os.ReadFile(cfg.oysterConfPath())
	require.NoError(t, err)
	text := string(content)
	// Existing settings preserved verbatim; only creds appended.
	assert.Contains(t, text, "usespv=1")
	assert.Contains(t, text, "addpeer=seed.example.com:44108")
	assert.NotEmpty(t, cfg.RPCUser)
	assert.NotEmpty(t, cfg.RPCPass)
	got := scrapeOysterConf(cfg.oysterConfPath())
	assert.Equal(t, cfg.RPCUser, got.username)
	assert.Equal(t, cfg.RPCPass, got.password)
	// A pre-existing conf without a listener is not given one (we only add
	// credentials), so we do not override the user's network choices.
	assert.NotContains(t, text, "rpclisten")
}

func TestAutoProvisionTightensPermsBeforeAppendingCreds(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("chmod semantics differ for root")
	}
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	// A world-readable conf without credentials: os.WriteFile alone would
	// leave the generated password readable by every local user.
	require.NoError(t, os.WriteFile(cfg.oysterConfPath(),
		[]byte("[Application Options]\nusespv=1\n"), 0o644))

	require.NoError(t, autoProvision(cfg))

	fi, err := os.Stat(cfg.oysterConfPath())
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), fi.Mode().Perm())
	assert.True(t, confHasCredentials(cfg))
}

func TestInsecureConfWarnings(t *testing.T) {
	tests := []struct {
		name    string
		conf    string
		wantSub string
	}{
		{"noservertls", "username=u\npassword=p\nnoservertls=1\n", "without TLS"},
		{"wildcard listener", "username=u\npassword=p\nrpclisten=0.0.0.0:44207\n", "beyond localhost"},
		{"empty host listener", "username=u\npassword=p\nrpclisten=:44207\n", "beyond localhost"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config{AppData: t.TempDir()}
			cfg.activeNet = mainNetForTest()
			require.NoError(t, os.WriteFile(cfg.oysterConfPath(), []byte(tt.conf), 0o600))
			warnings := insecureConfWarnings(cfg)
			require.NotEmpty(t, warnings)
			assert.Contains(t, warnings[0], tt.wantSub)
		})
	}
}

func TestInsecureConfWarningsSecureConf(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	require.NoError(t, os.WriteFile(cfg.oysterConfPath(),
		[]byte("username=u\npassword=p\nrpclisten=127.0.0.1:44207\n"), 0o600))
	assert.Empty(t, insecureConfWarnings(cfg))
}

func TestIsLoopbackListener(t *testing.T) {
	tests := []struct {
		listen string
		want   bool
	}{
		{"127.0.0.1:44207", true},
		{"localhost:44207", true},
		{"[::1]:44207", true},
		{"0.0.0.0:44207", false},
		{":44207", false},
		{"192.168.1.5:44207", false},
		{"[::]:44207", false},
	}
	for _, tt := range tests {
		t.Run(tt.listen, func(t *testing.T) {
			assert.Equal(t, tt.want, isLoopbackListener(tt.listen))
		})
	}
}

func TestRandomHex(t *testing.T) {
	a := randomHex(24)
	b := randomHex(24)
	assert.Len(t, a, 48)
	assert.NotEqual(t, a, b)
}
