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

func TestWriteOysterConfCreatesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oyster.conf")
	vals := bootstrapValues{
		username: "alice",
		password: "hunter2",
		useSPV:   true,
		addPeer:  "seed.example.com:44108",
	}
	require.NoError(t, writeOysterConf(path, oysterConfValues{}, vals))

	fi, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), fi.Mode().Perm())

	got := scrapeOysterConf(path)
	assert.Equal(t, "alice", got.username)
	assert.Equal(t, "hunter2", got.password)

	content, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(content), "[Application Options]")
	assert.Contains(t, string(content), "usespv=1")
	assert.Contains(t, string(content), "addpeer=seed.example.com:44108")
	assert.NotContains(t, string(content), "noservertls")
	assert.NotContains(t, string(content), "rpcconnect")
}

func TestWriteOysterConfAppendsOnlyMissingKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oyster.conf")
	original := "[Application Options]\nusername=existing\nlogdir=/custom/logs"
	require.NoError(t, os.WriteFile(path, []byte(original), 0o600))

	vals := bootstrapValues{
		username:    "ignored",
		password:    "newpass",
		rpcConnect:  "localhost:44107",
		noServerTLS: true,
	}
	require.NoError(t, writeOysterConf(path, scrapeOysterConf(path), vals))

	content, err := os.ReadFile(path)
	require.NoError(t, err)
	text := string(content)

	// Existing content is preserved verbatim, missing keys are appended.
	assert.Contains(t, text, "username=existing")
	assert.Contains(t, text, "logdir=/custom/logs")
	assert.NotContains(t, text, "username=ignored")
	assert.Contains(t, text, "password=newpass")
	assert.Contains(t, text, "rpcconnect=localhost:44107")
	assert.Contains(t, text, "noservertls=1")

	got := scrapeOysterConf(path)
	assert.Equal(t, "existing", got.username)
	assert.Equal(t, "newpass", got.password)
	assert.True(t, got.noServerTLS)
}

func TestRandomHex(t *testing.T) {
	a := randomHex(24)
	b := randomHex(24)
	assert.Len(t, a, 48)
	assert.NotEqual(t, a, b)
}

func TestConfHasCredentials(t *testing.T) {
	cfg := &config{AppData: t.TempDir()}
	cfg.activeNet = mainNetForTest()
	assert.False(t, confHasCredentials(cfg))

	require.NoError(t, os.WriteFile(cfg.oysterConfPath(), []byte("username=u\n"), 0o600))
	assert.False(t, confHasCredentials(cfg))

	require.NoError(t, os.WriteFile(cfg.oysterConfPath(), []byte("username=u\npassword=p\n"), 0o600))
	assert.True(t, confHasCredentials(cfg))
}
