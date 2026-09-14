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

func TestFindOysterBinary(t *testing.T) {
	// Make PATH lookups fail deterministically.
	t.Setenv("PATH", t.TempDir())

	t.Run("explicit path wins", func(t *testing.T) {
		bin := filepath.Join(t.TempDir(), "custom-oyster")
		writeFakeBinary(t, bin)
		cfg := &config{OysterBin: bin}
		p, src, err := findOysterBinary(cfg)
		require.NoError(t, err)
		assert.Equal(t, bin, p)
		assert.Equal(t, srcFlag, src)
	})

	t.Run("explicit path missing errors", func(t *testing.T) {
		cfg := &config{OysterBin: filepath.Join(t.TempDir(), "nope")}
		_, _, err := findOysterBinary(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "--oysterbin")
	})

	t.Run("found in PATH", func(t *testing.T) {
		dir := t.TempDir()
		writeFakeBinary(t, filepath.Join(dir, "oyster"))
		t.Setenv("PATH", dir)
		cfg := &config{OysterBin: "oyster"}
		p, src, err := findOysterBinary(cfg)
		require.NoError(t, err)
		assert.Equal(t, filepath.Join(dir, "oyster"), p)
		assert.Equal(t, srcPath, src)
	})

	// Binaries in untrusted implicit locations must never be picked up:
	// oystercli passes wallet passphrases and seeds to the daemon it runs.
	t.Run("cwd is not searched", func(t *testing.T) {
		dir := t.TempDir()
		writeFakeBinary(t, filepath.Join(dir, "oyster"))
		writeFakeBinary(t, filepath.Join(dir, "bin", "oyster"))
		t.Chdir(dir)
		cfg := &config{OysterBin: "oyster"}
		_, _, err := findOysterBinary(cfg)
		require.Error(t, err)
	})

	t.Run("executable dir is not searched", func(t *testing.T) {
		exe, err := os.Executable()
		require.NoError(t, err)
		sibling := filepath.Join(filepath.Dir(exe), "oyster")
		writeFakeBinary(t, sibling)
		t.Cleanup(func() { os.Remove(sibling) })

		cfg := &config{OysterBin: "oyster"}
		_, _, err = findOysterBinary(cfg)
		require.Error(t, err)
	})

	t.Run("not found reports remedies", func(t *testing.T) {
		cfg := &config{OysterBin: "oyster"}
		_, _, err := findOysterBinary(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not on your $PATH")
		assert.Contains(t, err.Error(), "--oysterbin")
	})

	t.Run("result is cached", func(t *testing.T) {
		bin := filepath.Join(t.TempDir(), "oyster")
		writeFakeBinary(t, bin)
		cfg := &config{OysterBin: bin}
		_, _, err := findOysterBinary(cfg)
		require.NoError(t, err)

		// Remove the file: the cached path must still be returned.
		require.NoError(t, os.Remove(bin))
		p, _, err := findOysterBinary(cfg)
		require.NoError(t, err)
		assert.Equal(t, bin, p)
	})
}

func TestIsExecutableFile(t *testing.T) {
	dir := t.TempDir()

	binary := filepath.Join(dir, "binary")
	writeFakeBinary(t, binary)
	assert.True(t, isExecutableFile(binary))

	plain := filepath.Join(dir, "plain")
	require.NoError(t, os.WriteFile(plain, []byte("x"), 0o644))
	assert.False(t, isExecutableFile(plain))

	assert.False(t, isExecutableFile(dir))
	assert.False(t, isExecutableFile(filepath.Join(dir, "missing")))
}

func TestShellQuote(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain", "/usr/local/bin/oyster", "/usr/local/bin/oyster"},
		{"spaces", "/Users/or/Library/Application Support/Oyster", "'/Users/or/Library/Application Support/Oyster'"},
		{"single quote", "it's", `'it'\''s'`},
		{"empty", "", "''"},
		{"tilde", "~/bin/oyster", "'~/bin/oyster'"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, shellQuote(tt.in))
		})
	}
}

func TestSpawnArgs(t *testing.T) {
	t.Run("default appdata mainnet has no args", func(t *testing.T) {
		cfg := &config{AppData: oysterHomeDir}
		cfg.activeNet = mainNetForTest()
		assert.Empty(t, spawnArgs(cfg))
	})

	t.Run("custom appdata and network are passed", func(t *testing.T) {
		cfg := &config{AppData: "/tmp/custom", TestNet2: true}
		cfg.activeNet = mainNetForTest()
		assert.Equal(t, []string{"--appdata=/tmp/custom", "--testnet2"}, spawnArgs(cfg))
	})
}
