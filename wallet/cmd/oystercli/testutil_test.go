// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/netparams"
	"github.com/stretchr/testify/require"
)

func mainNetForTest() *netparams.Params {
	return &netparams.MainNetParams
}

// configWithWalletDB returns a config whose appdata contains a mainnet
// wallet.db placeholder.
func configWithWalletDB(t *testing.T) *config {
	t.Helper()
	dir := t.TempDir()
	cfg := &config{AppData: dir}
	cfg.activeNet = mainNetForTest()
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "mainnet"), 0o700))
	require.NoError(t, os.WriteFile(cfg.walletDBPath(), []byte("db"), 0o600))
	return cfg
}

// writeFakeBinary creates an executable placeholder file.
func writeFakeBinary(t *testing.T, path string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755))
}
