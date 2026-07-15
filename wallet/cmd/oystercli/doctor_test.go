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

func TestCheckCertificate(t *testing.T) {
	t.Run("notls warns", func(t *testing.T) {
		got := checkCertificate(&config{NoTLS: true})
		assert.Equal(t, checkWarn, got.status)
	})

	t.Run("missing file fails", func(t *testing.T) {
		got := checkCertificate(&config{CAFile: filepath.Join(t.TempDir(), "rpc.cert")})
		assert.Equal(t, checkFail, got.status)
	})

	t.Run("garbage pem fails", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "rpc.cert")
		require.NoError(t, os.WriteFile(path, []byte("not a cert"), 0o600))
		got := checkCertificate(&config{CAFile: path})
		assert.Equal(t, checkFail, got.status)
	})
}

func TestExportDoctorReportRedactsSecrets(t *testing.T) {
	t.Chdir(t.TempDir())

	cfg := &config{
		AppData: "/home/user/.oyster",
		Connect: "localhost:44207",
		RPCUser: "topsecretuser",
		RPCPass: "topsecretpass",
	}
	cfg.activeNet = mainNetForTest()

	results := []checkResult{
		{"RPC auth", checkPass, "authenticated to oyster"},
		{"Chain sync", checkWarn, "still syncing"},
	}
	path, err := exportDoctorReport(cfg, results)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	report := string(data)

	assert.Contains(t, report, "[PASS] RPC auth")
	assert.Contains(t, report, "[WARN] Chain sync")
	assert.Contains(t, report, "mainnet")
	assert.NotContains(t, report, "topsecretuser")
	assert.NotContains(t, report, "topsecretpass")
}
