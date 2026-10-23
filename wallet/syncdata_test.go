// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/wallet/internal/cfgutil"
	"github.com/pearl-research-labs/pearl/wallet/netparams"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/stretchr/testify/require"
)

// useTestChain points the global config at a simnet SPV wallet and returns its network directory.
func useTestChain(t *testing.T, dbTimeout time.Duration) string {
	t.Helper()

	// Logging needs the log rotator, which tests don't set up.
	setLogLevels("off")

	appData := t.TempDir()
	prevCfg, prevNet := cfg, activeNet
	cfg = &config{
		UseSPV:     true,
		DBTimeout:  dbTimeout,
		AppDataDir: cfgutil.NewExplicitString(appData),
	}
	activeNet = &netparams.SimNetParams
	t.Cleanup(func() { cfg, activeNet = prevCfg, prevNet })

	netDir := networkDir(appData, activeNet.Params)
	require.NoError(t, os.MkdirAll(netDir, 0o700))

	return netDir
}

func writeFiles(t *testing.T, dir string, files map[string]string) {
	t.Helper()

	for name, content := range files {
		err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600)
		require.NoError(t, err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()

	content, err := os.ReadFile(path)
	require.NoError(t, err)

	return string(content)
}

// openSyncFiles lists the chain sync files in dir this process holds open. Only Linux lists them, through /proc.
func openSyncFiles(t *testing.T, dir string) []string {
	t.Helper()

	if runtime.GOOS != "linux" {
		return nil
	}
	fds, err := os.ReadDir("/proc/self/fd")
	require.NoError(t, err)

	var open []string
	for _, fd := range fds {
		target, _ := os.Readlink(filepath.Join("/proc/self/fd", fd.Name()))
		if filepath.Dir(target) == dir && slices.Contains(syncDataFiles, filepath.Base(target)) {
			open = append(open, filepath.Base(target))
		}
	}

	return open
}

// runConnectLoop starts the connect loop and returns a function that stops it.
func runConnectLoop(t *testing.T, loader *wallet.Loader) func() {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		rpcClientConnectLoop(ctx, nil, loader)
	}()

	stop := func() {
		cancel()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatal("connect loop did not stop")
		}
	}
	t.Cleanup(cancel)

	return stop
}

// TestResetSyncData makes sure only the chain sync files are moved aside.
func TestResetSyncData(t *testing.T) {
	netDir := useTestChain(t, time.Second)
	files := map[string]string{
		neutrinoDBName:                "neutrino",
		headerfs.BlockHeadersFileName: "blocks",
		"wallet.db":                   "wallet",
		"peers.json":                  "peers",
	}
	writeFiles(t, netDir, files)

	require.NoError(t, resetSyncData(netDir))

	for name, content := range files {
		path := filepath.Join(netDir, name)
		if !slices.Contains(syncDataFiles, name) {
			require.Equal(t, content, readFile(t, path))
			continue
		}

		require.NoFileExists(t, path)
		moved, err := filepath.Glob(path + ".corrupt-*")
		require.NoError(t, err)
		require.Len(t, moved, 1)
		require.Equal(t, content, readFile(t, moved[0]))
	}

	entries, err := os.ReadDir(netDir)
	require.NoError(t, err)
	require.Len(t, entries, len(files))
}

// TestConnectLoopResetsCorruptSyncData runs the connect loop on header files without the database that indexes them:
// the chain sync data is moved aside and the chain service starts over.
func TestConnectLoopResetsCorruptSyncData(t *testing.T) {
	netDir := useTestChain(t, 5*time.Second)
	headers := string(make([]byte, 5*headerfs.BlockHeaderSize))
	writeFiles(t, netDir, map[string]string{
		headerfs.BlockHeadersFileName: headers,
		"wallet.db":                   "wallet",
	})

	loader := wallet.NewLoader(activeNet.Params, netDir, true, cfg.DBTimeout, 250)
	stop := runConnectLoop(t, loader)

	// A fresh header file holds just the genesis header.
	path := filepath.Join(netDir, headerfs.BlockHeadersFileName)
	require.Eventually(t, func() bool {
		info, err := os.Stat(path)
		return err == nil && info.Size() == headerfs.BlockHeaderSize
	}, 20*time.Second, 20*time.Millisecond)
	stop()

	moved, err := filepath.Glob(path + ".corrupt-*")
	require.NoError(t, err)
	require.Len(t, moved, 1)
	require.Equal(t, headers, readFile(t, moved[0]))
	require.Equal(t, "wallet", readFile(t, filepath.Join(netDir, "wallet.db")))
	require.Empty(t, openSyncFiles(t, netDir))
}

// TestConnectLoopReconnectsAfterClientStops makes sure a chain client that goes away is replaced by one on the same
// database, which only works if the previous attempt let go of it.
func TestConnectLoopReconnectsAfterClientStops(t *testing.T) {
	netDir := useTestChain(t, 3*time.Second)

	prevScrypt := waddrmgr.DefaultScryptOptions
	waddrmgr.DefaultScryptOptions = waddrmgr.FastScryptOptions
	t.Cleanup(func() { waddrmgr.DefaultScryptOptions = prevScrypt })

	loader := wallet.NewLoader(activeNet.Params, netDir, true, cfg.DBTimeout, 250)
	w, err := loader.CreateNewWatchingOnlyWallet([]byte("public"), time.Now())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, loader.UnloadWallet()) })

	stop := runConnectLoop(t, loader)

	require.Eventually(t, func() bool {
		return w.ChainClient() != nil
	}, 20*time.Second, 20*time.Millisecond)
	first := w.ChainClient()
	first.Stop()

	require.Eventually(t, func() bool {
		current := w.ChainClient()
		return current != nil && current != first
	}, 20*time.Second, 20*time.Millisecond)

	stop()
	require.Empty(t, openSyncFiles(t, netDir))
}

// TestConnectLoopStopsWhileRetrying makes sure shutdown doesn't wait out the delay before the next attempt.
func TestConnectLoopStopsWhileRetrying(t *testing.T) {
	netDir := useTestChain(t, 50*time.Millisecond)

	prevMin, prevMax := chainClientRetryMin, chainClientRetryMax
	chainClientRetryMin, chainClientRetryMax = time.Minute, time.Minute
	t.Cleanup(func() { chainClientRetryMin, chainClientRetryMax = prevMin, prevMax })

	// Every attempt times out on a database held by someone else.
	held, err := walletdb.Create("bdb", filepath.Join(netDir, neutrinoDBName), true, time.Second, false)
	require.NoError(t, err)
	defer held.Close()

	loader := wallet.NewLoader(activeNet.Params, netDir, true, cfg.DBTimeout, 250)
	stop := runConnectLoop(t, loader)
	time.Sleep(300 * time.Millisecond)
	stop()
}
