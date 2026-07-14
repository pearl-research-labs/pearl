// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Daemon binary discovery and lifecycle: locating the oyster executable,
// rendering start commands, and spawning it detached from this process.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// spawnReadyTimeout bounds how long startOysterNow waits for the spawned
// daemon's RPC to come up.
const spawnReadyTimeout = 30 * time.Second

// findOysterBinary resolves the daemon binary and reports where it was
// found. Search order: an explicit --oysterbin path, $PATH, the directory of
// the running oystercli executable (release archives and the repo bin/ ship
// the binaries side by side), and finally ./bin relative to the working
// directory. The result is cached on the config.
func findOysterBinary(cfg *config) (path, source string, err error) {
	if cfg.resolvedOysterBin != "" {
		return cfg.resolvedOysterBin, cfg.resolvedOysterSrc, nil
	}

	remember := func(p, src string) (string, string, error) {
		cfg.resolvedOysterBin, cfg.resolvedOysterSrc = p, src
		return p, src, nil
	}

	if strings.ContainsRune(cfg.OysterBin, os.PathSeparator) {
		if !isExecutableFile(cfg.OysterBin) {
			return "", "", fmt.Errorf("no executable oyster binary at %s (from --oysterbin)", cfg.OysterBin)
		}
		return remember(cfg.OysterBin, "--oysterbin")
	}

	if p, lerr := exec.LookPath(cfg.OysterBin); lerr == nil {
		return remember(p, "PATH")
	}
	for _, candidate := range siblingCandidates(cfg.OysterBin) {
		if isExecutableFile(candidate) {
			return remember(candidate, "next to "+appName)
		}
	}
	for _, name := range windowsAware(cfg.OysterBin) {
		if p := filepath.Join("bin", name); isExecutableFile(p) {
			return remember(p, "./bin")
		}
	}

	return "", "", fmt.Errorf("cannot find the oyster binary; searched $PATH, %s, and ./bin — point --oysterbin at it or locate it when asked",
		strings.Join(siblingDirs(), ", "))
}

// locateOysterBinary resolves the daemon binary, and when automatic
// discovery fails, asks the user for its location instead of dead-ending.
// The answer is remembered for the rest of the session.
func locateOysterBinary(cfg *config) (string, error) {
	path, _, err := findOysterBinary(cfg)
	if err == nil {
		return path, nil
	}
	printError(err)

	var entered string
	ok, ferr := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Path to the oyster binary").
			Description("It ships next to " + appName + " in release archives, or build it with `task build:oyster`.").
			Placeholder("/path/to/oyster").
			Validate(func(s string) error {
				p := cleanAndExpandPath(strings.TrimSpace(s))
				if p == "" || !isExecutableFile(p) {
					return fmt.Errorf("no executable file there")
				}
				return nil
			}).
			Value(&entered),
	)))
	if ferr != nil {
		return "", ferr
	}
	// Re-check outside the form: huh's accessible mode skips field
	// validators when stdin reaches EOF, handing back an empty value.
	path = cleanAndExpandPath(strings.TrimSpace(entered))
	if !ok || !isExecutableFile(path) {
		return "", err
	}
	cfg.resolvedOysterBin, cfg.resolvedOysterSrc = path, "located interactively"
	return path, nil
}

// siblingCandidates lists possible daemon locations in the directory of the
// running executable (following a symlinked oystercli to its real home).
func siblingCandidates(name string) []string {
	var candidates []string
	for _, dir := range siblingDirs() {
		for _, n := range windowsAware(name) {
			candidates = append(candidates, filepath.Join(dir, n))
		}
	}
	return candidates
}

// siblingDirs returns the directories the running executable lives in, both
// as invoked and with symlinks resolved.
func siblingDirs() []string {
	exe, err := os.Executable()
	if err != nil {
		return nil
	}
	dirs := []string{filepath.Dir(exe)}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		if dir := filepath.Dir(resolved); dir != dirs[0] {
			dirs = append(dirs, dir)
		}
	}
	return dirs
}

// windowsAware returns the file names to try for a binary: as-is, plus the
// .exe form on Windows.
func windowsAware(name string) []string {
	if runtime.GOOS == "windows" && !strings.HasSuffix(strings.ToLower(name), ".exe") {
		return []string{name + ".exe", name}
	}
	return []string{name}
}

// isExecutableFile reports whether path is a regular file the current user
// could plausibly execute.
func isExecutableFile(path string) bool {
	fi, err := os.Stat(path)
	if err != nil || fi.IsDir() {
		return false
	}
	if runtime.GOOS == "windows" {
		return true
	}
	return fi.Mode().Perm()&0o111 != 0
}

// oysterStartCommand renders a copy-pasteable daemon invocation for the
// active network, using the resolved binary path when discovery succeeds.
func oysterStartCommand(cfg *config) string {
	bin := cfg.OysterBin
	if p, _, err := findOysterBinary(cfg); err == nil {
		bin = p
	}
	cmd := shellQuote(bin)
	for _, arg := range spawnArgs(cfg) {
		cmd += " " + shellQuote(arg)
	}
	return cmd
}

// shellQuote makes a string safe to paste into a shell (paths like
// "~/Library/Application Support/..." contain spaces).
func shellQuote(s string) string {
	if s != "" && !strings.ContainsAny(s, " \t'\"\\$&|;<>()*?[]#~`") {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// networkFlag returns the daemon flag selecting the active network, empty
// for mainnet.
func networkFlag(cfg *config) string {
	switch {
	case cfg.TestNet:
		return "--testnet"
	case cfg.TestNet2:
		return "--testnet2"
	case cfg.SimNet:
		return "--simnet"
	case cfg.SigNet:
		return "--signet"
	}
	return ""
}

// spawnArgs builds the daemon invocation. Configuration lives in oyster.conf;
// only the network and a non-default appdata are passed explicitly (they
// select which conf/wallet to use, so they cannot come from the conf itself).
func spawnArgs(cfg *config) []string {
	var args []string
	if cfg.AppData != oysterHomeDir {
		args = append(args, "--appdata="+cfg.AppData)
	}
	if flag := networkFlag(cfg); flag != "" {
		args = append(args, flag)
	}
	return args
}

// startOysterNow launches the daemon detached from this process (it keeps
// running after oystercli exits) and waits until its RPC answers. Requires
// oyster.conf to carry the credentials, which the caller guarantees via the
// bootstrap wizard.
func startOysterNow(cfg *config) error {
	binPath, err := locateOysterBinary(cfg)
	if err != nil {
		return err
	}

	cmd := exec.Command(binPath, spawnArgs(cfg)...)
	// The daemon must outlive this process: detach it from our process
	// group (Ctrl+C here must not kill it) and never hand it pipes, since
	// a closed pipe would SIGPIPE-kill it on its next log write.
	cmd.SysProcAttr = detachSysProcAttr()
	devnull, err := os.OpenFile(os.DevNull, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer devnull.Close()
	cmd.Stdin, cmd.Stdout, cmd.Stderr = devnull, devnull, devnull

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start %s: %w", binPath, err)
	}
	pid := cmd.Process.Pid
	_ = cmd.Process.Release()

	lipgloss.Println(th.subtle.Render(fmt.Sprintf("Started %s (pid %d), waiting for its RPC to come up...", binPath, pid)))

	if err := waitForOysterRPC(cfg, spawnReadyTimeout); err != nil {
		printError(err)
		printWarn("The daemon did not become ready in time; its most recent log lines:")
		printLogTail(cfg.logFilePath(), 15)
		return fmt.Errorf("oyster (pid %d) failed to become ready", pid)
	}

	printSuccess(fmt.Sprintf("oyster is running (pid %d). It keeps running after you quit; stop it with: kill %d", pid, pid))
	lipgloss.Println(th.subtle.Render("Logs: " + cfg.logFilePath()))
	return nil
}

// waitForOysterRPC polls the wallet RPC until it responds or the timeout
// elapses.
func waitForOysterRPC(cfg *config, timeout time.Duration) error {
	c, err := dialClient(cfg)
	if err != nil {
		return err
	}
	defer c.shutdown()

	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if _, lastErr = c.walletLocked(); lastErr == nil {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
}
