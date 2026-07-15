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
	"runtime"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// spawnReadyTimeout bounds how long startOysterNow waits for the spawned
// daemon's RPC to come up.
const spawnReadyTimeout = 30 * time.Second

// Binary discovery source labels, also shown to the user in the resolution
// story, the doctor, and before the binary is executed.
const (
	srcFlag    = "--oysterbin"
	srcPath    = "PATH"
	srcEntered = "provided path"
)

// findOysterBinary resolves the daemon binary: an explicit --oysterbin path,
// then $PATH — nothing else. The release installers put oyster on PATH; that
// is the supported setup. Deliberately no cwd or executable-relative lookup:
// implicitly running a binary from the working directory is an untrusted
// search path (this binary receives wallet passphrases and seeds), and
// developers running from a `task build` tree can pass --oysterbin or answer
// the prompt. The result is cached on the config.
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
		return remember(cfg.OysterBin, srcFlag)
	}

	if p, lerr := exec.LookPath(cfg.OysterBin); lerr == nil {
		return remember(p, srcPath)
	}

	return "", "", fmt.Errorf("%s is not on your $PATH; install it with the release installer, pass --oysterbin, or provide its path when asked",
		cfg.OysterBin)
}

// locateOysterBinary resolves the daemon binary, and when it is not on
// $PATH, asks for its exact location instead of dead-ending (developers
// running from a `task build` tree point it at bin/oyster). The answer is
// remembered for the rest of the session, and the chosen binary is always
// announced with its origin so there is no ambiguity about what will run.
func locateOysterBinary(cfg *config) (string, error) {
	path, source, err := findOysterBinary(cfg)
	if err != nil {
		printError(err)

		var entered string
		ok, ferr := runForm(newForm(huh.NewGroup(
			huh.NewInput().
				Title("Path to the oyster binary").
				Description("Installed by install.sh/install.ps1, or built with `task build:oyster` into bin/.").
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
		source = srcEntered
		cfg.resolvedOysterBin, cfg.resolvedOysterSrc = path, source
	}

	lipgloss.Println(th.subtle.Render("Using oyster binary: ") + th.accent.Render(path) + th.subtle.Render(" (from "+source+")"))
	return path, nil
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
// oyster.conf to carry the credentials, which autoProvision guarantees.
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

	printSuccess(fmt.Sprintf("oyster is running (pid %d). It keeps running after you quit.", pid))
	lipgloss.Println(th.subtle.Render(fmt.Sprintf("Stop it later from Node & sync, or with: kill %d  ·  Logs: %s", pid, cfg.logFilePath())))
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
