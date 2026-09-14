// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/version"
)

type checkStatus int

const (
	checkPass checkStatus = iota
	checkWarn
	checkFail
)

type checkResult struct {
	name   string
	status checkStatus
	detail string
}

// runDoctor executes the diagnostic checklist and offers to export a
// redacted report. The client may be nil (pre-connection triage); RPC checks
// then dial a throwaway connection themselves.
func runDoctor(cfg *config, c *client) {
	var results []checkResult
	_ = withSpinner("Running checks...", func() error {
		results = collectChecks(cfg, c)
		return nil
	})

	printTitle("Doctor")
	var lines []string
	for _, r := range results {
		lines = append(lines, renderCheck(r))
	}
	printBox(strings.Join(lines, "\n"))

	failed := 0
	for _, r := range results {
		if r.status == checkFail {
			failed++
		}
	}
	if failed == 0 {
		printSuccess("No blocking problems found.")
	} else {
		printWarn(fmt.Sprintf("%d check(s) failed; see details above.", failed))
	}

	export := false
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewConfirm().
			Title("Export a report file?").
			Description("Credentials are never included, so it is safe to attach to a bug report.").
			Affirmative("Export").
			Negative("Skip").
			Value(&export),
	)))
	if err != nil || !ok || !export {
		return
	}
	path, err := exportDoctorReport(cfg, results)
	if err != nil {
		printError(err)
		return
	}
	printSuccess("Report written to " + path)
}

func renderCheck(r checkResult) string {
	var mark string
	switch r.status {
	case checkPass:
		mark = th.good.Render("✓")
	case checkWarn:
		mark = th.warn.Render("!")
	case checkFail:
		mark = th.bad.Render("✗")
	}
	return fmt.Sprintf("%s %s %s", mark, th.value.Render(fmt.Sprintf("%-22s", r.name)), th.subtle.Render(r.detail))
}

// collectChecks runs every diagnostic and returns the results in display
// order. Checks are ordered so each failure explains the ones after it.
func collectChecks(cfg *config, c *client) []checkResult {
	results := []checkResult{
		checkConfigFile(cfg),
		checkCredentials(cfg),
		checkConnectTarget(cfg),
		checkCertificate(cfg),
		checkWalletDB(cfg),
		checkOysterBinary(cfg),
		checkTCP(cfg),
	}
	if !cfg.NoTLS {
		results = append(results, checkTLSHandshake(cfg))
	}
	results = append(results, checkRPC(cfg, c)...)
	results = append(results, checkLogFile(cfg))
	return results
}

func checkConfigFile(cfg *config) checkResult {
	path := cfg.oysterConfPath()
	if fileExists(path) {
		return checkResult{"oyster.conf", checkPass, path}
	}
	return checkResult{"oyster.conf", checkWarn, path + " not found (a secure one is written automatically)"}
}

func checkCredentials(cfg *config) checkResult {
	source := ""
	if cfg.src.creds != "" {
		source = " (source: " + cfg.src.creds + ")"
	}
	switch {
	case cfg.RPCUser != "" && cfg.RPCPass != "":
		return checkResult{"RPC credentials", checkPass, "username and password are set" + source}
	case cfg.RPCUser != "":
		return checkResult{"RPC credentials", checkFail, "password is missing" + source}
	default:
		return checkResult{"RPC credentials", checkFail, "username and password are missing" + source}
	}
}

func checkConnectTarget(cfg *config) checkResult {
	return checkResult{"Connect target", checkPass, cfg.Connect + " (source: " + cfg.src.connect + ")"}
}

func checkOysterBinary(cfg *config) checkResult {
	if p, src, err := findOysterBinary(cfg); err == nil {
		return checkResult{"oyster binary", checkPass, p + " (" + src + ")"}
	}
	return checkResult{"oyster binary", checkWarn,
		"not on $PATH — only needed to create wallets or start the daemon from here (use --oysterbin or the prompt)"}
}

func checkCertificate(cfg *config) checkResult {
	if cfg.NoTLS {
		return checkResult{"TLS", checkWarn, "disabled (--notls); fine for localhost-only setups"}
	}
	pem, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return checkResult{"TLS certificate", checkFail, fmt.Sprintf("cannot read %s", cfg.CAFile)}
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return checkResult{"TLS certificate", checkFail, cfg.CAFile + " contains no valid certificates"}
	}
	return checkResult{"TLS certificate", checkPass, cfg.CAFile}
}

func checkWalletDB(cfg *config) checkResult {
	path := cfg.walletDBPath()
	if cfg.walletDBExists() {
		return checkResult{"wallet.db", checkPass, path}
	}
	return checkResult{"wallet.db", checkFail, path + " does not exist (create a wallet first)"}
}

func checkTCP(cfg *config) checkResult {
	conn, err := net.DialTimeout("tcp", cfg.Connect, 3*time.Second)
	if err != nil {
		return checkResult{"TCP reachability", checkFail, fmt.Sprintf("%s: %v (daemon not running?)", cfg.Connect, err)}
	}
	conn.Close()
	return checkResult{"TCP reachability", checkPass, cfg.Connect + " accepts connections"}
}

func checkTLSHandshake(cfg *config) checkResult {
	pem, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return checkResult{"TLS handshake", checkWarn, "skipped (certificate unreadable)"}
	}
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(pem)
	host, _, _ := net.SplitHostPort(cfg.Connect)
	conn, err := tls.DialWithDialer(
		&net.Dialer{Timeout: 3 * time.Second}, "tcp", cfg.Connect,
		&tls.Config{RootCAs: pool, ServerName: host},
	)
	if err != nil {
		return checkResult{"TLS handshake", checkFail, err.Error()}
	}
	conn.Close()
	return checkResult{"TLS handshake", checkPass, "certificate accepted"}
}

// checkRPC verifies auth and reports wallet, sync, and chain state. It
// reuses the session client when available.
func checkRPC(cfg *config, c *client) []checkResult {
	if c == nil {
		var err error
		c, err = dialClient(cfg)
		if err != nil {
			return []checkResult{{"RPC connection", checkFail, err.Error()}}
		}
		defer c.shutdown()
	}

	locked, err := c.walletLocked()
	if err != nil {
		return []checkResult{{"RPC auth", checkFail, friendlyError(err)}}
	}
	results := []checkResult{{"RPC auth", checkPass, "authenticated to oyster"}}

	lockDetail := "wallet is unlocked"
	if locked {
		lockDetail = "wallet is locked (normal at rest)"
	}
	results = append(results, checkResult{"Wallet lock", checkPass, lockDetail})

	if si, err := c.fetchSyncInfo(); err != nil {
		results = append(results, checkResult{"Chain sync", checkFail,
			"sync state unavailable: " + friendlyError(err)})
	} else if si.synced {
		detail := "synced"
		if si.height > 0 {
			detail = fmt.Sprintf("synced at height %d", si.height)
		}
		results = append(results, checkResult{"Chain sync", checkPass, detail})
	} else {
		results = append(results, checkResult{"Chain sync", checkWarn,
			"still syncing: " + syncPercent(si)})
	}

	if info, err := c.info(); err == nil {
		results = append(results, checkResult{"pearld", checkPass,
			fmt.Sprintf("version %d, %d peer(s)", info.Version, info.Connections)})
	} else {
		results = append(results, checkResult{"pearld", checkWarn,
			"getinfo failed (oyster cannot reach pearld): " + friendlyError(err)})
	}
	return results
}

func checkLogFile(cfg *config) checkResult {
	path := cfg.logFilePath()
	fi, err := os.Stat(path)
	if err != nil {
		return checkResult{"Log file", checkWarn, path + " not found"}
	}
	return checkResult{"Log file", checkPass, fmt.Sprintf("%s (%d KB)", path, fi.Size()/1024)}
}

// exportDoctorReport writes the checklist as plain text next to the current
// directory. Credentials are deliberately excluded.
func exportDoctorReport(cfg *config, results []checkResult) (string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "%s doctor report\n", appName)
	fmt.Fprintf(&b, "generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "version:   %s\n", version.Version())
	fmt.Fprintf(&b, "network:   %s\n", cfg.activeNet.Params.Name)
	fmt.Fprintf(&b, "rpc:       %s (tls: %v)\n", cfg.Connect, !cfg.NoTLS)
	fmt.Fprintf(&b, "appdata:   %s\n\n", cfg.AppData)
	for _, r := range results {
		status := map[checkStatus]string{checkPass: "PASS", checkWarn: "WARN", checkFail: "FAIL"}[r.status]
		fmt.Fprintf(&b, "[%s] %-22s %s\n", status, r.name, r.detail)
	}

	path := fmt.Sprintf("oystercli-doctor-%s.txt", time.Now().Format("20060102-150405"))
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		return "", err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return path, nil
	}
	return abs, nil
}
