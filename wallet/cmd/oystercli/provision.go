// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Zero-configuration provisioning: when there is no usable oyster config,
// oystercli writes a secure one itself (generated credentials, loopback RPC,
// TLS on, SPV backend) rather than asking the user to make choices. An
// existing config is always respected and never overridden.

package main

import (
	"cmp"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"

	"charm.land/lipgloss/v2"
)

// provisionedConf is the secure configuration oystercli writes when none
// exists.
type provisionedConf struct {
	username string
	password string
	useSPV   bool
}

// autoProvision ensures oystercli has RPC credentials to connect with or
// spawn from, choosing secure defaults with no prompts:
//
//   - If credentials are already resolved (flags or an existing oyster.conf),
//     do nothing.
//   - Otherwise write a secure oyster.conf: generated credentials, TLS on
//     (the oyster default), and SPV so no pearld is needed. A missing file
//     is created; an existing file only has the missing credential keys
//     appended — other settings are never touched.
//
// No rpclisten is written: oyster's own default is loopback-only listeners
// on the active network's RPC port, so omitting it keeps the wallet
// unreachable off-host while letting one conf serve every network (a pinned
// port would make a --testnet daemon listen on the mainnet port).
func autoProvision(cfg *config) error {
	if cfg.RPCUser != "" && cfg.RPCPass != "" {
		return nil
	}

	confPath := cfg.oysterConfPath()
	existing := scrapeOysterConf(confPath)
	pc := provisionedConf{
		username: cmp.Or(existing.username, appName+"-"+randomHex(4)),
		password: cmp.Or(existing.password, randomHex(24)),
		useSPV:   true,
	}

	created, err := writeProvisionedConf(confPath, existing, pc)
	if err != nil {
		return err
	}
	cfg.rescrapeConf()

	if created {
		printSuccess("No oyster config found; wrote a secure one to " + confPath)
		lipgloss.Println(th.subtle.Render("  generated credentials · loopback RPC · TLS on · SPV sync"))
	} else {
		printSuccess("Added the missing RPC credentials to " + confPath)
	}
	return nil
}

// writeProvisionedConf creates a secure config when none exists, or appends
// only the missing credential keys to an existing one. It reports whether a
// new file was created. Existing settings are never rewritten.
func writeProvisionedConf(path string, existing oysterConfValues, pc provisionedConf) (created bool, err error) {
	content, readErr := os.ReadFile(path)
	switch {
	case errors.Is(readErr, os.ErrNotExist):
		lines := []string{
			"[Application Options]",
			"username=" + pc.username,
			"password=" + pc.password,
		}
		if pc.useSPV {
			lines = append(lines, "usespv=1")
		}
		// TLS and rpclisten are deliberately left at oyster's defaults
		// (TLS on, loopback-only listeners on the active network's
		// port), so this one conf serves mainnet and testnet alike.
		lines = append(lines, "; rpc listeners default to loopback on the active network's port")
		return true, os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600)
	case readErr != nil:
		return false, readErr
	}

	// The config exists but lacks credentials: append only those, so we
	// never clobber the user's chosen backend/TLS/listener settings.
	var add []string
	if existing.username == "" {
		add = append(add, "username="+pc.username)
	}
	if existing.password == "" {
		add = append(add, "password="+pc.password)
	}
	if len(add) == 0 {
		return false, nil
	}

	// os.WriteFile only applies the mode when creating a file, so a
	// pre-existing world-readable config must be tightened before secrets
	// are written into it. Refuse to proceed if that fails.
	if err := os.Chmod(path, 0o600); err != nil {
		return false, fmt.Errorf("cannot make %s private before adding credentials: %w", path, err)
	}

	var b strings.Builder
	b.Write(content)
	if len(content) > 0 && content[len(content)-1] != '\n' {
		b.WriteString("\n")
	}
	b.WriteString("; RPC credentials added by " + appName + "\n")
	b.WriteString(strings.Join(add, "\n") + "\n")
	return false, os.WriteFile(path, []byte(b.String()), 0o600)
}

// insecureConfWarnings inspects an existing oyster.conf for settings that
// weaken network security. oystercli never overrides them (they are the
// user's explicit choice) but surfaces them so the exposure is not silent.
func insecureConfWarnings(cfg *config) []string {
	v := scrapeOysterConf(cfg.oysterConfPath())
	var warnings []string
	if v.noServerTLS {
		warnings = append(warnings, "oyster.conf sets noservertls=1 — the wallet RPC runs without TLS")
	}
	for _, listen := range v.rpcListen {
		if !isLoopbackListener(listen) {
			warnings = append(warnings,
				"oyster.conf listens on "+listen+" — the wallet RPC is reachable beyond localhost")
		}
	}
	return warnings
}

// isLoopbackListener reports whether an rpclisten value is bound to loopback
// only. An empty host (":port") or a wildcard address counts as non-loopback.
func isLoopbackListener(listen string) bool {
	host, _, err := net.SplitHostPort(listen)
	if err != nil {
		host = listen
	}
	switch host {
	case "localhost":
		return true
	case "", "0.0.0.0", "::", "*":
		return false
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// randomHex returns n random bytes hex-encoded (2n characters).
func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err) // crypto/rand failure is unrecoverable
	}
	return hex.EncodeToString(b)
}

// confHasCredentials reports whether oyster.conf provides a full credential
// pair, which is the precondition for spawning the daemon.
func confHasCredentials(cfg *config) bool {
	vals := scrapeOysterConf(cfg.oysterConfPath())
	return vals.username != "" && vals.password != ""
}
