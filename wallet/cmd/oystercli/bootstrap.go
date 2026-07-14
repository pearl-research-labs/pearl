// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// bootstrapValues is what the guided setup writes into oyster.conf.
type bootstrapValues struct {
	username    string
	password    string
	useSPV      bool
	addPeer     string
	rpcConnect  string
	noServerTLS bool
}

// bootstrapConfigWizard writes (or completes) oyster.conf so that both the
// daemon and this CLI share one source of truth for credentials and chain
// backend settings. This is the fix for the common "oyster was always run
// with ad-hoc flags" situation, where credentials are otherwise irrecoverable.
func bootstrapConfigWizard(cfg *config) error {
	confPath := cfg.oysterConfPath()

	existing := scrapeOysterConf(confPath)
	if existing.username != "" && existing.password != "" {
		printWarn(confPath + " already contains credentials; nothing to bootstrap.")
		cfg.rescrapeConf()
		return nil
	}

	const (
		backendSPV    = "spv"
		backendPearld = "pearld"
	)
	vals := bootstrapValues{
		username: "oyster-" + randomHex(4),
		password: randomHex(24),
		// Pre-fill the pearld address with the network's conventional
		// local port.
		rpcConnect: net.JoinHostPort("localhost", cfg.activeNet.RPCClientPort),
	}
	backend := backendSPV
	keepTLS := true

	// Sequential forms rather than conditionally hidden groups: huh's
	// accessible mode prompts hidden groups anyway, so branching must live
	// in control flow.
	submitted, err := runForm(newForm(
		huh.NewGroup(
			huh.NewNote().
				Title("Guided oyster setup").
				Description(fmt.Sprintf(
					"This writes %s.\nOyster reads it automatically on start, and %s uses the same\nfile to connect — no flags needed on either side afterwards.",
					confPath, appName)),
			huh.NewInput().
				Title("RPC username").
				Description("Pre-filled with a generated value; edit if you prefer your own.").
				Validate(nonEmpty("username")).
				Value(&vals.username),
			huh.NewInput().
				Title("RPC password").
				Description("Pre-filled with a generated secret. It is stored in the config file (0600).").
				Validate(nonEmpty("password")).
				Value(&vals.password),
			huh.NewSelect[string]().
				Title("Chain backend").
				Description("How oyster learns about the chain.").
				Options(
					huh.NewOption("SPV — sync headers over the P2P network (no pearld needed)", backendSPV),
					huh.NewOption("Local pearld — connect to a pearld you run yourself", backendPearld),
				).
				Value(&backend),
			huh.NewConfirm().
				Title("Keep TLS on the wallet RPC?").
				Description("Recommended. Choose No only for localhost-only setups; oystercli adapts either way.").
				Affirmative("Keep TLS").
				Negative("Disable").
				Value(&keepTLS),
		),
	))
	if err != nil || !submitted {
		return err
	}

	vals.useSPV = backend == backendSPV
	vals.noServerTLS = !keepTLS

	if vals.useSPV {
		vals.rpcConnect = ""
		submitted, err = runForm(newForm(huh.NewGroup(
			huh.NewInput().
				Title("Extra peer (optional)").
				Description("addpeer= for SPV; useful on test networks with few reachable peers.").
				Placeholder("host[:port]").
				Value(&vals.addPeer),
		)))
	} else {
		submitted, err = runForm(newForm(huh.NewGroup(
			huh.NewInput().
				Title("pearld RPC address").
				Description("rpcconnect=; oyster authenticates to pearld with the same username/password.").
				Validate(nonEmpty("address")).
				Value(&vals.rpcConnect),
		)))
	}
	if err != nil || !submitted {
		return err
	}

	if err := writeOysterConf(confPath, existing, vals); err != nil {
		return err
	}
	cfg.rescrapeConf()

	printSuccess("Configuration written to " + confPath)
	if backend == backendPearld {
		lipgloss.Println(th.subtle.Render(
			"Note: with a pearld backend, oyster also needs pearld's TLS certificate at\n" +
				cfg.AppData + "/pearld.cert (or run pearld with matching rpcuser/rpcpass)."))
	}
	return nil
}

// writeOysterConf creates the config file, or appends only the missing keys
// when one already exists so user content is never rewritten.
func writeOysterConf(path string, existing oysterConfValues, vals bootstrapValues) error {
	var lines []string
	if existing.username == "" {
		lines = append(lines, "username="+vals.username)
	}
	if existing.password == "" {
		lines = append(lines, "password="+vals.password)
	}
	if vals.useSPV {
		lines = append(lines, "usespv=1")
		if peer := strings.TrimSpace(vals.addPeer); peer != "" {
			lines = append(lines, "addpeer="+peer)
		}
	}
	if vals.rpcConnect != "" {
		lines = append(lines, "rpcconnect="+vals.rpcConnect)
	}
	if vals.noServerTLS && !existing.noServerTLS {
		lines = append(lines, "noservertls=1")
	}

	content, err := os.ReadFile(path)
	switch {
	case os.IsNotExist(err):
		body := "[Application Options]\n" + strings.Join(lines, "\n") + "\n"
		return os.WriteFile(path, []byte(body), 0o600)
	case err != nil:
		return err
	}

	var b strings.Builder
	b.Write(content)
	if len(content) > 0 && content[len(content)-1] != '\n' {
		b.WriteString("\n")
	}
	b.WriteString("; added by " + appName + " guided setup\n")
	b.WriteString(strings.Join(lines, "\n") + "\n")
	return os.WriteFile(path, []byte(b.String()), 0o600)
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
// pair, which is the precondition for spawning the daemon without flags.
func confHasCredentials(cfg *config) bool {
	vals := scrapeOysterConf(cfg.oysterConfPath())
	return vals.username != "" && vals.password != ""
}
