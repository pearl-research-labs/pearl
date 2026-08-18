// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

// sources records where each connection setting came from, so the CLI can
// explain its reasoning instead of leaving the user to reverse-engineer it.
type sources struct {
	network string
	appData string
	conf    string // "found" / "not found"
	creds   string // e.g. "flags", "oyster.conf", "oyster.conf (auto-provisioned)", "none found"
	tls     string
	connect string
}

// resolutionStory renders a compact explanation of where every connection
// setting came from, so the user never has to guess why the CLI is asking
// for something or connecting where it does.
func resolutionStory(cfg *config) string {
	tlsValue := "on"
	if cfg.NoTLS {
		tlsValue = "off"
	}
	credValue := "username+password"
	switch {
	case cfg.RPCUser == "" && cfg.RPCPass == "":
		credValue = "none"
	case cfg.RPCPass == "":
		credValue = "username only"
	case cfg.RPCUser == "":
		credValue = "password only"
	}
	walletValue, walletNote := cfg.walletDBPath(), "found"
	if !cfg.walletDBExists() {
		walletNote = "missing"
	}

	rows := [][3]string{
		{"network", cfg.activeNet.Params.Name, cfg.src.network},
		{"appdata", cfg.AppData, cfg.src.appData},
		{"config", cfg.oysterConfPath(), cfg.src.conf},
		{"credentials", credValue, cfg.src.creds},
		{"tls", tlsValue, cfg.src.tls},
	}
	if !cfg.NoTLS {
		certNote := "found"
		if !fileExists(cfg.CAFile) {
			certNote = "missing"
		}
		rows = append(rows, [3]string{"certificate", cfg.CAFile, certNote})
	}
	binValue, binNote := cfg.OysterBin, "not found"
	if p, src, err := findOysterBinary(cfg); err == nil {
		binValue, binNote = p, src
	}
	rows = append(rows,
		[3]string{"connect", cfg.Connect, cfg.src.connect},
		[3]string{"wallet", walletValue, walletNote},
		[3]string{"oyster bin", binValue, binNote},
	)

	valueWidth := 0
	for _, r := range rows {
		if len(r[1]) > valueWidth {
			valueWidth = len(r[1])
		}
	}
	var b strings.Builder
	for i, r := range rows {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(th.subtle.Render(fmt.Sprintf("%-13s", r[0])))
		b.WriteString(th.value.Render(fmt.Sprintf("%-*s", valueWidth+2, r[1])))
		note := th.subtle.Render(r[2])
		if r[2] == "missing" || r[2] == "not found" || r[2] == "none found" {
			note = th.warn.Render(r[2])
		}
		b.WriteString(note)
	}
	return b.String()
}
