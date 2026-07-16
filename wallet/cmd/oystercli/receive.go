// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/compat"
	qrterminal "github.com/mdp/qrterminal/v3"
	"github.com/pearl-research-labs/pearl/node/btcutil"
)

// receiveScreen hands out receiving addresses, rendered large with a
// scannable QR code.
func receiveScreen(c *client) error {
	names, err := c.accountNames()
	if err != nil {
		return err
	}

	account := "default"
	if len(names) > 0 {
		account = names[0]
	}
	accountOpts := make([]huh.Option[string], 0, len(names))
	for _, name := range names {
		accountOpts = append(accountOpts, huh.NewOption(name, name))
	}

	fresh := true
	form := newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Receive into account").
			Options(accountOpts...).
			Value(&account),
		huh.NewSelect[bool]().
			Title("Address").
			Options(
				huh.NewOption("Generate a fresh address (recommended)", true),
				huh.NewOption("Reuse the current unused address", false),
			).
			Value(&fresh),
	))
	submitted, err := runForm(form)
	if err != nil || !submitted {
		return err
	}

	var addr btcutil.Address
	err = withAutoUnlock(c, func() error {
		var addrErr error
		if fresh {
			addr, addrErr = c.newAddress(account)
		} else {
			addr, addrErr = c.currentAddress(account)
		}
		return addrErr
	})
	if err != nil {
		return err
	}

	printTitle("Receive to " + account)
	printBox(th.accent.Bold(true).Render(addr.String()))
	printQR(addr.String())
	lipgloss.Println(th.subtle.Render("Share the address or QR code with the sender. Each payment to a fresh\naddress keeps your history harder to link."))
	return nil
}

// printQR renders a half-block QR code, keeping proper contrast on both dark
// and light terminals by swapping module colors as needed.
func printQR(text string) {
	var sb strings.Builder
	cfg := qrterminal.Config{
		Level:      qrterminal.L,
		Writer:     &sb,
		HalfBlocks: true,
		QuietZone:  2,
	}
	if compat.HasDarkBackground {
		// Foreground blocks are light on dark terminals, so draw the QR
		// background cells with blocks and leave the modules dark.
		cfg.BlackChar = qrterminal.BLACK_BLACK
		cfg.WhiteChar = qrterminal.WHITE_WHITE
		cfg.BlackWhiteChar = qrterminal.BLACK_WHITE
		cfg.WhiteBlackChar = qrterminal.WHITE_BLACK
	} else {
		cfg.BlackChar = qrterminal.WHITE_WHITE
		cfg.WhiteChar = qrterminal.BLACK_BLACK
		cfg.BlackWhiteChar = qrterminal.WHITE_BLACK
		cfg.WhiteBlackChar = qrterminal.BLACK_WHITE
	}
	qrterminal.GenerateWithConfig(text, cfg)
	lipgloss.Println(strings.TrimRight(sb.String(), "\n"))
}
