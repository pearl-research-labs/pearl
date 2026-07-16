// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Visual identity: the shared lipgloss styles for printed output and the
// huh theme/keymap adaptations.

package main

import (
	"charm.land/bubbles/v2/key"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// ui holds the shared styles for printed (non-form) output.
//
// Only the terminal's own palette is used: basic ANSI colors and text
// attributes, which every terminal theme — light or dark — is responsible
// for keeping readable. Plain text stays at the default foreground and
// de-emphasis is the Faint attribute, so nothing here depends on background
// detection. The one detection-dependent piece (QR contrast) reads
// lipgloss's compat global.
type ui struct {
	title  lipgloss.Style
	subtle lipgloss.Style
	accent lipgloss.Style
	value  lipgloss.Style
	good   lipgloss.Style
	warn   lipgloss.Style
	bad    lipgloss.Style
	box    lipgloss.Style
	header lipgloss.Style
}

var th = ui{
	title:  lipgloss.NewStyle().Foreground(lipgloss.Magenta).Bold(true),
	subtle: lipgloss.NewStyle().Faint(true),
	accent: lipgloss.NewStyle().Foreground(lipgloss.Cyan),
	value:  lipgloss.NewStyle(),
	good:   lipgloss.NewStyle().Foreground(lipgloss.Green),
	warn:   lipgloss.NewStyle().Foreground(lipgloss.Yellow),
	bad:    lipgloss.NewStyle().Foreground(lipgloss.Red),
	box: lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.BrightBlack).
		Padding(0, 2),
	header: lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Magenta).
		Padding(0, 2),
}

// oysterTheme is huh's stock charm theme with one workaround: menu option
// foregrounds are unset so they render in the terminal's default color.
// huh v2.0.3 always renders its light palette (nothing requests the terminal
// background) whose option foreground is near-white — invisible on light
// terminals. Both defects are fixed upstream in charmbracelet/huh#776;
// once released, this collapses to a plain ThemeCharm.
func oysterTheme() huh.Theme {
	return huh.ThemeFunc(func(isDark bool) *huh.Styles {
		t := huh.ThemeCharm(isDark)
		t.Focused.Option = t.Focused.Option.UnsetForeground()
		t.Focused.UnselectedOption = t.Focused.UnselectedOption.UnsetForeground()
		t.Blurred.Option = t.Blurred.Option.UnsetForeground()
		t.Blurred.UnselectedOption = t.Blurred.UnselectedOption.UnsetForeground()
		return t
	})
}

// oysterKeyMap extends huh's defaults so Esc aborts any form (the default
// only binds Ctrl+C, which "locks" users into multi-field screens like Send
// unless they guess the combo). runForm treats the abort as "go back".
//
// The bottom help line is assembled exclusively from per-field bindings, so
// the form-level Quit binding never shows up there by itself; instead the
// hint rides along on the Next/Submit help text, of which exactly one is
// visible per field.
func oysterKeyMap() *huh.KeyMap {
	km := huh.NewDefaultKeyMap()
	km.Quit = key.NewBinding(key.WithKeys("ctrl+c", "esc"), key.WithHelp("esc", "back"))
	appendBackHint(
		&km.Input.Next, &km.Input.Submit,
		&km.Text.Next, &km.Text.Submit,
		&km.Select.Next, &km.Select.Submit,
		&km.MultiSelect.Next, &km.MultiSelect.Submit,
		&km.Confirm.Next, &km.Confirm.Submit,
	)
	return km
}

func appendBackHint(bindings ...*key.Binding) {
	for _, b := range bindings {
		h := b.Help()
		b.SetHelp(h.Key, h.Desc+" • esc back")
	}
}
