// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Visual identity: the pearl palette, the shared lipgloss styles, and the
// huh theme/keymap adaptations.

package main

import (
	"os"
	"strings"

	"charm.land/bubbles/v2/key"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// The pearl palette: nacre white, iridescent violet, and sea tones.
var (
	colorViolet = lipgloss.Color("#8B7CF6")
	colorTeal   = lipgloss.Color("#2DD4BF")
	colorPink   = lipgloss.Color("#F0A6CA")
	colorGold   = lipgloss.Color("#E8C268")
	colorRed    = lipgloss.Color("#ED567A")
	colorGreen  = lipgloss.Color("#02BF87")
)

// ui holds the resolved styles for the detected terminal background. It is
// initialized once at startup via initUI.
type ui struct {
	isDark bool

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

var th ui

// detectDarkBackground guesses the terminal background without any terminal
// I/O. Querying the terminal (lipgloss.HasDarkBackground) reads from stdin
// and its response can race with the first huh form, which then sees the
// leftover bytes as phantom input and auto-submits. The stakes are only
// slightly-off colors in printed output (forms detect the background safely
// themselves via Bubble Tea), so an environment heuristic is enough:
// COLORFGBG is "<fg>;<bg>" (some terminals "<fg>;<default>;<bg>"), where a
// background of 0-6 or 8 means dark. Unknown means dark, the common case.
func detectDarkBackground() bool {
	parts := strings.Split(os.Getenv("COLORFGBG"), ";")
	if len(parts) < 2 {
		return true
	}
	switch parts[len(parts)-1] {
	case "7", "15":
		return false
	}
	return true
}

func initUI(isDark bool) {
	ld := lipgloss.LightDark(isDark)
	text := ld(lipgloss.Color("235"), lipgloss.Color("252"))
	faint := ld(lipgloss.Color("246"), lipgloss.Color("243"))
	borderCol := ld(lipgloss.Color("250"), lipgloss.Color("238"))

	th = ui{
		isDark: isDark,
		title:  lipgloss.NewStyle().Foreground(colorViolet).Bold(true),
		subtle: lipgloss.NewStyle().Foreground(faint),
		accent: lipgloss.NewStyle().Foreground(colorTeal),
		value:  lipgloss.NewStyle().Foreground(text),
		good:   lipgloss.NewStyle().Foreground(colorGreen),
		warn:   lipgloss.NewStyle().Foreground(colorGold),
		bad:    lipgloss.NewStyle().Foreground(colorRed),
		box: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderCol).
			Padding(0, 2),
		header: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(colorViolet).
			Padding(0, 2),
	}
}

// oysterTheme adapts the charm theme to the pearl palette.
func oysterTheme() huh.Theme {
	return huh.ThemeFunc(func(isDark bool) *huh.Styles {
		t := huh.ThemeCharm(isDark)
		ld := lipgloss.LightDark(isDark)

		t.Focused.Title = t.Focused.Title.Foreground(colorViolet).Bold(true)
		t.Focused.NoteTitle = t.Focused.NoteTitle.Foreground(colorViolet).Bold(true)
		t.Focused.Description = t.Focused.Description.Foreground(ld(lipgloss.Color("246"), lipgloss.Color("243")))
		t.Focused.SelectSelector = t.Focused.SelectSelector.Foreground(colorTeal).SetString("❯ ")
		t.Focused.NextIndicator = t.Focused.NextIndicator.Foreground(colorTeal)
		t.Focused.PrevIndicator = t.Focused.PrevIndicator.Foreground(colorTeal)
		t.Focused.MultiSelectSelector = t.Focused.MultiSelectSelector.Foreground(colorTeal).SetString("❯ ")
		t.Focused.SelectedOption = t.Focused.SelectedOption.Foreground(colorTeal)
		t.Focused.SelectedPrefix = t.Focused.SelectedPrefix.Foreground(colorGreen)
		t.Focused.FocusedButton = t.Focused.FocusedButton.Background(colorViolet)
		t.Focused.TextInput.Prompt = t.Focused.TextInput.Prompt.Foreground(colorTeal)
		t.Focused.TextInput.Cursor = t.Focused.TextInput.Cursor.Foreground(colorPink)
		t.Focused.ErrorIndicator = t.Focused.ErrorIndicator.Foreground(colorRed)
		t.Focused.ErrorMessage = t.Focused.ErrorMessage.Foreground(colorRed)

		t.Blurred.Title = t.Blurred.Title.Foreground(colorViolet)
		t.Blurred.TextInput.Prompt = t.Blurred.TextInput.Prompt.Foreground(colorTeal)
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
