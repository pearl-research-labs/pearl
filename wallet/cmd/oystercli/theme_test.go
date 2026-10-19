// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/stretchr/testify/assert"
)

func TestOysterThemeUnsetsOptionForegrounds(t *testing.T) {
	for _, isDark := range []bool{true, false} {
		stock, mine := huh.ThemeCharm(isDark), oysterTheme().Theme(isDark)

		unset := map[string][2]lipgloss.Style{
			"focused option":   {stock.Focused.Option, mine.Focused.Option},
			"focused unselect": {stock.Focused.UnselectedOption, mine.Focused.UnselectedOption},
			"blurred option":   {stock.Blurred.Option, mine.Blurred.Option},
			"blurred unselect": {stock.Blurred.UnselectedOption, mine.Blurred.UnselectedOption},
		}
		for name, pair := range unset {
			assert.NotEqual(t, lipgloss.NoColor{}, pair[0].GetForeground(),
				"dark=%v stock %s has a foreground", isDark, name)
			assert.Equal(t, lipgloss.NoColor{}, pair[1].GetForeground(),
				"dark=%v %s keeps the terminal color", isDark, name)
		}

		assert.Equal(t, stock.Focused.Title.Render("x"), mine.Focused.Title.Render("x"), "dark=%v title", isDark)
		assert.Equal(t, stock.Focused.SelectedOption.Render("x"), mine.Focused.SelectedOption.Render("x"),
			"dark=%v selected option", isDark)
	}
}

func TestOysterThemeBuildsEachPaletteOnce(t *testing.T) {
	theme := oysterTheme()

	darkFirst, darkAgain := theme.Theme(true), theme.Theme(true)
	lightFirst, lightAgain := theme.Theme(false), theme.Theme(false)

	assert.Same(t, darkFirst, darkAgain)
	assert.Same(t, lightFirst, lightAgain)
	assert.NotSame(t, darkFirst, lightFirst)
}

// Every field and every frame shares one *huh.Styles per palette, so a huh release that wrote to the styles it was
// given would restyle the whole app from then on.
func TestFormsDoNotModifyTheSharedStyles(t *testing.T) {
	fields := []struct {
		title    string
		field    func() huh.Field
		wantView string
	}{
		{"Select", func() huh.Field {
			var choice string
			return huh.NewSelect[string]().Title("Select").Description("pick one").Height(6).
				Options(huh.NewOption("one", "1"), huh.NewOption("two", "2"), huh.NewOption("three", "3")).
				Value(&choice)
		}, "two"},
		{"Multi", func() huh.Field {
			var picks []string
			return huh.NewMultiSelect[string]().Title("Multi").
				Options(huh.NewOption("a", "a"), huh.NewOption("b", "b")).Value(&picks)
		}, "b"},
		{"Confirm", func() huh.Field {
			var confirmed bool
			return huh.NewConfirm().Title("Confirm").Affirmative("Yes").Negative("No").Value(&confirmed)
		}, "Yes"},
		{"Input", func() huh.Field {
			var text string
			return huh.NewInput().Title("Input").Validate(huh.ValidateNotEmpty()).Value(&text)
		}, "input cannot be empty"},
		{"Password", func() huh.Field {
			var secret string
			return huh.NewInput().Title("Password").EchoMode(huh.EchoModePassword).Value(&secret)
		}, "Password"},
		{"Text", func() huh.Field {
			var note string
			return huh.NewText().Title("Text").Value(&note)
		}, "Text"},
	}
	keys := []tea.KeyPressMsg{
		{Code: tea.KeyEnter}, {Code: 'x', Text: "x"}, {Code: tea.KeyBackspace},
		{Code: tea.KeyDown}, {Code: tea.KeyUp}, {Code: tea.KeyLeft}, {Code: tea.KeyRight},
		{Code: tea.KeySpace, Text: " "}, {Code: tea.KeyDown}, {Code: tea.KeySpace, Text: " "},
	}

	for _, tt := range fields {
		for _, background := range []color.Color{color.Black, color.White} {
			isDark := tea.BackgroundColorMsg{Color: background}.IsDark()
			t.Run(fmt.Sprintf("%s dark=%v", tt.title, isDark), func(t *testing.T) {
				theme := oysterTheme()
				shared := theme.Theme(isDark)
				before := *shared

				form := huh.NewForm(huh.NewGroup(tt.field())).WithTheme(theme)
				form.Init()
				form.Update(tea.BackgroundColorMsg{Color: background})
				form.Update(tea.WindowSizeMsg{Width: 80, Height: 24})

				var views strings.Builder
				views.WriteString(form.View())
				for _, key := range keys {
					form.Update(key)
					views.WriteString(form.View())
				}

				assert.Contains(t, views.String(), tt.title)
				assert.Contains(t, views.String(), tt.wantView, "the rendering path under test was not reached")
				assert.Equal(t, before, *shared)
			})
		}
	}
}
