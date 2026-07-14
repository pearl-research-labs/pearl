// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// createWalletWizard creates a wallet by driving `oyster --createfromfile`,
// the same mechanism the desktop wallet uses, then walks the user through
// backing up the seed.
func createWalletWizard(cfg *config) error {
	binPath, err := locateOysterBinary(cfg)
	if err != nil {
		return err
	}

	const (
		modeNew    = "new"
		modeImport = "import"
	)
	var (
		mode       = modeNew
		seedInput  string
		birthday   string
		passphrase string
		confirm    string
	)
	// Sequential forms: accessible mode prompts every group of a form even
	// when hidden, so the import-only questions live in their own form.
	submitted, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Create wallet").
			Options(
				huh.NewOption("Generate a new recovery seed", modeNew),
				huh.NewOption("Restore from an existing seed", modeImport),
			).
			Value(&mode),
	)))
	if err != nil || !submitted {
		return err
	}

	if mode == modeImport {
		submitted, err = runForm(newForm(huh.NewGroup(
			huh.NewText().
				Title("Recovery seed").
				Description("The 12-word BIP39 mnemonic (or legacy hex seed) to restore from.").
				Validate(nonEmpty("seed")).
				Value(&seedInput),
			huh.NewInput().
				Title("Wallet birthday (optional)").
				Description("Approximate creation date, YYYY-MM-DD. Speeds up the initial scan.").
				Placeholder("2026-01-31").
				Validate(validateBirthday).
				Value(&birthday),
		)))
		if err != nil || !submitted {
			return err
		}
	}

	submitted, err = runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Private passphrase").
			Description("Encrypts your keys; required for every spend. There is no recovery if lost.").
			EchoMode(huh.EchoModePassword).
			Validate(nonEmpty("passphrase")).
			Value(&passphrase),
		huh.NewInput().
			Title("Repeat passphrase").
			EchoMode(huh.EchoModePassword).
			Validate(func(s string) error {
				if s != passphrase {
					return fmt.Errorf("passphrases do not match")
				}
				return nil
			}).
			Value(&confirm),
	)))
	if err != nil || !submitted {
		return err
	}

	setup := map[string]string{"PrivatePassphrase": passphrase}
	if mode == modeImport {
		setup["Seed"] = strings.TrimSpace(seedInput)
		if b := strings.TrimSpace(birthday); b != "" {
			t, _ := time.Parse("2006-01-02", b)
			setup["Bday"] = fmt.Sprintf("%d", t.Unix())
		}
	}

	var seed string
	err = withSpinner("Creating the wallet...", func() error {
		var createErr error
		seed, createErr = runOysterCreate(cfg, binPath, setup)
		return createErr
	})
	if err != nil {
		return err
	}

	printSuccess("Wallet created at " + cfg.walletDBPath())

	if mode == modeNew {
		if err := seedBackupCeremony(seed); err != nil {
			return err
		}
	}

	if confHasCredentials(cfg) {
		lipgloss.Println(th.subtle.Render("Next: pick \"Start oyster now\" to launch the daemon and connect."))
	} else {
		lipgloss.Println(th.subtle.Render("Next: run the guided setup to write oyster.conf, then start the daemon."))
	}
	return nil
}

// runOysterCreate writes the wallet-setup JSON to a private temp file, runs
// the daemon in --createfromfile mode, and extracts the seed it prints.
func runOysterCreate(cfg *config, binPath string, setup map[string]string) (string, error) {
	f, err := os.CreateTemp("", "oystercli-setup-*.json")
	if err != nil {
		return "", err
	}
	defer os.Remove(f.Name())
	blob, err := json.Marshal(setup)
	if err != nil {
		f.Close()
		return "", err
	}
	if _, err := f.Write(blob); err != nil {
		f.Close()
		return "", err
	}
	f.Close()

	args := []string{"--appdata=" + cfg.AppData, "--createfromfile=" + f.Name()}
	if flag := networkFlag(cfg); flag != "" {
		args = append(args, flag)
	}
	out, err := exec.Command(binPath, args...).CombinedOutput()
	if err != nil {
		detail := strings.TrimSpace(string(out))
		if detail == "" {
			detail = err.Error()
		}
		return "", fmt.Errorf("oyster --createfromfile failed: %s", detail)
	}

	seed := extractSeed(string(out))
	if seed == "" {
		return "", fmt.Errorf("wallet was created but no seed found in the daemon output")
	}
	return seed, nil
}

// extractSeed finds the seed line in the create output: a BIP39 mnemonic
// (12+ words) or a long hex string, preferring the last matching line.
func extractSeed(output string) string {
	lines := strings.Split(output, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		words := strings.Fields(line)
		if len(words) >= 12 && isLowerWords(words) {
			return line
		}
		if len(words) == 1 && len(line) >= 32 && isHex(line) {
			return line
		}
	}
	return ""
}

func isLowerWords(words []string) bool {
	for _, w := range words {
		for _, r := range w {
			if r < 'a' || r > 'z' {
				return false
			}
		}
	}
	return true
}

func isHex(s string) bool {
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9', r >= 'a' && r <= 'f', r >= 'A' && r <= 'F':
		default:
			return false
		}
	}
	return true
}

// seedBackupCeremony shows the mnemonic and refuses to continue until the
// user asserts it is written down.
func seedBackupCeremony(seed string) error {
	printTitle("Recovery seed — write it down now")
	printBox(th.warn.Bold(true).Render(wrapWords(seed, 4)))
	lipgloss.Println(th.subtle.Render("Anyone with these words can take your funds; anyone without them cannot\nrecover your wallet if this machine dies. Store them offline."))

	for {
		saved := false
		ok, err := runForm(newForm(huh.NewGroup(
			huh.NewConfirm().
				Title("Have you written the seed down?").
				Affirmative("Yes, it is safely stored").
				Negative("Not yet").
				Value(&saved),
		)))
		if err != nil {
			return err
		}
		if ok && saved {
			return nil
		}
	}
}

// wrapWords lays out a mnemonic a few words per line so it is easy to copy
// by hand.
func wrapWords(s string, perLine int) string {
	words := strings.Fields(s)
	if len(words) == 1 {
		return s
	}
	var b strings.Builder
	for i, w := range words {
		if i > 0 {
			if i%perLine == 0 {
				b.WriteString("\n")
			} else {
				b.WriteString("  ")
			}
		}
		fmt.Fprintf(&b, "%2d.%s", i+1, w)
	}
	return b.String()
}

// validateBirthday accepts empty or YYYY-MM-DD dates in the past.
func validateBirthday(s string) error {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return fmt.Errorf("use YYYY-MM-DD")
	}
	if t.After(time.Now()) {
		return fmt.Errorf("birthday cannot be in the future")
	}
	return nil
}
