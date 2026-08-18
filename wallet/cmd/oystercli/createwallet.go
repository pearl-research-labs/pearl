// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
				Validate(huh.ValidateNotEmpty()).
				Value(&seedInput),
			huh.NewInput().
				Title("Wallet birthday (optional)").
				Description("Date of the wallet's first use, YYYY-MM-DD; narrows the recovery scan.\nLeave empty to scan the whole chain.").
				Placeholder("2026-05-01").
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
			Validate(huh.ValidateNotEmpty()).
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
		setup["Bday"] = fmt.Sprintf("%d", importBirthday(cfg, birthday).Unix())
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

	lipgloss.Println(th.subtle.Render("Next: pick \"Start oyster now\" to launch the daemon and connect."))
	return nil
}

// runOysterCreate writes the wallet-setup JSON to a private temp file, runs
// the daemon in --createfromfile mode, and extracts the seed it prints.
func runOysterCreate(cfg *config, binPath string, setup map[string]string) (string, error) {
	// The setup blob carries the private passphrase and seed, and
	// oyster --createfromfile only accepts a file path. Put it in a
	// private 0700 directory (so other users cannot even observe the
	// filename), write the file 0600, and remove the whole directory
	// afterwards. It exists only for the brief lifetime of the create call.
	dir, err := os.MkdirTemp("", "oystercli-setup-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)

	blob, err := json.Marshal(setup)
	if err != nil {
		return "", err
	}
	setupPath := filepath.Join(dir, "wallet-setup.json")
	if err := os.WriteFile(setupPath, blob, 0o600); err != nil {
		return "", err
	}

	args := []string{"--appdata=" + cfg.AppData, "--createfromfile=" + setupPath}
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

// importBirthday resolves the recovery start time for a restore. A restored
// wallet only finds its addresses and balance by rescanning the chain from
// the birthday, and oyster defaults a missing Bday to time.Now() — which
// skips all history and leaves the wallet empty. An omitted birthday must
// therefore mean "scan everything": the chain's genesis time.
func importBirthday(cfg *config, input string) time.Time {
	if s := strings.TrimSpace(input); s != "" {
		if t, err := time.Parse("2006-01-02", s); err == nil {
			return t
		}
	}
	return cfg.activeNet.Params.GenesisBlock.BlockHeader().Timestamp
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
