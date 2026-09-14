// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// withAutoUnlock runs op and, when it fails because the wallet is locked,
// prompts for the passphrase and retries once. Declining the prompt returns
// the original error.
func withAutoUnlock(c *client, op func() error) error {
	err := op()
	if !isRPCErrorCode(err, btcjson.ErrRPCWalletUnlockNeeded) {
		return err
	}
	printWarn("This action needs the wallet unlocked.")
	unlocked, promptErr := promptUnlock(c)
	if promptErr != nil {
		return promptErr
	}
	if !unlocked {
		return err
	}
	return op()
}

// unlockDurations are the offered unlock windows; 0 keeps the wallet
// unlocked until it is locked explicitly (we lock on exit if we unlocked).
var unlockDurations = []huh.Option[int64]{
	huh.NewOption("1 minute", int64(60)),
	huh.NewOption("5 minutes", int64(300)),
	huh.NewOption("30 minutes", int64(1800)),
	huh.NewOption("Until I lock it (or this session ends)", int64(0)),
}

// promptUnlock asks for the passphrase and unlocks the wallet. Returns false
// without error when the user backs out.
func promptUnlock(c *client) (bool, error) {
	var (
		passphrase string
		timeout    = int64(300)
	)
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Wallet passphrase").
			EchoMode(huh.EchoModePassword).
			Validate(huh.ValidateNotEmpty()).
			Value(&passphrase),
		huh.NewSelect[int64]().
			Title("Stay unlocked for").
			Options(unlockDurations...).
			Value(&timeout),
	)))
	if err != nil || !ok {
		return false, err
	}
	if err := c.unlock(passphrase, timeout); err != nil {
		return false, err
	}
	printSuccess("Wallet unlocked.")
	return true, nil
}

// securityScreen groups the lock/passphrase/key operations.
func securityScreen(c *client) error {
	for {
		const (
			opUnlock = "unlock"
			opLock   = "lock"
			opChange = "change"
			opImport = "import"
			opDump   = "dump"
			opSign   = "sign"
			opVerify = "verify"
			opBack   = "back"
		)

		lockState := "unknown"
		if locked, err := c.walletLocked(); err == nil {
			if locked {
				lockState = "locked"
			} else {
				lockState = "unlocked"
			}
		}

		choice := opUnlock
		submitted, err := runForm(newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title("Security").
				Description("Wallet is currently "+lockState+".").
				Options(
					huh.NewOption("Unlock wallet", opUnlock),
					huh.NewOption("Lock wallet", opLock),
					huh.NewOption("Change passphrase", opChange),
					huh.NewOption("Import private key (WIF)", opImport),
					huh.NewOption("Reveal private key (dangerous)", opDump),
					huh.NewOption("Sign a message", opSign),
					huh.NewOption("Verify a message", opVerify),
					huh.NewOption("Back", opBack),
				).
				Value(&choice),
		)))
		if err != nil {
			return err
		}
		if !submitted || choice == opBack {
			return nil
		}

		var opErr error
		switch choice {
		case opUnlock:
			_, opErr = promptUnlock(c)
		case opLock:
			if opErr = c.lock(); opErr == nil {
				printSuccess("Wallet locked.")
			}
		case opChange:
			opErr = changePassphraseFlow(c)
		case opImport:
			opErr = importKeyFlow(c)
		case opDump:
			opErr = dumpKeyFlow(c)
		case opSign:
			opErr = signMessageFlow(c)
		case opVerify:
			opErr = verifyMessageFlow(c)
		}
		if opErr != nil {
			printError(opErr)
		}
	}
}

func changePassphraseFlow(c *client) error {
	var oldPass, newPass, confirm string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Current passphrase").
			EchoMode(huh.EchoModePassword).
			Validate(huh.ValidateNotEmpty()).
			Value(&oldPass),
		huh.NewInput().
			Title("New passphrase").
			EchoMode(huh.EchoModePassword).
			Validate(huh.ValidateNotEmpty()).
			Value(&newPass),
		huh.NewInput().
			Title("Repeat new passphrase").
			EchoMode(huh.EchoModePassword).
			Validate(func(s string) error {
				if s != newPass {
					return fmt.Errorf("passphrases do not match")
				}
				return nil
			}).
			Value(&confirm),
	)))
	if err != nil || !ok {
		return err
	}
	if err := c.changePassphrase(oldPass, newPass); err != nil {
		return err
	}
	printSuccess("Passphrase changed.")
	return nil
}

func importKeyFlow(c *client) error {
	var (
		wif    string
		rescan = true
	)
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Private key (WIF)").
			EchoMode(huh.EchoModePassword).
			Validate(huh.ValidateNotEmpty()).
			Value(&wif),
		huh.NewConfirm().
			Title("Rescan the chain for its history?").
			Description("A rescan can take a long time; skip it only if the key never received funds.").
			Affirmative("Rescan").
			Negative("Skip").
			Value(&rescan),
	)))
	if err != nil || !ok {
		return err
	}
	err = withAutoUnlock(c, func() error {
		return withSpinner("Importing key (this can take a while when rescanning)...", func() error {
			return c.importPrivKey(strings.TrimSpace(wif), rescan)
		})
	})
	if err != nil {
		return err
	}
	printSuccess("Key imported into the 'imported' account.")
	return nil
}

func dumpKeyFlow(c *client) error {
	var (
		address string
		typed   string
	)
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Address to reveal the key for").
			Validate(func(s string) error { return validateRecipient(c, strings.TrimSpace(s)) }).
			Value(&address),
		huh.NewInput().
			Title("Type REVEAL to confirm").
			Description("Anyone who sees this key can irrevocably take the funds it controls.").
			Validate(func(s string) error {
				if s != "REVEAL" {
					return fmt.Errorf("type REVEAL (all caps) to proceed")
				}
				return nil
			}).
			Value(&typed),
	)))
	if err != nil || !ok {
		return err
	}
	var wif string
	err = withAutoUnlock(c, func() error {
		var dumpErr error
		wif, dumpErr = c.dumpPrivKey(strings.TrimSpace(address))
		return dumpErr
	})
	if err != nil {
		return err
	}
	printWarn("Private key follows. Clear your terminal afterwards (e.g. with reset).")
	printBox(th.bad.Bold(true).Render(wif))
	return nil
}

func signMessageFlow(c *client) error {
	var address, message string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Sign with address").
			Validate(func(s string) error { return validateRecipient(c, strings.TrimSpace(s)) }).
			Value(&address),
		huh.NewText().
			Title("Message").
			Validate(huh.ValidateNotEmpty()).
			Value(&message),
	)))
	if err != nil || !ok {
		return err
	}
	var sig string
	err = withAutoUnlock(c, func() error {
		var signErr error
		sig, signErr = c.signMessage(strings.TrimSpace(address), message)
		return signErr
	})
	if err != nil {
		return err
	}
	printSuccess("Signature:")
	printBox(th.accent.Render(sig))
	return nil
}

func verifyMessageFlow(c *client) error {
	var address, signature, message string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("Signer address").
			Validate(func(s string) error { return validateRecipient(c, strings.TrimSpace(s)) }).
			Value(&address),
		huh.NewInput().
			Title("Signature (base64)").
			Validate(huh.ValidateNotEmpty()).
			Value(&signature),
		huh.NewText().
			Title("Message").
			Validate(huh.ValidateNotEmpty()).
			Value(&message),
	)))
	if err != nil || !ok {
		return err
	}
	valid, err := c.verifyMessage(strings.TrimSpace(address), strings.TrimSpace(signature), message)
	if err != nil {
		return err
	}
	if valid {
		printSuccess("Signature is valid.")
	} else {
		printError(fmt.Errorf("signature is NOT valid for this address and message"))
	}
	return nil
}
