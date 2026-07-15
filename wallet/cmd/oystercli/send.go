// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
	"strings"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// minRelayFeeRate is the network's minimum relay fee rate in PRL/kB
// (1000 grain/kB, mempool.DefaultMinRelayTxFee). oyster's send RPCs treat the
// fee rate as a literal value with no daemon-side default — passing 0 builds
// a zero-fee transaction that every mempool rejects — so this floor is what
// an empty fee field selects.
const minRelayFeeRate = 1e-5

// sendScreen walks through paying a single recipient: pick the source
// account, enter and validate the destination and amount, review a summary,
// then sign and broadcast (unlocking on demand).
func sendScreen(c *client) error {
	accounts, err := c.listAccounts(1)
	if err != nil {
		return err
	}

	names := sortedKeys(accounts)
	fromAccount := "default"
	if len(names) > 0 {
		fromAccount = names[0]
	}

	accountOpts := make([]huh.Option[string], 0, len(names))
	for _, name := range names {
		label := fmt.Sprintf("%-16s %s", name, fmtPRL(accounts[name]))
		accountOpts = append(accountOpts, huh.NewOption(label, name))
	}

	var (
		address    string
		amountStr  string
		feeRateStr string
	)

	form := newForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("Send from").
				Description("Account to spend from.").
				Options(accountOpts...).
				Value(&fromAccount),
			huh.NewInput().
				Title("Recipient address").
				Placeholder(c.cfg.activeNet.Params.Name+" address").
				Validate(func(s string) error {
					return validateRecipient(c, strings.TrimSpace(s))
				}).
				Value(&address),
			huh.NewInput().
				Title("Amount").
				DescriptionFunc(func() string {
					return "Available: " + fmtPRL(accounts[fromAccount])
				}, &fromAccount).
				Placeholder("0.00 PRL").
				Validate(func(s string) error {
					amt, err := parsePRL(s)
					if err != nil {
						return err
					}
					if amt > accounts[fromAccount] {
						return fmt.Errorf("exceeds spendable balance (%s)", fmtPRL(accounts[fromAccount]))
					}
					return nil
				}).
				Value(&amountStr),
			huh.NewInput().
				Title("Fee rate (optional)").
				Description("PRL per kB; leave empty for the network minimum (0.00001).").
				Placeholder("0.00001").
				Validate(validateFeeRate).
				Value(&feeRateStr),
		),
	)
	submitted, err := runForm(form)
	if err != nil || !submitted {
		return err
	}

	address = strings.TrimSpace(address)
	amount, err := parsePRL(amountStr)
	if err != nil {
		return err
	}
	feeRate, feeLabel, err := parseFeeRate(feeRateStr)
	if err != nil {
		return err
	}

	printBox(kvLines([][2]string{
		{"From account", fromAccount},
		{"To", address},
		{"Amount", fmtPRL(amount)},
		{"Fee rate", feeLabel},
		{"Network", c.cfg.activeNet.Params.Name},
	}))

	// Default to Send: this confirm is the final step of a transaction the
	// user just built and reviewed, so Enter should commit it (matching the
	// Enter-to-advance rhythm of the fields above). When the wallet is
	// locked, the passphrase prompt in withAutoUnlock is the real gate.
	confirmed := true
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewConfirm().
			Title("Broadcast this transaction?").
			Description("This cannot be undone once confirmed by the network.").
			Affirmative("Send").
			Negative("Cancel").
			Value(&confirmed),
	)))
	if err != nil || !ok || !confirmed {
		if err == nil {
			printWarn("Send cancelled. Nothing was broadcast.")
		}
		return err
	}

	var txid *chainhash.Hash
	err = withAutoUnlock(c, func() error {
		return withSpinner("Signing and broadcasting...", func() error {
			var sendErr error
			txid, sendErr = c.send(fromAccount, address, amount, feeRate, 1)
			return sendErr
		})
	})
	if err != nil {
		return err
	}

	printSuccess("Transaction broadcast.")
	printBox(kvLines([][2]string{
		{"Txid", txid.String()},
		{"Amount", fmtPRL(amount)},
		{"To", address},
	}))
	return nil
}

// validateRecipient checks the address decodes and belongs to the active
// network. Validation is local so it can run on every submit attempt without
// touching the daemon.
func validateRecipient(c *client, s string) error {
	if s == "" {
		return fmt.Errorf("address is required")
	}
	addr, err := btcutil.DecodeAddress(s, c.cfg.activeNet.Params)
	if err != nil {
		return fmt.Errorf("not a valid address")
	}
	if !addr.IsForNet(c.cfg.activeNet.Params) {
		return fmt.Errorf("address is not for %s", c.cfg.activeNet.Params.Name)
	}
	return nil
}

// validateFeeRate accepts an empty value (network minimum) or a rate at or
// above the relay floor; anything lower is guaranteed to be rejected at
// broadcast, so it is refused here instead.
func validateFeeRate(s string) error {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return fmt.Errorf("must be a number")
	}
	if v < minRelayFeeRate {
		return fmt.Errorf("below the network minimum relay fee (%.5f PRL/kB)", minRelayFeeRate)
	}
	return nil
}

// parseFeeRate converts optional fee input to the RPC value, defaulting to
// the minimum relay fee when left empty.
func parseFeeRate(s string) (float64, string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return minRelayFeeRate, fmtPRLFloat(minRelayFeeRate) + "/kB (network minimum)", nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, "", fmt.Errorf("invalid fee rate")
	}
	return v, fmtPRLFloat(v) + "/kB", nil
}
