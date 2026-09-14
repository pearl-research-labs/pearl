// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// accountsScreen manages wallet accounts: list, create, rename, and address
// inspection.
func accountsScreen(c *client) error {
	for {
		accounts, err := c.listAccounts(0)
		if err != nil {
			return err
		}

		rows := make([][2]string, 0, len(accounts))
		for _, name := range sortedKeys(accounts) {
			rows = append(rows, [2]string{name, fmtPRL(accounts[name])})
		}
		printTitle("Accounts")
		printBox(kvLines(rows))

		const (
			opNew       = "new"
			opRename    = "rename"
			opAddresses = "addresses"
			opBack      = "back"
		)
		choice := opNew
		form := newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title("Accounts").
				Options(
					huh.NewOption("Create a new account", opNew),
					huh.NewOption("Rename an account", opRename),
					huh.NewOption("Show addresses of an account", opAddresses),
					huh.NewOption("Back", opBack),
				).
				Value(&choice),
		))
		submitted, err := runForm(form)
		if err != nil {
			return err
		}
		if !submitted || choice == opBack {
			return nil
		}

		var opErr error
		switch choice {
		case opNew:
			opErr = createAccountFlow(c)
		case opRename:
			opErr = renameAccountFlow(c, sortedKeys(accounts))
		case opAddresses:
			opErr = showAccountAddresses(c, sortedKeys(accounts))
		}
		if opErr != nil {
			printError(opErr)
		}
	}
}

func validateAccountName(s string) error {
	s = strings.TrimSpace(s)
	if s == "" {
		return fmt.Errorf("name is required")
	}
	if s == "*" || strings.EqualFold(s, "imported") {
		return fmt.Errorf("%q is a reserved account name", s)
	}
	return nil
}

func createAccountFlow(c *client) error {
	var name string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title("New account name").
			Description("Creating accounts requires an unlocked wallet.").
			Validate(validateAccountName).
			Value(&name),
	)))
	if err != nil || !ok {
		return err
	}
	name = strings.TrimSpace(name)
	if err := withAutoUnlock(c, func() error { return c.createAccount(name) }); err != nil {
		return err
	}
	printSuccess(fmt.Sprintf("Account %q created.", name))
	return nil
}

func renameAccountFlow(c *client, names []string) error {
	opts := make([]huh.Option[string], 0, len(names))
	for _, n := range names {
		if n == "imported" {
			continue
		}
		opts = append(opts, huh.NewOption(n, n))
	}
	var oldName, newName string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Rename which account?").
			Options(opts...).
			Value(&oldName),
		huh.NewInput().
			Title("New name").
			Validate(validateAccountName).
			Value(&newName),
	)))
	if err != nil || !ok {
		return err
	}
	newName = strings.TrimSpace(newName)
	if err := withAutoUnlock(c, func() error { return c.renameAccount(oldName, newName) }); err != nil {
		return err
	}
	printSuccess(fmt.Sprintf("Renamed %q to %q.", oldName, newName))
	return nil
}

func showAccountAddresses(c *client, names []string) error {
	account := ""
	opts := make([]huh.Option[string], 0, len(names))
	for _, n := range names {
		opts = append(opts, huh.NewOption(n, n))
	}
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Account").
			Options(opts...).
			Value(&account),
	)))
	if err != nil || !ok {
		return err
	}

	addrs, err := c.addressesByAccount(account)
	if err != nil {
		return err
	}
	printTitle(fmt.Sprintf("Addresses in %q (%d)", account, len(addrs)))
	if len(addrs) == 0 {
		lipgloss.Println(th.subtle.Render("No addresses yet; use Receive to generate one."))
		return nil
	}
	var b strings.Builder
	for i, a := range addrs {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(th.accent.Render(a.String()))
	}
	printBox(b.String())
	return nil
}
