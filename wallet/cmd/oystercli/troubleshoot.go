// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"charm.land/huh/v2"
)

// troubleshootScreen groups the diagnostic tools.
func troubleshootScreen(c *client) error {
	for {
		const (
			opConsole = "console"
			opDoctor  = "doctor"
			opLogs    = "logs"
			opBack    = "back"
		)
		choice := opConsole
		submitted, err := runForm(newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title("Troubleshoot").
				Options(
					huh.NewOption("RPC console         run any wallet or node RPC", opConsole),
					huh.NewOption("Doctor              full connectivity and state checkup", opDoctor),
					huh.NewOption("Logs                inspect the oyster log file", opLogs),
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
		case opConsole:
			opErr = consoleScreen(c)
		case opDoctor:
			runDoctor(c.cfg, c)
		case opLogs:
			opErr = logsScreen(c.cfg)
		}
		if opErr != nil {
			printError(opErr)
		}
	}
}
