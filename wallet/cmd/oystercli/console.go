// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
)

// consoleUnusableFlags marks commands that cannot work over the HTTP POST
// transport used here (mirrors prlctl).
const consoleUnusableFlags = btcjson.UFWebsocketOnly | btcjson.UFNotification

// consoleScreen is an interactive raw JSON-RPC prompt with method
// autocompletion. Unknown methods are still sent verbatim: oyster forwards
// anything it does not handle to pearld, so node RPCs work here too.
func consoleScreen(c *client) error {
	methods := usableMethods()

	lipgloss.Println(th.subtle.Render(
		"Type a method with positional args (e.g. getbalance, listunspent 1),\n" +
			"help <method> for usage, or an empty line to leave. Tab-completion\n" +
			"suggestions cover wallet and node methods; unknown methods are sent as-is."))

	var history []string
	for {
		line := ""
		input := huh.NewInput().
			Title("rpc").
			Prompt("❯ ").
			Suggestions(append(historySuggestions(history), methods...)).
			Value(&line)
		ok, err := runForm(newForm(huh.NewGroup(input)))
		if err != nil {
			return err
		}
		line = strings.TrimSpace(line)
		if !ok || line == "" || line == "exit" || line == "quit" {
			return nil
		}
		history = append([]string{line}, history...)

		runConsoleCommand(c, line)
	}
}

// runConsoleCommand executes one console line and pretty-prints the outcome.
func runConsoleCommand(c *client, line string) {
	fields := strings.Fields(line)
	method := fields[0]
	args := fields[1:]

	lipgloss.Println(th.accent.Render("❯ " + line))

	result, err := executeRPCLine(c, method, args)
	if err != nil {
		printError(err)
		return
	}
	printJSONResult(result)
}

// executeRPCLine turns a method plus string args into an RPC call. Registered
// commands get their positional args coerced to the right JSON types via
// btcjson; unregistered ones are passed through as raw JSON when possible.
func executeRPCLine(c *client, method string, args []string) (json.RawMessage, error) {
	params := make([]interface{}, len(args))
	for i, a := range args {
		params[i] = a
	}

	if _, err := btcjson.MethodUsageFlags(method); err == nil {
		cmd, err := btcjson.NewCmd(method, params...)
		if err != nil {
			usage, uerr := btcjson.MethodUsageText(method)
			if uerr == nil {
				return nil, fmt.Errorf("%w\nusage: %s", err, usage)
			}
			return nil, err
		}
		return traced(c, method, func() (json.RawMessage, error) {
			return rpcclient.ReceiveFuture(c.rpc.SendCmd(cmd))
		})
	}

	// Not a registered command: send verbatim, JSON-encoding each arg (bare
	// words become strings, numbers/objects pass through untouched).
	raw := make([]json.RawMessage, len(args))
	for i, a := range args {
		if json.Valid([]byte(a)) {
			raw[i] = json.RawMessage(a)
		} else {
			b, _ := json.Marshal(a)
			raw[i] = b
		}
	}
	return traced(c, method, func() (json.RawMessage, error) {
		return c.rpc.RawRequest(method, raw)
	})
}

// printJSONResult renders an RPC result with indentation; scalar results are
// printed bare, like prlctl.
func printJSONResult(result json.RawMessage) {
	s := string(result)
	switch {
	case s == "" || s == "null":
		printSuccess("ok (no result)")
	case strings.HasPrefix(s, "{") || strings.HasPrefix(s, "["):
		var dst bytes.Buffer
		if err := json.Indent(&dst, result, "", "  "); err != nil {
			lipgloss.Println(th.value.Render(s))
			return
		}
		lipgloss.Println(th.value.Render(dst.String()))
	case strings.HasPrefix(s, `"`):
		var str string
		if err := json.Unmarshal(result, &str); err == nil {
			lipgloss.Println(th.value.Render(str))
		} else {
			lipgloss.Println(th.value.Render(s))
		}
	default:
		lipgloss.Println(th.value.Render(s))
	}
}

// usableMethods returns all registered RPC methods usable over this
// transport, sorted for the completion list.
func usableMethods() []string {
	all := btcjson.RegisteredCmdMethods()
	usable := make([]string, 0, len(all))
	for _, m := range all {
		flags, err := btcjson.MethodUsageFlags(m)
		if err != nil || flags&consoleUnusableFlags != 0 {
			continue
		}
		usable = append(usable, m)
	}
	sort.Strings(usable)
	return usable
}

// historySuggestions surfaces previous console lines in the autocomplete.
func historySuggestions(history []string) []string {
	if len(history) > 20 {
		history = history[:20]
	}
	return history
}
