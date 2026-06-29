package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	flags "github.com/jessevdk/go-flags"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
)

const (
	showHelpMessage = "Specify -h to show available options"
	listCmdMessage  = "Specify -l to list available commands"
)

// commandUsage returns the usage for a specific command.
func commandUsage(method string) string {
	usage, err := btcjson.MethodUsageText(method)
	if err != nil {
		// This should never happen since the method was already checked
		// before calling this function, but be safe.
		return fmt.Sprintf("Failed to obtain command usage: %v", err)
	}

	return fmt.Sprintf("Usage:\n  %s", usage)
}

// usage returns the general usage when the help flag is not displayed and
// an invalid command was specified.  The commandUsage function is used
// instead when a valid command was specified.
func usage(errorMessage string) string {
	appName := filepath.Base(os.Args[0])
	appName = strings.TrimSuffix(appName, filepath.Ext(appName))
	return fmt.Sprintf("%s\nUsage:\n  %s [OPTIONS] <command> <args...>\n\n%s\n%s",
		errorMessage, appName, showHelpMessage, listCmdMessage)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run executes prlctl, returning an error on failure.  main is responsible for
// reporting the error and setting the exit code, keeping all process exit
// handling in a single place.
func run() error {
	cfg, args, err := loadConfig()
	if err != nil {
		return err
	}
	if cfg.ShowHelp {
		writeHelp(cfg)
		return nil
	}
	if cfg.ShowVersion {
		fmt.Println(filepath.Base(os.Args[0]), "version", version())
		return nil
	}
	if cfg.ListCommands {
		listCommands()
		return nil
	}
	if len(args) < 1 {
		return errors.New(usage("no command specified"))
	}

	// Ensure the specified method identifies a valid registered command and
	// is one of the usable types.
	method := args[0]
	usageFlags, err := btcjson.MethodUsageFlags(method)
	if err != nil {
		return fmt.Errorf("unrecognized command %q\n%s", method, listCmdMessage)
	}
	if usageFlags&unusableFlags != 0 {
		return fmt.Errorf("the %q command can only be used via "+
			"websockets\n%s", method, listCmdMessage)
	}

	// Convert remaining command line args to a slice of interface values
	// to be passed along as parameters to new command creation function.
	//
	// Since some commands, such as submitblock, can involve data which is
	// too large for the Operating System to allow as a normal command line
	// parameter, support using '-' as an argument to allow the argument
	// to be read from a stdin pipe.
	bio := bufio.NewReader(os.Stdin)
	params := make([]interface{}, 0, len(args[1:]))
	for _, arg := range args[1:] {
		if arg == "-" {
			param, err := bio.ReadString('\n')
			if err != nil && err != io.EOF {
				return fmt.Errorf("failed to read data from stdin: %w", err)
			}
			if err == io.EOF && len(param) == 0 {
				return fmt.Errorf("not enough lines provided on stdin")
			}
			param = strings.TrimRight(param, "\r\n")
			params = append(params, param)
			continue
		}

		params = append(params, arg)
	}

	// Attempt to create the appropriate command using the arguments
	// provided by the user.  This validates the parameters and coerces the
	// positional string arguments into their proper JSON types.  On failure
	// the command usage is printed before returning.
	cmd, err := btcjson.NewCmd(method, params...)
	if err != nil {
		// Include the error code when it's a
		// btcjson.Error as it reallistcally will always be since the
		// NewCmd function is only supposed to return errors of that
		// type.
		if jerr, ok := err.(btcjson.Error); ok {
			return fmt.Errorf("%s command: %w (code: %s)\n%s",
				method, err, jerr.ErrorCode, commandUsage(method))
		}

		// The error is not a btcjson.Error and this really should not
		// happen.  Nevertheless, fallback to just showing the error if it
		// should happen due to a bug in the package.
		return fmt.Errorf("%s command: %w\n%s", method, err,
			commandUsage(method))
	}

	// Send the request to the server using the user-specified connection
	// configuration.
	client, err := newRPCClient(cfg)
	if err != nil {
		return err
	}
	defer client.Shutdown()

	result, err := rpcclient.ReceiveFuture(client.SendCmd(cmd))
	if err != nil {
		return err
	}

	// Choose how to display the result based on its type.
	strResult := string(result)
	switch {
	case strings.HasPrefix(strResult, "{") || strings.HasPrefix(strResult, "["):
		var dst bytes.Buffer
		if err := json.Indent(&dst, result, "", "  "); err != nil {
			return fmt.Errorf("failed to format result: %w", err)
		}
		fmt.Println(dst.String())

	case strings.HasPrefix(strResult, `"`):
		var str string
		if err := json.Unmarshal(result, &str); err != nil {
			return fmt.Errorf("failed to unmarshal result: %w", err)
		}
		fmt.Println(str)

	case strResult != "null":
		fmt.Println(strResult)
	}

	return nil
}

// writeHelp writes the go-flags generated help plus prlctl's stdin convention.
func writeHelp(cfg *config) {
	parser := flags.NewParser(cfg, flags.Default)
	parser.WriteHelp(os.Stdout)
	fmt.Println()
	fmt.Println("The special parameter `-` " +
		"indicates that a parameter should be read " +
		"from the\nnext unread line from standard " +
		"input.")
}

func newRPCClient(cfg *config) (*rpcclient.Client, error) {
	var certs []byte
	if !cfg.NoTLS && cfg.RPCCert != "" {
		pem, err := os.ReadFile(cfg.RPCCert)
		if err != nil {
			return nil, err
		}
		certs = pem
	}

	return rpcclient.New(&rpcclient.ConnConfig{
		Host:          cfg.RPCServer,
		User:          cfg.RPCUser,
		Pass:          cfg.RPCPassword,
		Proxy:         cfg.Proxy,
		ProxyUser:     cfg.ProxyUser,
		ProxyPass:     cfg.ProxyPass,
		DisableTLS:    cfg.NoTLS,
		Certificates:  certs,
		TLSSkipVerify: cfg.TLSSkipVerify,
		HTTPPostMode:  true,
		// prlctl is a one-shot CLI; fail fast instead of retrying.
		HTTPPostTries: 1,
	}, nil)
}
