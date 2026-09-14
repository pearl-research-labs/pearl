// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	flags "github.com/jessevdk/go-flags"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/wallet/netparams"
)

const (
	// unusableFlags are the command usage flags which this utility are not
	// able to use.  In particular it doesn't support websockets and
	// consequently notifications.
	unusableFlags = btcjson.UFWebsocketOnly | btcjson.UFNotification
)

var (
	pearldHomeDir         = btcutil.AppDataDir("pearld", false)
	prlctlHomeDir         = btcutil.AppDataDir("prlctl", false)
	oysterHomeDir         = btcutil.AppDataDir("oyster", false)
	defaultConfigFile     = filepath.Join(prlctlHomeDir, "prlctl.conf")
	defaultRPCServer      = "localhost"
	defaultRPCCertFile    = filepath.Join(pearldHomeDir, "rpc.cert")
	defaultWalletCertFile = filepath.Join(oysterHomeDir, "rpc.cert")
)

// listCommands categorizes and lists all of the usable commands along with
// their one-line usage.
func listCommands() {
	const (
		categoryChain uint8 = iota
		categoryWallet
		numCategories
	)

	// Get a list of registered commands and categorize and filter them.
	cmdMethods := btcjson.RegisteredCmdMethods()
	categorized := make([][]string, numCategories)
	for _, method := range cmdMethods {
		flags, err := btcjson.MethodUsageFlags(method)
		if err != nil {
			// This should never happen since the method was just
			// returned from the package, but be safe.
			continue
		}

		// Skip the commands that aren't usable from this utility.
		if flags&unusableFlags != 0 {
			continue
		}

		usage, err := btcjson.MethodUsageText(method)
		if err != nil {
			// This should never happen since the method was just
			// returned from the package, but be safe.
			continue
		}

		// Categorize the command based on the usage flags.
		category := categoryChain
		if flags&btcjson.UFWalletOnly != 0 {
			category = categoryWallet
		}
		categorized[category] = append(categorized[category], usage)
	}

	// Display the command according to their categories.
	categoryTitles := make([]string, numCategories)
	categoryTitles[categoryChain] = "Chain Server Commands:"
	categoryTitles[categoryWallet] = "Wallet Server Commands (--wallet):"
	for category := uint8(0); category < numCategories; category++ {
		fmt.Println(categoryTitles[category])
		for _, usage := range categorized[category] {
			fmt.Println(usage)
		}
		fmt.Println()
	}
}

// config defines the configuration options for prlctl.
//
// See loadConfig for details on the configuration load process.
type config struct {
	ShowHelp       bool   `no-flag:"true"`
	ConfigFile     string `short:"C" long:"configfile" description:"Path to configuration file"`
	ListCommands   bool   `short:"l" long:"listcommands" description:"List all of the supported commands and exit"`
	NoTLS          bool   `long:"notls" description:"Disable TLS"`
	Proxy          string `long:"proxy" description:"Connect via SOCKS5 proxy (eg. 127.0.0.1:9050)"`
	ProxyPass      string `long:"proxypass" default-mask:"-" description:"Password for proxy server"`
	ProxyUser      string `long:"proxyuser" description:"Username for proxy server"`
	RegressionTest bool   `long:"regtest" description:"Connect to the regression test network"`
	RPCCert        string `short:"c" long:"rpccert" description:"RPC server certificate chain for validation"`
	RPCPassword    string `short:"P" long:"rpcpass" default-mask:"-" description:"RPC password"`
	RPCServer      string `short:"s" long:"rpcserver" description:"RPC server to connect to"`
	RPCUser        string `short:"u" long:"rpcuser" description:"RPC username"`
	SimNet         bool   `long:"simnet" description:"Connect to the simulation test network"`
	TLSSkipVerify  bool   `long:"skipverify" description:"Do not verify tls certificates (not recommended!)"`
	TestNet        bool   `long:"testnet" description:"Connect to testnet"`
	TestNet2       bool   `long:"testnet2" description:"Connect to testnet v2"`
	SigNet         bool   `long:"signet" description:"Connect to signet"`
	ShowVersion    bool   `short:"V" long:"version" description:"Display version information and exit"`
	Wallet         bool   `long:"wallet" description:"Connect to wallet"`
}

// normalizeAddress returns addr with the passed default port appended if
// there is not already a port specified.
func normalizeAddress(addr string, chain *chaincfg.Params, useWallet bool) (string, error) {
	_, _, err := net.SplitHostPort(addr)
	if err != nil {
		var defaultPort string
		switch chain {
		case &chaincfg.TestNetParams:
			if useWallet {
				defaultPort = netparams.TestNetParams.RPCServerPort
			} else {
				defaultPort = netparams.TestNetParams.RPCClientPort
			}
		case &chaincfg.TestNet2Params:
			if useWallet {
				defaultPort = netparams.TestNet2Params.RPCServerPort
			} else {
				defaultPort = netparams.TestNet2Params.RPCClientPort
			}
		case &chaincfg.SimNetParams:
			if useWallet {
				defaultPort = netparams.SimNetParams.RPCServerPort
			} else {
				defaultPort = netparams.SimNetParams.RPCClientPort
			}
		case &chaincfg.RegressionNetParams:
			if useWallet {
				// TODO Or: add port once regtest is supported in the wallet
				paramErr := fmt.Errorf("cannot use -wallet with -regtest, wallet not yet compatible with regtest")
				return "", paramErr
			} else {
				defaultPort = netparams.RegressionNetParams.RPCClientPort
			}
		case &chaincfg.SigNetParams:
			if useWallet {
				defaultPort = netparams.SigNetParams.RPCServerPort
			} else {
				defaultPort = netparams.SigNetParams.RPCClientPort
			}
		default:
			if useWallet {
				defaultPort = netparams.MainNetParams.RPCServerPort
			} else {
				defaultPort = netparams.MainNetParams.RPCClientPort
			}
		}

		return net.JoinHostPort(addr, defaultPort), nil
	}
	return addr, nil
}

// cleanAndExpandPath expands environment variables and leading ~ in the
// passed path, cleans the result, and returns it.
func cleanAndExpandPath(path string) string {
	// Expand initial ~ to OS specific home directory.
	if strings.HasPrefix(path, "~") {
		homeDir := filepath.Dir(prlctlHomeDir)
		path = strings.Replace(path, "~", homeDir, 1)
	}

	// NOTE: The os.ExpandEnv doesn't work with Windows-style %VARIABLE%,
	// but they variables can still be expanded via POSIX-style $VARIABLE.
	return filepath.Clean(os.ExpandEnv(path))
}

// loadConfig initializes and parses the config using a config file and command
// line options.
//
// The configuration proceeds as follows:
//  1. Start with a default config with sane settings
//  2. Pre-parse the command line to check for an alternative config file
//  3. Load configuration file overwriting defaults with any specified options
//  4. Parse CLI options and overwrite/add any specified options
//
// The above results in functioning properly without any config settings
// while still allowing the user to override settings with config files and
// command line options.  Command line options always take precedence.
func loadConfig() (*config, []string, error) {
	// Default config.
	cfg := config{
		ConfigFile: defaultConfigFile,
		RPCServer:  defaultRPCServer,
		RPCCert:    defaultRPCCertFile,
	}

	// Pre-parse the command line options to see if an alternative config
	// file, the version flag, or the list commands flag was specified.  Any
	// errors aside from the help message error can be ignored here since
	// they will be caught by the final parse below.
	preCfg := cfg
	preParser := flags.NewParser(&preCfg, flags.HelpFlag)
	_, err := preParser.Parse()
	if err != nil {
		if e, ok := err.(*flags.Error); ok && e.Type == flags.ErrHelp {
			preCfg.ShowHelp = true
			return &preCfg, nil, nil
		}
	}
	if preCfg.ShowVersion || preCfg.ListCommands {
		return &preCfg, nil, nil
	}

	appName := filepath.Base(os.Args[0])
	appName = strings.TrimSuffix(appName, filepath.Ext(appName))
	usageMessage := fmt.Sprintf("Use %s -h to show options", appName)

	if _, err := os.Stat(preCfg.ConfigFile); os.IsNotExist(err) {
		// Use config file for RPC server to create default prlctl config
		var serverConfigPath string
		if preCfg.Wallet {
			serverConfigPath = filepath.Join(oysterHomeDir, "oyster.conf")
		} else {
			serverConfigPath = filepath.Join(pearldHomeDir, "pearld.conf")
		}

		err := createDefaultConfigFile(preCfg.ConfigFile, serverConfigPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating a default config file: %v\n", err)
		}
	}

	// Load additional config from file.  PrintErrors is intentionally
	// omitted (unlike flags.Default) so that this function owns all error
	// reporting via the returned error instead of go-flags also writing to
	// stderr, which would print parse errors twice.
	parser := flags.NewParser(&cfg, flags.HelpFlag|flags.PassDoubleDash)
	err = flags.NewIniParser(parser).ParseFile(preCfg.ConfigFile)
	if err != nil {
		if _, ok := err.(*os.PathError); !ok {
			return nil, nil, fmt.Errorf("error parsing config file: %w\n%s",
				err, usageMessage)
		}
	}

	// Parse command line options again to ensure they take precedence.
	// args holds the positional arguments (the RPC command and its params).
	args, err := parser.Parse()
	if err != nil {
		if e, ok := err.(*flags.Error); ok && e.Type == flags.ErrHelp {
			cfg.ShowHelp = true
			return &cfg, nil, nil
		}
		return nil, nil, fmt.Errorf("%w\n%s", err, usageMessage)
	}

	// default network is mainnet
	network := &chaincfg.MainNetParams

	// Multiple networks can't be selected simultaneously.
	numNets := 0
	if cfg.TestNet {
		numNets++
		network = &chaincfg.TestNetParams
	}
	if cfg.TestNet2 {
		numNets++
		network = &chaincfg.TestNet2Params
	}
	if cfg.SimNet {
		numNets++
		network = &chaincfg.SimNetParams
	}
	if cfg.RegressionTest {
		numNets++
		network = &chaincfg.RegressionNetParams
	}
	if cfg.SigNet {
		numNets++
		network = &chaincfg.SigNetParams
	}

	if numNets > 1 {
		str := "%s: Multiple network params (testnet, testnet2, " +
			"simnet, regtest, signet) can't be used together -- choose one"
		err := fmt.Errorf(str, "loadConfig")
		return nil, nil, err
	}

	// Override the RPC certificate if the --wallet flag was specified and
	// the user did not specify one.
	if cfg.Wallet && cfg.RPCCert == defaultRPCCertFile {
		cfg.RPCCert = defaultWalletCertFile
	}

	// Handle environment variable expansion in the RPC certificate path.
	cfg.RPCCert = cleanAndExpandPath(cfg.RPCCert)

	// Add default port to RPC server based on --testnet and --wallet flags
	// if needed.
	cfg.RPCServer, err = normalizeAddress(cfg.RPCServer, network, cfg.Wallet)
	if err != nil {
		return nil, nil, err
	}

	return &cfg, args, nil
}

// createDefaultConfigFile creates a basic config file at the given destination path.
// For this it tries to read the config file for the RPC server (either pearld or
// oyster), and extract the RPC user and password from it.
func createDefaultConfigFile(destinationPath, serverConfigPath string) error {
	// Read the RPC server config
	serverConfigFile, err := os.Open(serverConfigPath)
	if err != nil {
		return err
	}
	defer serverConfigFile.Close()
	content, err := io.ReadAll(serverConfigFile)
	if err != nil {
		return err
	}

	// Extract the rpcuser
	rpcUserRegexp := regexp.MustCompile(`(?m)^\s*rpcuser=([^\s]+)`)
	userSubmatches := rpcUserRegexp.FindSubmatch(content)
	if userSubmatches == nil {
		// No user found, nothing to do
		return nil
	}

	// Extract the rpcpass
	rpcPassRegexp := regexp.MustCompile(`(?m)^\s*rpcpass=([^\s]+)`)
	passSubmatches := rpcPassRegexp.FindSubmatch(content)
	if passSubmatches == nil {
		// No password found, nothing to do
		return nil
	}

	// Extract the notls
	noTLSRegexp := regexp.MustCompile(`(?m)^\s*notls=(0|1)(?:\s|$)`)
	noTLSSubmatches := noTLSRegexp.FindSubmatch(content)

	// Create the destination directory if it does not exists
	err = os.MkdirAll(filepath.Dir(destinationPath), 0700)
	if err != nil {
		return err
	}

	// Create the destination file and write the rpcuser and rpcpass to it
	dest, err := os.OpenFile(destinationPath,
		os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer dest.Close()

	destString := fmt.Sprintf("rpcuser=%s\nrpcpass=%s\n",
		string(userSubmatches[1]), string(passSubmatches[1]))
	if noTLSSubmatches != nil {
		destString += fmt.Sprintf("notls=%s\n", noTLSSubmatches[1])
	}

	if _, err := dest.WriteString(destString); err != nil {
		return err
	}

	return nil
}
