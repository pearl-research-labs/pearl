// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"net"
	"net/http"
	_ "net/http/pprof" // nolint:gosec
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	neutrino "github.com/pearl-research-labs/pearl/spv"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/rpc/legacyrpc"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

var (
	cfg *config

	// chainClientRetryMin and chainClientRetryMax bound the delay between attempts to start the chain client.
	chainClientRetryMin = time.Second
	chainClientRetryMax = 30 * time.Second
)

// chainLoopStopTimeout bounds how long shutdown waits for the chain client to close its files, so a chain service that
// is slow to stop can't hold up the exit.
const chainLoopStopTimeout = 5 * time.Second

func main() {
	// Use all processor cores.
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Work around defer not working after os.Exit.
	if err := walletMain(); err != nil {
		os.Exit(1)
	}
}

// walletMain is a work-around main function that is required since deferred
// functions (such as log flushing) are not called with calls to os.Exit.
// Instead, main runs this function and checks for a non-nil error, at which
// point any defers have already run, and if the error is non-nil, the program
// can be exited with an error exit status.
func walletMain() error {
	// Load configuration and parse command line.  This function also
	// initializes logging and configures it accordingly.
	tcfg, _, err := loadConfig()
	if err != nil {
		return err
	}
	cfg = tcfg
	defer func() {
		if logRotator != nil {
			logRotator.Close()
		}
	}()

	// Initialise key scopes to match the active network's coin type.
	// This ensures the BIP-86 derivation path,
	// m/86'/1'/0'/0/0 on testnet, m/86'/808276'/0'/0/0 on mainnet.
	waddrmgr.InitKeyScopes(activeNet.Params.HDCoinType)

	// Show version at startup.
	log.Infof("Version %s", version())

	if cfg.Profile != "" {
		go func() {
			listenAddr := net.JoinHostPort("", cfg.Profile)
			log.Infof("Profile server listening on %s", listenAddr)
			profileRedirect := http.RedirectHandler("/debug/pprof",
				http.StatusSeeOther)
			http.Handle("/", profileRedirect)
			log.Errorf("%v", http.ListenAndServe(listenAddr, nil))
		}()
	}

	dbDir := networkDir(cfg.AppDataDir.Value, activeNet.Params)
	loader := wallet.NewLoader(
		activeNet.Params, dbDir, true, cfg.DBTimeout, 250,
	)

	// Create and start HTTP server to serve wallet client connections.
	// This will be updated with the wallet and chain server RPC client
	// created below after each is created.
	rpcs, legacyRPCServer, err := startRPCServers(loader)
	if err != nil {
		log.Errorf("Unable to create RPC servers: %v", err)
		return err
	}

	// Create and start chain RPC client so it's ready to connect to
	// the wallet when loaded later.
	chainCtx, cancelChain := context.WithCancel(context.Background())
	defer cancelChain()
	chainLoopDone := make(chan struct{})
	if !cfg.NoInitialLoad {
		go func() {
			defer close(chainLoopDone)
			rpcClientConnectLoop(chainCtx, legacyRPCServer, loader)
		}()
	} else {
		close(chainLoopDone)
	}

	loader.RunAfterLoad(func(w *wallet.Wallet) {
		startWalletRPCServices(w, rpcs, legacyRPCServer)
	})

	if !cfg.NoInitialLoad {
		// Load the wallet database.  It must have been created already
		// or this will return an appropriate error.
		_, err = loader.OpenExistingWallet([]byte(cfg.WalletPass), true)
		if err != nil {
			log.Error(err)
			return err
		}
	}

	// Add interrupt handlers to shutdown the various process components
	// before exiting.  Interrupt handlers run in LIFO order, so the wallet
	// (which should be closed last) is added first.
	//
	// Waiting for the chain client to close the chain database and header files is added before it, as the last step.
	addInterruptHandler(func() {
		select {
		case <-chainLoopDone:
		case <-time.After(chainLoopStopTimeout):
			log.Warn("Timed out waiting for the chain client to stop")
		}
	})
	addInterruptHandler(func() {
		err := loader.UnloadWallet()
		if err != nil && err != wallet.ErrNotLoaded {
			log.Errorf("Failed to close wallet: %v", err)
		}
	})
	if rpcs != nil {
		addInterruptHandler(func() {
			// TODO: Does this need to wait for the grpc server to
			// finish up any requests?
			log.Warn("Stopping RPC server...")
			rpcs.Stop()
			log.Info("RPC server shutdown")
		})
	}
	if legacyRPCServer != nil {
		addInterruptHandler(func() {
			log.Warn("Stopping legacy RPC server...")
			legacyRPCServer.Stop()
			log.Info("Legacy RPC server shutdown")
		})
		go func() {
			<-legacyRPCServer.RequestProcessShutdown()
			simulateInterrupt()
		}()
	}

	// Added last so it runs first: whatever the wallet still waits for from the chain client, such as a recovery batch,
	// then ends at once instead of holding up the shutdown.
	addInterruptHandler(cancelChain)

	<-interruptHandlersDone
	log.Info("Shutdown complete")
	return nil
}

// rpcClientConnectLoop continuously attempts a connection to the consensus RPC
// server.  When a connection is established, the client is used to sync the
// loaded wallet, either immediately or when loaded at a later time.
//
// The legacy RPC is optional.  If set, the connected RPC client will be
// associated with the server for RPC passthrough and to enable additional
// methods.
func rpcClientConnectLoop(ctx context.Context, legacyRPCServer *legacyrpc.Server, loader *wallet.Loader) {
	var certs []byte
	if !cfg.UseSPV {
		certs = readCAFile()
	}

	netDir := networkDir(cfg.AppDataDir.Value, activeNet.Params)
	retryDelay := chainClientRetryMin
	syncDataReset := false

	for ctx.Err() == nil {
		chainClient, release, err := startChainClient(ctx, netDir, certs)
		if err != nil {
			// Resetting more than once can't help and would only throw away the sync progress over and over.
			if !syncDataReset && isSyncDataCorrupt(err) && resetSyncData(netDir) == nil {
				syncDataReset = true
				continue
			}

			select {
			case <-time.After(retryDelay):
			case <-ctx.Done():
				return
			}
			retryDelay = min(2*retryDelay, chainClientRetryMax)

			continue
		}
		retryDelay = chainClientRetryMin

		// Rather than inlining this logic directly into the loader
		// callback, a function variable is used to avoid running any of
		// this after the client disconnects by setting it to nil.  This
		// prevents the callback from associating a wallet loaded at a
		// later time with a client that has already disconnected.  A
		// mutex is used to make this concurrent safe.
		associateRPCClient := func(w *wallet.Wallet) {
			w.SynchronizeRPC(chainClient)
			if legacyRPCServer != nil {
				legacyRPCServer.SetChainServer(chainClient)
			}
		}
		mu := new(sync.Mutex)
		loader.RunAfterLoad(func(w *wallet.Wallet) {
			mu.Lock()
			associate := associateRPCClient
			mu.Unlock()
			if associate != nil {
				associate(w)
			}
		})

		stopOnCancel := context.AfterFunc(ctx, chainClient.Stop)
		chainClient.WaitForShutdown()
		stopOnCancel()

		mu.Lock()
		associateRPCClient = nil
		mu.Unlock()

		// The next attempt opens the same files.
		release()
		if ctx.Err() != nil {
			return
		}

		loadedWallet, ok := loader.LoadedWallet()
		if ok {
			// Do not attempt a reconnect when the wallet was
			// explicitly stopped.
			if loadedWallet.ShuttingDown() {
				return
			}

			loadedWallet.SetChainSynced(false)

			// TODO: Rework the wallet so changing the RPC client
			// does not require stopping and restarting everything.
			loadedWallet.Stop()
			loadedWallet.WaitForShutdown()
			loadedWallet.Start()
		}
	}
}

// startChainClient starts the chain client and returns what releases it once it has shut down. On failure nothing is
// left open, so the next attempt can open the files again or move them aside.
func startChainClient(ctx context.Context, netDir string, certs []byte) (chain.Interface, func(), error) {
	if !cfg.UseSPV {
		chainClient, err := startChainRPC(certs)
		if err != nil {
			log.Errorf("Unable to open connection to consensus RPC server: %v", err)
			return nil, nil, err
		}

		return chainClient, func() {}, nil
	}

	spvdb, err := walletdb.Create("bdb", filepath.Join(netDir, neutrinoDBName), true, cfg.DBTimeout, false)
	if err != nil {
		log.Errorf("Unable to create Neutrino DB: %s", err)
		return nil, nil, err
	}
	chainService, err := neutrino.NewChainService(neutrino.Config{
		DataDir:      netDir,
		Database:     spvdb,
		ChainParams:  *activeNet.Params,
		ConnectPeers: cfg.ConnectPeers,
		AddPeers:     cfg.AddPeers,
	})
	if err != nil {
		log.Errorf("Couldn't create Neutrino ChainService: %s", err)
		_ = spvdb.Close()
		return nil, nil, err
	}

	// The chain service uses the database until it has stopped.
	release := func() {
		if err := chainService.Stop(); err != nil {
			log.Errorf("Couldn't stop Neutrino ChainService: %s", err)
		}
		_ = spvdb.Close()
	}

	chainClient := chain.NewNeutrinoClient(activeNet.Params, chainService)
	if err := chainClient.Start(ctx); err != nil {
		log.Errorf("Couldn't start Neutrino client: %s", err)
		release()
		return nil, nil, err
	}

	return chainClient, release, nil
}

func readCAFile() []byte {
	// Read certificate file if TLS is not disabled.
	var certs []byte
	if !cfg.DisableClientTLS {
		var err error
		certs, err = os.ReadFile(cfg.CAFile.Value)
		if err != nil {
			log.Warnf("Cannot open CA file: %v", err)
			// If there's an error reading the CA file, continue
			// with nil certs and without the client connection.
			certs = nil
		}
	} else {
		log.Info("Chain server RPC TLS is disabled")
	}

	return certs
}

// startChainRPC opens a RPC client connection to a pearld server for blockchain
// services.  This function uses the RPC options from the global config and
// there is no recovery in case the server is not available or if there is an
// authentication error.  Instead, all requests to the client will simply error.
func startChainRPC(certs []byte) (*chain.RPCClient, error) {
	log.Infof("Attempting RPC client connection to %v", cfg.RPCConnect)
	rpcc, err := chain.NewRPCClient(activeNet.Params, cfg.RPCConnect,
		cfg.PearldUsername, cfg.PearldPassword, certs, cfg.DisableClientTLS, 0)
	if err != nil {
		return nil, err
	}
	err = rpcc.Start(context.Background())
	return rpcc, err
}
