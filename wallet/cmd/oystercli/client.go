// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// client wraps the JSON-RPC connection to oyster with helpers for the oyster
// extension methods that have no typed rpcclient bindings, session unlock
// tracking, and optional call tracing.
type client struct {
	rpc *rpcclient.Client
	cfg *config

	// unlockedByUs is set when this session unlocked the wallet, so it can
	// be locked again on exit as a courtesy.
	unlockedByUs bool
}

// dialClient establishes the HTTP POST mode RPC connection to oyster. No
// network traffic happens until the first call.
func dialClient(cfg *config) (*client, error) {
	connCfg := &rpcclient.ConnConfig{
		Host:         cfg.Connect,
		User:         cfg.RPCUser,
		Pass:         cfg.RPCPass,
		Params:       cfg.activeNet.Params.Name,
		HTTPPostMode: true,
		DisableTLS:   cfg.NoTLS,
		// Interactive tool: fail fast so errors surface immediately instead
		// of silently retrying with backoff.
		HTTPPostTries: 1,
	}
	if !cfg.NoTLS {
		certs, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("cannot read certificate file %s: %w", cfg.CAFile, err)
		}
		connCfg.Certificates = certs
	}

	rpc, err := rpcclient.New(connCfg, nil)
	if err != nil {
		return nil, err
	}
	return &client{rpc: rpc, cfg: cfg}, nil
}

// traced runs fn and, when --verbose is set, logs the method name, duration,
// and error outcome to stderr. Secrets never appear because only the method
// name is logged.
func traced[T any](c *client, method string, fn func() (T, error)) (T, error) {
	start := time.Now()
	v, err := fn()
	if c.cfg.Verbose {
		status := "ok"
		if err != nil {
			status = rawErrorDetail(err)
		}
		fmt.Fprintf(os.Stderr, "rpc %-24s %8s  %s\n", method, time.Since(start).Round(time.Millisecond), status)
	}
	return v, err
}

// rawCall invokes a parameterless RPC method and decodes the result into out
// (skipped when out is nil). It exists for the oyster extension methods that
// have no typed rpcclient binding.
func (c *client) rawCall(method string, out interface{}) error {
	res, err := traced(c, method, func() (json.RawMessage, error) {
		return c.rpc.RawRequest(method, nil)
	})
	if err != nil {
		return err
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(res, out)
}

// isRPCErrorCode reports whether err is a btcjson RPC error with the given
// code.
func isRPCErrorCode(err error, code btcjson.RPCErrorCode) bool {
	var rpcErr *btcjson.RPCError
	return errors.As(err, &rpcErr) && rpcErr.Code == code
}

// --- Status ---

// syncStatus decodes into the canonical btcjson result type; the RPC uses
// snake_case field names, so hand-rolled mirror structs decode to zeros.
func (c *client) syncStatus() (*btcjson.GetSyncProgressResult, error) {
	var sp btcjson.GetSyncProgressResult
	if err := c.rawCall("getsyncprogress", &sp); err != nil {
		return nil, err
	}
	return &sp, nil
}

func (c *client) walletLocked() (bool, error) {
	var locked bool
	err := c.rawCall("walletislocked", &locked)
	return locked, err
}

// chainSynced reports the wallet's own caught-up flag. Unlike
// getsyncprogress it works for both SPV and pearld-backed daemons.
func (c *client) chainSynced() (bool, error) {
	var synced bool
	err := c.rawCall("chainsynced", &synced)
	return synced, err
}

// syncInfo is a backend-agnostic view of the wallet's sync state.
type syncInfo struct {
	synced     bool
	height     int32 // wallet height (best effort; 0 when unknown)
	peerHeight int32 // best known peer height; only meaningful in SPV mode
	spv        bool  // true when getsyncprogress (SPV detail) was available
}

// fetchSyncInfo prefers the detailed SPV progress and falls back to the
// chainsynced flag plus the wallet's best block for pearld-backed daemons,
// where getsyncprogress always errors.
func (c *client) fetchSyncInfo() (*syncInfo, error) {
	if sp, err := c.syncStatus(); err == nil {
		return &syncInfo{
			synced:     sp.Synced,
			height:     sp.BlockHeight,
			peerHeight: sp.BestPeerHeight,
			spv:        true,
		}, nil
	}

	synced, err := c.chainSynced()
	if err != nil {
		return nil, err
	}
	info := &syncInfo{synced: synced}
	if _, height, err := c.bestBlock(); err == nil {
		info.height = height
	}
	return info, nil
}

func (c *client) info() (*btcjson.InfoWalletResult, error) {
	return traced(c, "getinfo", c.rpc.GetInfo)
}

func (c *client) bestBlock() (*chainhash.Hash, int32, error) {
	type result struct {
		hash   *chainhash.Hash
		height int32
	}
	res, err := traced(c, "getbestblock", func() (result, error) {
		hash, height, err := c.rpc.GetBestBlock()
		return result{hash, height}, err
	})
	return res.hash, res.height, err
}

// --- Balances and accounts ---

func (c *client) balance(minConf int) (btcutil.Amount, error) {
	return traced(c, "getbalance", func() (btcutil.Amount, error) {
		return c.rpc.GetBalanceMinConf("*", minConf)
	})
}

func (c *client) listAccounts(minConf int) (map[string]btcutil.Amount, error) {
	return traced(c, "listaccounts", func() (map[string]btcutil.Amount, error) {
		return c.rpc.ListAccountsMinConf(minConf)
	})
}

func (c *client) accountNames() ([]string, error) {
	accounts, err := c.listAccounts(0)
	if err != nil {
		return nil, err
	}
	return sortedKeys(accounts), nil
}

func (c *client) createAccount(name string) error {
	_, err := traced(c, "createnewaccount", func() (struct{}, error) {
		return struct{}{}, c.rpc.CreateNewAccount(name)
	})
	return err
}

func (c *client) renameAccount(oldName, newName string) error {
	_, err := traced(c, "renameaccount", func() (struct{}, error) {
		return struct{}{}, c.rpc.RenameAccount(oldName, newName)
	})
	return err
}

func (c *client) addressesByAccount(account string) ([]btcutil.Address, error) {
	return traced(c, "getaddressesbyaccount", func() ([]btcutil.Address, error) {
		return c.rpc.GetAddressesByAccount(account)
	})
}

// --- Addresses ---

func (c *client) newAddress(account string) (btcutil.Address, error) {
	return traced(c, "getnewaddress", func() (btcutil.Address, error) {
		return c.rpc.GetNewAddress(account)
	})
}

func (c *client) currentAddress(account string) (btcutil.Address, error) {
	return traced(c, "getaccountaddress", func() (btcutil.Address, error) {
		return c.rpc.GetAccountAddress(account)
	})
}

// --- Transactions ---

func (c *client) listTransactions(count, from int) ([]btcjson.ListTransactionsResult, error) {
	return traced(c, "listtransactions", func() ([]btcjson.ListTransactionsResult, error) {
		return c.rpc.ListTransactionsCountFrom("*", count, from)
	})
}

func (c *client) transaction(txid string) (*btcjson.GetTransactionResult, error) {
	hash, err := chainhash.NewHashFromStr(txid)
	if err != nil {
		return nil, err
	}
	return traced(c, "gettransaction", func() (*btcjson.GetTransactionResult, error) {
		return c.rpc.GetTransaction(hash)
	})
}

func (c *client) send(fromAccount, toAddress string, amount btcutil.Amount, feeRatePerKb float64, minConf int) (*chainhash.Hash, error) {
	addr, err := btcutil.DecodeAddress(toAddress, c.cfg.activeNet.Params)
	if err != nil {
		return nil, err
	}
	// Use sendmany rather than sendfrom: oyster gates sendfrom on an
	// RPC-typed chain client, so it fails with "Chain RPC is inactive" in
	// SPV mode even though the broadcast itself goes out over P2P. sendmany
	// takes the same from-account, fee rate, and minconf and works with
	// either backend (it is also what the desktop wallet uses).
	amounts := map[btcutil.Address]btcutil.Amount{addr: amount}
	return traced(c, "sendmany", func() (*chainhash.Hash, error) {
		return c.rpc.SendManyMinConf(fromAccount, amounts, feeRatePerKb, minConf)
	})
}

// --- Coins ---

func (c *client) listUnspent(minConf int) ([]btcjson.ListUnspentResult, error) {
	return traced(c, "listunspent", func() ([]btcjson.ListUnspentResult, error) {
		return c.rpc.ListUnspentMinMax(minConf, 9999999)
	})
}

func (c *client) listLocked() ([]*wire.OutPoint, error) {
	return traced(c, "listlockunspent", c.rpc.ListLockUnspent)
}

func (c *client) lockUnspent(unlock bool, ops []*wire.OutPoint) error {
	_, err := traced(c, "lockunspent", func() (struct{}, error) {
		return struct{}{}, c.rpc.LockUnspent(unlock, ops)
	})
	return err
}

// --- Security ---

func (c *client) unlock(passphrase string, timeoutSecs int64) error {
	_, err := traced(c, "walletpassphrase", func() (struct{}, error) {
		return struct{}{}, c.rpc.WalletPassphrase(passphrase, timeoutSecs)
	})
	if err == nil {
		c.unlockedByUs = true
	}
	return err
}

func (c *client) lock() error {
	_, err := traced(c, "walletlock", func() (struct{}, error) {
		return struct{}{}, c.rpc.WalletLock()
	})
	if err == nil {
		c.unlockedByUs = false
	}
	return err
}

func (c *client) changePassphrase(old, new string) error {
	_, err := traced(c, "walletpassphrasechange", func() (struct{}, error) {
		return struct{}{}, c.rpc.WalletPassphraseChange(old, new)
	})
	return err
}

func (c *client) importPrivKey(wif string, rescan bool) error {
	decoded, err := btcutil.DecodeWIF(wif)
	if err != nil {
		return fmt.Errorf("invalid WIF key: %w", err)
	}
	_, err = traced(c, "importprivkey", func() (struct{}, error) {
		return struct{}{}, c.rpc.ImportPrivKeyRescan(decoded, "imported", rescan)
	})
	return err
}

func (c *client) dumpPrivKey(addr string) (string, error) {
	decoded, err := btcutil.DecodeAddress(addr, c.cfg.activeNet.Params)
	if err != nil {
		return "", err
	}
	wif, err := traced(c, "dumpprivkey", func() (*btcutil.WIF, error) {
		return c.rpc.DumpPrivKey(decoded)
	})
	if err != nil {
		return "", err
	}
	return wif.String(), nil
}

func (c *client) signMessage(addr, message string) (string, error) {
	decoded, err := btcutil.DecodeAddress(addr, c.cfg.activeNet.Params)
	if err != nil {
		return "", err
	}
	return traced(c, "signmessage", func() (string, error) {
		return c.rpc.SignMessage(decoded, message)
	})
}

func (c *client) verifyMessage(addr, signature, message string) (bool, error) {
	decoded, err := btcutil.DecodeAddress(addr, c.cfg.activeNet.Params)
	if err != nil {
		return false, err
	}
	return traced(c, "verifymessage", func() (bool, error) {
		return c.rpc.VerifyMessage(decoded, signature, message)
	})
}

// stopDaemon asks oyster to shut down gracefully via its authenticated
// "stop" RPC (wallet unload + clean exit). Only a client holding the
// daemon's credentials can stop it, so this only ever affects "our" daemon.
func (c *client) stopDaemon() error {
	if err := c.rawCall("stop", nil); err != nil {
		return err
	}
	// The daemon is exiting, so there is nothing left to re-lock on our way
	// out; a stopped process holds no decrypted keys.
	c.unlockedByUs = false
	return nil
}

// lockOnExitIfNeeded locks the wallet again if this session unlocked it.
func (c *client) lockOnExitIfNeeded() {
	if c.unlockedByUs {
		_ = c.lock()
	}
}

// shutdown releases the underlying RPC client resources.
func (c *client) shutdown() {
	c.rpc.Shutdown()
}
