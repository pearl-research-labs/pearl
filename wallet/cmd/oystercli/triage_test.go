// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestClassifyConnectError(t *testing.T) {
	cfgNoWallet := &config{AppData: t.TempDir()}
	cfgNoWallet.activeNet = mainNetForTest()

	cfgWithWallet := configWithWalletDB(t)

	cfgRemote := &config{AppData: t.TempDir(), Connect: "wallet.example.com:44207"}
	cfgRemote.activeNet = mainNetForTest()

	tests := []struct {
		name string
		cfg  *config
		err  error
		want triageKind
	}{
		{"missing wallet wins", cfgNoWallet, errors.New("connection refused"), triageNoWallet},
		// A remote daemon has its own wallet: the local wallet.db must
		// not steer triage toward creating one here.
		{"remote ignores local wallet", cfgRemote, errors.New("connection refused"), triageNotRunning},
		{"refused", cfgWithWallet, errors.New("dial tcp 127.0.0.1:44207: connection refused"), triageNotRunning},
		{"timeout", cfgWithWallet, errors.New("i/o timeout"), triageNotRunning},
		{"tls", cfgWithWallet, errors.New("x509: certificate signed by unknown authority"), triageTLS},
		{"missing cert", cfgWithWallet, errors.New("cannot read certificate file /x/rpc.cert: no such file"), triageTLS},
		{"auth", cfgWithWallet, errors.New("status code: 401, response: \"\""), triageAuth},
		{"unknown", cfgWithWallet, errors.New("boom"), triageUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, classifyConnectError(tt.cfg, tt.err))
		})
	}
}
