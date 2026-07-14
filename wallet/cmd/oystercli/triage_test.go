// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClassifyConnectError(t *testing.T) {
	cfgNoWallet := &config{AppData: t.TempDir()}
	cfgNoWallet.activeNet = mainNetForTest()

	cfgWithWallet := configWithWalletDB(t)

	tests := []struct {
		name string
		cfg  *config
		err  error
		want triageKind
	}{
		{"missing wallet wins", cfgNoWallet, errors.New("connection refused"), triageNoWallet},
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

func TestProbeTCP(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer l.Close()

	assert.True(t, probeTCP(l.Addr().String()))

	l.Close()
	assert.False(t, probeTCP(l.Addr().String()))
}
