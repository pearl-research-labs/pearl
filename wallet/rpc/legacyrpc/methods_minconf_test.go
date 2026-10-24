// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package legacyrpc

import (
	"math"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// The btcjson confirmation-count parameters (minconf, maxconf) are *int,
// but every legacyrpc handler narrowed them to int32 with a plain cast.
// Values above math.MaxInt32 wrapped: minconf 2**32+1 reached the wallet
// as 1, so balance, received, and UTXO queries — and the send paths —
// silently used a different confirmation threshold than the caller
// requested.
//
// These tests drive the handlers with a nil wallet: any call that gets
// past parameter validation dereferences the wallet and panics, while a
// rejected parameter returns an error first. Out-of-range values must
// return an error without panicking; in-range values (including the
// exact int32 boundary) must pass validation and reach the wallet,
// observed here as the nil-wallet panic.
func TestHandlersRejectOutOfRangeConfCounts(t *testing.T) {
	intPtr := func(v int) *int { return &v }

	// wrapToOne narrows to int32(1) without validation.
	wrapToOne := 1<<32 + 1
	// wrapToNegative narrows to a negative int32 without validation.
	wrapToNegative := 1 << 31

	validAddr := "prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d"

	handlers := []struct {
		name string
		call func(minConf int) (interface{}, error)
	}{
		{"getbalance", func(minConf int) (interface{}, error) {
			return getBalance(&btcjson.GetBalanceCmd{
				MinConf: intPtr(minConf),
			}, nil)
		}},
		{"getreceivedbyaccount", func(minConf int) (interface{}, error) {
			return getReceivedByAccount(&btcjson.GetReceivedByAccountCmd{
				Account: "default",
				MinConf: intPtr(minConf),
			}, nil)
		}},
		{"getreceivedbyaddress", func(minConf int) (interface{}, error) {
			return getReceivedByAddress(&btcjson.GetReceivedByAddressCmd{
				Address: validAddr,
				MinConf: intPtr(minConf),
			}, nil)
		}},
		{"listaccounts", func(minConf int) (interface{}, error) {
			return listAccounts(&btcjson.ListAccountsCmd{
				MinConf: intPtr(minConf),
			}, nil)
		}},
		{"listreceivedbyaccount", func(minConf int) (interface{}, error) {
			return listReceivedByAccount(
				&btcjson.ListReceivedByAccountCmd{
					MinConf: intPtr(minConf),
				}, nil)
		}},
		{"listreceivedbyaddress", func(minConf int) (interface{}, error) {
			includeEmpty := false
			return listReceivedByAddress(
				&btcjson.ListReceivedByAddressCmd{
					MinConf:      intPtr(minConf),
					IncludeEmpty: &includeEmpty,
				}, nil)
		}},
		{"listunspent", func(minConf int) (interface{}, error) {
			return listUnspent(&btcjson.ListUnspentCmd{
				MinConf: intPtr(minConf),
				MaxConf: intPtr(9999999),
			}, nil)
		}},
		{"sendfrom", func(minConf int) (interface{}, error) {
			return sendFrom(&btcjson.SendFromCmd{
				FromAccount: "default",
				ToAddress:   validAddr,
				Amount:      1,
				MinConf:     intPtr(minConf),
			}, nil, nil)
		}},
		{"sendmany", func(minConf int) (interface{}, error) {
			return sendMany(&btcjson.SendManyCmd{
				FromAccount: "default",
				Amounts:     map[string]float64{validAddr: 1},
				MinConf:     intPtr(minConf),
			}, nil)
		}},
	}

	try := func(call func() (interface{}, error)) (err error, panicked bool) {
		defer func() {
			if recover() != nil {
				panicked = true
				err = nil
			}
		}()
		_, err = call()
		return err, false
	}

	for _, h := range handlers {
		for _, tc := range []struct {
			name      string
			minConf   int
			wantError bool
		}{
			{"wraps to one", wrapToOne, true},
			{"wraps to negative", wrapToNegative, true},
			{"negative", -1, true},
			{"zero", 0, false},
			{"one", 1, false},
			{"max int32", math.MaxInt32, false},
		} {
			err, panicked := try(func() (interface{}, error) {
				return h.call(tc.minConf)
			})
			switch {
			case tc.wantError && panicked:
				t.Errorf("%s minconf=%d: got panic, want a "+
					"parameter error (value reached the "+
					"wallet unchecked)", h.name, tc.minConf)
			case tc.wantError && err == nil:
				t.Errorf("%s minconf=%d: got no error, want a "+
					"parameter error", h.name, tc.minConf)
			case !tc.wantError && !panicked:
				t.Errorf("%s minconf=%d: got err=%v, want the "+
					"value to pass validation and reach "+
					"the wallet", h.name, tc.minConf, err)
			}
		}
	}

	// listunspent validates maxconf as well as minconf.
	err, panicked := try(func() (interface{}, error) {
		return listUnspent(&btcjson.ListUnspentCmd{
			MinConf: intPtr(1),
			MaxConf: intPtr(wrapToOne),
		}, nil)
	})
	if panicked || err == nil {
		t.Errorf("listunspent maxconf=%d: got panic=%v err=%v, want a "+
			"parameter error", wrapToOne, panicked, err)
	}
}
