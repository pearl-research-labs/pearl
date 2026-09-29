// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
)

func TestTxPage(t *testing.T) {
	tests := []struct {
		name      string
		txids     []string
		wantPage  []string
		wantShown int
	}{
		{"empty", nil, nil, 0},
		{"short page", []string{"a", "a"}, []string{"a", "a"}, 1},
		{"exactly a page", []string{"a", "b", "b"}, []string{"a", "b", "b"}, 2},
		{"past the page", []string{"a", "a", "b", "b", "c"}, []string{"a", "a", "b", "b"}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entries := make([]btcjson.ListTransactionsResult, len(tt.txids))
			for i, txid := range tt.txids {
				entries[i].TxID = txid
			}

			page, gotShown := txPage(entries, 2)

			var gotPage []string
			for _, entry := range page {
				gotPage = append(gotPage, entry.TxID)
			}
			assert.Equal(t, tt.wantPage, gotPage)
			assert.Equal(t, tt.wantShown, gotShown)
		})
	}
}
