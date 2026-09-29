// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
)

func testCoin(n int, amount float64, address string, locked bool) coinRow {
	op := wire.OutPoint{Hash: chainhash.HashH([]byte(fmt.Sprintf("coin-%d", n))), Index: uint32(n % 3)}
	return coinRow{
		op:     op,
		key:    op.String(),
		meta:   coinMeta{amount: amount, address: address, spendable: true},
		known:  true,
		confs:  10,
		locked: locked,
	}
}

func keysOf(rows []coinRow) []string {
	keys := make([]string, len(rows))
	for i, row := range rows {
		keys[i] = row.key
	}
	return keys
}

func TestSearchCoins(t *testing.T) {
	big := testCoin(1, 9.49822153, "rprl1pnxczdmv80shm2av2", false)
	small := testCoin(2, 0.001, "rprl1pzzzzzzzzzzzzzzzzz", false)
	lockedBig := testCoin(3, 9.49999999, "rprl1pqqqqqqqqqqqqqqqqq", true)
	unpriced := coinRow{op: wire.OutPoint{Hash: chainhash.HashH([]byte("locked-before-start"))}, locked: true}
	unpriced.key = unpriced.op.String()
	rows := []coinRow{big, small, lockedBig, unpriced}

	tests := []struct {
		name  string
		query string
		want  []coinRow
	}{
		{"empty query keeps everything in order", "  ", rows},
		{"address fragment", "nxczdmv", []coinRow{big}},
		{"the whole address, not the shortened text a row shows", "rprl1pzzzzzzzzzzzzzzzzz", []coinRow{small}},
		{"txid fragment", big.op.Hash.String()[:12], []coinRow{big}},
		{"outpoint with index", big.key, []coinRow{big}},
		{"amount", "9.498", []coinRow{big}},
		{"locked outputs", "locked", []coinRow{lockedBig, unpriced}},
		{"every word must match", "locked 9.4999", []coinRow{lockedBig}},
		{"case is ignored", "RPRL1PNXCZDMV", []coinRow{big}},
		{"no match", "nothing-here", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, keysOf(tt.want), keysOf(searchCoins(rows, tt.query)))
		})
	}
}

func TestCoinsShownNote(t *testing.T) {
	assert.Equal(t, "Showing the largest 500 of 30031 outputs.", coinsShownNote(500, 30031, 30031, ""))
	assert.Equal(t, `Showing 500 of the 3120 outputs matching "9.49" (30031 in the wallet).`,
		coinsShownNote(500, 3120, 30031, "9.49"))
	assert.Equal(t, `12 of 30031 outputs match "locked".`, coinsShownNote(12, 12, 30031, "locked"))
}

// TestLockDeltaIgnoresUnlistedCoins guards the rule a searched list relies on: an apply reconciles only the rows it
// listed, so a locked output the search did not show must stay locked.
func TestLockDeltaIgnoresUnlistedCoins(t *testing.T) {
	listedLocked := testCoin(1, 5, "a", true)
	listedFree := testCoin(2, 4, "b", false)
	hiddenLocked := testCoin(3, 0.5, "c", true)
	hiddenFree := testCoin(4, 0.4, "d", false)
	all := []coinRow{listedLocked, listedFree, hiddenLocked, hiddenFree}
	listed := []coinRow{listedLocked, listedFree}

	toLock, toUnlock := lockDelta(listed, []string{listedLocked.key, listedFree.key})
	assert.Equal(t, []wire.OutPoint{listedFree.op}, outpointsOf(toLock))
	assert.Empty(t, toUnlock, "the hidden locked output must not be unlocked")

	toLock, toUnlock = lockDelta(listed, nil)
	assert.Empty(t, toLock)
	assert.Equal(t, []wire.OutPoint{listedLocked.op}, outpointsOf(toUnlock))

	// Reconciling the whole wallet with the same picks would have unlocked the hidden output.
	_, toUnlockAll := lockDelta(all, []string{listedLocked.key, listedFree.key})
	assert.Equal(t, []wire.OutPoint{hiddenLocked.op}, outpointsOf(toUnlockAll))
}

func outpointsOf(ops []*wire.OutPoint) []wire.OutPoint {
	var out []wire.OutPoint
	for _, op := range ops {
		out = append(out, *op)
	}
	return out
}
