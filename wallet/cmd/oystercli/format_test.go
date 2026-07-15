// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFmtPRLFloat(t *testing.T) {
	tests := []struct {
		name string
		v    float64
		want string
	}{
		{"integral", 5, "5.00 PRL"},
		{"one decimal", 1.5, "1.50 PRL"},
		{"full precision", 0.12345678, "0.12345678 PRL"},
		{"trailing zeros trimmed", 2.50000000, "2.50 PRL"},
		{"zero", 0, "0.00 PRL"},
		{"negative", -1.5, "-1.50 PRL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, fmtPRLFloat(tt.v))
		})
	}
}

func TestParsePRL(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    float64
		wantErr bool
	}{
		{"simple", "1.5", 1.5, false},
		{"whitespace", "  2 ", 2, false},
		{"zero rejected", "0", 0, true},
		{"negative rejected", "-1", 0, true},
		{"garbage rejected", "abc", 0, true},
		{"empty rejected", "", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePRL(tt.input)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.InDelta(t, tt.want, got.ToPRL(), 1e-8)
		})
	}
}

func TestShortID(t *testing.T) {
	tests := []struct {
		name string
		s    string
		max  int
		want string
	}{
		{"short unchanged", "abcdef", 10, "abcdef"},
		{"long truncated", "0123456789abcdef0123456789abcdef", 17, "01234567…89abcdef"},
		{"tiny max unchanged", "0123456789", 4, "0123456789"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, shortID(tt.s, tt.max))
		})
	}
}

func TestSortedKeys(t *testing.T) {
	m := map[string]btcutil.Amount{
		"zebra":    1,
		"alpha":    2,
		"default":  3,
		"imported": 4,
	}
	assert.Equal(t, []string{"default", "alpha", "imported", "zebra"}, sortedKeys(m))
}

func TestSyncPercent(t *testing.T) {
	assert.Equal(t, "50.0% (50/100)", syncPercent(&syncInfo{height: 50, peerHeight: 100}))
	// Nearly-but-not-fully synced must not round up to 100.0%.
	assert.Equal(t, "99.9% (86840/86864)", syncPercent(&syncInfo{height: 86840, peerHeight: 86864}))
	// Genuinely at the tip shows 100.0%; ahead of the peer is capped.
	assert.Equal(t, "100.0% (100/100)", syncPercent(&syncInfo{height: 100, peerHeight: 100}))
	assert.Equal(t, "100.0% (120/100)", syncPercent(&syncInfo{height: 120, peerHeight: 100}))
	assert.Equal(t, "height 42", syncPercent(&syncInfo{height: 42}))
}
