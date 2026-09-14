// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestExtractSeed(t *testing.T) {
	mnemonic := "abandon ability able about above absent absorb abstract absurd abuse access accident"
	tests := []struct {
		name   string
		output string
		want   string
	}{
		{"bare mnemonic", mnemonic + "\n", mnemonic},
		{"mnemonic after log noise", "2026-07-14 [INF] OYST: Version 1.1.5\n" + mnemonic + "\n", mnemonic},
		{"hex seed", "Creating the wallet...\nDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF00112233\n", "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF00112233"},
		{"no seed present", "some log line\nanother [INF] line\n", ""},
		{"empty output", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, extractSeed(tt.output))
		})
	}
}

func TestWrapWords(t *testing.T) {
	got := wrapWords("alpha beta gamma delta epsilon", 2)
	want := " 1.alpha   2.beta\n 3.gamma   4.delta\n 5.epsilon"
	assert.Equal(t, want, got)
}

// An omitted birthday must resolve to genesis, not "now": oyster starts the
// restore recovery scan at the birthday, so a too-recent one silently skips
// all history and the wallet comes up with no addresses and no balance.
func TestImportBirthday(t *testing.T) {
	cfg := &config{}
	cfg.activeNet = mainNetForTest()
	genesis := cfg.activeNet.Params.GenesisBlock.BlockHeader().Timestamp

	assert.Equal(t, genesis, importBirthday(cfg, ""))
	assert.Equal(t, genesis, importBirthday(cfg, "   "))

	want, err := time.Parse("2006-01-02", "2026-05-01")
	assert.NoError(t, err)
	assert.Equal(t, want, importBirthday(cfg, " 2026-05-01 "))
}

func TestValidateBirthday(t *testing.T) {
	assert.NoError(t, validateBirthday(""))
	assert.NoError(t, validateBirthday("2024-02-29"))
	assert.Error(t, validateBirthday("not-a-date"))
	assert.Error(t, validateBirthday("2999-01-01"))
}
