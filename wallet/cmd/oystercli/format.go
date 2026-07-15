// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
)

// fmtPRL renders an amount as "1.234 PRL", trimming insignificant zeros but
// always keeping at least two decimal places for scanability.
func fmtPRL(a btcutil.Amount) string {
	return fmtPRLFloat(a.ToPRL())
}

// fmtPRLFloat renders a PRL value received as a float from the RPC layer.
func fmtPRLFloat(v float64) string {
	s := strconv.FormatFloat(v, 'f', 8, 64)
	s = strings.TrimRight(s, "0")
	// Keep a minimum of two decimals so columns of amounts line up sanely.
	if i := strings.IndexByte(s, '.'); i == len(s)-1 {
		s += "00"
	} else if i == len(s)-2 {
		s += "0"
	}
	return s + " PRL"
}

// parsePRL converts user input to an amount, rejecting non-positive values.
func parsePRL(s string) (btcutil.Amount, error) {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, fmt.Errorf("not a valid amount")
	}
	amt, err := btcutil.NewAmount(v)
	if err != nil {
		return 0, err
	}
	if amt <= 0 {
		return 0, fmt.Errorf("amount must be positive")
	}
	return amt, nil
}

// shortID renders long identifiers (txids, addresses) with a middle ellipsis
// so rows stay compact while remaining recognizable.
func shortID(s string, max int) string {
	if len(s) <= max || max < 8 {
		return s
	}
	half := (max - 1) / 2
	return s[:half] + "…" + s[len(s)-half:]
}

// fmtUnixTime renders a unix timestamp compactly, in local time.
func fmtUnixTime(unix int64) string {
	if unix == 0 {
		return "-"
	}
	return time.Unix(unix, 0).Format("2006-01-02 15:04")
}

// fmtConfs renders a confirmation count, flagging unconfirmed transactions.
func fmtConfs(confs int64) string {
	if confs <= 0 {
		return "unconfirmed"
	}
	return fmt.Sprintf("%d conf", confs)
}

// syncPercent formats sync progress as a percentage of the best known peer
// height, falling back to raw heights when peers haven't reported yet. The
// value is truncated (not rounded) to one decimal so it never displays
// "100.0%" while still behind — e.g. 86840/86864 shows 99.9%, not 100.0%.
func syncPercent(si *syncInfo) string {
	if si.peerHeight > 0 {
		pct := float64(si.height) / float64(si.peerHeight) * 100
		pct = math.Floor(pct*10) / 10
		if pct > 100 {
			pct = 100
		}
		return fmt.Sprintf("%.1f%% (%d/%d)", pct, si.height, si.peerHeight)
	}
	return fmt.Sprintf("height %d", si.height)
}

// sortedKeys returns the map keys in sorted order with the default account
// listed first, since it is almost always the one the user wants.
func sortedKeys(m map[string]btcutil.Amount) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i] == "default" {
			return true
		}
		if keys[j] == "default" {
			return false
		}
		return keys[i] < keys[j]
	})
	return keys
}
