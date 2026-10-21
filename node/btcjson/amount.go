// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package btcjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/pearl-research-labs/pearl/node/btcutil"
)

// Amount is an exact RPC amount in grains, where one PRL is 100,000,000 grains.
// Its JSON representation is a number in PRL, with eight decimal places when
// encoded. For example, Amount(1) represents one grain and encodes as 0.00000001.
// Conversion to or from btcutil.Amount preserves the integer grain value.
// Amount supports the full int64 range; consensus limits are enforced elsewhere.
type Amount int64

// MarshalJSON encodes the amount as a PRL number without using floating point.
func (a Amount) MarshalJSON() ([]byte, error) {
	whole := int64(a) / btcutil.GrainPerPearl
	fraction := int64(a) % btcutil.GrainPerPearl
	sign := ""
	if a < 0 {
		sign = "-"
		whole = -whole
		fraction = -fraction
	}
	return []byte(fmt.Sprintf("%s%d.%08d", sign, whole, fraction)), nil
}

// UnmarshalJSON decodes a PRL number, including exponent notation, into grains.
// It rejects non-numbers, fractional grains, and values outside the int64 range.
// On error the receiver is unchanged. Parsing takes linear time in the input
// size; exponent notation never causes allocation proportional to its value.
func (a *Amount) UnmarshalJSON(data []byte) error {
	data = bytes.Trim(data, " \t\r\n")
	if !json.Valid(data) || (data[0] != '-' && (data[0] < '0' || data[0] > '9')) {
		return fmt.Errorf("amount must be a JSON number")
	}

	number := string(data)
	negative := number[0] == '-'
	if negative {
		number = number[1:]
	}
	mantissa, exponentText := number, "0"
	if i := strings.IndexAny(number, "eE"); i >= 0 {
		mantissa, exponentText = number[:i], number[i+1:]
	}
	fractionDigits := 0
	if i := strings.IndexByte(mantissa, '.'); i >= 0 {
		fractionDigits = len(mantissa) - i - 1
	}
	digits := strings.TrimLeft(strings.ReplaceAll(mantissa, ".", ""), "0")
	if digits == "" {
		*a = 0
		return nil
	}
	significant := strings.TrimRight(digits, "0")
	trailingZeros := len(digits) - len(significant)
	if len(significant) > 19 {
		return fmt.Errorf("amount exceeds int64 grain precision")
	}
	exponent, err := strconv.ParseInt(exponentText, 10, 64)
	if err != nil {
		return fmt.Errorf("amount exponent is out of range")
	}

	// Removing trailing zeros lets us decide whether the result is integral
	// and fits in at most 19 digits before performing bounded arithmetic.
	minExponent := int64(fractionDigits) - int64(trailingZeros) - 8
	if exponent < minExponent {
		return fmt.Errorf("amount contains a fractional grain")
	}
	if exponent > minExponent+19-int64(len(significant)) {
		return fmt.Errorf("amount is outside the int64 grain range")
	}
	value, err := strconv.ParseUint(significant, 10, 64)
	if err != nil {
		return err
	}
	for zeros := exponent - minExponent; zeros > 0; zeros-- {
		value *= 10
	}
	// MinInt64's magnitude cannot be represented as a positive int64.
	if negative && value == uint64(math.MaxInt64)+1 {
		*a = Amount(math.MinInt64)
		return nil
	}
	if value > uint64(math.MaxInt64) {
		return fmt.Errorf("amount is outside the int64 grain range")
	}
	amount := Amount(value)
	if negative {
		amount = -amount
	}
	*a = amount
	return nil
}
