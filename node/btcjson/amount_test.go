// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package btcjson_test

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAmountJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		amount btcjson.Amount
		json   string
	}{
		{0, "0.00000000"},
		{1, "0.00000001"},
		{-1, "-0.00000001"},
		{100000000, "1.00000000"},
		{-125000000, "-1.25000000"},
		{6710886400000002, "67108864.00000002"},
		{6710886400000019, "67108864.00000019"},
		{-6710886400000002, "-67108864.00000002"},
		{9007199254740993, "90071992.54740993"},
		{2100000000000000000, "21000000000.00000000"},
		{math.MaxInt64 - 1, "92233720368.54775806"},
		{math.MaxInt64, "92233720368.54775807"},
		{math.MinInt64 + 1, "-92233720368.54775807"},
		{math.MinInt64, "-92233720368.54775808"},
	}
	for _, test := range tests {
		t.Run(test.json, func(t *testing.T) {
			encoded, err := json.Marshal(test.amount)
			require.NoError(t, err)
			assert.Equal(t, test.json, string(encoded))
			var decoded btcjson.Amount
			require.NoError(t, json.Unmarshal(encoded, &decoded))
			assert.Equal(t, test.amount, decoded)
		})
	}
}

func TestAmountUnmarshalJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		json string
		want btcjson.Amount
	}{
		{"0", 0},
		{"-0.0", 0},
		{" \t\r\n1.25\n", 125000000},
		{"1e-8", 1},
		{"-1E-8", -1},
		{"1e+1", 1000000000},
		{"1.000000000", 100000000},
		{"0.0000000100", 1},
		{"100000000000000000000e-28", 1},
		{"0.000000000000000000001e21", 100000000},
		{"6.710886400000002e7", 6710886400000002},
		{"9223372036854775807e-8", math.MaxInt64},
		{"-9223372036854775808e-8", math.MinInt64},
		{"1000000000e-17", 1},
		{"0e" + strings.Repeat("9", 4096), 0},
		{"-0.000e-" + strings.Repeat("9", 4096), 0},
		{"1e" + strings.Repeat("0", 4096), 100000000},
		{"1" + strings.Repeat("0", 4096) + "e-4104", 1},
	}
	for _, test := range tests {
		var got btcjson.Amount
		require.NoError(t, json.Unmarshal([]byte(test.json), &got), "Unmarshal %.80q", test.json)
		assert.Equal(t, test.want, got, "Unmarshal %.80q", test.json)
	}
}

func TestAmountUnmarshalJSONErrors(t *testing.T) {
	t.Parallel()

	for _, number := range []string{
		"", " ", "null", "true", "false", `"1.25"`, `"1"`, "[]", "{}",
		"+1", "01", ".1", "1.", "--1", "NaN", "Infinity", "1e", "1 2", "\u00a01",
		"0.000000001", "-0.000000001", "1e-9", "1.000000001",
		"92233720368.54775808", "-92233720368.54775809", "100000000000",
		"123456789012345678901e-9", "1e9223372036854775807", "1e-9223372036854775808",
		"1e" + strings.Repeat("9", 4096), "1e-" + strings.Repeat("9", 4096),
	} {
		got := btcjson.Amount(123)
		assert.Error(t, got.UnmarshalJSON([]byte(number)), "Unmarshal %.80q", number)
		assert.Equal(t, btcjson.Amount(123), got, "Unmarshal %.80q changed receiver on error", number)
	}
}

func TestAmountResultRoundTrip(t *testing.T) {
	t.Parallel()

	const amount btcjson.Amount = 6710886400000002
	for _, result := range []interface{}{
		btcjson.Vout{Value: amount},
		btcjson.PrevOut{Value: amount},
		btcjson.GetTxOutResult{Value: amount},
	} {
		encoded, err := json.Marshal(result)
		require.NoError(t, err)
		assert.Contains(t, string(encoded), `"value":67108864.00000002`, "%T has inexact JSON", result)
		decoded := reflect.New(reflect.TypeOf(result))
		require.NoError(t, json.Unmarshal(encoded, decoded.Interface()))
		assert.Equal(t, result, decoded.Elem().Interface(), "%T changed after JSON round trip", result)
	}
}

func TestAmountNotificationRoundTrip(t *testing.T) {
	t.Parallel()

	const amount btcjson.Amount = 6710886400000002
	for _, arg := range []interface{}{amount, "67108864.00000002", "6.710886400000002e7"} {
		command, err := btcjson.NewCmd("txaccepted", "123", arg)
		require.NoError(t, err)
		assert.Equal(t, btcjson.NewTxAcceptedNtfn("123", amount), command, "NewCmd(%v)", arg)
		encoded, err := btcjson.MarshalCmd(btcjson.RpcVersion1, nil, command)
		require.NoError(t, err)
		var request btcjson.Request
		require.NoError(t, json.Unmarshal(encoded, &request))
		require.Len(t, request.Params, 2)
		assert.Equal(t, "67108864.00000002", string(request.Params[1]), "request rounded amount")
		decoded, err := btcjson.UnmarshalCmd(&request)
		require.NoError(t, err)
		assert.Equal(t, command, decoded, "notification round trip")
	}
	for _, arg := range []interface{}{1.25, "1e-9", `"1.25"`} {
		_, err := btcjson.NewCmd("txaccepted", "123", arg)
		assert.Error(t, err, "NewCmd unexpectedly accepted %v (%T)", arg, arg)
	}
}

func TestRequestPreservesParameterJSON(t *testing.T) {
	t.Parallel()

	input := `{
		"jsonrpc":"1.0",
		"method":"test",
		"params":[67108864.00000002,1e-8,{"amount":9007199254740993},null,true,"text"],
		"id":1
	}`
	var request btcjson.Request
	require.NoError(t, json.Unmarshal([]byte(input), &request))
	want := []string{"67108864.00000002", "1e-8", `{"amount":9007199254740993}`, "null", "true", `"text"`}
	require.Len(t, request.Params, len(want))
	for i, param := range request.Params {
		assert.Equal(t, want[i], string(param), "param %d", i)
	}
	for _, input := range []string{
		`{"method":"test"}`,
		`{"method":"test","params":null}`,
		`{"method":"test","params":[]}`,
	} {
		require.NoError(t, json.Unmarshal([]byte(input), &request))
		assert.NotNil(t, request.Params, "missing or null params should decode to an empty array: %s", input)
		assert.Empty(t, request.Params, "missing or null params should decode to an empty array: %s", input)
	}
	for _, input := range []string{"null", `{"params":{}}`, `{"params":1}`, `{"params":"x"}`} {
		assert.Error(t, json.Unmarshal([]byte(input), &request),
			"invalid request unexpectedly accepted: %s", input)
	}
}

func TestAmountHelp(t *testing.T) {
	t.Parallel()

	examples, complex := btcjson.TstReflectTypeToJSONExample(func(key string) string {
		return key
	}, reflect.TypeOf(btcjson.Amount(0)), 0, "amount")
	assert.False(t, complex)
	assert.Equal(t, []string{"n.nnn"}, examples)
	field := reflect.TypeOf(struct{ Vout btcjson.Vout }{}).Field(0)
	assert.Contains(t, btcjson.TstFieldUsage(field, nil), `"value":n.nnn`,
		"amount usage should show a decimal number")
}

func FuzzAmountRoundTrip(f *testing.F) {
	for _, amount := range []int64{0, 1, -1, 6710886400000002, 9007199254740993, math.MinInt64, math.MaxInt64} {
		f.Add(amount)
	}
	f.Fuzz(func(t *testing.T, grains int64) {
		encoded, err := json.Marshal(btcjson.Amount(grains))
		require.NoError(t, err)
		var decoded btcjson.Amount
		require.NoError(t, json.Unmarshal(encoded, &decoded))
		assert.Equal(t, grains, int64(decoded), "round trip through %s", encoded)
	})
}
