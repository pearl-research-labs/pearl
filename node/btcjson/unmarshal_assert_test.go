package btcjson_test

import (
	"encoding/json"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// Malformed scriptPubKey objects must produce an error, not a panic: the
// address field used to be extracted with an unchecked type assertion, and
// this unmarshaller runs while the server parses an importmulti request,
// before any handler dispatch.
func TestScriptPubKeyUnmarshalMalformed(t *testing.T) {
	malformed := []string{
		`{"address":123}`,
		`{"address":null}`,
		`{"address":["x"]}`,
		`{}`,
	}
	for _, payload := range malformed {
		var spk btcjson.ScriptPubKey
		if err := json.Unmarshal([]byte(payload), &spk); err == nil {
			t.Errorf("payload %s: expected error, got value %#v",
				payload, spk.Value)
		}
	}

	// The full command path must also error, not panic.
	_, err := btcjson.UnmarshalCmd(&btcjson.Request{
		Method: "importmulti",
		Params: []json.RawMessage{
			json.RawMessage(`[{"scriptPubKey":{"address":123},"timestamp":0}]`),
		},
	})
	if err == nil {
		t.Error("importmulti with numeric address: expected error, got nil")
	}
}

// Well-formed scriptPubKey values keep parsing exactly as before.
func TestScriptPubKeyUnmarshalValid(t *testing.T) {
	var spk btcjson.ScriptPubKey
	if err := json.Unmarshal([]byte(`{"address":"prl1pabc"}`), &spk); err != nil {
		t.Fatalf("address object: %v", err)
	}
	addr, ok := spk.Value.(btcjson.ScriptPubKeyAddress)
	if !ok || addr.Address != "prl1pabc" {
		t.Errorf("address object: got %#v", spk.Value)
	}

	var raw btcjson.ScriptPubKey
	if err := json.Unmarshal([]byte(`"76a91400"`), &raw); err != nil {
		t.Fatalf("string form: %v", err)
	}
	if raw.Value != "76a91400" {
		t.Errorf("string form: got %#v", raw.Value)
	}
}

// Malformed scanning objects must produce an error, not a panic: duration
// and progress used to be extracted with unchecked type assertions, and
// this unmarshaller runs in RPC clients parsing a getwalletinfo response.
func TestScanningOrFalseUnmarshalMalformed(t *testing.T) {
	malformed := []string{
		`{"duration":"x","progress":0.5}`,
		`{"duration":3,"progress":"x"}`,
		`{"progress":0.5}`,
		`{"duration":3}`,
		`{}`,
	}
	for _, payload := range malformed {
		var sof btcjson.ScanningOrFalse
		if err := json.Unmarshal([]byte(payload), &sof); err == nil {
			t.Errorf("payload %s: expected error, got value %#v",
				payload, sof.Value)
		}
	}
}

// Well-formed scanning values keep parsing exactly as before.
func TestScanningOrFalseUnmarshalValid(t *testing.T) {
	var b btcjson.ScanningOrFalse
	if err := json.Unmarshal([]byte(`false`), &b); err != nil {
		t.Fatalf("bool form: %v", err)
	}
	if b.Value != false {
		t.Errorf("bool form: got %#v", b.Value)
	}

	var p btcjson.ScanningOrFalse
	if err := json.Unmarshal([]byte(`{"duration":3,"progress":0.5}`), &p); err != nil {
		t.Fatalf("progress form: %v", err)
	}
	prog, ok := p.Value.(btcjson.ScanProgress)
	if !ok || prog.Duration != 3 || prog.Progress != 0.5 {
		t.Errorf("progress form: got %#v", p.Value)
	}
}
