package btcjson_test

import (
	"encoding/json"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

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
