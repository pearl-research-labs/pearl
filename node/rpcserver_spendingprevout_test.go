package main

import (
	"encoding/json"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// An explicit JSON null inside the outputs array is not rejected by
// btcjson: unmarshalling null into a []*T element yields a nil element,
// the same mechanism as the getblock null-verbosity case (#401) but one
// level down. Before the fix the handler dereferenced the nil element
// (o.Txid) and panicked; it must return an invalid-parameter error.
func TestHandleGetTxSpendingPrevOutNullElement(t *testing.T) {
	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"gettxspendingprevout",` +
		`"params":[[null]]}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.GetTxSpendingPrevOutCmd)
	if !ok {
		t.Fatalf("expected *btcjson.GetTxSpendingPrevOutCmd, got %T", cmd)
	}

	// Producer pin: the wire really does produce a nil element.
	if len(got.Outputs) != 1 || got.Outputs[0] != nil {
		t.Fatalf("producer pin broken: Outputs=%v, want one nil element",
			got.Outputs)
	}

	// The panic happens while converting outpoints, before any mempool
	// access, so a bare server is enough.
	s := &rpcServer{}
	results, err := handleGetTxSpendingPrevOut(s, got, make(chan struct{}))
	if err == nil {
		t.Fatalf("null element: got results %v, want error", results)
	}
	rpcErr, ok := err.(*btcjson.RPCError)
	if !ok || rpcErr.Code != btcjson.ErrRPCInvalidParameter {
		t.Fatalf("null element: got error %v, want RPCError "+
			"ErrRPCInvalidParameter", err)
	}
	if results != nil {
		t.Fatalf("null element: got results %v, want nil", results)
	}
}
