package mempool

// Regression test for charge-before-acceptance in the free-relay
// limiter: the penny-flooding budget must only be charged once a
// transaction has passed EVERY acceptance check and is inserted.
// Run: go test ./node/mempool/ -run TestFreeTxRelayChargeOnAccept -v

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// A below-min-fee transaction is evaluated against the free-relay
// limiter inside checkMempoolAcceptance, but two fallible checks run
// AFTER that evaluation: validateReplacement and
// ValidateTransactionScripts. If the limiter charge is committed during
// the evaluation, a transaction that fails either later check burns the
// penny-flooding budget without ever entering the pool — and any P2P
// peer can repeat that with invalid-signature transactions until the
// node's free-relay budget is exhausted.
func TestFreeTxRelayChargeOnAccept(t *testing.T) {
	harness, spendableOuts, err := newPoolHarness(&chaincfg.MainNetParams)
	if err != nil {
		t.Fatalf("unable to create test pool: %v", err)
	}
	harness.txPool.cfg.Policy.DisableRelayPriority = false
	// FreeTxRelayLimit stays at the harness/node default of 15.0.

	// Zero-fee transaction with a corrupted witness signature: it
	// passes sanity, input, standardness and sequence-lock checks (so
	// it reaches and passes the free-relay fee evaluation), then fails
	// script validation — after the point where the limiter used to
	// commit its charge.
	badTx, err := harness.CreateSignedTx(
		[]spendableOutput{spendableOuts[0]}, 1, btcutil.Amount(0), false,
	)
	if err != nil {
		t.Fatalf("unable to create zero-fee tx: %v", err)
	}
	msgTx := badTx.MsgTx()
	if len(msgTx.TxIn[0].Witness) == 0 ||
		len(msgTx.TxIn[0].Witness[0]) == 0 {

		t.Fatalf("expected a witness signature to corrupt")
	}
	msgTx.TxIn[0].Witness[0][0] ^= 0xff

	_, err = harness.txPool.ProcessTransaction(badTx, true, true, 0)
	if err == nil {
		t.Fatalf("bad-signature zero-fee tx was accepted")
	}
	if harness.txPool.IsTransactionInPool(badTx.Hash()) {
		t.Fatalf("bad-signature tx is in the pool after rejection")
	}
	if harness.txPool.pennyTotal != 0 {
		t.Fatalf("rejected tx consumed the free-relay budget: "+
			"pennyTotal=%v, want 0", harness.txPool.pennyTotal)
	}
	if harness.txPool.lastPennyUnix != 0 {
		t.Fatalf("rejected tx mutated lastPennyUnix: %v, want 0",
			harness.txPool.lastPennyUnix)
	}

	// The full budget must still be available to a valid free
	// transaction spending the same output, and that acceptance must
	// be charged exactly once.
	goodTx, err := harness.CreateSignedTx(
		[]spendableOutput{spendableOuts[0]}, 1, btcutil.Amount(0), false,
	)
	if err != nil {
		t.Fatalf("unable to create zero-fee tx: %v", err)
	}
	accepted, err := harness.txPool.ProcessTransaction(goodTx, true, true, 0)
	if err != nil {
		t.Fatalf("valid zero-fee tx rejected after bad tx: %v", err)
	}
	if len(accepted) != 1 {
		t.Fatalf("expected 1 accepted tx, got %d", len(accepted))
	}
	if harness.txPool.pennyTotal <= 0 {
		t.Fatalf("accepted tx was not charged: pennyTotal=%v",
			harness.txPool.pennyTotal)
	}
}
