package mempool

// Regression test for the validateRelayFeeMet dead rate-limiter bug:
// Run: go test ./node/mempool/ -run TestFreeTxRelayRateLimiter -v

import (
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// A zero-fee transaction relayed to a node whose policy enables free
// relay (DisableRelayPriority=false, the node's default
// FreeTxRelayLimit=15.0 KB/min, per node/config.go `--limitfreerelay`:
// "Limit relay of transactions with no transaction fee to the given
// amount in thousands of bytes per minute") must be ADMITTED while the
// penny-flooding budget is untouched, and REJECTED BY THE RATE LIMITER
// once the budget is configured to zero. On the unpatched code both
// cases fail: the tx is blanket-rejected up front with "under the
// required amount" and the rate limiter never runs.
func TestFreeTxRelayRateLimiter(t *testing.T) {
	// Case 1: free tx, relay enabled, fresh budget -> must be accepted.
	harness, spendableOuts, err := newPoolHarness(&chaincfg.MainNetParams)
	if err != nil {
		t.Fatalf("unable to create test pool: %v", err)
	}
	harness.txPool.cfg.Policy.DisableRelayPriority = false
	// FreeTxRelayLimit stays at the harness/node default of 15.0.

	zeroFeeTx, err := harness.CreateSignedTx(
		[]spendableOutput{spendableOuts[0]}, 1, btcutil.Amount(0), false,
	)
	if err != nil {
		t.Fatalf("unable to create zero-fee tx: %v", err)
	}

	accepted, err := harness.txPool.ProcessTransaction(zeroFeeTx, true, true, 0)
	if err != nil {
		t.Fatalf("CASE 1 FAIL: zero-fee tx within free-relay budget "+
			"was rejected: %v", err)
	}
	if len(accepted) != 1 {
		t.Fatalf("CASE 1 FAIL: expected 1 accepted tx, got %d", len(accepted))
	}
	if !harness.txPool.IsTransactionInPool(zeroFeeTx.Hash()) {
		t.Fatalf("CASE 1 FAIL: zero-fee tx not in pool after acceptance")
	}
	if harness.txPool.pennyTotal <= 0 {
		t.Fatalf("CASE 1 FAIL: pennyTotal was not charged (rate "+
			"limiter never ran): pennyTotal=%v",
			harness.txPool.pennyTotal)
	}

	// Case 2: free tx, relay enabled, budget configured to zero ->
	// must be rejected specifically by the rate limiter.
	harness2, spendableOuts2, err := newPoolHarness(&chaincfg.MainNetParams)
	if err != nil {
		t.Fatalf("unable to create test pool: %v", err)
	}
	harness2.txPool.cfg.Policy.DisableRelayPriority = false
	harness2.txPool.cfg.Policy.FreeTxRelayLimit = 0

	zeroFeeTx2, err := harness2.CreateSignedTx(
		[]spendableOutput{spendableOuts2[0]}, 1, btcutil.Amount(0), false,
	)
	if err != nil {
		t.Fatalf("unable to create zero-fee tx: %v", err)
	}

	_, err = harness2.txPool.ProcessTransaction(zeroFeeTx2, true, true, 0)
	if err == nil {
		t.Fatalf("CASE 2 FAIL: zero-fee tx accepted despite zero " +
			"free-relay budget")
	}
	if !strings.Contains(err.Error(), "rate limiter") {
		t.Fatalf("CASE 2 FAIL: expected rejection by the rate "+
			"limiter, got: %v", err)
	}
}
