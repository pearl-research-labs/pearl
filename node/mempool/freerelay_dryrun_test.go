package mempool

// Regression test for the dry-run side effect in CheckMempoolAcceptance:
// Run: go test ./node/mempool/ -run TestFreeTxRelayDryRun -race -v

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// CheckMempoolAcceptance (the testmempoolaccept RPC) is a dry run: it
// holds only the pool read lock and never inserts the transaction. It
// must therefore EVALUATE the free-relay rate limiter without CHARGING
// it. Charging on this path both consumes the penny-flooding budget for
// transactions that were never accepted and writes pennyTotal /
// lastPennyUnix under a read lock, racing concurrent dry runs.
func TestFreeTxRelayDryRunDoesNotConsumeBudget(t *testing.T) {
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

	// Pre-warm the transaction's lazily cached hashes: btcutil.Tx is
	// not safe for concurrent first use, and each RPC dry run decodes
	// its own transaction, so sharing an unwarmed Tx here would race
	// in the test fixture rather than in the pool.
	_ = zeroFeeTx.Hash()
	_ = zeroFeeTx.WitnessHash()

	// Concurrent dry runs: under -race this flags any write to the
	// limiter state made while holding only the read lock.
	const dryRuns = 8
	var wg sync.WaitGroup
	errs := make([]error, dryRuns)
	for i := 0; i < dryRuns; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			result, err := harness.txPool.CheckMempoolAcceptance(zeroFeeTx)
			if err == nil && result == nil {
				err = errNilResult
			}
			errs[i] = err
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("dry run %d: zero-fee tx within free-relay "+
				"budget was rejected: %v", i, err)
		}
	}

	if harness.txPool.IsTransactionInPool(zeroFeeTx.Hash()) {
		t.Fatalf("dry run inserted the transaction into the pool")
	}
	if harness.txPool.pennyTotal != 0 {
		t.Fatalf("dry runs consumed the free-relay budget: "+
			"pennyTotal=%v, want 0", harness.txPool.pennyTotal)
	}
	if harness.txPool.lastPennyUnix != 0 {
		t.Fatalf("dry runs mutated lastPennyUnix: %v, want 0",
			harness.txPool.lastPennyUnix)
	}

	// A real acceptance afterwards must still find the full budget:
	// the same transaction is admitted and charged exactly once.
	accepted, err := harness.txPool.ProcessTransaction(zeroFeeTx, true, true, 0)
	if err != nil {
		t.Fatalf("real acceptance after dry runs was rejected: %v", err)
	}
	if len(accepted) != 1 {
		t.Fatalf("expected 1 accepted tx, got %d", len(accepted))
	}
	if harness.txPool.pennyTotal <= 0 {
		t.Fatalf("real acceptance did not charge the budget: "+
			"pennyTotal=%v", harness.txPool.pennyTotal)
	}

	// With the budget configured to zero, a dry run must still be
	// rejected by the rate limiter (the limiter is evaluated), while
	// leaving the limiter state untouched.
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

	_, err = harness2.txPool.CheckMempoolAcceptance(zeroFeeTx2)
	if err == nil {
		t.Fatalf("dry run accepted zero-fee tx despite zero " +
			"free-relay budget")
	}
	if !strings.Contains(err.Error(), "rate limiter") {
		t.Fatalf("expected dry-run rejection by the rate limiter, "+
			"got: %v", err)
	}
	if harness2.txPool.pennyTotal != 0 || harness2.txPool.lastPennyUnix != 0 {
		t.Fatalf("rejected dry run mutated limiter state: "+
			"pennyTotal=%v lastPennyUnix=%v",
			harness2.txPool.pennyTotal, harness2.txPool.lastPennyUnix)
	}
}

var errNilResult = errors.New(
	"CheckMempoolAcceptance returned nil result and nil error")
