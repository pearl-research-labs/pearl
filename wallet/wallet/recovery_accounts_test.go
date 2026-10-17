package wallet

import (
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
)

// Regression: account 1 must never reuse account 0's derivation path.
// A funded account-1 receiving address was previously invisible after
// restoring a wallet even though the wallet recognized the address as owned.
func TestRecoveryKeyPathsPreserveAccount(t *testing.T) {
	for _, account := range []uint32{0, 1, 2} {
		for _, index := range []uint32{0, 7, 250} {
			ext := externalKeyPath(account, index)
			if ext.Account != account ||
				ext.InternalAccount != account ||
				ext.Branch != waddrmgr.ExternalBranch ||
				ext.Index != index {
				t.Fatalf("external path for account %d index %d: %+v",
					account, index, ext)
			}
			in := internalKeyPath(account, index)
			if in.Account != account ||
				in.InternalAccount != account ||
				in.Branch != waddrmgr.InternalBranch ||
				in.Index != index {
				t.Fatalf("internal path for account %d index %d: %+v",
					account, index, in)
			}
		}
	}
}

// Each account's lookahead must remain independent: a match for one account
// must not advance the recovery horizon in any other account.
func TestAccountRecoveryHorizonsAreIndependent(t *testing.T) {
	const window = uint32(10)
	scope := waddrmgr.KeyScopeBIP0086
	account0 := NewRecoveryManager(window, 10, &chainParams).State().StateForScope(scope)
	account1 := NewRecoveryManager(window, 10, &chainParams).State().StateForScope(scope)

	_, delta0 := account0.ExternalBranch.ExtendHorizon()
	_, delta1 := account1.ExternalBranch.ExtendHorizon()
	if delta0 != window || delta1 != window {
		t.Fatalf("initial windows: account0=%d account1=%d", delta0, delta1)
	}

	account1.ExternalBranch.ReportFound(3)
	_, delta0 = account0.ExternalBranch.ExtendHorizon()
	_, delta1 = account1.ExternalBranch.ExtendHorizon()
	if delta0 != 0 || delta1 != 4 {
		t.Fatalf("account horizons interfered: account0=%d account1=%d", delta0, delta1)
	}
}
