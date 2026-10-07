package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// TestDeploymentStateOutOfRange checks that deploymentState honours its
// contract for IDs that do not exist: it must return a DeploymentError for
// every out-of-range ID, including the exact boundary
// ID == len(Deployments) (== chaincfg.DefinedDeployments), instead of
// indexing the fixed-size Deployments array and panicking.
func TestDeploymentStateOutOfRange(t *testing.T) {
	params := chaincfg.RegressionNetParams
	b := &BlockChain{
		chainParams:      &params,
		deploymentCaches: newThresholdCaches(chaincfg.DefinedDeployments),
	}

	// A valid ID resolves without error.
	if _, err := b.deploymentState(nil, chaincfg.DeploymentTestDummy); err != nil {
		t.Fatalf("valid ID %d: unexpected error %v",
			chaincfg.DeploymentTestDummy, err)
	}

	outOfRange := []uint32{
		uint32(len(params.Deployments)),     // exact boundary
		uint32(len(params.Deployments)) + 1, // already handled before the fix
		1 << 20,                             // far out of range
	}
	for _, id := range outOfRange {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("deploymentState(id=%d) panicked: %v "+
						"(expected DeploymentError)", id, r)
				}
			}()
			_, err := b.deploymentState(nil, id)
			if err == nil {
				t.Fatalf("deploymentState(id=%d): expected "+
					"DeploymentError, got nil", id)
			}
			if _, ok := err.(DeploymentError); !ok {
				t.Fatalf("deploymentState(id=%d): expected "+
					"DeploymentError, got %T (%v)", id, err, err)
			}
		}()
	}
}
