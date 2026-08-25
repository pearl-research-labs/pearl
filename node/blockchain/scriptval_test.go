// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/txscript"
)

// TestCheckBlockScripts ensures that validating the all of the scripts in a
// known-good block doesn't return an error.
func TestCheckBlockScripts(t *testing.T) {
	t.Skip("Test files rely on Bitcoin block format, which is no longer supported") // TODO Or: re-enable with Pearl-format test fixtures
	testBlockNum := 277647
	blockDataFile := fmt.Sprintf("%d.dat.bz2", testBlockNum)
	blocks, err := loadBlocks(blockDataFile)
	if err != nil {
		t.Errorf("Error loading file: %v\n", err)
		return
	}
	if len(blocks) > 1 {
		t.Errorf("The test block file must only have one block in it")
		return
	}
	if len(blocks) == 0 {
		t.Errorf("The test block file may not be empty")
		return
	}

	storeDataFile := fmt.Sprintf("%d.utxostore.bz2", testBlockNum)
	view, err := loadUtxoView(storeDataFile)
	if err != nil {
		t.Errorf("Error loading txstore: %v\n", err)
		return
	}

	scriptFlags := txscript.StandardVerifyFlags
	hashCache := txscript.NewHashCache(0)
	err = checkBlockScripts(blocks[0], view, scriptFlags, nil, hashCache)
	if err != nil {
		t.Errorf("Transaction script validation failed: %v\n", err)
		return
	}
}
