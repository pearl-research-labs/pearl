// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package bloom_test

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/btcutil/bloom"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

func TestMerkleBlock3(t *testing.T) {
	blockStr := "" +
		// CertificateV1 (212 bytes): Version(4) + BlockHash(32) + PublicData(164) + ProofLen(4) + ProofData(8)
		"01000000" + // Version (1 = ZK)
		"0000000000000000000000000000000000000000000000000000000000000000" + // BlockHash (32 zeros)
		// PublicData (164 bytes = 5×32 + 4 bytes,, all zeros)
		"0000000000000000000000000000000000000000000000000000000000000000" +
		"0000000000000000000000000000000000000000000000000000000000000000" +
		"0000000000000000000000000000000000000000000000000000000000000000" +
		"0000000000000000000000000000000000000000000000000000000000000000" +
		"0000000000000000000000000000000000000000000000000000000000000000" +
		"00000000" + // last 4 bytes
		"08000000" + // ProofLen (8 bytes)
		"deadbeefcafebabe" + // ProofData
		// Block header (108 bytes): Version(4) + PrevBlock(32) + MerkleRoot(32) + Timestamp(4) + Bits(4) + ProofCommitment(32)
		"0100000079cda856b143d9db2c1caff01d1aecc8630d30625d10e8b" +
		"4b8b0000000000000b50cc069d6a3e33e3ff84a5c41d9d3febe7c770fdc" +
		"c96b2c3ff60abe184f196367291b4d4c86041b" +
		"0000000000000000000000000000000000000000000000000000000000000000" + // ProofCommitment (32 zeros)
		// Transaction count + transactions:
		"01010000000100000000000000000000000000000000000000000000000" +
		"00000000000000000ffffffff08044c86041b020a02ffffffff0100f205" +
		"2a01000000434104ecd3229b0571c3be876feaac0442a9f13c5a5727429" +
		"27af1dc623353ecf8c202225f64868137a18cdd85cbbb4c74fbccfd4f49" +
		"639cf1bdc94a5672bb15ad5d4cac00000000"
	blockBytes, err := hex.DecodeString(blockStr)
	if err != nil {
		t.Errorf("TestMerkleBlock3 DecodeString failed: %v", err)
		return
	}
	blk, err := btcutil.NewBlockFromBytes(blockBytes)
	if err != nil {
		t.Errorf("TestMerkleBlock3 NewBlockFromBytes failed: %v", err)
		return
	}

	f := bloom.NewFilter(10, 0, 0.000001, wire.BloomUpdateAll)

	inputStr := "63194f18be0af63f2c6bc9dc0f777cbefed3d9415c4af83f3ee3a3d669c00cb5"
	hash, err := chainhash.NewHashFromStr(inputStr)
	if err != nil {
		t.Errorf("TestMerkleBlock3 NewHashFromStr failed: %v", err)
		return
	}

	f.AddHash(hash)

	mBlock, _ := bloom.NewMerkleBlock(blk, f)

	wantStr := "0100000079cda856b143d9db2c1caff01d1aecc8630d30625d10e8b4" +
		"b8b0000000000000b50cc069d6a3e33e3ff84a5c41d9d3febe7c770fdcc" +
		"96b2c3ff60abe184f196367291b4d4c86041b" +
		"0000000000000000000000000000000000000000000000000000000000000000" + // ProofCommitment (32 zeros)
		// MerkleBlock data:
		"0100000001b50cc069d6a3e33e3ff84a5c41d9d3febe7c770fdcc96b2c3" +
		"ff60abe184f19630101"
	want, err := hex.DecodeString(wantStr)
	if err != nil {
		t.Errorf("TestMerkleBlock3 DecodeString failed: %v", err)
		return
	}

	got := bytes.NewBuffer(nil)
	err = mBlock.PrlEncode(got, wire.ProtocolVersion, wire.LatestEncoding)
	if err != nil {
		t.Errorf("TestMerkleBlock3 PrlEncode failed: %v", err)
		return
	}

	if !bytes.Equal(want, got.Bytes()) {
		t.Errorf("TestMerkleBlock3 failed merkle block comparison: "+
			"got %v want %v", got.Bytes(), want)
		return
	}
}
