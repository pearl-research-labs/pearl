// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

// TestSequenceLocksActive tests the SequenceLockActive function to ensure it
// works as expected in all possible combinations/scenarios.
func TestSequenceLocksActive(t *testing.T) {
	seqLock := func(h int32, s int64) *SequenceLock {
		return &SequenceLock{
			Seconds:     s,
			BlockHeight: h,
		}
	}

	tests := []struct {
		seqLock     *SequenceLock
		blockHeight int32
		mtp         time.Time

		want bool
	}{
		// Block based sequence lock with equal block height.
		{seqLock: seqLock(1000, -1), blockHeight: 1001, mtp: time.Unix(9, 0), want: true},

		// Time based sequence lock with mtp past the absolute time.
		{seqLock: seqLock(-1, 30), blockHeight: 2, mtp: time.Unix(31, 0), want: true},

		// Block based sequence lock with current height below seq lock block height.
		{seqLock: seqLock(1000, -1), blockHeight: 90, mtp: time.Unix(9, 0), want: false},

		// Time based sequence lock with current time before lock time.
		{seqLock: seqLock(-1, 30), blockHeight: 2, mtp: time.Unix(29, 0), want: false},

		// Block based sequence lock at the same height, so shouldn't yet be active.
		{seqLock: seqLock(1000, -1), blockHeight: 1000, mtp: time.Unix(9, 0), want: false},

		// Time based sequence lock with current time equal to lock time, so shouldn't yet be active.
		{seqLock: seqLock(-1, 30), blockHeight: 2, mtp: time.Unix(30, 0), want: false},
	}

	t.Logf("Running %d sequence locks tests", len(tests))
	for i, test := range tests {
		got := SequenceLockActive(test.seqLock,
			test.blockHeight, test.mtp)
		if got != test.want {
			t.Fatalf("SequenceLockActive #%d got %v want %v", i,
				got, test.want)
		}
	}
}

// TestCheckConnectBlockTemplate tests the CheckConnectBlockTemplate function to
// ensure it fails.
func TestCheckConnectBlockTemplate(t *testing.T) {
	t.Skip("Test files rely on Bitcoin block format, which is no longer supported") // TODO Or: re-enable with Pearl-format test fixtures
	// Create a new database and chain instance to run tests against.
	chain, teardownFunc, err := chainSetup("checkconnectblocktemplate",
		&chaincfg.MainNetParams)
	if err != nil {
		t.Errorf("Failed to setup chain instance: %v", err)
		return
	}
	defer teardownFunc()

	// Since we're not dealing with the real block chain, set the coinbase
	// maturity to 1.
	chain.TstSetCoinbaseMaturity(1)

	// Load up blocks such that there is a side chain.
	// (genesis block) -> 1 -> 2 -> 3 -> 4
	//                          \-> 3a
	testFiles := []string{
		"blk_0_to_4.dat.bz2",
		"blk_3A.dat.bz2",
	}

	var blocks []*btcutil.Block
	for _, file := range testFiles {
		blockTmp, err := loadBlocks(file)
		if err != nil {
			t.Fatalf("Error loading file: %v\n", err)
		}
		blocks = append(blocks, blockTmp...)
	}

	for i := 1; i <= 3; i++ {
		isMainChain, _, err := chain.ProcessBlock(blocks[i], BFNone)
		if err != nil {
			t.Fatalf("CheckConnectBlockTemplate: Received unexpected error "+
				"processing block %d: %v", i, err)
		}
		if !isMainChain {
			t.Fatalf("CheckConnectBlockTemplate: Expected block %d to connect "+
				"to main chain", i)
		}
	}

	// Block 3 should fail to connect since it's already inserted.
	err = chain.CheckConnectBlockTemplate(blocks[3])
	if err == nil {
		t.Fatal("CheckConnectBlockTemplate: Did not received expected error " +
			"on block 3")
	}

	// Block 4 should connect successfully to tip of chain.
	err = chain.CheckConnectBlockTemplate(blocks[4])
	if err != nil {
		t.Fatalf("CheckConnectBlockTemplate: Received unexpected error on "+
			"block 4: %v", err)
	}

	// Block 3a should fail to connect since does not build on chain tip.
	err = chain.CheckConnectBlockTemplate(blocks[5])
	if err == nil {
		t.Fatal("CheckConnectBlockTemplate: Did not received expected error " +
			"on block 3a")
	}

	// Block 4 should connect with a valid template (ZK PoW is not checked in
	// CheckConnectBlockTemplate since it only validates the template structure).
	err = chain.CheckConnectBlockTemplate(blocks[4])
	if err != nil {
		t.Fatalf("CheckConnectBlockTemplate: Received unexpected error on "+
			"block 4: %v", err)
	}

	// Invalid block building on chain tip should fail to connect.
	invalidBlock := *blocks[4].MsgBlock()
	invalidBlock.BlockHeader().Bits--
	err = chain.CheckConnectBlockTemplate(btcutil.NewBlock(&invalidBlock))
	if err == nil {
		t.Fatal("CheckConnectBlockTemplate: Did not received expected error " +
			"on block 4 with invalid difficulty bits")
	}
}

// TestCheckBlockSanity tests the CheckBlockSanity function to ensure it works
// as expected.
func TestCheckBlockSanity(t *testing.T) {
	// Create a block with a valid V1 certificate.
	// Use RegTest chainParams so PoW is actually verified (SimNet auto-skips PoW).
	chainParams := &chaincfg.RegressionNetParams
	header := wire.BlockHeader{
		Version:    1,
		PrevBlock:  chainhash.Hash{},
		MerkleRoot: Block100000.BlockHeader().MerkleRoot,
		Timestamp:  time.Unix(time.Now().Unix(), 0), // Truncate to second precision
		Bits:       chainParams.PowLimitBits,
	}

	// Mine a valid certificate (only 1 block, so longer ZK proof time is acceptable)
	cert, err := zkpow.Mine(&header)
	if err != nil {
		t.Fatalf("Mine failed: %v", err)
	}

	block := btcutil.NewBlock(&wire.MsgBlock{
		MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		},
		Transactions: Block100000.Transactions,
	})

	timeSource := NewMedianTime()
	err = CheckBlockSanity(block, chainParams, timeSource)
	if err != nil {
		t.Errorf("CheckBlockSanity: %v", err)
	}

	// Ensure a block that has a timestamp with a precision higher than one
	// second fails.
	timestamp := block.MsgBlock().BlockHeader().Timestamp
	block.MsgBlock().BlockHeader().Timestamp = timestamp.Add(time.Nanosecond)
	err = CheckBlockSanity(block, chainParams, timeSource)
	if err == nil {
		t.Errorf("CheckBlockSanity: error is nil when it shouldn't be")
	}
}

// TestCheckSerializedHeight tests the CheckSerializedHeight function with
// various serialized heights and also does negative tests to ensure errors
// and handled properly.
func TestCheckSerializedHeight(t *testing.T) {
	// Create an empty coinbase template to be used in the tests below.
	coinbaseOutpoint := wire.NewOutPoint(&chainhash.Hash{}, math.MaxUint32)
	coinbaseTx := wire.NewMsgTx(1)
	coinbaseTx.AddTxIn(wire.NewTxIn(coinbaseOutpoint, nil, nil))

	// Expected rule errors.
	missingHeightError := RuleError{
		ErrorCode: ErrMissingCoinbaseHeight,
	}
	badHeightError := RuleError{
		ErrorCode: ErrBadCoinbaseHeight,
	}

	tests := []struct {
		sigScript  []byte // Serialized data
		wantHeight int32  // Expected height
		err        error  // Expected error type
	}{
		// No serialized height length.
		{[]byte{}, 0, missingHeightError},
		// Serialized height length with no height bytes.
		{[]byte{0x02}, 0, missingHeightError},
		// Serialized height length with too few height bytes.
		{[]byte{0x02, 0x4a}, 0, missingHeightError},
		// Serialized height that needs 2 bytes to encode.
		{[]byte{0x02, 0x4a, 0x52}, 21066, nil},
		// Serialized height that needs 2 bytes to encode, but backwards
		// endianness.
		{[]byte{0x02, 0x4a, 0x52}, 19026, badHeightError},
		// Serialized height that needs 3 bytes to encode.
		{[]byte{0x03, 0x40, 0x0d, 0x03}, 200000, nil},
		// Serialized height that needs 3 bytes to encode, but backwards
		// endianness.
		{[]byte{0x03, 0x40, 0x0d, 0x03}, 1074594560, badHeightError},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		msgTx := coinbaseTx.Copy()
		msgTx.TxIn[0].SignatureScript = test.sigScript
		tx := btcutil.NewTx(msgTx)

		err := CheckSerializedHeight(tx, test.wantHeight)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("CheckSerializedHeight #%d wrong error type "+
				"got: %v <%T>, want: %T", i, err, err, test.err)
			continue
		}

		if rerr, ok := err.(RuleError); ok {
			trerr := test.err.(RuleError)
			if rerr.ErrorCode != trerr.ErrorCode {
				t.Errorf("CheckSerializedHeight #%d wrong "+
					"error code got: %v, want: %v", i,
					rerr.ErrorCode, trerr.ErrorCode)
				continue
			}
		}
	}
}

// p2trScript returns a P2TR pkScript: OP_1 OP_DATA_32 <32-byte key>.
func p2trScript(key byte) []byte {
	script := make([]byte, 34)
	script[0] = txscript.OP_1
	script[1] = txscript.OP_DATA_32
	script[2] = key
	return script
}

// p2mrScript returns a P2MR pkScript: OP_2 OP_DATA_32 <32-byte merkle root>.
func p2mrScript(root byte) []byte {
	script := make([]byte, 34)
	script[0] = txscript.OP_2
	script[1] = txscript.OP_DATA_32
	script[2] = root
	return script
}

// TestCheckTransactionSanityOutputScriptTypes verifies the consensus-level
// rule in CheckTransactionSanity that restricts transaction output scripts
// to the three types Pearl supports: P2TR (SegWit v1), P2MR (SegWit v2,
// BIP 360), and OP_RETURN (null data). Anything else must be rejected.
func TestCheckTransactionSanityOutputScriptTypes(t *testing.T) {
	t.Parallel()

	mkTx := func(pkScript []byte) *btcutil.Tx {
		tx := wire.NewMsgTx(wire.TxVersion)
		tx.AddTxIn(&wire.TxIn{
			PreviousOutPoint: wire.OutPoint{
				Hash:  chainhash.Hash{0x01},
				Index: 0,
			},
			Sequence: 0xffffffff,
		})
		tx.AddTxOut(&wire.TxOut{Value: 1000, PkScript: pkScript})
		return btcutil.NewTx(tx)
	}

	// OP_RETURN 0x01 0x02.
	opReturnScript := []byte{txscript.OP_RETURN, 0x02, 0x01, 0x02}

	// A legacy P2PKH script: OP_DUP OP_HASH160 OP_DATA_20 <20B> OP_EQUALVERIFY OP_CHECKSIG.
	legacyP2PKH := append([]byte{
		txscript.OP_DUP, txscript.OP_HASH160, txscript.OP_DATA_20,
	}, make([]byte, 20)...)
	legacyP2PKH = append(legacyP2PKH,
		txscript.OP_EQUALVERIFY, txscript.OP_CHECKSIG)

	tests := []struct {
		name     string
		pkScript []byte
		wantErr  bool
	}{
		{"P2TR accepted", p2trScript(0x01), false},
		{"P2MR accepted", p2mrScript(0x02), false},
		{"OP_RETURN accepted", opReturnScript, false},
		{"legacy P2PKH rejected", legacyP2PKH, true},
		{"empty script rejected", []byte{}, true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := CheckTransactionSanity(mkTx(test.pkScript))
			if test.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// Block100000 is a test block with 4 transactions using P2TR outputs.
var Block100000 = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash: chainhash.Hash([32]byte{ // Block hash placeholder
					0xb4, 0x66, 0x51, 0x7c, 0x4c, 0xa8, 0x50, 0xc5,
					0x7c, 0x21, 0x9c, 0x77, 0xbe, 0xe8, 0xdf, 0xcb,
					0x3f, 0x56, 0x9b, 0xd5, 0x5a, 0xc6, 0x7c, 0xc8,
					0x43, 0x8d, 0x00, 0x29, 0xd5, 0x2a, 0xa1, 0x60,
				}),
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe},
			},
		},
		BlockHeader: wire.BlockHeader{
			Version: 1,
			PrevBlock: chainhash.Hash([32]byte{ // Make go vet happy.
				0x50, 0x12, 0x01, 0x19, 0x17, 0x2a, 0x61, 0x04,
				0x21, 0xa6, 0xc3, 0x01, 0x1d, 0xd3, 0x30, 0xd9,
				0xdf, 0x07, 0xb6, 0x36, 0x16, 0xc2, 0xcc, 0x1f,
				0x1c, 0xd0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
			}),
			MerkleRoot: chainhash.Hash([32]byte{
				0xf8, 0xba, 0xbf, 0x35, 0x35, 0x35, 0xef, 0x0e,
				0xbe, 0x12, 0x96, 0xb8, 0x75, 0xbb, 0x7f, 0x6b,
				0x64, 0xdb, 0x2d, 0xc0, 0xf5, 0x51, 0x41, 0x39,
				0x2f, 0x5e, 0x95, 0x3f, 0x58, 0x76, 0x24, 0xda,
			}),
			Timestamp: time.Unix(1293623863, 0), // 2010-12-29 11:57:43 +0000 UTC
			Bits:      0x1b04864c,               // 453281356
		},
	},
	Transactions: []*wire.MsgTx{
		{
			Version: 1,
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: wire.OutPoint{
					Hash:  chainhash.Hash{},
					Index: 0xffffffff,
				},
				SignatureScript: []byte{
					0x04, 0x4c, 0x86, 0x04, 0x1b, 0x02, 0x06, 0x02,
				},
				Sequence: 0xffffffff,
			}},
			TxOut: []*wire.TxOut{
				{Value: 0x12a05f200, PkScript: p2trScript(0x01)},
			},
		},
		{
			Version: 1,
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: wire.OutPoint{
					Hash: chainhash.Hash([32]byte{ // Make go vet happy.
						0x03, 0x2e, 0x38, 0xe9, 0xc0, 0xa8, 0x4c, 0x60,
						0x46, 0xd6, 0x87, 0xd1, 0x05, 0x56, 0xdc, 0xac,
						0xc4, 0x1d, 0x27, 0x5e, 0xc5, 0x5f, 0xc0, 0x07,
						0x79, 0xac, 0x88, 0xfd, 0xf3, 0x57, 0xa1, 0x87,
					}), // 87a157f3fd88ac7907c05fc55e271dc4acdc5605d187d646604ca8c0e9382e03
					Index: 0,
				},
				Sequence: 0xffffffff,
			}},
			TxOut: []*wire.TxOut{
				{Value: 0x2123e300, PkScript: p2trScript(0x02)},
				{Value: 0x108e20f00, PkScript: p2trScript(0x03)},
			},
		},
		{
			Version: 1,
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: wire.OutPoint{
					Hash: chainhash.Hash([32]byte{ // Make go vet happy.
						0xc3, 0x3e, 0xbf, 0xf2, 0xa7, 0x09, 0xf1, 0x3d,
						0x9f, 0x9a, 0x75, 0x69, 0xab, 0x16, 0xa3, 0x27,
						0x86, 0xaf, 0x7d, 0x7e, 0x2d, 0xe0, 0x92, 0x65,
						0xe4, 0x1c, 0x61, 0xd0, 0x78, 0x29, 0x4e, 0xcf,
					}), // cf4e2978d0611ce46592e02d7e7daf8627a316ab69759a9f3df109a7f2bf3ec3
					Index: 1,
				},
				Sequence: 0xffffffff,
			}},
			TxOut: []*wire.TxOut{
				{Value: 0xf4240, PkScript: p2trScript(0x04)},
				{Value: 0x11d260c0, PkScript: p2trScript(0x05)},
			},
		},
		{
			Version: 1,
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: wire.OutPoint{
					Hash: chainhash.Hash([32]byte{ // Make go vet happy.
						0x0b, 0x60, 0x72, 0xb3, 0x86, 0xd4, 0xa7, 0x73,
						0x23, 0x52, 0x37, 0xf6, 0x4c, 0x11, 0x26, 0xac,
						0x3b, 0x24, 0x0c, 0x84, 0xb9, 0x17, 0xa3, 0x90,
						0x9b, 0xa1, 0xc4, 0x3d, 0xed, 0x5f, 0x51, 0xf4,
					}), // f4515fed3dc4a19b90a317b9840c243bac26114cf637522373a7d486b372600b
					Index: 0,
				},
				Sequence: 0xffffffff,
			}},
			TxOut: []*wire.TxOut{
				{Value: 0xf4240, PkScript: p2trScript(0x06)},
			},
		},
	},
}
