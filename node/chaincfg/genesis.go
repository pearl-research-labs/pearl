// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package chaincfg

import (
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// genesisCoinbaseTx is the coinbase transaction for the genesis blocks for
// the main network, regression test network, and test network (version 3).
var genesisCoinbaseTx = wire.MsgTx{
	Version: 1,
	TxIn: []*wire.TxIn{
		{
			PreviousOutPoint: wire.OutPoint{
				Hash:  chainhash.Hash{},
				Index: 0xffffffff,
			},
			SignatureScript: []byte{
				0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04, 0x4c, /* |.......L| */
				0xe1, 0x54, 0x68, 0x65, 0x20, 0x6e, 0x61, 0x74, /* |.The nat| */
				0x69, 0x76, 0x65, 0x20, 0x63, 0x75, 0x72, 0x72, /* |ive curr| */
				0x65, 0x6e, 0x63, 0x79, 0x20, 0x66, 0x6f, 0x72, /* |ency for| */
				0x20, 0x6d, 0x61, 0x63, 0x68, 0x69, 0x6e, 0x65, /* | machine| */
				0x73, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x68, 0x75, /* |s and hu| */
				0x6d, 0x61, 0x6e, 0x73, 0x2e, 0x20, 0x54, 0x6f, /* |mans. To| */
				0x74, 0x61, 0x6c, 0x20, 0x32, 0x2e, 0x31, 0x65, /* |tal 2.1e| */
				0x39, 0x20, 0x50, 0x65, 0x61, 0x72, 0x6c, 0x73, /* |9 Pearls| */
				0x2c, 0x20, 0x43, 0x75, 0x6d, 0x75, 0x6c, 0x61, /* |, Cumula| */
				0x74, 0x69, 0x76, 0x65, 0x20, 0x65, 0x6d, 0x69, /* |tive emi| */
				0x73, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x68, 0x2f, /* |ssion h/| */
				0x28, 0x68, 0x2b, 0x36, 0x35, 0x30, 0x32, 0x32, /* |(h+65022| */
				0x36, 0x29, 0x2c, 0x20, 0x57, 0x54, 0x45, 0x4d, /* |6), WTEM| */
				0x41, 0x20, 0x33, 0x6d, 0x31, 0x34, 0x73, 0x20, /* |A 3m14s | */
				0x37, 0x64, 0x2c, 0x20, 0x42, 0x69, 0x74, 0x63, /* |7d, Bitc| */
				0x6f, 0x69, 0x6e, 0x20, 0x62, 0x6c, 0x6f, 0x63, /* |oin bloc| */
				0x6b, 0x20, 0x23, 0x39, 0x34, 0x36, 0x38, 0x34, /* |k #94684| */
				0x32, 0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x4d, /* |2 with M| */
				0x54, 0x50, 0x20, 0x31, 0x37, 0x37, 0x37, 0x32, /* |TP 17772| */
				0x37, 0x30, 0x35, 0x32, 0x34, 0x20, 0x69, 0x73, /* |70524 is| */
				0x20, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, /* | 0000000| */
				0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, /* |00000000| */
				0x30, 0x30, 0x30, 0x30, 0x30, 0x33, 0x61, 0x39, /* |000003a9| */
				0x32, 0x32, 0x31, 0x38, 0x36, 0x31, 0x65, 0x63, /* |221861ec| */
				0x37, 0x34, 0x31, 0x31, 0x34, 0x32, 0x37, 0x64, /* |7411427d| */
				0x64, 0x32, 0x34, 0x66, 0x39, 0x37, 0x39, 0x66, /* |d24f979f| */
				0x62, 0x39, 0x31, 0x61, 0x33, 0x62, 0x38, 0x66, /* |b91a3b8f| */
				0x37, 0x32, 0x36, 0x35, 0x65, 0x31, 0x64, 0x62, /* |7265e1db| */
				0x63, 0x2e, /* |c.| */
			},
			Sequence: 0xffffffff,
		},
	},
	TxOut: []*wire.TxOut{
		{
			Value: 0x0,
			PkScript: []byte{
				0x6a, /* |j| */
			},
		},
	},
	LockTime: 0,
}

// genesisHash is the hash of the first block in the block chain for the main
// network (genesis block).
var genesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0xd7, 0xad, 0x8d, 0xc5, 0x41, 0x5f, 0xb5, 0x48,
	0xff, 0x53, 0xef, 0xe7, 0x5f, 0x10, 0xe0, 0xe2,
	0x74, 0x03, 0xd1, 0xa3, 0x3f, 0x07, 0xdd, 0x0c,
	0x8f, 0x61, 0xff, 0xb7, 0x93, 0x30, 0x8d, 0xa1,
})

// genesisMerkleRoot is the hash of the first transaction in the genesis block
// for the main network.
var genesisMerkleRoot = chainhash.Hash([chainhash.HashSize]byte{
	0xc1, 0x75, 0x16, 0x37, 0x38, 0x2f, 0xbd, 0x52,
	0xf5, 0xe4, 0xe2, 0x6d, 0xc9, 0xf3, 0x8d, 0x8c,
	0xcc, 0x2a, 0xb2, 0x16, 0xdf, 0xed, 0x38, 0xbd,
	0x48, 0x96, 0x23, 0xf1, 0xc6, 0x59, 0x2f, 0x30,
})

// genesisBlock defines the genesis block of the block chain which serves as the
// public transaction ledger for the main network.
var genesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      genesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},         // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: genesisMerkleRoot,        // 302f59c6f1239648bd38eddf16b22acc8c8df3c96de2e4f552bd2f38371675c1
			Timestamp:  time.Unix(1777280400, 0), // 2026-04-27 09:00:00 +0000 UTC
			Bits:       0x1b00ffff,               // [000000000000ffff000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&genesisCoinbaseTx},
}

// regTestGenesisHash is the hash of the first block in the block chain for the
// regression test network (genesis block).
var regTestGenesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0x32, 0x58, 0x7e, 0x24, 0x47, 0xb0, 0x43, 0x2f,
	0x2c, 0xce, 0x83, 0xb0, 0x8c, 0x81, 0xb3, 0xbe,
	0xbb, 0x1e, 0xca, 0xff, 0xef, 0xbd, 0xdb, 0xe9,
	0xa8, 0x51, 0x6a, 0xdf, 0x27, 0xab, 0xd0, 0x78,
})

// regTestGenesisMerkleRoot is the hash of the first transaction in the genesis
// block for the regression test network. It is the same as the merkle root for
// the main network.
var regTestGenesisMerkleRoot = genesisMerkleRoot

// regTestGenesisBlock defines the genesis block of the block chain which serves
// as the public transaction ledger for the regression test network.
var regTestGenesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      regTestGenesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},         // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: regTestGenesisMerkleRoot, // 302f59c6f1239648bd38eddf16b22acc8c8df3c96de2e4f552bd2f38371675c1
			Timestamp:  time.Unix(1776675600, 0), // 2026-04-20 09:00:00 +0000 UTC
			Bits:       0x1e010000,               // [0000010000000000000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&genesisCoinbaseTx},
}

// testNetGenesisTx is the coinbase transaction for the testnet genesis block.
var testNetGenesisTx = wire.MsgTx{
	Version: 1,
	TxIn: []*wire.TxIn{
		{
			PreviousOutPoint: wire.OutPoint{
				Hash:  chainhash.Hash{},
				Index: 0xffffffff,
			},
			SignatureScript: []byte{
				// Message: `03/May/2024 000000000000000000001ebd58c244970b3aa9d783bb001011fbe8ea8e98e00e`
				0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04, 0x4c, /* |.......L| */
				0x4c, 0x30, 0x33, 0x2f, 0x4d, 0x61, 0x79, 0x2f, /* |L03/May/| */
				0x32, 0x30, 0x32, 0x34, 0x20, 0x30, 0x30, 0x30, /* |2024 000| */
				0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, /* |00000000| */
				0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, /* |00000000| */
				0x30, 0x31, 0x65, 0x62, 0x64, 0x35, 0x38, 0x63, /* |01ebd58c| */
				0x32, 0x34, 0x34, 0x39, 0x37, 0x30, 0x62, 0x33, /* |244970b3| */
				0x61, 0x61, 0x39, 0x64, 0x37, 0x38, 0x33, 0x62, /* |aa9d783b| */
				0x62, 0x30, 0x30, 0x31, 0x30, 0x31, 0x31, 0x66, /* |b001011f| */
				0x62, 0x65, 0x38, 0x65, 0x61, 0x38, 0x65, 0x39, /* |be8ea8e9| */
				0x38, 0x65, 0x30, 0x30, 0x65, /* |8e00e| */
			},
			Sequence: 0xffffffff,
		},
	},
	TxOut: []*wire.TxOut{
		{
			Value: 0x12a05f200,
			PkScript: []byte{
				0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0xac,
			},
		},
	},
	LockTime: 0,
}

// testNetGenesisHash is the hash of the first block in the block chain for the
// test network.
var testNetGenesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0x2f, 0x3a, 0xf9, 0x10, 0xc5, 0x24, 0x39, 0x45,
	0x4b, 0xe1, 0xb6, 0xc9, 0x52, 0x1d, 0x35, 0x4b,
	0x17, 0x42, 0x9e, 0x6f, 0xcd, 0x7e, 0xba, 0x3d,
	0x77, 0xa9, 0x22, 0x71, 0x50, 0xbd, 0x6a, 0x47,
})

// testNetGenesisMerkleRoot is the hash of the first transaction in the genesis
// block for the test network.
var testNetGenesisMerkleRoot = chainhash.Hash([chainhash.HashSize]byte{
	0x4e, 0x7b, 0x2b, 0x91, 0x28, 0xfe, 0x02, 0x91,
	0xdb, 0x06, 0x93, 0xaf, 0x2a, 0xe4, 0x18, 0xb7,
	0x67, 0xe6, 0x57, 0xcd, 0x40, 0x7e, 0x80, 0xcb,
	0x14, 0x34, 0x22, 0x1e, 0xae, 0xa7, 0xa0, 0x7a,
})

// testNetGenesisBlock defines the genesis block of the block chain which serves
// as the public transaction ledger for the test network.
var testNetGenesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      testNetGenesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},         // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: testNetGenesisMerkleRoot, // 7aa0a7ae1e223414cb807e40cd57e667b718e42aaf9306db9102fe28912b7b4e
			Timestamp:  time.Unix(1777215600, 0), // 2026-04-26 15:00:00 +0000 UTC
			Bits:       0x1b00ffff,               // [000000000000ffff000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&testNetGenesisTx},
}

// testNet2GenesisHash is the hash of the first block in the block chain for the
// test network v2 (fresh genesis).
var testNet2GenesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0x96, 0x6f, 0x33, 0xa0, 0x2e, 0x2e, 0x5f, 0x2b,
	0xd8, 0xd5, 0x11, 0x38, 0x41, 0xce, 0xee, 0x29,
	0x04, 0x47, 0x6f, 0xe4, 0x2d, 0x0d, 0x26, 0x72,
	0x4d, 0x5f, 0x82, 0xd8, 0x0c, 0x6d, 0xc8, 0xb0,
})

// testNet2GenesisMerkleRoot is the hash of the first transaction in the genesis
// block for the test network v2. Same as testnet since it uses the same
// coinbase tx.
var testNet2GenesisMerkleRoot = testNetGenesisMerkleRoot

// testNet2GenesisBlock defines the genesis block of the block chain which
// serves as the public transaction ledger for the test network v2.
var testNet2GenesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      testNet2GenesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},          // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: testNet2GenesisMerkleRoot, // same as testnet
			Timestamp:  time.Unix(1776675600, 0),  // 2026-04-20 09:00:00 +0000 UTC
			Bits:       0x1b00ffff,                // [000000000000ffff000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&testNetGenesisTx},
}

// simNetGenesisHash is the hash of the first block in the block chain for the
// simulation test network.
var simNetGenesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0x32, 0x58, 0x7e, 0x24, 0x47, 0xb0, 0x43, 0x2f,
	0x2c, 0xce, 0x83, 0xb0, 0x8c, 0x81, 0xb3, 0xbe,
	0xbb, 0x1e, 0xca, 0xff, 0xef, 0xbd, 0xdb, 0xe9,
	0xa8, 0x51, 0x6a, 0xdf, 0x27, 0xab, 0xd0, 0x78,
})

// simNetGenesisMerkleRoot is the hash of the first transaction in the genesis
// block for the simulation test network. It is the same as the merkle root for
// the main network.
var simNetGenesisMerkleRoot = genesisMerkleRoot

// simNetGenesisBlock defines the genesis block of the block chain which serves
// as the public transaction ledger for the simulation test network.
var simNetGenesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      simNetGenesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},         // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: simNetGenesisMerkleRoot,  // 302f59c6f1239648bd38eddf16b22acc8c8df3c96de2e4f552bd2f38371675c1
			Timestamp:  time.Unix(1776675600, 0), // 2026-04-20 09:00:00 +0000 UTC
			Bits:       0x1e010000,               // [0000010000000000000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&genesisCoinbaseTx},
}

// sigNetGenesisHash is the hash of the first block in the block chain for the
// signet test network.
var sigNetGenesisHash = chainhash.Hash([chainhash.HashSize]byte{
	0x34, 0x0f, 0xb0, 0x93, 0x0e, 0x9d, 0x8d, 0xd2,
	0x2c, 0xcf, 0x1f, 0x11, 0x04, 0x4d, 0x53, 0xdd,
	0xc9, 0x78, 0x2f, 0x9e, 0x56, 0xfb, 0xc9, 0x52,
	0x74, 0x46, 0x25, 0x61, 0x1e, 0x23, 0xef, 0x2f,
})

// sigNetGenesisMerkleRoot is the hash of the first transaction in the genesis
// block for the signet test network. It is the same as the merkle root for the
// main network.
var sigNetGenesisMerkleRoot = genesisMerkleRoot

// sigNetGenesisBlock defines the genesis block of the block chain which serves
// as the public transaction ledger for the signet test network.
var sigNetGenesisBlock = wire.MsgBlock{
	MsgHeader: wire.MsgHeader{
		MsgCertificate: wire.MsgCertificate{
			Certificate: &wire.CertificateV1{
				Hash:      sigNetGenesisHash,
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef}, // Vanity proof for genesis block
			},
		},
		BlockHeader: wire.BlockHeader{
			Version:    1,
			PrevBlock:  chainhash.Hash{},         // 0000000000000000000000000000000000000000000000000000000000000000
			MerkleRoot: sigNetGenesisMerkleRoot,  // 302f59c6f1239648bd38eddf16b22acc8c8df3c96de2e4f552bd2f38371675c1
			Timestamp:  time.Unix(1776675600, 0), // 2026-04-20 09:00:00 +0000 UTC
			Bits:       0x1d0fffff,               // [0000000fffff0000000000000000000000000000000000000000000000000000]
		},
	},
	Transactions: []*wire.MsgTx{&genesisCoinbaseTx},
}
