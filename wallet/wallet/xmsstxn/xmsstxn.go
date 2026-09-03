// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package xmsstxn builds P2MR transactions that spend with XMSS
// post-quantum signatures.
//
// Script-level primitives such as the leaf script, witness layout, and
// chunk constants live in node/txscript. This package is the wallet-facing
// layer: it derives spendable P2MR-XMSS addresses, builds transactions, signs
// them, and applies wallet policy such as the minimum relay fee.
package xmsstxn

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txrules"
	"github.com/pearl-research-labs/pearl/xmss"
)

// Descriptor describes a P2MR-XMSS output and carries the script material
// and key pair needed to recognize and spend a UTXO paid to it.
//
// SecretKey is sensitive. Clear it after use if the caller does not need to
// keep spending authority in memory.
type Descriptor struct {
	Addr         *btcutil.AddressMerkleRoot
	PkScript     []byte
	LeafScript   []byte
	ControlBlock []byte
	PublicKey    [xmss.PKLen]byte
	SecretKey    [xmss.SKLen]byte
}

// UTXO describes a previous P2MR-XMSS output being spent.
type UTXO struct {
	OutPoint wire.OutPoint
	Value    int64
}

// SpendRequest describes a 1-in / 1-out P2MR-XMSS spend.
type SpendRequest struct {
	PrevOut             UTXO
	DestinationPkScript []byte
	Fee                 int64
	Descriptor          *Descriptor
	MsgUID              uint32
}

// DeriveDescriptor deterministically derives a spendable P2MR-XMSS
// descriptor from raw XMSS seeds for the given chain.
func DeriveDescriptor(privSeed [xmss.PrivateSeedLen]byte,
	pubSeed [xmss.PublicSeedLen]byte, net *chaincfg.Params) (*Descriptor, error) {

	xmssPK, xmssSK, err := xmss.Keygen(privSeed, pubSeed)
	if err != nil {
		return nil, fmt.Errorf("xmss keygen: %w", err)
	}

	leafScript, err := txscript.XMSSLeafScript(xmssPK)
	if err != nil {
		return nil, fmt.Errorf("build leaf script: %w", err)
	}

	tree := txscript.AssembleTaprootScriptTree(
		txscript.NewBaseTapLeaf(leafScript),
	)
	merkleRoot := tree.RootNode.TapHash()

	addr, err := btcutil.NewAddressMerkleRoot(merkleRoot[:], net)
	if err != nil {
		return nil, fmt.Errorf("encode p2mr address: %w", err)
	}

	pkScript, err := txscript.PayToAddrScript(addr)
	if err != nil {
		return nil, fmt.Errorf("build p2mr pkScript: %w", err)
	}

	controlBlock, err := (&txscript.MerkleRootControlBlock{
		LeafVersion:    txscript.BaseLeafVersion,
		InclusionProof: tree.LeafMerkleProofs[0].InclusionProof,
	}).ToBytes()
	if err != nil {
		return nil, fmt.Errorf("serialize control block: %w", err)
	}

	return &Descriptor{
		Addr:         addr,
		PkScript:     pkScript,
		LeafScript:   leafScript,
		ControlBlock: controlBlock,
		PublicKey:    xmssPK,
		SecretKey:    xmssSK,
	}, nil
}

// BuildSpend signs and returns a 1-in / 1-out P2MR-XMSS transaction.
// A non-nil return is consensus-valid and meets the network minimum relay fee.
func BuildSpend(req SpendRequest) (*wire.MsgTx, error) {
	if req.Descriptor == nil {
		return nil, fmt.Errorf("descriptor is nil")
	}
	if req.Fee < 0 {
		return nil, fmt.Errorf("fee must be non-negative, got %d", req.Fee)
	}
	if req.PrevOut.Value <= req.Fee {
		return nil, fmt.Errorf("fundingValue %d must exceed fee %d",
			req.PrevOut.Value, req.Fee)
	}
	if len(req.DestinationPkScript) == 0 {
		return nil, fmt.Errorf("destPkScript is empty")
	}

	desc := req.Descriptor
	if len(desc.LeafScript) == 0 {
		return nil, fmt.Errorf("leafScript is empty")
	}
	if len(desc.ControlBlock) == 0 {
		return nil, fmt.Errorf("controlBlock is empty")
	}

	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: req.PrevOut.OutPoint,
		Sequence:         wire.MaxTxInSequenceNum,
	})
	tx.AddTxOut(&wire.TxOut{
		Value:    req.PrevOut.Value - req.Fee,
		PkScript: req.DestinationPkScript,
	})

	prevFetcher := txscript.NewCannedPrevOutputFetcher(
		desc.PkScript, req.PrevOut.Value,
	)
	sigHashes := txscript.NewTxSigHashes(tx, prevFetcher)

	tapLeaf := txscript.NewBaseTapLeaf(desc.LeafScript)
	sigHash, err := txscript.CalcTapscriptSignaturehash(
		sigHashes, txscript.SigHashDefault, tx, 0, prevFetcher, tapLeaf,
	)
	if err != nil {
		return nil, fmt.Errorf("compute tapscript sighash: %w", err)
	}

	var msg [xmss.MsgLen]byte
	copy(msg[:], sigHash)
	sig, err := xmss.Sign(req.MsgUID, desc.SecretKey, msg)
	if err != nil {
		return nil, fmt.Errorf("xmss sign: %w", err)
	}

	tx.TxIn[0].Witness = txscript.XMSSScriptPathWitness(
		sig, desc.LeafScript, desc.ControlBlock,
	)

	finalSigHashes := txscript.NewTxSigHashes(tx, prevFetcher)
	vm, err := txscript.NewEngine(
		desc.PkScript, tx, 0, txscript.StandardVerifyFlags, nil,
		finalSigHashes, req.PrevOut.Value, prevFetcher,
	)
	if err != nil {
		return nil, fmt.Errorf("create script engine: %w", err)
	}
	if err := vm.Execute(); err != nil {
		return nil, fmt.Errorf("script engine rejected XMSS spend: %w", err)
	}

	if err := checkRelayFee(tx, req.Fee); err != nil {
		return nil, err
	}

	return tx, nil
}

// checkRelayFee returns an error if fee is below DefaultRelayFeePerKb
// scaled to the signed virtual size of tx.
func checkRelayFee(tx *wire.MsgTx, fee int64) error {
	baseSize := tx.SerializeSizeStripped()
	witnessSize := tx.SerializeSize() - baseSize
	const witnessScaleFactor = 4
	vsize := baseSize + (witnessSize+witnessScaleFactor-1)/witnessScaleFactor

	minFee := txrules.FeeForSerializeSize(
		txrules.DefaultRelayFeePerKb, vsize,
	)
	if btcutil.Amount(fee) < minFee {
		return fmt.Errorf(
			"fee %d grains below min relay fee %d grains for "+
				"vsize=%d (relay=%d grains/kvB)",
			fee, int64(minFee), vsize,
			int64(txrules.DefaultRelayFeePerKb),
		)
	}
	return nil
}
