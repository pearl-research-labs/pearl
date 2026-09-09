// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const IncompleteBlockHeaderSize = wire.IncompleteBlockHeaderSize

// CertificateHeaderContext contains the available ancestor headers needed
// to validate the certificate of a proposed block header. Parent and
// Grandparent may be nil near genesis.
type CertificateHeaderContext struct {
	Parent      *wire.BlockHeader
	Grandparent *wire.BlockHeader
}

// Advance moves the ancestor window forward after a header is accepted.
func (headers *CertificateHeaderContext) Advance(accepted *wire.BlockHeader) {
	headers.Grandparent = headers.Parent
	parent := *accepted // necessary copy, so that the pointer will not vanish
	headers.Parent = &parent
}

// CheckCertificateContext requires a V4 certificate's proof-carried ancestor
// header to be the proposed header, its parent, or its grandparent.
func CheckCertificateContext(proposed *wire.BlockHeader,
	headers CertificateHeaderContext,
	cert wire.BlockCertificate, flags BehaviorFlags) error {

	if flags&BFNoPoWCheck == BFNoPoWCheck {
		return nil
	}

	v4, ok := cert.(*wire.CertificateV4)
	if !ok || v4 == nil {
		return nil
	}

	publicData := v4.PublicDataBytes()
	if len(publicData) < IncompleteBlockHeaderSize {
		str := fmt.Sprintf("v4 public data is %d bytes, want at least %d",
			len(publicData), IncompleteBlockHeaderSize)
		return ruleError(ErrHighHash, str)
	}
	// The v4 certificate's public data is expected to start with the
	// serialized ancestor header, checked against the recent headers context.
	ancestorHeader := publicData[:IncompleteBlockHeaderSize]

	for _, candidate := range [...]*wire.BlockHeader{
		proposed, headers.Parent, headers.Grandparent,
	} {
		if candidate == nil {
			continue
		}
		candidateBytes := candidate.IncompleteHeaderBytes()
		if bytes.Equal(ancestorHeader, candidateBytes[:]) {
			return nil
		}
	}

	str := "v4 ancestor header is not the proposed header, its parent, " +
		"or its grandparent"
	return ruleError(ErrHighHash, str)
}

// header reconstructs a wire.BlockHeader from the blockNode's stored
// header fields; prev_block is the node's parent hash (genesis carries
// the zero hash). It is used to recover parent and grandparent headers
// for CheckCertificateContext without a database fetch.
func (node *blockNode) header() wire.BlockHeader {
	var prevBlock chainhash.Hash
	if node.parent != nil {
		prevBlock = node.parent.hash
	}
	return wire.BlockHeader{
		Version:         node.version,
		PrevBlock:       prevBlock,
		MerkleRoot:      node.merkleRoot,
		Timestamp:       time.Unix(node.timestamp, 0),
		Bits:            node.bits,
		ProofCommitment: node.proofCommitment,
	}
}
