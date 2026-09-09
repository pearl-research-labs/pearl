// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/wire"
)

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
	// Retain a snapshot if the caller later reuses or modifies accepted.
	parent := *accepted
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

	if cert == nil || cert.Version() != wire.CertificateVersionV4 {
		return nil
	}
	// Preserve the absent-certificate case when the interface holds a nil V4.
	if cert == (*wire.CertificateV4)(nil) {
		return nil
	}

	publicData := cert.PublicDataBytes()
	if len(publicData) < wire.IncompleteBlockHeaderSize {
		str := fmt.Sprintf("v4 public data is %d bytes, want at least %d",
			len(publicData), wire.IncompleteBlockHeaderSize)
		return ruleError(ErrHighHash, str)
	}
	// The v4 certificate's public data is expected to start with the
	// serialized ancestor header, checked against the recent headers context.
	ancestorHeader := publicData[:wire.IncompleteBlockHeaderSize]

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
