// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// checkCertificateAncestors requires a V4 certificate's proof-carried ancestor
// header to be the proposed header, its parent, or its grandparent. The
// certificate's ancestor headers must form a chain ending at the proposed block.
func checkCertificateAncestors(proposed *wire.BlockHeader,
	cert wire.BlockCertificate, flags BehaviorFlags) error {

	v4, ok := cert.(*wire.CertificateV4)
	if flags&BFNoPoWCheck != 0 || !ok || v4 == nil {
		return nil
	}
	if len(v4.AncestorHeaders) > wire.MaxCertificateV4AncestorHeaders {
		str := fmt.Sprintf("v4 certificate has %d ancestor headers, max %d",
			len(v4.AncestorHeaders), wire.MaxCertificateV4AncestorHeaders)
		return ruleError(ErrHighHash, str)
	}

	publicData := v4.PublicDataBytes()
	if len(publicData) < wire.IncompleteBlockHeaderSize {
		str := fmt.Sprintf("v4 public data is %d bytes, want at least %d",
			len(publicData), wire.IncompleteBlockHeaderSize)
		return ruleError(ErrHighHash, str)
	}
	// Authenticate every supplied header, including those after a match.
	ancestorHeader := publicData[:wire.IncompleteBlockHeaderSize]
	proposedBytes := proposed.IncompleteHeaderBytes()
	matched := bytes.Equal(ancestorHeader, proposedBytes[:])
	prevHash := proposed.PrevBlock
	for i := range v4.AncestorHeaders {
		header := &v4.AncestorHeaders[i]
		if header.BlockHash() != prevHash {
			str := fmt.Sprintf("v4 ancestor header at depth %d does not connect", i+1)
			return ruleError(ErrHighHash, str)
		}
		candidateBytes := header.IncompleteHeaderBytes()
		matched = matched || bytes.Equal(ancestorHeader, candidateBytes[:])
		prevHash = header.PrevBlock
	}
	if matched {
		return nil
	}

	str := "v4 ancestor header is not the proposed header, its parent, " +
		"or its grandparent"
	return ruleError(ErrHighHash, str)
}
