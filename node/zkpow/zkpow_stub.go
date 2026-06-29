//go:build !zkpow

package zkpow

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	DefaultNBits     = 0x1E01FFFF
	DefaultM         = 256
	DefaultN         = 512
	DefaultNoiseRank = 32
	DefaultMMAType   = 0
)

func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func VerifyZKProofFFI(
	header *wire.BlockHeader,
	certHash chainhash.Hash,
	proofCommitment chainhash.Hash,
	publicData []byte,
	proofData []byte,
	nbitsOverride *uint32,
) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func Mine(header *wire.BlockHeader) (*wire.CertificateV2, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable mining")
}

func MineMoE(header *wire.BlockHeader, m, n, e, topK uint32) (*wire.CertificateV2, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable MoE mining")
}
