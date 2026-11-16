//go:build !zkpow

package zkpow

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// Mining parameters the node passes to the miner. The rest of the mining
// configuration, including the noise rank, comes from the Rust FFI and so is only
// available in the zkpow build.
const (
	DefaultNBits = 0x1E01FFFF
	DefaultM     = 256
	DefaultN     = 512
)

func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func BatchVerify(headers []*wire.BlockHeader, certs []wire.BlockCertificate) (int, error) {
	if len(headers) != len(certs) {
		return 0, fmt.Errorf("BatchVerify: %d headers but %d certificates", len(headers), len(certs))
	}
	for i := range certs {
		if err := VerifyCertificate(headers[i], certs[i]); err != nil {
			return i, err
		}
	}

	return len(certs), nil
}

func VerifyZKProofFFI(
	header *wire.BlockHeader,
	cert wire.BlockCertificate,
	nbitsOverride *uint32,
) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func CheckRankPenalty(bits uint32, publicData []byte) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func Mine(header *wire.BlockHeader, certVersion wire.CertificateVersion) (wire.BlockCertificate, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable mining")
}

func MineMoE(header *wire.BlockHeader, m, n, e, topK uint32,
	certVersion wire.CertificateVersion) (wire.BlockCertificate, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable MoE mining")
}
