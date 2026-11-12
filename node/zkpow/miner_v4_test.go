//go:build zkpow

package zkpow

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestMineV4AndVerify(t *testing.T) {
	for name, device := range map[string]int{"h100": 0, "b200": 1} {
		t.Run(name, func(t *testing.T) {
			parent := &wire.BlockHeader{Bits: 0x207fffff, Timestamp: time.Unix(1, 0)}
			header := &wire.BlockHeader{PrevBlock: parent.BlockHash(), Bits: 0x207fffff, Timestamp: time.Unix(2, 0)}
			cert, err := MineV4(header, parent, device)
			require.NoError(t, err)
			require.Equal(t, cert.ProofCommitment(), header.ProofCommitment)
			require.Equal(t, header.BlockHash(), cert.BlockHash())
			require.Empty(t, cert.AncestorHeaders)
			require.NoError(t, VerifyCertificate(header, cert))

			var buf bytes.Buffer
			require.NoError(t, cert.Serialize(&buf))
			var restored wire.CertificateV4
			require.NoError(t, restored.Deserialize(&buf))
			require.NoError(t, VerifyCertificate(header, &restored))

			header.MerkleRoot[0] ^= 1
			require.Error(t, VerifyCertificate(header, &restored))
		})
	}
}

func TestMineV4RejectsInvalidInput(t *testing.T) {
	parent := &wire.BlockHeader{}
	header := &wire.BlockHeader{Bits: 0x207fffff}
	_, err := MineV4(nil, parent, 1)
	require.Error(t, err)
	_, err = MineV4(header, nil, 1)
	require.Error(t, err)
	_, err = MineV4(header, parent, 1)
	require.ErrorContains(t, err, "parent does not connect")
	header.PrevBlock = parent.BlockHash()
	_, err = MineV4(header, parent, 256)
	require.ErrorContains(t, err, "unsupported V4 mining device")
	header.Bits = 0
	_, err = MineV4(header, parent, 1)
	require.ErrorContains(t, err, "target is zero")
}
