package chain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/require"
)

// TestFilterPeersServicesFormats pins the wire contract between pearld's
// getpeerinfo response and filterPeers: the Services field is the hex
// encoding of the 8-byte big-endian service flags (the same shape the
// dispatcher's peer harness encodes, and the shape handleGetPeerInfo now
// emits). A Services value that does not decode to exactly 8 bytes must
// produce an error, never a panic: the value crosses a trust boundary
// (it originates in a remote peer's version message, relayed by the
// backend node's RPC), so a malformed relay must not crash the wallet.
func TestFilterPeersServicesFormats(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		services    string
		expectAddrs int
		expectErr   bool
	}{
		{
			// SFNodeNetwork|SFNodeWitness = 9, as an 8-byte
			// big-endian hex string.
			name:        "hex 8-byte segwit full node is eligible",
			services:    "0000000000000009",
			expectAddrs: 1,
		},
		{
			// What pearld's handleGetPeerInfo emitted before the
			// format fix: fmt.Sprintf("%08d", 9) = "00000009",
			// a decimal string that hex-decodes to only 4 bytes
			// and panicked binary.BigEndian.Uint64.
			name:      "decimal-shaped services errors, no panic",
			services:  "00000009",
			expectErr: true,
		},
		{
			name:      "short hex services errors, no panic",
			services:  "040d",
			expectErr: true,
		},
		{
			name:      "empty services errors, no panic",
			services:  "",
			expectErr: true,
		},
		{
			name:      "non-hex services errors",
			services:  "not-hex!",
			expectErr: true,
		},
		{
			// 8 valid bytes but without the required flags:
			// filtered out, not an error.
			name:        "hex 8-byte without witness is filtered",
			services:    "0000000000000001",
			expectAddrs: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			peers := []btcjson.GetPeerInfoResult{{
				Addr:     "192.168.0.1:44108",
				Services: test.services,
			}}

			addrs, err := filterPeers(peers)
			if test.expectErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Len(t, addrs, test.expectAddrs)
		})
	}
}
