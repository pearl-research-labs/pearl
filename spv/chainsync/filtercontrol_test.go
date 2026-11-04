package chainsync

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Swaps the package checkpoint map, so this test must stay sequential.
func TestControlCFHeader(t *testing.T) {
	original := filterHeaderCheckpoints
	t.Cleanup(func() {
		filterHeaderCheckpoints = original
	})

	height := uint32(999)
	header := hashFromStr(
		"4a242283a406a7c089f671bb8df7671e5d5e9ba577cea1047d30a7f4919df193",
	)
	require.NotNil(t, header)
	filterHeaderCheckpoints = map[wire.PearlNet]map[uint32]*chainhash.Hash{
		chaincfg.MainNetParams.Net: {
			height: header,
		},
	}

	require.NoError(t, ControlCFHeader(
		chaincfg.MainNetParams, wire.GCSFilterRegular, height, header,
	))

	bad := hashFromStr(
		"000000000006a7c089f671bb8df7671e5d5e9ba577cea1047d30a7f4919df193",
	)
	require.NotNil(t, bad)
	require.ErrorIs(t, ControlCFHeader(
		chaincfg.MainNetParams, wire.GCSFilterRegular, height, bad,
	), ErrCheckpointMismatch)

	require.NoError(t, ControlCFHeader(
		chaincfg.MainNetParams, wire.GCSFilterRegular, 99, bad,
	))
}

func TestFilterHeaderCheckpoints(t *testing.T) {
	t.Parallel()

	paramsByNet := map[wire.PearlNet]chaincfg.Params{
		wire.MainNet:  chaincfg.MainNetParams,
		wire.TestNet:  chaincfg.TestNetParams,
		wire.TestNet2: chaincfg.TestNet2Params,
	}

	mainPins := filterHeaderCheckpoints[wire.MainNet]
	require.NotEmpty(t, mainPins)
	for _, cp := range chaincfg.MainNetParams.Checkpoints {
		_, ok := mainPins[uint32(cp.Height)]
		assert.Truef(t, ok, "missing filter checkpoint at height %d", cp.Height)
	}

	testnet2Pins := filterHeaderCheckpoints[wire.TestNet2]
	require.NotEmpty(t, testnet2Pins)
	for height := uint32(10000); height <= 100000; height += 10000 {
		_, ok := testnet2Pins[height]
		assert.Truef(t, ok, "missing testnet2 filter checkpoint at height %d", height)
	}

	for net, pins := range filterHeaderCheckpoints {
		params, ok := paramsByNet[net]
		require.Truef(t, ok, "unexpected network %v", net)
		for height, header := range pins {
			t.Run(fmt.Sprintf("%s/%d", net, height), func(t *testing.T) {
				assert.NotZero(t, height)
				assert.Zero(t, height%10000)
				require.NotNil(t, header)

				assert.NoError(t, ValidateCFHeader(
					params, wire.GCSFilterRegular, height, header,
				))

				flipped := *header
				flipped[0] ^= 0xff
				assert.ErrorIs(t, ValidateCFHeader(
					params, wire.GCSFilterRegular, height, &flipped,
				), ErrCheckpointMismatch)
			})
		}
	}
}
