// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package chainhash

import (
	"encoding/hex"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const mainNetGenesisHashStr = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

func TestNewHashFromStrStrict(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		in      string
		want    Hash
		wantErr error
	}{
		{
			name: "genesis hash",
			in:   mainNetGenesisHashStr,
			want: mainNetGenesisHash,
		},
		{
			name:    "stripped leading zeros",
			in:      "19d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
			wantErr: ErrHashStrSizeMismatch,
		},
		{
			name:    "odd length hash",
			in:      "1",
			wantErr: ErrHashStrSizeMismatch,
		},
		{
			name:    "empty string",
			wantErr: ErrHashStrSizeMismatch,
		},
		{
			name:    "even length hash that is too short",
			in:      "deadbeef" + strings.Repeat("0", 24),
			wantErr: ErrHashStrSizeMismatch,
		},
		{
			name:    "hash string that is too long",
			in:      mainNetGenesisHashStr + "0",
			wantErr: ErrHashStrSizeMismatch,
		},
		{
			name:    "non-hex chars",
			in:      mainNetGenesisHashStr[:63] + "g",
			wantErr: hex.InvalidByteError('g'),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewHashFromStrStrict(tt.in)
			if tt.wantErr != nil {
				require.ErrorIs(t, err, tt.wantErr)
				assert.Nil(t, got)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, *got)

			var decoded Hash
			require.NoError(t, DecodeStrict(&decoded, tt.in))
			assert.Equal(t, tt.want, decoded)
		})
	}
}

// The lenient parser must keep accepting what the strict one rejects, since existing callers depend on the zero
// padding.
func TestDecodeStrictVersusLenient(t *testing.T) {
	t.Parallel()

	short := "19d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

	var lenient Hash
	require.NoError(t, Decode(&lenient, short))
	assert.Equal(t, mainNetGenesisHash, lenient)

	var strict Hash
	assert.ErrorIs(t, DecodeStrict(&strict, short), ErrHashStrSizeMismatch)
}

func BenchmarkDecodeStrict(b *testing.B) {
	var result Hash

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := DecodeStrict(&result, mainNetGenesisHashStr); err != nil {
			b.Fatalf("unexpected decode error: %v", err)
		}
	}
}
