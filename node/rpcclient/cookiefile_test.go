// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpcclient

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadCookieFile(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		filename     string
		contents     string
		wantUsername string
		wantPassword string
		wantErr      bool
	}{
		{
			name:         "standard credentials",
			filename:     ".cookie",
			contents:     "__cookie__:secret\n",
			wantUsername: "__cookie__",
			wantPassword: "secret",
		},
		{
			name:         "password containing colons",
			filename:     ".cookie",
			contents:     "__cookie__:secret:with:colons\n",
			wantUsername: "__cookie__",
			wantPassword: "secret:with:colons",
		},
		{
			name:         "CRLF line ending",
			filename:     ".cookie",
			contents:     "__cookie__:secret\r\n",
			wantUsername: "__cookie__",
			wantPassword: "secret",
		},
		{
			name:     "empty file",
			filename: ".cookie",
			wantErr:  true,
		},
		{
			name:     "missing separator",
			filename: ".cookie",
			contents: "__cookie__\n",
			wantErr:  true,
		},
		{
			name:     "scanner token too long",
			filename: ".cookie",
			contents: strings.Repeat("a", bufio.MaxScanTokenSize+1),
			wantErr:  true,
		},
		{
			name:    "missing file",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cookiePath := filepath.Join(t.TempDir(), "missing.cookie")
			if tt.filename != "" {
				cookiePath = filepath.Join(t.TempDir(), tt.filename)
				require.NoError(t, os.WriteFile(cookiePath, []byte(tt.contents), 0o600))
			}

			username, password, err := readCookieFile(cookiePath)
			if tt.wantErr {
				require.Error(t, err)
				assert.Empty(t, username)
				assert.Empty(t, password)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.wantUsername, username)
			assert.Equal(t, tt.wantPassword, password)
		})
	}
}
