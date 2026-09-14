// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cfgutil

import (
	"os/user"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCleanAndExpandPath(t *testing.T) {
	t.Setenv("CFGUTIL_TEST_DIR", "/tmp/cfgutil")

	u, err := user.Current()
	require.NoError(t, err)

	tests := []struct {
		name string
		path string
		want string
	}{
		{"plain path", "/var/log/oyster.log", "/var/log/oyster.log"},
		{"cleaned", "/var/./log/../log/oyster.log", "/var/log/oyster.log"},
		{"env expansion", "$CFGUTIL_TEST_DIR/logs", "/tmp/cfgutil/logs"},
		{"tilde expansion", "~/wallet", filepath.Join(u.HomeDir, "wallet")},
		{"bare tilde", "~", u.HomeDir},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, CleanAndExpandPath(tt.path))
		})
	}
}
