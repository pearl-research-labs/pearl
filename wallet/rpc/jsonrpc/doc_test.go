// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package jsonrpc

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONRPCReferenceListsMethods(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "docs", "user", "json-rpc.md"))
	require.NoError(t, err)
	text := string(doc)

	methods := serverMethods()
	for method, handlerData := range rpcHandlers {
		if handlerData.noHelp {
			methods[method] = struct{}{}
		}
	}
	for method := range methods {
		assert.Contains(t, text, "`"+method+"`")
	}
}
