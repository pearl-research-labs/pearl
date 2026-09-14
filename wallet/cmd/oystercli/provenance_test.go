// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResolutionStory(t *testing.T) {

	cfg := &config{
		AppData: t.TempDir(),
		Connect: "localhost:44207",
		NoTLS:   true,
	}
	cfg.activeNet = mainNetForTest()
	cfg.src = sources{
		network: "default",
		appData: "--appdata",
		conf:    "not found",
		creds:   "none found",
		tls:     "--notls",
		connect: "default mainnet port",
	}

	story := resolutionStory(cfg)

	assert.Contains(t, story, "credentials")
	assert.Contains(t, story, "none found")
	assert.Contains(t, story, cfg.oysterConfPath())
	assert.Contains(t, story, "not found")
	assert.Contains(t, story, cfg.walletDBPath())
	assert.Contains(t, story, "missing")
	// TLS is off, so no certificate row.
	assert.NotContains(t, story, "certificate")
}
