// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package version provides the canonical version for all Pearl binaries.
// All components in the monorepo delegate to this package so the version
// is defined in a single place.
package version

import (
	"bytes"
	"fmt"
	"strings"
)

const semanticAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"

// These constants define the application version and follow the semantic
// versioning 2.0.0 spec (http://semver.org/).
const (
	Major uint = 1
	Minor uint = 0
	Patch uint = 3

	// PreRelease MUST only contain characters from semanticAlphabet
	// per the semantic versioning spec.
	PreRelease = ""
)

// Build may be overridden at link time via
// -ldflags "-X github.com/pearl-research-labs/pearl/version.Build=<value>".
var Build string

// Version returns the application version as a properly formed string per the
// semantic versioning 2.0.0 spec (http://semver.org/).
func Version() string {
	v := fmt.Sprintf("%d.%d.%d", Major, Minor, Patch)

	if preRelease := normalizeVerString(PreRelease); preRelease != "" {
		v = fmt.Sprintf("%s-%s", v, preRelease)
	}

	if build := normalizeVerString(Build); build != "" {
		v = fmt.Sprintf("%s+%s", v, build)
	}

	return v
}

// normalizeVerString returns the passed string stripped of all characters which
// are not valid according to the semantic versioning guidelines for pre-release
// version and build metadata strings.  In particular they MUST only contain
// characters in semanticAlphabet.
func normalizeVerString(str string) string {
	var result bytes.Buffer
	for _, r := range str {
		if strings.ContainsRune(semanticAlphabet, r) {
			result.WriteRune(r)
		}
	}
	return result.String()
}
