// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// allMessageCommands is every command makeEmptyMessage understands. Add new
// commands here so they are covered by TestMessageCapsFitProtocolLimit.
var allMessageCommands = []string{
	CmdVersion, CmdVerAck, CmdGetAddr, CmdAddr, CmdGetBlocks, CmdInv,
	CmdGetData, CmdNotFound, CmdBlock, CmdTx, CmdGetHeaders, CmdHeaders,
	CmdPing, CmdPong, CmdMemPool, CmdFilterAdd, CmdFilterClear, CmdFilterLoad,
	CmdMerkleBlock, CmdReject, CmdSendHeaders, CmdFeeFilter, CmdGetCFilters,
	CmdGetCFHeaders, CmdGetCFCheckpt, CmdCFilter, CmdCFHeaders, CmdCFCheckpt,
	CmdWTxIdRelay,
}

// allCertificateVersions is every certificate version this build can place on
// the wire. Add new versions here so the size assertions cover them.
var allCertificateVersions = []CertificateVersion{
	CertificateVersionV1, CertificateVersionV2, CertificateVersionV3,
	CertificateVersionV4,
}

// TestMessageCapsFitProtocolLimit asserts that no message type declares a
// per-type payload cap larger than the protocol-wide message limit. A type that
// does is unreadable at its own declared maximum: the read path rejects the
// message on MaxProtocolMessageLength before the per-type cap is consulted, so
// legitimate traffic at that size can never be exchanged.
func TestMessageCapsFitProtocolLimit(t *testing.T) {
	for _, command := range allMessageCommands {
		t.Run(command, func(t *testing.T) {
			msg, err := makeEmptyMessage(command)
			if err != nil {
				t.Skipf("makeEmptyMessage(%q): %v", command, err)
			}

			if declared := msg.MaxPayloadLength(ProtocolVersion); declared > MaxProtocolMessageLength {
				t.Errorf("%s declares a %d-byte payload cap, which exceeds the "+
					"%d-byte protocol message limit; a message of that size can "+
					"never be sent or received", command, declared,
					uint32(MaxProtocolMessageLength))
			}
		})
	}
}

// TestBlockCapHasRoomForEveryCertificateVersion asserts that the BLOCK message
// cap leaves room for a certificate of any version on top of the consensus
// transaction payload. Certificates are excluded from block vsize, so that room
// has to be budgeted explicitly rather than assumed to be inside
// MaxBlockPayload. A version whose certificate does not fit makes
// consensus-valid blocks unrelayable.
func TestBlockCapHasRoomForEveryCertificateVersion(t *testing.T) {
	declared := int((&MsgBlock{}).MaxPayloadLength(ProtocolVersion))
	room := declared - MaxBlockPayload

	for _, version := range allCertificateVersions {
		needed := MaxCertificateSize(version)
		if room < needed {
			t.Errorf("BLOCK declares a %d-byte cap, leaving %d bytes above the "+
				"%d-byte transaction payload, but a version-%d certificate can "+
				"be %d bytes; a maximum-size block carrying one would be "+
				"rejected on receipt", declared, room, MaxBlockPayload, version,
				needed)
		}
	}
}

// TestFp8ProofSizeMatchesRust asserts that the Go and Rust copies of the FP8
// blob limit agree. Go bounds what the node accepts off the wire; Rust bounds
// what the verifier will process. If they drift, the node accepts certificates
// its own verifier refuses, or rejects ones that would have verified. The
// relationship is currently asserted only in a comment.
func TestFp8ProofSizeMatchesRust(t *testing.T) {
	const rustPath = "../../zk-pow/bindings/go/src/common.rs"

	source, err := os.ReadFile(rustPath)
	if err != nil {
		t.Fatalf("reading %s: %v", rustPath, err)
	}

	matches := regexp.MustCompile(
		`MAX_FP8_PROOF_SIZE\s*:\s*usize\s*=\s*(\d+)`,
	).FindSubmatch(source)
	if matches == nil {
		t.Fatalf("MAX_FP8_PROOF_SIZE not found in %s; if it moved, update this "+
			"test rather than deleting it", rustPath)
	}

	rustValue, err := strconv.Atoi(string(matches[1]))
	if err != nil {
		t.Fatalf("parsing MAX_FP8_PROOF_SIZE from %s: %v", rustPath, err)
	}

	if rustValue != MaxFp8ProofSize {
		t.Errorf("MaxFp8ProofSize is %d in Go but MAX_FP8_PROOF_SIZE is %d in "+
			"%s; the node and the verifier disagree on what they accept",
			MaxFp8ProofSize, rustValue, rustPath)
	}
}
