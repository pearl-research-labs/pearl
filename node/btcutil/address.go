// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package btcutil

import (
	"bytes"
	"errors"
	"fmt"
	"strings"

	"github.com/pearl-research-labs/pearl/node/btcutil/bech32"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// encodeSegWitAddress creates a bech32m encoded address string from a
// witness version and witness program. Pearl only supports witness versions
// 1+ (Taproot, P2MR), all of which use bech32m encoding per BIP 350.
func encodeSegWitAddress(hrp string, witnessVersion byte, witnessProgram []byte) (string, error) {
	if witnessVersion < 1 || witnessVersion > 16 {
		return "", fmt.Errorf("unsupported witness version %d",
			witnessVersion)
	}

	converted, err := bech32.ConvertBits(witnessProgram, 8, 5, true)
	if err != nil {
		return "", err
	}

	combined := make([]byte, len(converted)+1)
	combined[0] = witnessVersion
	copy(combined[1:], converted)

	bech, err := bech32.EncodeM(hrp, combined)
	if err != nil {
		return "", err
	}

	version, program, err := decodeSegWitAddress(bech)
	if err != nil {
		return "", fmt.Errorf("invalid segwit address: %v", err)
	}

	if version != witnessVersion || !bytes.Equal(program, witnessProgram) {
		return "", fmt.Errorf("invalid segwit address")
	}

	return bech, nil
}

// Address is an interface type for any type of destination a transaction
// output may spend to. All Pearl addresses are witness-based (bech32m).
type Address interface {
	// String returns the string encoding of the address.
	String() string

	// EncodeAddress returns the bech32m string encoding of the address.
	EncodeAddress() string

	// ScriptAddress returns the raw bytes of the address to be used
	// when inserting the address into a txout's script.
	ScriptAddress() []byte

	// IsForNet returns whether or not the address is associated with the
	// passed network.
	IsForNet(*chaincfg.Params) bool

	// WitnessVersion returns the witness version of the address
	// (1 for Taproot, 2 for P2MR).
	WitnessVersion() byte

	// WitnessProgram returns the witness program of the address.
	// For Taproot this is the tweaked output key; for P2MR this is
	// the Merkle root of the script tree.
	WitnessProgram() []byte
}

// ErrWrongNetwork describes an error where an address is well formed but
// belongs to a different network than the one it was decoded for.
var ErrWrongNetwork = errors.New("address is for the wrong network")

// DecodeAddress decodes the string encoding of an address and returns
// the Address if addr is a valid encoding for a known address type that
// belongs to net. Taproot (witness version 1) and P2MR (witness version 2)
// are supported.
//
// The network of an address is determined from its bech32m human-readable
// part (HRP). An address with a different HRP than net is rejected with an
// error that wraps ErrWrongNetwork.
func DecodeAddress(addr string, net *chaincfg.Params) (Address, error) {
	decoded, err := decodeAddress(addr)
	if err != nil {
		return nil, err
	}

	if !decoded.IsForNet(net) {
		return nil, fmt.Errorf("%w: expected %s", ErrWrongNetwork, net.Name)
	}

	return decoded, nil
}

// decodeAddress decodes the address of any registered network. Only
// DecodeAddress may call it, otherwise the network check can be skipped.
func decodeAddress(addr string) (Address, error) {
	// Bech32 encoded segwit addresses start with a human-readable part
	// (hrp) followed by '1'. For mainnet the hrp is "bc", and for
	// testnet it is "tb". If the address string has a prefix that matches
	// one of the prefixes for the known networks, we try to decode it as
	// a segwit address.
	// Further reading: https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki
	oneIndex := strings.LastIndexByte(addr, '1')
	if oneIndex > 1 {
		prefix := addr[:oneIndex+1]
		if chaincfg.IsBech32SegwitPrefix(prefix) {
			witnessVer, witnessProg, err := decodeSegWitAddress(addr)
			if err != nil {
				return nil, err
			}

			// The HRP is everything before the found '1'.
			hrp := prefix[:len(prefix)-1]

			switch witnessVer {
			case 1:
				return newAddressTaproot(hrp, witnessProg)
			case 2:
				return newAddressMerkleRoot(hrp, witnessProg)
			default:
				return nil, fmt.Errorf("unsupported witness version: %d", witnessVer)
			}
		}
	}

	return nil, errors.New("only Taproot and P2MR addresses are supported")
}

// decodeSegWitAddress parses a bech32m encoded segwit address string and
// returns the witness version and witness program byte representation.
// Pearl only supports witness versions 1+ (bech32m). Version 0 (bech32)
// addresses are rejected.
func decodeSegWitAddress(address string) (byte, []byte, error) {
	_, data, bech32version, err := bech32.DecodeGeneric(address)
	if err != nil {
		return 0, nil, err
	}

	if len(data) < 1 {
		return 0, nil, fmt.Errorf("no witness version")
	}

	version := data[0]
	if version == 0 || version > 16 {
		return 0, nil, fmt.Errorf("unsupported witness version: %v",
			version)
	}

	// All supported versions use bech32m encoding (BIP 350).
	if bech32version != bech32.VersionM {
		return 0, nil, fmt.Errorf("invalid checksum: expected bech32m "+
			"encoding for witness version %d", version)
	}

	regrouped, err := bech32.ConvertBits(data[1:], 5, 8, false)
	if err != nil {
		return 0, nil, err
	}

	if len(regrouped) < 2 || len(regrouped) > 40 {
		return 0, nil, fmt.Errorf("invalid data length")
	}

	return version, regrouped, nil
}

// address is the shared implementation for all Pearl address types. Each
// concrete type (AddressTaproot, AddressMerkleRoot) embeds this and is
// distinguished only by the witness version.
type address struct {
	hrp            string
	witnessVersion byte
	witnessProgram []byte
}

// EncodeAddress returns the bech32m string encoding of the address.
func (a *address) EncodeAddress() string {
	str, err := encodeSegWitAddress(
		a.hrp, a.witnessVersion, a.witnessProgram,
	)
	if err != nil {
		return ""
	}
	return str
}

// ScriptAddress returns the raw bytes of the witness program to be used
// when inserting the address into a txout's script.
func (a *address) ScriptAddress() []byte {
	return a.witnessProgram[:]
}

// IsForNet returns whether the address is associated with the passed network,
// determined by comparing the human-readable part (HRP) of the bech32m
// encoding against the network's configured segwit HRP.
func (a *address) IsForNet(net *chaincfg.Params) bool {
	return a.hrp == net.Bech32HRPSegwit
}

// String returns the bech32m string encoding of the address.
func (a *address) String() string {
	return a.EncodeAddress()
}

// Hrp returns the human-readable part of the bech32m encoded address
// (e.g. "prl" for mainnet, "tprl" for testnet).
func (a *address) Hrp() string {
	return a.hrp
}

// WitnessVersion returns the witness version byte (1 for Taproot, 2 for P2MR).
func (a *address) WitnessVersion() byte {
	return a.witnessVersion
}

// WitnessProgram returns the 32-byte witness program. For Taproot this is
// the tweaked output key; for P2MR this is the Merkle root of the script tree.
func (a *address) WitnessProgram() []byte {
	return a.witnessProgram[:]
}

// newAddress creates an address with the given witness version and a
// 32-byte witness program.
func newAddress(hrp string, version byte, witnessProg []byte) (*address, error) {
	if len(witnessProg) != 32 {
		return nil, fmt.Errorf("witness program must be 32 bytes for "+
			"version %d", version)
	}
	return &address{
		hrp:            strings.ToLower(hrp),
		witnessVersion: version,
		witnessProgram: witnessProg,
	}, nil
}

// AddressTaproot is an Address for a pay-to-taproot (P2TR) output (SegWit v1).
type AddressTaproot struct{ *address }

// NewAddressTaproot returns a new AddressTaproot.
func NewAddressTaproot(witnessProg []byte,
	net *chaincfg.Params) (*AddressTaproot, error) {

	return newAddressTaproot(net.Bech32HRPSegwit, witnessProg)
}

func newAddressTaproot(hrp string,
	witnessProg []byte) (*AddressTaproot, error) {

	base, err := newAddress(hrp, 0x01, witnessProg)
	if err != nil {
		return nil, err
	}
	return &AddressTaproot{base}, nil
}

// AddressMerkleRoot is an Address for a pay-to-merkle-root (P2MR, BIP 360)
// output (SegWit v2).
type AddressMerkleRoot struct{ *address }

// NewAddressMerkleRoot returns a new AddressMerkleRoot.
func NewAddressMerkleRoot(witnessProg []byte,
	net *chaincfg.Params) (*AddressMerkleRoot, error) {

	return newAddressMerkleRoot(net.Bech32HRPSegwit, witnessProg)
}

func newAddressMerkleRoot(hrp string,
	witnessProg []byte) (*AddressMerkleRoot, error) {

	base, err := newAddress(hrp, 0x02, witnessProg)
	if err != nil {
		return nil, err
	}
	return &AddressMerkleRoot{base}, nil
}
