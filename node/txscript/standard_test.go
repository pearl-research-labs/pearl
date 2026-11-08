// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// mustParseShortForm parses the passed short form script and returns the
// resulting bytes.  It panics if an error occurs.  This is only used in the
// tests as a helper since the only way it can fail is if there is an error in
// the test source code.
func mustParseShortForm(script string) []byte {
	s, err := parseShortForm(script)
	if err != nil {
		panic("invalid short form script in test source: err " +
			err.Error() + ", script: " + script)
	}

	return s
}

// newAddressTaproot returns a new btcutil.AddressTaproot from the
// provided hash.  It panics if an error occurs.  This is only used in the tests
// as a helper since the only way it can fail is if there is an error in the
// test source code.
func newAddressTaproot(scriptHash []byte) btcutil.Address {
	addr, err := btcutil.NewAddressTaproot(scriptHash,
		&chaincfg.MainNetParams)
	if err != nil {
		panic("invalid script hash in test source")
	}

	return addr
}

// TestExtractPkScriptAddrs ensures that extracting the type, addresses, and
// number of required signatures from PkScripts works as intended.
func TestExtractPkScriptAddrs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		script  []byte
		addrs   []btcutil.Address
		reqSigs int
		class   ScriptClass
	}{
		{
			name: "valid signature from a sigscript - no addresses",
			script: hexToBytes("47304402204e45e16932b8af514961a1d" +
				"3a1a25fdf3f4f7732e9d624c6c61548ab5fb8cd41022" +
				"0181522ec8eca07de4860a4acdd12909d831cc56cbba" +
				"c4622082221a8768d1d0901"),
			addrs:   nil,
			reqSigs: 0,
			class:   NonStandardTy,
		},
		{
			name: "v1 p2tr witness-script-hash",
			script: hexToBytes("51201a82f7457a9ba6ab1074e9f50" +
				"053eefc637f8b046e389b636766bdc7d1f676f8"),
			addrs: []btcutil.Address{newAddressTaproot(
				hexToBytes("1a82f7457a9ba6ab1074e9f50053eefc6" +
					"37f8b046e389b636766bdc7d1f676f8"))},
			reqSigs: 1,
			class:   WitnessV1TaprootTy,
		},
		{
			name:    "empty script",
			script:  []byte{},
			addrs:   nil,
			reqSigs: 0,
			class:   NonStandardTy,
		},
		{
			name:    "script that does not parse",
			script:  []byte{OP_DATA_45},
			addrs:   nil,
			reqSigs: 0,
			class:   NonStandardTy,
		},
	}

	t.Logf("Running %d tests.", len(tests))
	for i, test := range tests {
		class, addrs, reqSigs, err := ExtractPkScriptAddrs(
			test.script, &chaincfg.MainNetParams)
		if err != nil {
		}

		if !reflect.DeepEqual(addrs, test.addrs) {
			t.Errorf("ExtractPkScriptAddrs #%d (%s) unexpected "+
				"addresses\ngot  %v\nwant %v", i, test.name,
				addrs, test.addrs)
			continue
		}

		if reqSigs != test.reqSigs {
			t.Errorf("ExtractPkScriptAddrs #%d (%s) unexpected "+
				"number of required signatures - got %d, "+
				"want %d", i, test.name, reqSigs, test.reqSigs)
			continue
		}

		if class != test.class {
			t.Errorf("ExtractPkScriptAddrs #%d (%s) unexpected "+
				"script type - got %s, want %s", i, test.name,
				class, test.class)
			continue
		}
	}
}

// bogusAddress implements the btcutil.Address interface so the tests can ensure
// unsupported address types are handled properly.
type bogusAddress struct{}

// EncodeAddress simply returns an empty string.  It exists to satisfy the
// btcutil.Address interface.
func (b *bogusAddress) EncodeAddress() string {
	return ""
}

// ScriptAddress simply returns an empty byte slice.  It exists to satisfy the
// btcutil.Address interface.
func (b *bogusAddress) ScriptAddress() []byte {
	return nil
}

// IsForNet lies blatantly to satisfy the btcutil.Address interface.
func (b *bogusAddress) IsForNet(chainParams *chaincfg.Params) bool {
	return true // why not?
}

// String simply returns an empty string.  It exists to satisfy the
// btcutil.Address interface.
func (b *bogusAddress) String() string {
	return ""
}

func (b *bogusAddress) WitnessVersion() byte   { return 0xff }
func (b *bogusAddress) WitnessProgram() []byte { return nil }

// TestPayToAddrScript ensures the PayToAddrScript function generates the
// correct scripts for the various types of addresses.
func TestPayToAddrScript(t *testing.T) {
	t.Parallel()

	p2tr, err := btcutil.NewAddressTaproot(hexToBytes("3a8e170b546c3b122ab9c175e"+
		"ff36fb344db2684fe96497eb51b440e75232709"), &chaincfg.MainNetParams)
	if err != nil {
		t.Fatalf("Unable to create p2tr address: %v",
			err)
	}

	// Errors used in the tests below defined here for convenience and to
	// keep the horizontal test size shorter.
	errUnsupportedAddress := scriptError(ErrUnsupportedAddress, "")

	tests := []struct {
		in       btcutil.Address
		expected string
		err      error
	}{
		// pay-to-taproot address on mainnet.
		{
			p2tr,
			"OP_1 DATA_32 0x3a8e170b546c3b122ab9c175eff36fb344db2684" +
				"fe96497eb51b440e75232709",
			nil,
		},

		// Supported address types with nil pointers.
		{(*btcutil.AddressTaproot)(nil), "", errUnsupportedAddress},

		// Unsupported address type.
		{&bogusAddress{}, "", errUnsupportedAddress},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		pkScript, err := PayToAddrScript(test.in)
		if e := tstCheckScriptError(err, test.err); e != nil {
			t.Errorf("PayToAddrScript #%d unexpected error - "+
				"got %v, want %v", i, err, test.err)
			continue
		}

		expected := mustParseShortForm(test.expected)
		if !bytes.Equal(pkScript, expected) {
			t.Errorf("PayToAddrScript #%d got: %x\nwant: %x",
				i, pkScript, expected)
			continue
		}
	}
}

// scriptClassTests houses several test scripts used to ensure various class
// determination is working as expected.  It's defined as a test global versus
// inside a function scope since this spans both the standard tests and the
// consensus tests (pay-to-script-hash is part of consensus).
var scriptClassTests = []struct {
	name   string
	script string
	class  ScriptClass
}{
	// All unsupported scripts (non-Taproot) should resolve to NonStandardTy
	{
		name: "Pay Pubkey",
		script: "DATA_65 0x0411db93e1dcdb8a016b49840f8c53bc1eb68a382e" +
			"97b1482ecad7b148a6909a5cb2e0eaddfb84ccf9744464f82e16" +
			"0bfa9b8b64f9d4c03f999b8643f656b412a3 CHECKSIG",
		class: NonStandardTy,
	},
	// tx 599e47a8114fe098103663029548811d2651991b62397e057f0c863c2bc9f9ea
	{
		name: "Pay PubkeyHash",
		script: "DUP HASH160 DATA_20 0x660d4ef3a743e3e696ad990364e555" +
			"c271ad504b EQUALVERIFY CHECKSIG",
		class: NonStandardTy,
	},
	// part of tx 6d36bc17e947ce00bb6f12f8e7a56a1585c5a36188ffa2b05e10b4743273a74b
	// codeseparator parts have been elided. (bitcoin core's checks for
	// multisig type doesn't have codesep either).
	{
		name: "multisig",
		script: "1 DATA_33 0x0232abdc893e7f0631364d7fd01cb33d24da4" +
			"5329a00357b3a7886211ab414d55a 1 CHECKMULTISIG",
		class: NonStandardTy,
	},
	// tx e5779b9e78f9650debc2893fd9636d827b26b4ddfa6a8172fe8708c924f5c39d
	{
		name: "P2SH",
		script: "HASH160 DATA_20 0x433ec2ac1ffa1b7b7d027f564529c57197f" +
			"9ae88 EQUAL",
		class: NonStandardTy,
	},

	{
		// Nulldata with no data at all.
		name:   "nulldata no data",
		script: "RETURN",
		class:  NullDataTy,
	},
	{
		// Nulldata with single zero push.
		name:   "nulldata zero",
		script: "RETURN 0",
		class:  NullDataTy,
	},
	{
		// Nulldata with small integer push.
		name:   "nulldata small int",
		script: "RETURN 1",
		class:  NullDataTy,
	},
	{
		// Nulldata with max small integer push.
		name:   "nulldata max small int",
		script: "RETURN 16",
		class:  NullDataTy,
	},
	{
		// Nulldata with small data push.
		name:   "nulldata small data",
		script: "RETURN DATA_8 0x046708afdb0fe554",
		class:  NullDataTy,
	},
	{
		// Canonical nulldata with 60-byte data push.
		name: "canonical nulldata 60-byte push",
		script: "RETURN 0x3c 0x046708afdb0fe5548271967f1a67130b7105cd" +
			"6a828e03909a67962e0ea1f61deb649f6bc3f4cef3046708afdb" +
			"0fe5548271967f1a67130b7105cd6a",
		class: NullDataTy,
	},
	{
		// Non-canonical nulldata with 60-byte data push.
		name: "non-canonical nulldata 60-byte push",
		script: "RETURN PUSHDATA1 0x3c 0x046708afdb0fe5548271967f1a67" +
			"130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef3" +
			"046708afdb0fe5548271967f1a67130b7105cd6a",
		class: NullDataTy,
	},
	{
		// Nulldata with max allowed data to be considered standard.
		name: "nulldata max standard push",
		script: "RETURN PUSHDATA1 0x50 0x046708afdb0fe5548271967f1a67" +
			"130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef3" +
			"046708afdb0fe5548271967f1a67130b7105cd6a828e03909a67" +
			"962e0ea1f61deb649f6bc3f4cef3",
		class: NullDataTy,
	},
	{
		// Nulldata with more than the old 80-byte limit. Any OP_RETURN
		// is now classified as NullData regardless of push size.
		name: "nulldata large push",
		script: "RETURN PUSHDATA1 0x51 0x046708afdb0fe5548271967f1a67" +
			"130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef3" +
			"046708afdb0fe5548271967f1a67130b7105cd6a828e03909a67" +
			"962e0ea1f61deb649f6bc3f4cef308",
		class: NullDataTy,
	},
	{
		// OP_RETURN followed by multiple opcodes is still NullData —
		// any script starting with OP_RETURN is provably unspendable.
		name:   "nulldata multiple opcodes",
		script: "RETURN 4 TRUE",
		class:  NullDataTy,
	},

	// The next few are almost multisig (it is the more complex script type)
	// but with various changes to make it fail.
	{
		// Multisig but invalid nsigs.
		name: "strange 1",
		script: "DUP DATA_33 0x0232abdc893e7f0631364d7fd01cb33d24da45" +
			"329a00357b3a7886211ab414d55a 1 CHECKMULTISIG",
		class: NonStandardTy,
	},
	{
		// Multisig but invalid pubkey.
		name:   "strange 2",
		script: "1 1 1 CHECKMULTISIG",
		class:  NonStandardTy,
	},
	{
		// Multisig but no matching npubkeys opcode.
		name: "strange 3",
		script: "1 DATA_33 0x0232abdc893e7f0631364d7fd01cb33d24da4532" +
			"9a00357b3a7886211ab414d55a DATA_33 0x0232abdc893e7f0" +
			"631364d7fd01cb33d24da45329a00357b3a7886211ab414d55a " +
			"CHECKMULTISIG",
		class: NonStandardTy,
	},
	{
		// Multisig but with multisigverify.
		name: "strange 4",
		script: "1 DATA_33 0x0232abdc893e7f0631364d7fd01cb33d24da4532" +
			"9a00357b3a7886211ab414d55a 1 CHECKMULTISIGVERIFY",
		class: NonStandardTy,
	},
	{
		// Multisig but wrong length.
		name:   "strange 5",
		script: "1 CHECKMULTISIG",
		class:  NonStandardTy,
	},
	{
		name:   "doesn't parse",
		script: "DATA_5 0x01020304",
		class:  NonStandardTy,
	},
	{
		name: "multisig script with wrong number of pubkeys",
		script: "2 " +
			"DATA_33 " +
			"0x027adf5df7c965a2d46203c781bd4dd8" +
			"21f11844136f6673af7cc5a4a05cd29380 " +
			"DATA_33 " +
			"0x02c08f3de8ee2de9be7bd770f4c10eb0" +
			"d6ff1dd81ee96eedd3a9d4aeaf86695e80 " +
			"3 CHECKMULTISIG",
		class: NonStandardTy,
	},
	{
		// A pay to witness pub key hash pk script.
		name:   "Pay To Witness PubkeyHash",
		script: "0 DATA_20 0x1d0f172a0ecb48aee1be1f2687d2963ae33f71a1",
		class:  NonStandardTy,
	},
	{
		// A pay to witness scripthash pk script.
		name:   "Pay To Witness Scripthash",
		script: "0 DATA_32 0x9f96ade4b41d5433f4eda31e1738ec2b36f6e7d1420d94a6af99801a88f7f7ff",
		class:  NonStandardTy,
	},
	{
		name:   "Pay To Taproot",
		script: "1 DATA_32 0xef46d1aa78101e3350600a5d36045ba97c2670daa91e9f3a48c43c6e739754e6",
		class:  WitnessV1TaprootTy,
	},
}

// TestScriptClass ensures all the scripts in scriptClassTests have the expected
// class.
func TestScriptClass(t *testing.T) {
	t.Parallel()

	for _, test := range scriptClassTests {
		script := mustParseShortForm(test.script)
		class := GetScriptClass(script)
		if class != test.class {
			t.Errorf("%s: expected %s got %s (script %x)", test.name,
				test.class, class, script)
			continue
		}
	}
}

// TestNullDataScript tests whether NullDataScript returns a valid script.
func TestNullDataScript(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected []byte
		err      error
		class    ScriptClass
	}{
		{
			name:     "small int",
			data:     hexToBytes("01"),
			expected: mustParseShortForm("RETURN 1"),
			err:      nil,
			class:    NullDataTy,
		},
		{
			name:     "max small int",
			data:     hexToBytes("10"),
			expected: mustParseShortForm("RETURN 16"),
			err:      nil,
			class:    NullDataTy,
		},
		{
			name: "data of size before OP_PUSHDATA1 is needed",
			data: hexToBytes("0102030405060708090a0b0c0d0e0f10111" +
				"2131415161718"),
			expected: mustParseShortForm("RETURN 0x18 0x01020304" +
				"05060708090a0b0c0d0e0f101112131415161718"),
			err:   nil,
			class: NullDataTy,
		},
		{
			name: "just right",
			data: hexToBytes("000102030405060708090a0b0c0d0e0f101" +
				"112131415161718191a1b1c1d1e1f202122232425262" +
				"728292a2b2c2d2e2f303132333435363738393a3b3c3" +
				"d3e3f404142434445464748494a4b4c4d4e4f"),
			expected: mustParseShortForm("RETURN PUSHDATA1 0x50 " +
				"0x000102030405060708090a0b0c0d0e0f101112131" +
				"415161718191a1b1c1d1e1f20212223242526272829" +
				"2a2b2c2d2e2f303132333435363738393a3b3c3d3e3" +
				"f404142434445464748494a4b4c4d4e4f"),
			err:   nil,
			class: NullDataTy,
		},
		{
			name: "too big",
			data: hexToBytes("000102030405060708090a0b0c0d0e0f101" +
				"112131415161718191a1b1c1d1e1f202122232425262" +
				"728292a2b2c2d2e2f303132333435363738393a3b3c3" +
				"d3e3f404142434445464748494a4b4c4d4e4f50"),
			expected: nil,
			err:      scriptError(ErrTooMuchNullData, ""),
			class:    NonStandardTy,
		},
	}

	for i, test := range tests {
		script, err := NullDataScript(test.data)
		if e := tstCheckScriptError(err, test.err); e != nil {
			t.Errorf("NullDataScript: #%d (%s): %v", i, test.name,
				e)
			continue

		}

		// Check that the expected result was returned.
		if !bytes.Equal(script, test.expected) {
			t.Errorf("NullDataScript: #%d (%s) wrong result\n"+
				"got: %x\nwant: %x", i, test.name, script,
				test.expected)
			continue
		}

		// Check that the script has the correct type.
		scriptType := GetScriptClass(script)
		if scriptType != test.class {
			t.Errorf("GetScriptClass: #%d (%s) wrong result -- "+
				"got: %v, want: %v", i, test.name, scriptType,
				test.class)
			continue
		}
	}
}

// TestNewScriptClass tests whether NewScriptClass returns a valid ScriptClass.
func TestNewScriptClass(t *testing.T) {
	tests := []struct {
		name       string
		scriptName string
		want       *ScriptClass
		wantErr    error
	}{
		{
			name:       "NewScriptClass - ok",
			scriptName: NullDataTy.String(),
			want: func() *ScriptClass {
				s := NullDataTy
				return &s
			}(),
		},
		{
			name:       "NewScriptClass - invalid",
			scriptName: "foo",
			wantErr:    ErrUnsupportedScriptType,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewScriptClass(tt.scriptName)
			if err != nil && !errors.Is(err, tt.wantErr) {
				t.Errorf("NewScriptClass() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewScriptClass() got = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestScriptClassString ensures String names every defined class and
// returns "Invalid" — without panicking — for out-of-range values,
// including the exact boundary len(scriptClassToName).
func TestScriptClassString(t *testing.T) {
	for class, want := range scriptClassToName {
		if got := ScriptClass(class).String(); got != want {
			t.Errorf("ScriptClass(%d).String() = %q, want %q", class, got, want)
		}
	}
	for _, class := range []ScriptClass{
		ScriptClass(len(scriptClassToName)),
		ScriptClass(len(scriptClassToName) + 1),
		ScriptClass(255),
	} {
		if got := class.String(); got != "Invalid" {
			t.Errorf("ScriptClass(%d).String() = %q, want %q", byte(class), got, "Invalid")
		}
	}
}
