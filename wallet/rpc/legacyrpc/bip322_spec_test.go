package legacyrpc

import (
	"encoding/base64"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// The helpers below build the BIP-322 virtual transactions exactly as
// BIP-322 specifies, using only node primitives. They deliberately
// share no code with the implementation under test, so the tests in
// this file fail if the implementation's construction drifts from
// the spec in either direction.

func specBIP322VirtualTxs(t *testing.T, addr *btcutil.AddressTaproot,
	message string) (*wire.MsgTx, []byte) {

	t.Helper()

	// message_hash is the BIP340 tagged hash of the message with
	// tag "BIP0322-signed-message".
	msgHash := chainhash.TaggedHash(
		[]byte("BIP0322-signed-message"), []byte(message),
	)

	scriptPubKey, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	// to_spend: nVersion = 0, nLockTime = 0, one input with the null
	// prevout, sequence 0 and scriptSig = OP_0 PUSH32[message_hash],
	// one output paying 0 to the address's scriptPubKey.
	toSpend := wire.NewMsgTx(0)
	nullIn := wire.NewTxIn(
		&wire.OutPoint{Hash: chainhash.Hash{}, Index: 0xffffffff},
		nil, nil,
	)
	nullIn.Sequence = 0
	nullIn.SignatureScript = append(
		[]byte{txscript.OP_0, 0x20}, msgHash[:]...,
	)
	toSpend.AddTxIn(nullIn)
	toSpend.AddTxOut(wire.NewTxOut(0, scriptPubKey))

	// to_sign: nVersion = 0, nLockTime = 0, one input spending
	// to_spend:0 with sequence 0, one output of 0 with a bare
	// OP_RETURN scriptPubKey.
	toSign := wire.NewMsgTx(0)
	spendHash := toSpend.TxHash()
	signIn := wire.NewTxIn(wire.NewOutPoint(&spendHash, 0), nil, nil)
	signIn.Sequence = 0
	toSign.AddTxIn(signIn)
	toSign.AddTxOut(wire.NewTxOut(0, []byte{txscript.OP_RETURN}))

	return toSign, scriptPubKey
}

// specBIP322Sign produces a BIP-322 simple signature the way an
// external, spec-conformant wallet would.
func specBIP322Sign(t *testing.T, privKey *btcec.PrivateKey,
	addr *btcutil.AddressTaproot, message string) string {

	t.Helper()

	toSign, scriptPubKey := specBIP322VirtualTxs(t, addr, message)

	prevFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, 0)
	sigHashes := txscript.NewTxSigHashes(toSign, prevFetcher)
	witness, err := txscript.TaprootWitnessSignature(
		toSign, sigHashes, 0, 0, scriptPubKey,
		txscript.SigHashDefault, privKey,
	)
	require.NoError(t, err)

	witnessBytes, err := SerializeWitnessStack(witness)
	require.NoError(t, err)
	return base64.StdEncoding.EncodeToString(witnessBytes)
}

// specBIP322Verify checks a simple signature the way an external,
// spec-conformant verifier would: rebuild the spec virtual
// transactions and run the script engine.
func specBIP322Verify(t *testing.T, addr *btcutil.AddressTaproot,
	signature, message string) bool {

	t.Helper()

	sigBytes, err := base64.StdEncoding.DecodeString(signature)
	require.NoError(t, err)
	witness, err := ParseWitnessStack(sigBytes)
	require.NoError(t, err)

	toSign, scriptPubKey := specBIP322VirtualTxs(t, addr, message)
	toSign.TxIn[0].Witness = witness

	prevFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, 0)
	sigHashes := txscript.NewTxSigHashes(toSign, prevFetcher)
	vm, err := txscript.NewEngine(
		scriptPubKey, toSign, 0, txscript.StandardVerifyFlags,
		nil, sigHashes, 0, prevFetcher,
	)
	if err != nil {
		return false
	}
	return vm.Execute() == nil
}

func testBIP322KeyAddr(t *testing.T) (*btcec.PrivateKey,
	*btcutil.AddressTaproot) {

	t.Helper()
	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	addr, err := createTaprootAddress(privKey.PubKey(), &chaincfg.SimNetParams)
	require.NoError(t, err)
	return privKey, addr
}

// A signature produced by BIP322SignMessageSimple must verify under
// an independent BIP-322 construction, or no external BIP-322 tool
// can check the signatures this wallet hands to users.
func TestBIP322SignsVerifyUnderSpecConstruction(t *testing.T) {
	privKey, addr := testBIP322KeyAddr(t)

	for _, message := range []string{
		"Hello BIP-322",
		"",
		strings.Repeat("long message ", 20), // 260 bytes
	} {
		sig, err := BIP322SignMessageSimple(privKey, addr, message)
		require.NoError(t, err)
		require.True(t, specBIP322Verify(t, addr, sig, message),
			"signature for %q must verify under the BIP-322 construction",
			message)
		require.False(t, specBIP322Verify(t, addr, sig, message+"x"),
			"signature must not verify for a different message")
	}
}

// A signature produced by an external BIP-322 signer must verify
// here, or users cannot check signatures made by other wallets.
func TestBIP322VerifiesSpecConstructionSignatures(t *testing.T) {
	privKey, addr := testBIP322KeyAddr(t)

	for _, message := range []string{"Hello BIP-322", ""} {
		sig := specBIP322Sign(t, privKey, addr, message)
		valid, err := BIP322VerifyMessageSimple(addr, sig, message)
		require.NoError(t, err)
		require.True(t, valid,
			"spec-constructed signature for %q must verify", message)

		valid, err = BIP322VerifyMessageSimple(addr, sig, message+"x")
		require.NoError(t, err)
		require.False(t, valid)
	}
}

// BIP-322 commits to a 32-byte hash of the message, so message
// length is unbounded. Signing must not fail just because a
// message is longer than the 80-byte data-carrier limit that
// applies to OP_RETURN outputs.
func TestBIP322SignsLongMessages(t *testing.T) {
	privKey, addr := testBIP322KeyAddr(t)

	message := strings.Repeat("a", 500)
	sig, err := BIP322SignMessageSimple(privKey, addr, message)
	require.NoError(t, err, "signing a 500-byte message should succeed")
	require.NotEmpty(t, sig)

	valid, err := BIP322VerifyMessageSimple(addr, sig, message)
	require.NoError(t, err)
	require.True(t, valid)
}
