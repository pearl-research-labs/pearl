// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build xmss

package wallet

// On-chain proof artifact for a P2MR + XMSS round trip on Pearl mainnet:
// the spending tx is unlocked by an XMSS signature against seed A and
// pays to a P2MR pkScript committing to a second XMSS leaf derived from
// seed B, so both ends are post-quantum.
//
// The file is filled in by the operator in three passes:
//
//	Pass 1 (pre-funding): set the four xmss*SeedHex constants and
//	    mainnetFee, then `go test -tags xmss -run TestP2MRXMSSDeriveAddress`
//	    to print the funding (A) and destination (B) addresses.
//
//	Pass 2 (post-funding): send to address A from oyster, set the
//	    mainnetFunding* constants from the confirmed UTXO, then
//	    `go test -tags xmss -run TestP2MRXMSSBuildSpend` to print the
//	    raw spend hex; broadcast it via `oyster sendrawtransaction`.
//
//	Pass 3 (post-broadcast): set the four expected* constants from
//	    the mined tx. The committed test then enforces byte-exact
//	    reproducibility forever.
//
// Each XMSS keypair is single-use: msgUID=0 over two distinct sighashes
// leaks the seed. Seed A signs the spend once; seed B is committed to
// on chain only as part of the destination pkScript. Neither may be
// reused for any other sighash.

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/wallet/xmsstxn"
	"github.com/pearl-research-labs/pearl/xmss"
	"github.com/stretchr/testify/require"
)

// Pass 1: seeds for the funding (A) and destination (B) P2MR-XMSS
// addresses, plus the absolute fee (grains). Seed pairs MUST be
// distinct from each other and from any seed used elsewhere.
const (
	// Seed A: bytes 0x00–0x3f (private) and 0x40–0x5f (public).
	xmssAPrivSeedHex = "000102030405060708090a0b0c0d0e0f" +
		"101112131415161718191a1b1c1d1e1f" +
		"202122232425262728292a2b2c2d2e2f" +
		"303132333435363738393a3b3c3d3e3f"
	xmssAPubSeedHex = "404142434445464748494a4b4c4d4e4f" +
		"505152535455565758595a5b5c5d5e5f"

	// Seed B: bytes 0x80–0xbf (private) and 0xc0–0xdf (public).
	xmssBPrivSeedHex = "808182838485868788898a8b8c8d8e8f" +
		"909192939495969798999a9b9c9d9e9f" +
		"a0a1a2a3a4a5a6a7a8a9aaabacadaeaf" +
		"b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
	xmssBPubSeedHex = "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf" +
		"d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"

	mainnetFee = int64(50_000) // grains; ~670 vB XMSS spend
)

// Pass 2: the confirmed funding UTXO. fundingValue must match the
// on-chain output exactly -- it is hashed into the BIP-341 sighash.
const (
	mainnetFundingTxID  = "fb7b5dfd0e9a2ff7a86ec89a697b611b5e0e98fb02e6f33ab44d32967875356d"
	mainnetFundingVout  = uint32(1)            // vout 1: P2MR output; vout 0 is the P2TR change
	mainnetFundingValue = int64(1_000_000_000) // 10 PRL, confirmed in block 56303
)

// Pass 3: the on-chain artifacts asserted byte-exact for reproducibility.
const (
	expectedP2MRAddressA = "prl1zjrmn44f7n9cqekf9h4f2wl5hgp3apn0a2t7l09gcyc580gqsxm3ss67e3p"
	expectedP2MRAddressB = "prl1zxxlkld5f4c723urfqemt3mzv9h8z6kf8yrxd8txa3fv2l0r6hkhqhryssk"
	expectedSpendTxID    = "1fc2168472af90dfbcd9cd6392b914c40150ce5eedaefa5d0bd60a02fbaa2b53"
	// Mined in block 56313. Input: P2MR_A (10 PRL, XMSS-signed). Output: P2MR_B (9.9995 PRL).
	// Witness: 2340-byte XMSS signature split into 5×468-byte chunks, leaf script, control block.
	expectedSpendTxHex = "020000000001016d35757896324db43af3e602fb980e5e1b617b699ac86ea8f72f9a0efd5d7bfb" +
		"0100000000ffffffff01b0069a3b0000000022522031bf6fb689ae3ca8f0690676b8ec4c2dce2d59272" +
		"0ccd3acdd8a58afbc7abdae07fdd40100000000180b7c7b62823b7db1b25e6daeea27cdaa01759f50f6" +
		"a99e097b7a3be1855ba21e8bbeb5aac2e73539b7bca95c904277e21ed1d11557f85bcc11666300a255a" +
		"9c0ab39d3a832a4d57b8dc2d026262d3820f2f95d555f02d759b82936034bf57345d4f06007557a7419" +
		"194d151e61f7aeb9c8459d422a633db0ce805bffaf5606a75bc4fd040c660ab75e42c9735cbaa149b7e" +
		"dc5b10f673aa246c3a97408a262bc9f824a54eacc3ad2d12a7fa3de63db900b277fd2ea673f22caf0aa" +
		"a45b464e54300815c28642f4de8f6170a37d7ff1e548d1415276d98a2b23d2d134ea2d76646d8707320" +
		"644afd6c96e3ea2172bdad47866f94b57e25dd7cb84b5065b7413eda42615920f6fe8316f39b41b215e" +
		"66c2a43ef35982df863f443487e360d1f62f753271f96b988d070710bcabff7ca0597864a468ff1436f" +
		"ed10bd740f93703870ec6feb75db0cc8fcb8fe4b7bcc0dfebfc7d0fd271f2dbf6056209c5427ef6e721" +
		"642d71e313d6249ad6b94b1fbbf172b43cc16bdd80cc1761e7fdc9af11028e6135149a1b85952c46fd5" +
		"2c1f02cc2c7e5a7c7bbcbb43351b38c2df0152c94de4e88cba4699647bab5a1912a84e52fa739aedf61" +
		"fb616f5db433ff02c45ee82b9129ae782e9627ffc716cb977c4858fdd40184837066a4f7c11edd26678" +
		"834d697bb1c82515431e78f58f980f69b4180a23c5d378aa0137fe8861f476c90aae9275c91df760a71" +
		"f4f11c5b88cfb0861014b709f0d0c9634878494709c0d420987cbb0e0eb346458d8154f6afd4060e281" +
		"6e6701aeebc08653c280c83facc2978a63a828e4a8adabffe6e33369b0bf2030160a1f4021d85d40efd" +
		"cc5ba9fd6074082a985de4cd82922f8de2d093e813183651ded943136e3cb36237b6654f05aad13000d" +
		"c2764d3a91abdaa62d12c201cf2ec2dcf35bcb79eaca358e329a6e95e0f3067e76e6491847971846784" +
		"5717533205d1dd54679b155e42004cf9471d586c375abc5a9761fa852a3ee90a1108db959feb0aff4df" +
		"edfc50df0f11aac959da216a115e82c14a5bdab5dcba6311c63b3d57ef708bd08c05fe50e1b05135f94" +
		"264dce8d037e4ffd2d40b7ace51cd34fd22f581aa8882b235cb619c05ee46af62aee3a722a0010cd3c4" +
		"bd59b7d764caf9c898fd184ec129ae27544b467feee02d5db75cf1bdd1bef5cf82802061b317c009c24" +
		"76443459e1add5e4fb4f7062dbc8609372d8ee8db7a09f94e0b7001d5491041f5f6ac03d8c54b538eba" +
		"e26409b7cc003a422a454fe67e7b56ff361a6ceaa5a790a94a874d7103a0805a19db518a06d11a8e1b6" +
		"fdd4015a969316a1a1610b505ca43f2c00460a77c49612b137c0e939a645a1ee365badd023c434da884" +
		"f515f9774b981a86ce8c22e73e98b3d1b062db23caa3509af869773f3b27a9582f6c596c37b3c9c3ab4" +
		"3b68415bf777a198fb528743cd495a07cb64bcea892807cfab44f37c5e0b88853c5a9fe701ec515be74" +
		"725332e8cad64a9809d81b488dbedf8447024e592813645eb5f3405a1310a49905bf29983cb77e859829" +
		"670b8b755478ae166beae6369e1f7878e644399f10beac23e10270a7bed2adf00cb7d86e36e69a66b51" +
		"5b64cf277002f95dbfc0d6040ec9e088e608bbd5b9726f01b86ee58b0de7ed4a1434159e9abf81ea9d4" +
		"13c653ccc9e2abbdd389481350a003fc9649fa6f531b80f02408285d9a7cf40417bb37d1d74e8b23116" +
		"d2cad1fff4926b7dbaaaf7cf0aed12153607005a365fbbb9e26ecee0099023880d190d59a249929133b" +
		"989619c92dd664ab0242eed643f50b8ab48c9a0d923007817dc868ff808c5a9b35468dc1dc0556d6128" +
		"65c6c474698a53f0ef7a63414a4bdc6c44156faffa39981433193922528c21091de9bedee885b23f86f" +
		"cbd8436653e2da09ff0ed70bc4b9430fd13515c154506509cea4018de35b9d05bb4fc5125ebfe302a43" +
		"e2efe166b92878000d26b6f06da7fdd40193c9d29fc80fbf335024c52f1aaae2850b25ccfd5142e6430" +
		"e26da5a9704a7e9c4d9885c2041d16e838365465e3995a38681c5ba33bbc2845063ab64df60d2a00742" +
		"a3a5c0fdecc1edeb80c3626cb5496d33474202dcbea23fa006c844385ca85840a57dd724e494ca94a2b" +
		"56e202b8b33e149803678aaf4419786546978798336650132e6fce577b244e1e876f8028e7294d0a489" +
		"066fb44caab85963b39c076df605b27ec371f8b4edb795778dd2b3828e476c129e5199f045580bdae42" +
		"90a0d074a23285fac88522d4a511340466994eb1c89cf909e46829a4a981083b702bbad0b8d0fdf86737" +
		"8c6285aaa2e83f76976f4188ba4c2e2d27ef52b00d37fc4cdb7930b585cdfb1c0758135599631fffd6a" +
		"c0b3c8b97a29a943a19818b351a6daa1da791572597f0e3738628a5c937918ffc170b420b8878e32f7f" +
		"164a5a541dd72cfd5598eb6f2c35785a58fb8782813b6e465df57ae9255fb09439f4cd70abb147676de" +
		"f99c8b086cdf6f43fe009cfeb8758a095d6f6fadb44d965699a7356f3cec0df3f69988f42faa6303c83" +
		"d20c58bb1574bc3c6e586b312fb2d680fbc40eefc1e171b3282ca0eb8ff50c1246f26e9eda94986d16f" +
		"300df9390e5cecf4331a4ac55d0d496092202f643364be89fb068a61fdd4014d1878275cb7bd241664b" +
		"a4596b54f44f5d25f29dae18236a0b1bf0c036cbfb1a92ed0b4e89fe83b13223633bc2cdcfdcfca3bfc" +
		"391f6d16917e5db8adf6c3e0890d18da7217a17cd4dfce9db5c5f81365a658333ee553ed629ed73eea4" +
		"bf3c30b6fad648cf6b942dc5d2affbdec0a07ea7a12f35cd1c2b1776089d296374c272fd309dd1e5c3d" +
		"a577e35fefc4236ca6d91a0d3c8e5bde3cde9e5a2554fdbecd4b885509d483a2628760dc43137cd1080" +
		"347092c055d69622aade712ac5a86391ebf64ae2de07185f862906e95eedac52c704547dc8d6c46c7fb" +
		"d317489664f8420fe88c0d479d227dd16dec247f8d467a44dbe159aa4d30450b99436c56a84e07a2b80" +
		"4ca56eb2879f9bc1c03435dea3c9ed9bef5f2a17ddca13159604e1fd929bea02f0da6717edba686612b" +
		"0bc79f99142c9db1d89b62839a63788e6b4c6484f7e336eff0e7b6609db5d5788d3cf7ba18d7c3dc049" +
		"7558e55fa0cc761046b62999592bab476047eeac222fd23ae00281a1c24831f49d39924c4ba42364d3bc" +
		"ee83236052b4ecf65a41f01cccd437d78b45f3e132d4666b2cb35b58e6f6e6c347f2ec48aa584e90249" +
		"9ea79b11e95163ed52fbb3507e9c9b9e823cc166f2965795784cf97b0c23a0f7b5d1c5ca02c457c84a2" +
		"d42409df35bcfadb2a6626b7dec561f5f470525f3b755ba4b6c3d78cdeb11d83af1fb404142434445464" +
		"748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5fde01c100000000"
)

// loadSeed decodes the hex seed constants for the named address into
// fixed-size arrays.
func loadSeed(t *testing.T, privHex, pubHex, label string) (
	priv [xmss.PrivateSeedLen]byte, pub [xmss.PublicSeedLen]byte) {

	t.Helper()
	privBytes, err := hex.DecodeString(privHex)
	require.NoError(t, err, "decode %s private seed", label)
	require.Len(t, privBytes, xmss.PrivateSeedLen,
		"%s private seed must decode to %d bytes",
		label, xmss.PrivateSeedLen)

	pubBytes, err := hex.DecodeString(pubHex)
	require.NoError(t, err, "decode %s public seed", label)
	require.Len(t, pubBytes, xmss.PublicSeedLen,
		"%s public seed must decode to %d bytes",
		label, xmss.PublicSeedLen)

	copy(priv[:], privBytes)
	copy(pub[:], pubBytes)
	return priv, pub
}

// TestP2MRXMSSDeriveAddress prints the funding (A) and destination (B)
// mainnet P2MR-XMSS addresses derived from the committed seeds. After
// pass 3 it also asserts both match the expectedP2MRAddress* constants.
func TestP2MRXMSSDeriveAddress(t *testing.T) {
	privA, pubA := loadSeed(t, xmssAPrivSeedHex, xmssAPubSeedHex, "A")
	privB, pubB := loadSeed(t, xmssBPrivSeedHex, xmssBPubSeedHex, "B")

	descA, err := xmsstxn.DeriveDescriptor(
		privA, pubA, &chaincfg.MainNetParams,
	)
	require.NoError(t, err)
	require.True(t, descA.Addr.IsForNet(&chaincfg.MainNetParams))

	descB, err := xmsstxn.DeriveDescriptor(
		privB, pubB, &chaincfg.MainNetParams,
	)
	require.NoError(t, err)
	require.True(t, descB.Addr.IsForNet(&chaincfg.MainNetParams))

	require.NotEqual(t, descA.Addr.String(), descB.Addr.String(),
		"xmssA*SeedHex must differ from xmssB*SeedHex")

	t.Logf("address A (fund this): %s", descA.Addr.String())
	t.Logf("  merkle root: %x", descA.Addr.WitnessProgram())
	t.Logf("  xmss pk:     %x", descA.PublicKey[:])
	t.Logf("  leaf script: %x", descA.LeafScript)
	t.Logf("  ctrl block:  %x", descA.ControlBlock)
	t.Logf("  pkScript:    %x", descA.PkScript)
	t.Logf("address B (spend destination): %s", descB.Addr.String())
	t.Logf("  merkle root: %x", descB.Addr.WitnessProgram())
	t.Logf("  xmss pk:     %x", descB.PublicKey[:])
	t.Logf("  leaf script: %x", descB.LeafScript)
	t.Logf("  ctrl block:  %x", descB.ControlBlock)
	t.Logf("  pkScript:    %x", descB.PkScript)

	// Pass 3 reproducibility gate.
	require.Equal(t, expectedP2MRAddressA, descA.Addr.String())
	require.Equal(t, expectedP2MRAddressB, descB.Addr.String())
}

// TestP2MRXMSSBuildSpend reconstructs the P2MR_A -> P2MR_B spending tx,
// prints its hex for broadcast in pass 2, and asserts byte-exact equality
// with expectedSpendTxHex in pass 3. The underlying helper runs the
// script VM and min-relay-fee guard, so a green test implies the hex is
// ready to broadcast.
func TestP2MRXMSSBuildSpend(t *testing.T) {
	privA, pubA := loadSeed(t, xmssAPrivSeedHex, xmssAPubSeedHex, "A")
	privB, pubB := loadSeed(t, xmssBPrivSeedHex, xmssBPubSeedHex, "B")

	descA, err := xmsstxn.DeriveDescriptor(
		privA, pubA, &chaincfg.MainNetParams,
	)
	require.NoError(t, err)
	require.True(t, descA.Addr.IsForNet(&chaincfg.MainNetParams))

	descB, err := xmsstxn.DeriveDescriptor(
		privB, pubB, &chaincfg.MainNetParams,
	)
	require.NoError(t, err)
	require.True(t, descB.Addr.IsForNet(&chaincfg.MainNetParams))

	fundingHash, err := chainhash.NewHashFromStr(mainnetFundingTxID)
	require.NoError(t, err, "parse mainnetFundingTxID")
	outpoint := wire.OutPoint{
		Hash:  *fundingHash,
		Index: mainnetFundingVout,
	}

	spendTx, err := xmsstxn.BuildSpend(xmsstxn.SpendRequest{
		PrevOut: xmsstxn.UTXO{
			OutPoint: outpoint,
			Value:    mainnetFundingValue,
		},
		DestinationPkScript: descB.PkScript,
		Fee:                 mainnetFee,
		Descriptor:          descA,
	})
	require.NoError(t, err)
	require.Len(t, spendTx.TxIn, 1)
	require.Len(t, spendTx.TxOut, 1)
	require.Equal(t, mainnetFundingValue-mainnetFee,
		spendTx.TxOut[0].Value)
	require.Equal(t, descB.PkScript, spendTx.TxOut[0].PkScript)

	spendHex := mustSerializeHex(t, spendTx)
	spendTxID := spendTx.TxHash().String()
	t.Logf("P2MR_A -> P2MR_B spend tx:")
	t.Logf("  txid:        %s", spendTxID)
	t.Logf("  destination: %s", descB.Addr.String())
	t.Logf("  hex:         %s", spendHex)

	require.Equal(t, expectedSpendTxHex, spendHex)
	require.Equal(t, expectedSpendTxID, spendTxID)
}

// mustSerializeHex returns tx serialized to hex (witness included).
func mustSerializeHex(t *testing.T, tx *wire.MsgTx) string {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, tx.Serialize(&buf))
	return hex.EncodeToString(buf.Bytes())
}
