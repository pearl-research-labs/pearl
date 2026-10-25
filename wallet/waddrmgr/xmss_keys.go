// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package waddrmgr

import (
	"crypto/sha256"
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/xmss"
	"golang.org/x/crypto/hkdf"
)

// DeriveXMSSSeeds derives the XMSS private and public seeds for the given
// derivation path. This uses purpose 222 (m/222'/coin'/account'/branch/index)
// to derive a (private) key, then uses HKDF to expand it into XMSS seeds.
// The corresponding EC public key must never be used.
//
// The path should match the BIP-86 path used for the corresponding Schnorr key.
// The seeds are derived independently to ensure quantum computers cannot
// compromise XMSS even if Schnorr keys are broken.
func DeriveXMSSSeeds(pqMgr *ScopedKeyManager, ns walletdb.ReadWriteBucket, path DerivationPath) (
	privSeed [xmss.PrivateSeedLen]byte, pubSeed [xmss.PublicSeedLen]byte, err error) {

	// Ensure the account exists in PQ scope (auto-create if needed)
	if _, err := pqMgr.AccountProperties(ns, path.InternalAccount); err != nil {
		if createErr := pqMgr.NewRawAccount(ns, path.InternalAccount); createErr != nil {
			return privSeed, pubSeed, fmt.Errorf("failed to create PQ account %d: %w",
				path.InternalAccount, createErr)
		}
	}

	addr, err := pqMgr.DeriveFromKeyPath(ns, path)
	if err != nil {
		return privSeed, pubSeed, fmt.Errorf("failed to derive XMSS key path: %w", err)
	}

	pubKeyAddr, ok := addr.(ManagedPubKeyAddress)
	if !ok {
		return privSeed, pubSeed, fmt.Errorf("derived address is not a public key address")
	}

	ecPrivKey, err := pubKeyAddr.PrivKey()
	if err != nil {
		return privSeed, pubSeed, fmt.Errorf("failed to get private key: %w", err)
	}

	// Expand via HKDF: 32 bytes → 96 bytes (64 priv seed + 32 pub seed)
	ikm := ecPrivKey.Serialize()
	hkdfGen := hkdf.New(sha256.New, ikm, nil, []byte("XMSS-SEED-EXPANSION"))

	// Read 64 bytes for private seed, 32 bytes for public seed
	xmssSeed := make([]byte, xmss.PrivateSeedLen+xmss.PublicSeedLen)
	if _, err := io.ReadFull(hkdfGen, xmssSeed); err != nil {
		return privSeed, pubSeed, fmt.Errorf("failed to derive XMSS seeds: %w", err)
	}

	copy(privSeed[:], xmssSeed[:xmss.PrivateSeedLen])
	copy(pubSeed[:], xmssSeed[xmss.PrivateSeedLen:])
	clear(xmssSeed)

	return privSeed, pubSeed, nil
}

// DeriveXMSSPublicKey is a convenience function that derives the XMSS seeds
// and generates the public key.
func DeriveXMSSPublicKey(pqMgr *ScopedKeyManager, ns walletdb.ReadWriteBucket,
	path DerivationPath) ([xmss.PKLen]byte, error) {

	var pubKey [xmss.PKLen]byte

	privSeed, pubSeed, err := DeriveXMSSSeeds(pqMgr, ns, path)
	if err != nil {
		return pubKey, err
	}

	// Generate keypair (this is the slow XMSS keygen operation)
	pk, sk, err := xmss.Keygen(privSeed, pubSeed)
	clear(sk[:])
	if err != nil {
		return pk, fmt.Errorf("failed to generate XMSS keypair: %w", err)
	}

	// Clear sensitive data
	clear(privSeed[:])
	clear(pubSeed[:])

	return pk, nil
}

// DeriveXMSSTapscriptRoot returns the root of the one-leaf script tree
// <xmss_pubkey> OP_CHECKXMSSSIG for path. The manager must be unlocked because
// the XMSS seeds come from a private key.
func DeriveXMSSTapscriptRoot(pqMgr *ScopedKeyManager, ns walletdb.ReadWriteBucket,
	path DerivationPath) ([]byte, error) {

	xmssPubKey, err := DeriveXMSSPublicKey(pqMgr, ns, path)
	if err != nil {
		return nil, fmt.Errorf("failed to derive XMSS public key: %w", err)
	}

	xmssScript, err := txscript.NewScriptBuilder().
		AddData(xmssPubKey[:]).
		AddOp(txscript.OP_CHECKXMSSSIG).
		Script()
	if err != nil {
		return nil, fmt.Errorf("failed to build XMSS script: %w", err)
	}

	tapLeaf := txscript.NewBaseTapLeaf(xmssScript)
	tapTree := txscript.AssembleTaprootScriptTree(tapLeaf)
	rootHash := tapTree.RootNode.TapHash()
	return rootHash[:], nil
}
