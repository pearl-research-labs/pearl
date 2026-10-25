// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

/*
Package btcutil provides Pearl-specific convenience functions and types.

# Block Overview

A Block defines a block that provides easier and more efficient
manipulation of raw wire protocol blocks.  It also memoizes hashes for the
block and its transactions on their first access so subsequent accesses don't
have to repeat the relatively expensive hashing operations.

# Tx Overview

A Tx defines a transaction that provides more efficient manipulation of
raw wire protocol transactions.  It memoizes the hash for the transaction on its
first access so subsequent accesses don't have to repeat the relatively
expensive hashing operations.

# Address Overview

The Address interface provides an abstraction for a Pearl address.  While the
most common type is a pay-to-pubkey-hash, the protocol already supports others
and may well support more in the future.  This package currently provides
implementations for the pay-to-pubkey, pay-to-pubkey-hash, and
pay-to-script-hash address types.

To decode/encode an address:

	// NOTE: The address must belong to the given network, otherwise the
	// error returned wraps btcutil.ErrWrongNetwork.
	addrString := "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv"
	net := &chaincfg.MainNetParams
	addr, err := btcutil.DecodeAddress(addrString, net)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(addr.EncodeAddress())
*/
package btcutil
