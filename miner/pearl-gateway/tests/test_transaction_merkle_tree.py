"""
Tests for merkle root calculation using real Bitcoin block data.
"""

import json
import os

import pytest
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    calculate_merkle_root,
    double_sha256,
)


class TestMerkleBtcpy:
    def test_bitcoin_block_100k_merkle_root(self):
        """
        Test merkle root calculation using real Bitcoin block 100k data.

        Block 100k is a famous Bitcoin block and this test verifies our implementation
        against the actual merkle root from the Bitcoin network.
        """
        # Load the Bitcoin block 100k data
        test_data_path = os.path.join(os.path.dirname(__file__), "bitcoin_block_100k.json")
        with open(test_data_path) as f:
            block_data = json.load(f)

        # Extract the expected merkle root from the block data
        expected_merkle_root = bytes.fromhex(block_data["Data"]["METADATA"]["merkleroot"])

        # Collect txids from each transaction
        txids = []
        for tx in block_data["Data"]["TRANSACTIONS"]:
            try:
                parsed_tx = Transaction.from_raw(tx["hex"])
                txids.append(parsed_tx.get_txid())
            except Exception as e:
                pytest.fail(f"Failed to parse transaction hex {tx['hex'][:16]}...: {e}")

        # Calculate merkle root using our function
        calculated_merkle_root = calculate_merkle_root(txids)

        # Verify the merkle roots match
        assert calculated_merkle_root == expected_merkle_root

    def test_empty_transactions(self):
        """Test merkle root calculation with only coinbase transaction."""
        coinbase_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08044c86041b020602ffffffff0100f2052a010000004341041b0e8c2567c12536aa13357b79a073dc4444acb83c4ec7a0e2f99dd7457516c5817242da796924ca4e99947d087fedf9ce467cb9f7c6287078f801df276fdf84ac00000000"
        coinbase_tx = Transaction.from_raw(coinbase_hex)
        coinbase_txid = coinbase_tx.get_txid()

        result = calculate_merkle_root([coinbase_txid])

        expected = bytes.fromhex(coinbase_txid)
        assert result == expected

    def test_single_additional_transaction(self):
        """Test merkle root with coinbase + one additional transaction."""
        coinbase_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08044c86041b020602ffffffff0100f2052a010000004341041b0e8c2567c12536aa13357b79a073dc4444acb83c4ec7a0e2f99dd7457516c5817242da796924ca4e99947d087fedf9ce467cb9f7c6287078f801df276fdf84ac00000000"
        tx1_hex = "0100000001032e38e9c0a84c6046d687d10556dcacc41d275ec55fc00779ac88fdf357a187000000008c493046022100c352d3dd993a981beba4a63ad15c209275ca9470abfcd57da93b58e4eb5dce82022100840792bc1f456062819f15d33ee7055cf7b5ee1af1ebcc6028d9cdb1c3af7748014104f46db5e9d61a9dc27b8d64ad23e7383a4e6ca164593c2527c038c0857eb67ee8e825dca65046b82c9331586c82e0fd1f633f25f87c161bc6f8a630121df2b3d3ffffffff0200e32321000000001976a914c398efa9c392ba6013c5e04ee729755ef7f58b3288ac000fe208010000001976a914948c765a6914d43f2a7ac177da2c2f6b52de3d7c88ac00000000"

        coinbase_tx = Transaction.from_raw(coinbase_hex)
        tx1 = Transaction.from_raw(tx1_hex)

        result = calculate_merkle_root([coinbase_tx.get_txid(), tx1.get_txid()])

        coinbase_hash = bytes.fromhex(coinbase_tx.get_txid())[::-1]
        tx1_hash = bytes.fromhex(tx1.get_txid())[::-1]
        expected = double_sha256(coinbase_hash + tx1_hash)[::-1]

        assert result == expected
