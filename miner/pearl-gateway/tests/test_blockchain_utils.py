import pytest
from bitcoinutils.bech32 import Encoding, bech32_encode, convertbits
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    bits_to_target,
    create_coinbase_transaction,
    target_to_bits,
)


class TestCreateCoinbaseTransaction:
    """Unit tests for create_coinbase_transaction function."""

    def test_build_coinbase_from_template_matches_node(self):
        """Test that our coinbase built from template matches the node's coinbasetxn 100%."""
        block_template = {
            "height": 2591,
            "coinbasevalue": 297454395584,
            "coinbaseaux": {"flags": "0b2f503253482f627463642f"},
            "transactions": [],
            "coinbasetxn": {
                "data": "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff10021f0a000b2f503253482f627463642fffffffff01c0e0a941450000002251208635cb51e0601a2f55b17b1ba41b21a511b3753a0bf4610bd52eb1a15d69a28100000000",
            },
        }

        node_coinbase_hex = block_template["coinbasetxn"]["data"]
        node_coinbase = Transaction.from_raw(node_coinbase_hex)

        node_scriptpubkey = bytes.fromhex(node_coinbase.outputs[0].script_pubkey.to_hex())
        witness_program = node_scriptpubkey[2:]
        witness_program_5bit = convertbits(witness_program, 8, 5, True)
        node_mining_address = bech32_encode("tprl", [1] + witness_program_5bit, Encoding.BECH32M)

        our_coinbase = create_coinbase_transaction(
            height=block_template["height"],
            coinbase_value=block_template["coinbasevalue"],
            mining_address=node_mining_address,
            coinbase_aux=block_template["coinbaseaux"],
            default_witness_commitment=None,
        )

        assert our_coinbase.to_hex() == node_coinbase_hex


@pytest.mark.parametrize("bits", [0x1B00FFFF, 0x1A7FFF80, 0x1D00FFFF, 0x1E010000, 0x207FFFFF])
def test_target_to_bits_round_trips_compact_targets(bits):
    assert bits_to_target(target_to_bits(bits_to_target(bits))) == bits_to_target(bits)


def test_target_to_bits_keeps_the_sign_bit_clear():
    # 0x800000 as a mantissa would be negative, so it moves to the next exponent.
    assert target_to_bits(0x800000) == 0x04008000
