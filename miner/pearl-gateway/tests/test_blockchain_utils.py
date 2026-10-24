from bitcoinutils.bech32 import Encoding, bech32_encode, convertbits
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    bits_to_target,
    create_coinbase_transaction,
)


class TestBitsToTarget:
    """bits_to_target must mirror the node's workmath.CompactToBig:
    23-bit mantissa, bit 23 is the sign, exponent <= 3 shifts right.
    """

    def test_network_pow_limit_bits(self):
        # The PowLimitBits of the node's networks (chaincfg params).
        assert bits_to_target(0x1B00FFFF) == 0xFFFF * 256**24
        assert bits_to_target(0x1E010000) == 0x010000 * 256**27
        assert bits_to_target(0x1D0FFFFF) == 0x0FFFFF * 256**26

    def test_exponent_above_three(self):
        assert bits_to_target(0x0400FFFF) == 0xFFFF00
        assert bits_to_target(0x03000001) == 1

    def test_exponent_at_or_below_three_shifts_right(self):
        # These raised "ValueError: negative shift count" before the fix.
        assert bits_to_target(0x02008000) == 0x80
        assert bits_to_target(0x02003456) == 0x34
        assert bits_to_target(0x01003456) == 0
        assert bits_to_target(0x0000FFFF) == 0

    def test_sign_bit_is_not_part_of_the_mantissa(self):
        # Go masks the mantissa to 23 bits and negates when bit 23 is set;
        # the old 24-bit mask folded the sign bit into the magnitude.
        assert bits_to_target(0x1D800000) == 0
        assert bits_to_target(0x1C800001) == -(256**25)
        assert bits_to_target(0x1B80FFFF) == -(0xFFFF * 256**24)


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
