from hashlib import sha256

from bitcoinutils.bech32 import Encoding, bech32_decode, convertbits
from bitcoinutils.script import Script
from bitcoinutils.transactions import Transaction, TxInput, TxOutput, TxWitnessInput


def double_sha256(data: bytes) -> bytes:
    """Calculate double SHA256 hash as used in Pearl."""
    return sha256(sha256(data).digest()).digest()


def calculate_merkle_root(txids: list[str]) -> bytes:
    """
    Calculate the merkle root from a list of txid hex strings.

    Args:
        txids: List of txid hex strings in display order (big-endian),
               with the coinbase txid first.

    Returns:
        bytes: The 32-byte merkle root hash
    """
    # Txids are in display order (big-endian); reverse to little-endian for merkle calc
    tx_hashes = [bytes.fromhex(txid)[::-1] for txid in txids]

    # Calculate merkle root using the standard algorithm
    return _compute_merkle_root(tx_hashes)


def _compute_merkle_root(hashes: list[bytes]) -> bytes:
    """
    Compute merkle root from a list of transaction hashes.
    Implements the standard merkle tree algorithm.
    """
    assert len(hashes) > 0, "No hashes to compute merkle root"

    if len(hashes) == 1:
        return hashes[0][::-1]  # Convert back to big-endian for final result

    # Pair up hashes and compute next level
    current_level = hashes.copy()
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            # Concatenate the pair and double SHA256
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                # If odd number of hashes, duplicate the last one
                combined = current_level[i] + current_level[i]
            next_level.append(double_sha256(combined))
        current_level = next_level

    return current_level[0][::-1]  # Convert back to big-endian for final result


def bits_to_target(bits: int) -> int:
    """Convert bits to target.
    bits is a 4-byte value: 1 byte exponent, 3 bytes mantissa (big-endian)
    target = mantissa * 2**(8*(exponent-3))
    """
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    target = mantissa * (1 << (8 * (exponent - 3)))
    return target


def create_coinbase_transaction(
    height: int,
    coinbase_value: int,
    mining_address: str,
    coinbase_aux: dict[str, str] | None = None,
    default_witness_commitment: str | None = None,
) -> Transaction:
    """
    Create a coinbase transaction from scratch.
    """
    script_pubkey = get_script_pubkey_from_p2tr_address(mining_address)

    # Build coinbase script (scriptSig)
    # BIP34: Height must be first item in coinbase scriptSig
    # Use Script() to encode height as proper script number
    height_script = Script([height])
    coinbase_script_bytes = bytes.fromhex(height_script.to_hex())

    # Add extra nonce byte (matches node's behavior)
    coinbase_script_bytes += b"\x00"

    if coinbase_aux and "flags" in coinbase_aux:
        aux_flags = bytes.fromhex(coinbase_aux["flags"])
        coinbase_script_bytes += aux_flags

    coinbase_input = TxInput(
        txid="0" * 64,  # Null transaction hash
        txout_index=0xFFFFFFFF,  # Max uint32
        script_sig=Script([coinbase_script_bytes.hex()]),
        sequence=b"\xff\xff\xff\xff",  # Max sequence (4 bytes)
    )

    coinbase_output = TxOutput(coinbase_value, script_pubkey)

    outputs = [coinbase_output]

    has_witness = default_witness_commitment is not None

    if has_witness:
        witness_script = Script(["OP_RETURN", "aa21a9ed" + default_witness_commitment])
        witness_output = TxOutput(0, witness_script)
        outputs.append(witness_output)

    coinbase_tx = Transaction(
        inputs=[coinbase_input],
        outputs=outputs,
        locktime=b"\x00\x00\x00\x00",  # 4-byte little-endian locktime (0)
        version=b"\x01\x00\x00\x00",  # 4-byte little-endian version (1)
        has_segwit=has_witness,
        witnesses=[TxWitnessInput([bytes(32).hex()])] if has_witness else None,
    )

    return coinbase_tx


def get_script_pubkey_from_p2tr_address(address: str) -> Script:
    """
    Extract the script_pub_key from a Taproot (P2TR) address with ANY HRP.
    Enforces witness v1 + 32-byte program + bech32m encoding.
    """
    hrp, data, encoding = bech32_decode(address)

    if hrp is None or data is None:
        raise ValueError("Invalid bech32/bech32m address")

    if len(data) == 0:
        raise ValueError("Invalid witness program (empty data)")

    witness_version = data[0]
    if witness_version != 1:
        raise ValueError(f"Expected Taproot witness version 1, got {witness_version}")

    # Taproot (v1) must use bech32m
    if encoding != Encoding.BECH32M:
        raise ValueError(f"Taproot address must be bech32m, got {encoding}")

    witness_program_5bit = data[1:]
    witness_program_8bit = convertbits(witness_program_5bit, 5, 8, False)

    if witness_program_8bit is None:
        raise ValueError("Invalid witness program (convertbits failed)")

    witness_program_bytes = bytes(witness_program_8bit)

    if len(witness_program_bytes) != 32:
        raise ValueError(
            f"Taproot witness program must be 32 bytes, got {len(witness_program_bytes)}"
        )

    # scriptPubKey = OP_1 (0x51) PUSH32 (0x20) <32-byte program>
    script_bytes = b"\x51\x20" + witness_program_bytes
    return Script.from_raw(script_bytes.hex())
