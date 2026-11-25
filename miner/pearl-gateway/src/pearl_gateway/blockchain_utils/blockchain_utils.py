from hashlib import sha256

from bitcoinutils.bech32 import Encoding, bech32_decode, convertbits
from bitcoinutils.script import Script
from bitcoinutils.transactions import Transaction, TxInput, TxOutput, TxWitnessInput


def double_sha256(data: bytes) -> bytes:
    return sha256(sha256(data).digest()).digest()


def calculate_merkle_root(txids: list[str]) -> bytes:
    """Calculate the 32-byte merkle root from big-endian display txid hex strings."""
    current_level = [bytes.fromhex(txid)[::-1] for txid in txids]
    if not current_level:
        raise ValueError("No hashes to compute merkle root")
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
            next_level.append(double_sha256(current_level[i] + right))
        current_level = next_level
    return current_level[0][::-1]


def reconstruct_stratum_merkle_root(coinbase_hash: bytes, merkle_branch: list[bytes]) -> bytes:
    """Walk a Stratum merkle branch (internal-order hashes) up from the coinbase hash.

    Returns the root in display order, matching calculate_merkle_root and what
    IncompleteBlockHeader expects.
    """
    current = coinbase_hash
    for sibling in merkle_branch:
        current = double_sha256(current + sibling)
    return current[::-1]


def bits_to_target(bits: int) -> int:
    """Convert compact nBits representation to 256-bit target."""
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent < 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa * (1 << (8 * (exponent - 3)))


def create_coinbase_transaction(
    height: int,
    coinbase_value: int,
    mining_address: str,
    coinbase_aux: dict[str, str] | None = None,
    default_witness_commitment: str | None = None,
) -> Transaction:
    script_pubkey = get_script_pubkey_from_p2tr_address(mining_address)

    # BIP34: height first in scriptSig, followed by extra nonce byte and aux flags.
    coinbase_script_bytes = bytes.fromhex(Script([height]).to_hex()) + b"\x00"
    if coinbase_aux and "flags" in coinbase_aux:
        coinbase_script_bytes += bytes.fromhex(coinbase_aux["flags"])

    coinbase_input = TxInput(
        txid="0" * 64,
        txout_index=0xFFFFFFFF,
        script_sig=Script([coinbase_script_bytes.hex()]),
        sequence=b"\xff\xff\xff\xff",
    )
    outputs = [TxOutput(coinbase_value, script_pubkey)]
    has_witness = default_witness_commitment is not None
    if has_witness:
        outputs.append(TxOutput(0, Script(["OP_RETURN", "aa21a9ed" + default_witness_commitment])))

    return Transaction(
        inputs=[coinbase_input],
        outputs=outputs,
        locktime=b"\x00\x00\x00\x00",
        version=b"\x01\x00\x00\x00",
        has_segwit=has_witness,
        witnesses=[TxWitnessInput([bytes(32).hex()])] if has_witness else None,
    )


def get_script_pubkey_from_p2tr_address(address: str) -> Script:
    """Extract Taproot (P2TR) scriptPubKey from a bech32m witness-v1 address with any HRP."""
    hrp, data, encoding = bech32_decode(address)
    if hrp is None or data is None:
        raise ValueError("Invalid bech32/bech32m address")
    if not data:
        raise ValueError("Invalid witness program (empty data)")
    if data[0] != 1:
        raise ValueError(f"Expected Taproot witness version 1, got {data[0]}")
    if encoding != Encoding.BECH32M:
        raise ValueError(f"Taproot address must be bech32m, got {encoding}")

    program_8bit = convertbits(data[1:], 5, 8, False)
    if program_8bit is None:
        raise ValueError("Invalid witness program (convertbits failed)")
    program = bytes(program_8bit)
    if len(program) != 32:
        raise ValueError(f"Taproot witness program must be 32 bytes, got {len(program)}")
    # OP_1 (0x51) PUSH32 (0x20) <32-byte program>
    return Script.from_raw((b"\x51\x20" + program).hex())
