from dataclasses import dataclass

from pearl_mining import IncompleteBlockHeader

PROOF_COMMITMENT_SIZE = 32


@dataclass
class PearlHeader:
    """Pearl block header wrapping IncompleteBlockHeader + optional 32-byte proof_commitment."""

    incomplete_header: IncompleteBlockHeader
    proof_commitment: bytes | None = None

    def __post_init__(self) -> None:
        if (
            self.proof_commitment is not None
            and len(self.proof_commitment) != PROOF_COMMITMENT_SIZE
        ):
            raise ValueError(
                f"Proof commitment must be {PROOF_COMMITMENT_SIZE} bytes, got {len(self.proof_commitment)}"
            )

    @property
    def previous_block_hash(self) -> bytes:
        return bytes(self.incomplete_header.prev_block)

    @property
    def timestamp(self) -> int:
        return self.incomplete_header.timestamp

    @property
    def target_bits(self) -> int:
        return self.incomplete_header.nbits

    @classmethod
    def get_serialized_header_size(cls) -> int:
        return IncompleteBlockHeader.SERIALIZED_SIZE + PROOF_COMMITMENT_SIZE

    @classmethod
    def deserialize(cls, data: bytes) -> "PearlHeader":
        header_size = IncompleteBlockHeader.SERIALIZED_SIZE
        header = IncompleteBlockHeader.from_bytes(data[:header_size])
        proof_commitment = data[header_size : header_size + PROOF_COMMITMENT_SIZE]
        return cls(incomplete_header=header, proof_commitment=proof_commitment)

    def serialize_without_proof_commitment(self) -> bytes:
        return bytes(self.incomplete_header.to_bytes())

    def serialize(self) -> bytes:
        if self.proof_commitment is None:
            raise ValueError("Proof commitment is not set")
        return self.serialize_without_proof_commitment() + self.proof_commitment
