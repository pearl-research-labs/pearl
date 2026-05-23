from bitcoinutils.utils import encode_varint
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import ZKCertificate


class PearlBlock:
    def __init__(
        self,
        header: PearlHeader,
        raw_txns: list[bytes],
        zk_certificate: ZKCertificate,
    ):
        self.header = header

        if header.proof_commitment is None:
            header.proof_commitment = zk_certificate.get_proof_commitment()
        elif header.proof_commitment != zk_certificate.get_proof_commitment():
            raise ValueError("Proof commitment mismatch")

        self.raw_txns = raw_txns
        self.zk_certificate = zk_certificate

    def serialize(self) -> bytes:
        """
        Format: ZK_CERTIFICATE|BLOCK_HEADER|TX_COUNT (varint)|TRANSACTIONS
        """
        zk_certificate_bytes = self.zk_certificate.serialize()
        header_bytes = self.header.serialize()
        tx_count_bytes = encode_varint(len(self.raw_txns))
        transactions_bytes = b"".join(self.raw_txns)
        return zk_certificate_bytes + header_bytes + tx_count_bytes + transactions_bytes
