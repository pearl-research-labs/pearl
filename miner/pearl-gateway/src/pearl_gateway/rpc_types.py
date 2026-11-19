"""Type definitions for pearld getblocktemplate RPC responses."""

from pydantic import BaseModel, Field

from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion


class BlockTemplateTx(BaseModel):
    data: str
    txid: str


class CoinbaseAux(BaseModel):
    flags: str


class GetBlockTemplateResponse(BaseModel):
    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    previousblockhash: str
    transactions: list[BlockTemplateTx]
    version: int
    target: str
    coinbaseaux: CoinbaseAux | None = None
    coinbasevalue: int = Field(ge=0)
    default_witness_commitment: str | None = None
    requiredcertversion: CertificateVersion = CertificateVersion.ZK_DENSE
    ancestorheaders: list[str] = Field(default_factory=list)
