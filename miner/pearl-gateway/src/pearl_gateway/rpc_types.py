"""Type definitions for pearld RPC responses."""

from pydantic import BaseModel, Field

from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion


class BlockTemplateTx(BaseModel):
    """Regular transaction in getblocktemplate response."""

    data: str
    hash: str
    txid: str
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)


class CoinbaseAux(BaseModel):
    """Coinbase auxiliary data in getblocktemplate response."""

    flags: str


# Original struct from node/btcjson/chainsvrresults.go:GetBlockTemplateResult
class GetBlockTemplateResponse(BaseModel):
    """Fields in getblocktemplate response with runtime validation."""

    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    previousblockhash: str
    vsizelimit: int = Field(gt=0)
    transactions: list[BlockTemplateTx]
    version: int
    longpollid: str
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    noncerange: str
    capabilities: list[str]
    coinbaseaux: CoinbaseAux
    coinbasevalue: int = Field(ge=0)
    default_witness_commitment: str | None = None
    # Absent from nodes that predate the field; those only ever require V1.
    requiredcertversion: CertificateVersion = CertificateVersion.ZK_DENSE
