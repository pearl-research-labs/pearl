import copy

import pytest
from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.zk_certificate import (
    PUBLICDATA_SIZE,
    CertificateProof,
    CertificateVersion,
    ZKCertificate,
)
from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob
from pearl_gateway.config import PearlConfig
from pearl_gateway.rpc_types import GetBlockTemplateResponse
from pearl_mining import MatrixMerkleProof, MerkleProof, MoEProofParams, PlainProof


def _build_plain_proof(*, moe: bool) -> PlainProof:
    merkle_proof = MerkleProof(
        total_leaves=1,
        leaf_data=[b"\x00" * 1024],
        leaf_indices=[0],
        root=b"\x00" * 32,
        siblings=[],
    )
    matrix_proof = MatrixMerkleProof(proof=merkle_proof, row_indices=[0])
    moe_params = (
        MoEProofParams(
            e=2,
            top_k=1,
            expert_idx=0,
            routing_end_offsets=[1, 2],
            inner_a_rows=[0],
            routing_proof=merkle_proof,
        )
        if moe
        else None
    )
    return PlainProof(
        m=4,
        n=4,
        k=4,
        noise_rank=1,
        a_merkle_proof=matrix_proof,
        bt_merkle_proof=matrix_proof,
        moe=moe_params,
    )


@pytest.fixture
def make_dummy_plain_proof():
    return _build_plain_proof


@pytest.fixture
def dummy_moe_proof(make_dummy_plain_proof):
    return make_dummy_plain_proof(moe=True)


@pytest.fixture(scope="session")
def mining_address():
    return "tprl1p50ecun6548wx3per3hz7hp7qsnaqqdxtweyz7s37ty3kumszqscqe5v7x6"


@pytest.fixture(scope="session")
def sample_block_template_data():
    """Block template fixture with difficulty 2**180."""
    return {
        "bits": "17100000",
        "curtime": 1293623863,
        "height": 100000,
        "previousblockhash": "000000000002d01c1fccc21636b607dfd930d31d01c3a62104612a1719011250",
        "transactions": [
            {
                "data": "0100000001032e38e9c0a84c6046d687d10556dcacc41d275ec55fc00779ac88fdf357a187000000008c493046022100c352d3dd993a981beba4a63ad15c209275ca9470abfcd57da93b58e4eb5dce82022100840792bc1f456062819f15d33ee7055cf7b5ee1af1ebcc6028d9cdb1c3af7748014104f46db5e9d61a9dc27b8d64ad23e7383a4e6ca164593c2527c038c0857eb67ee8e825dca65046b82c9331586c82e0fd1f633f25f87c161bc6f8a630121df2b3d3ffffffff0200e32321000000001976a914c398efa9c392ba6013c5e04ee729755ef7f58b3288ac000fe208010000001976a914948c765a6914d43f2a7ac177da2c2f6b52de3d7c88ac00000000",
                "txid": "87a157f3fd88ac7907c05fc55e271dc4acdc5605d187d646604ca8c0e9382e03",
            },
            {
                "data": "0100000001c33ebff2a709f13d9f9a7569ab16a32786af7d7e2de09265e41c61d078294ecf010000008a4730440220032d30df5ee6f57fa46cddb5eb8d0d9fe8de6b342d27942ae90a3231e0ba333e02203deee8060fdc70230a7f5b4ad7d7bc3e628cbe219a886b84269eaeb81e26b4fe014104ae31c31bf91278d99b8377a35bbce5b27d9fff15456839e919453fc7b3f721f0ba403ff96c9deeb680e5fd341c0fc3a7b90da4631ee39560639db462e9cb850fffffffff0240420f00000000001976a914b0dcbf97eabf4404e31d952477ce822dadbe7e1088acc060d211000000001976a9146b1281eec25ab4e1e0793ff4e08ab1abb3409cd988ac00000000",
                "txid": "cf4e2978d0611ce46592e02d7e7daf8627a316ab69759a9f3df109a7f2bf3ec3",
            },
        ],
        "version": 536870912,
        "coinbaseaux": {"flags": "00"},
        "coinbasevalue": 10000000000,
        "default_witness_commitment": "6a24aa21a9ed2ce1c6b7d1b481e1d941d45380ffc4c65532dd01c504c33bb7119d2d539a1560",
        "target": "1000000000000000000000000000000000000000000000",
        "requiredcertversion": 2,
    }


@pytest.fixture(scope="session")
def sample_block_template(sample_block_template_data, mining_address):
    return BlockTemplate.from_get_block_template(
        GetBlockTemplateResponse.model_validate(sample_block_template_data),
        mining_address=mining_address,
    )


@pytest.fixture
def sample_plain_proof():
    a_proof = MerkleProof(
        total_leaves=1024,
        leaf_data=[bytes(1024) for _ in range(4)],
        leaf_indices=[320, 321, 322, 323],
        root=b"\x08" * 32,
        siblings=[bytes([i]) * 32 for i in range(8)],
    )
    bt_proof = MerkleProof(
        total_leaves=1024,
        leaf_data=[bytes(1024) for _ in range(16)],
        leaf_indices=list(range(160, 176)),
        root=b"\x67" * 32,
        siblings=[bytes([i]) * 32 for i in range(6)],
    )
    return PlainProof(
        m=1024,
        n=1024,
        k=1024,
        noise_rank=64,
        a_merkle_proof=MatrixMerkleProof(proof=a_proof, row_indices=[320, 321, 322, 323]),
        bt_merkle_proof=MatrixMerkleProof(proof=bt_proof, row_indices=list(range(160, 176))),
        moe=None,
    )


@pytest.fixture
def sample_mining_job():
    return MiningJob.from_dict(
        {
            "incomplete_header_bytes": "AAAAICcOqViZtp4qeqp5mnm0R3Yl7+D2GJ/5coUqW1OAhhKyiYZG4kAi7Bg+/A440ZG+21mb+7on6Nyg4JUtABGkU8SLg39pAAABHg==",
            "target": 6901746346790563787434755862277025452451108972170386555162524223799296,
            "cert_version": int(CertificateVersion.ZK_MOE),
        }
    )


@pytest.fixture
def submit_plain_proof_params(sample_plain_proof, sample_mining_job):
    return {
        "plain_proof": sample_plain_proof.to_base64(),
        "mining_job": sample_mining_job.to_dict(),
    }


@pytest.fixture
def sample_pearl_block(sample_block_template):
    header = copy.copy(sample_block_template.header)
    zk_certificate = ZKCertificate.from_pearl_header(
        header,
        CertificateProof(bytes(PUBLICDATA_SIZE), bytes(10000)),
    )
    return PearlBlock(
        header,
        raw_txns=sample_block_template.get_raw_transactions(),
        zk_certificate=zk_certificate,
    )


@pytest.fixture
def valid_jsonrpc_request():
    return {"jsonrpc": "2.0", "method": "getMiningInfo", "params": {}, "id": 1}


@pytest.fixture
def sample_pearl_config(mining_address):
    return PearlConfig(
        rpc_url="https://127.0.0.1:18334",
        rpc_user="test_user",
        rpc_password="test_password",
        refresh_interval_seconds=5,
        mining_address=mining_address,
    )


@pytest.fixture
def config_env_vars(monkeypatch, mining_address):
    env_vars = {
        "PEARLD_RPC_URL": "https://127.0.0.1:18334",
        "PEARLD_RPC_USER": "test_user",
        "PEARLD_RPC_PASSWORD": "test_password",
        "PEARLD_REFRESH_INTERVAL_SECONDS": "5",
        "PEARLD_MINING_ADDRESS": mining_address,
        "MINER_RPC_TRANSPORT": "uds",
        "MINER_RPC_SOCKET_PATH": "/tmp/test_pearlgw.sock",
        "MINER_RPC_PORT": "8337",
        "MINER_RPC_HOST": "localhost",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    yield None
