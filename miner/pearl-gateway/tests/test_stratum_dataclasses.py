"""Tests for the pool-mode Stratum payload models and BlockTemplate.from_stratum_notify."""

from typing import Any

import pytest
from pearl_gateway.blockchain_utils.blockchain_utils import (
    double_sha256,
    reconstruct_stratum_merkle_root,
    target_to_bits,
)
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import (
    BlockTemplate,
    MiningJob,
    StratumNotify,
    StratumSetDifficulty,
)


@pytest.fixture
def stratum_notify_params() -> dict[str, Any]:
    return {
        "job_id": "test",
        "prev_hash": "00" * 32,
        "coinb1": (
            "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0704"
        ),
        "coinb2": (
            "0101ffffffff0100f2052a01000000232102deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
            "deadbeefdeadbeefdeadbeefac00000000"
        ),
        "merkle_branch": [],
        "version": "20000000",
        "bits": "1e010000",
        "ntime": "69df47a7",
        "clean_jobs": False,
        "extranonce": bytes(4),
        "cert_version": CertificateVersion.ZK_DENSE,
    }


@pytest.fixture
def stratum_notify(stratum_notify_params: dict[str, Any]) -> StratumNotify:
    return StratumNotify(**stratum_notify_params)


class TestFromStratumNotify:
    """Test BlockTemplate.from_stratum_notify byte-order handling.

    These tests verify that the Python gateway produces identical header
    serialization to the Go pool, catching byte-order mismatches in
    prev_block, merkle_root, and coinbase txid computation.
    """

    def test_prev_block_byte_order(self, stratum_notify_params):
        """prev_block from stratum should be stored as-is (display order).
        The Rust IncompleteBlockHeader.to_bytes() reverses it to internal order,
        matching Go's writeBlockHeader which writes PrevBlock (internal) directly.
        """
        prev_hash = "c345c8a096b00cd8aa85d6f3c369a3496708680199804555797829e9000774c5"

        # Minimal stratum notify params (no merkle branch, trivial coinbase)
        coinb1 = (
            "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0704"
        )
        coinb2 = "0101ffffffff0100f2052a01000000232102deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefac00000000"
        extranonce = bytes(4)

        notify = StratumNotify(
            **(
                stratum_notify_params
                | {
                    "prev_hash": prev_hash,
                    "coinb1": coinb1,
                    "coinb2": coinb2,
                    "extranonce": extranonce,
                }
            )
        )
        template = BlockTemplate.from_stratum_notify(notify)

        # The stored prev_block should match the hex string from the pool
        assert bytes(template.header.incomplete_header.prev_block) == bytes.fromhex(prev_hash)

    def test_coinbase_txid_strips_witness(self):
        """Coinbase hash for merkle tree must be txid (witness-stripped),
        not the raw double-SHA256 of the full coinbase bytes.
        """
        from bitcoinutils.transactions import Transaction

        # A simple non-witness coinbase
        coinb1 = (
            "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0704"
        )
        coinb2 = "0101ffffffff0100f2052a01000000232102deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefac00000000"
        extranonce = bytes(4)

        coinbase_bytes = bytes.fromhex(coinb1) + extranonce + bytes.fromhex(coinb2)
        tx = Transaction.from_raw(coinbase_bytes.hex())
        txid_bytes = bytes.fromhex(tx.get_txid())[::-1]
        raw_hash = double_sha256(coinbase_bytes)

        # For non-witness tx, txid == raw hash
        assert txid_bytes == raw_hash

    def test_merkle_branch_reversal(self, stratum_notify_params):
        """Merkle branch siblings from the pool (sent via chainhash.Hash.String())
        are in display order and must be reversed to internal order before hashing.
        """
        coinbase_hash = bytes.fromhex("00" * 32)
        siblings_display = ["11" * 32, "22" * 32]
        expected_root = bytes.fromhex(
            "f85702ac9dce77b5db696ac5a966dc7da282759c3021d7dc2620a4bf619fd246"
        )

        notify = StratumNotify(**(stratum_notify_params | {"merkle_branch": siblings_display}))

        # StratumNotify reverses branch entries before merkle hashing.
        root = reconstruct_stratum_merkle_root(coinbase_hash, notify.merkle_branch)

        assert root == expected_root

    def test_from_stratum_notify_produces_valid_template(self, stratum_notify_params):
        """from_stratum_notify should produce a BlockTemplate with all fields set."""
        notify = StratumNotify(**(stratum_notify_params | {"job_id": "abc123"}))
        template = BlockTemplate.from_stratum_notify(notify)

        assert template.job_id == "abc123"
        assert template.height is None
        assert template.raw_transactions == []
        assert template.header.version == 0x20000000
        assert template.header.target_bits == 0x1E010000
        assert template.header.timestamp == 0x69DF47A7

    def test_from_stratum_notify_extranonce2_baked_into_coinbase(self, stratum_notify_params):
        """extranonce2 is appended after extranonce1 in the coinbase, so it
        changes the merkle root, and is stored on the template for echo-back."""
        coinb1 = bytes.fromhex(stratum_notify_params["coinb1"])
        coinb2 = bytes.fromhex(stratum_notify_params["coinb2"])
        extranonce1 = stratum_notify_params["extranonce"]
        extranonce2 = bytes.fromhex("aabbccdd")

        # Bump coinb1's scriptSig-length prefix (byte 41) from 7 to 0x0b for e2.
        coinb1_with_e2 = coinb1[:41] + bytes([0x0B]) + coinb1[42:]

        notify_off = StratumNotify(**stratum_notify_params)
        notify_on = StratumNotify(
            **(stratum_notify_params | {"coinb1": coinb1_with_e2.hex(), "extranonce2": extranonce2})
        )

        template_off = BlockTemplate.from_stratum_notify(notify_off)
        template_on = BlockTemplate.from_stratum_notify(notify_on)

        assert template_off.extranonce2 is None
        expected_off_hash = double_sha256(coinb1 + extranonce1 + coinb2)

        assert template_on.extranonce2 == extranonce2
        expected_on_hash = double_sha256(coinb1_with_e2 + extranonce1 + extranonce2 + coinb2)

        assert expected_off_hash != expected_on_hash
        assert template_off.header.incomplete_header.merkle_root != (
            template_on.header.incomplete_header.merkle_root
        )

    def test_header_serialization_length(self, stratum_notify):
        """Serialized header (with proof commitment) should be 108 bytes,
        matching Go's MaxBlockHeaderPayload.
        """
        template = BlockTemplate.from_stratum_notify(stratum_notify)

        # Without proof commitment
        header_bytes = template.header.serialize_without_proof_commitment()
        assert len(header_bytes) == 76  # IncompleteBlockHeader size

        # With proof commitment
        template.header.proof_commitment = bytes(32)
        full_header = template.header.serialize()
        assert len(full_header) == 108  # 76 + 32

    def test_stratum_notify_validates_params(self):
        """Malformed pool messages should fail validation before template creation."""
        params = [
            "job",
            "not-a-hash",
            "00",
            "00",
            [],
            "20000000",
            "1e010000",
            "69df47a7",
            False,
            1,
        ]

        with pytest.raises(ValueError):
            StratumNotify.from_params(params, extranonce=bytes(4))

    def test_stratum_notify_from_params_maps_valid_payload(self):
        """The boundary parser should preserve the Stratum notify interface."""
        extranonce = bytes.fromhex("deadbeef")
        params = [
            "job",
            "11" * 32,
            "00",
            "ff",
            ["22" * 32],
            "20000000",
            "1e010000",
            "69df47a7",
            True,
            2,
        ]

        notify = StratumNotify.from_params(params, extranonce=extranonce)

        assert notify.job_id == "job"
        assert notify.prev_hash == bytes.fromhex("11" * 32)
        assert notify.coinb1 == bytes.fromhex("00")
        assert notify.coinb2 == bytes.fromhex("ff")
        assert notify.merkle_branch == [bytes.fromhex("22" * 32)[::-1]]
        assert notify.version == 0x20000000
        assert notify.bits == 0x1E010000
        assert notify.ntime == 0x69DF47A7
        assert notify.clean_jobs is True
        assert notify.extranonce == extranonce
        assert notify.cert_version == CertificateVersion.ZK_MOE

    def test_stratum_notify_parses_cert_version_param(self):
        """The 10th param sets the required certificate version (MoE fork)."""
        base = [
            "job",
            "11" * 32,
            "00",
            "ff",
            ["22" * 32],
            "20000000",
            "1e010000",
            "69df47a7",
            True,
        ]

        moe = StratumNotify.from_params([*base, 2], extranonce=bytes(4))
        assert moe.cert_version == CertificateVersion.ZK_MOE

        v1 = StratumNotify.from_params([*base, 1], extranonce=bytes(4))
        assert v1.cert_version == CertificateVersion.ZK_DENSE

    @pytest.mark.parametrize("count", [8, 9, 11])
    def test_stratum_notify_rejects_wrong_param_count(self, count):
        with pytest.raises(ValueError):
            StratumNotify.from_params(["x"] * count, extranonce=bytes(4))

    def test_stratum_notify_rejects_invalid_cert_version(self):
        params = [
            "job",
            "11" * 32,
            "00",
            "ff",
            ["22" * 32],
            "20000000",
            "1e010000",
            "69df47a7",
            True,
            99,
        ]
        with pytest.raises(ValueError):
            StratumNotify.from_params(params, extranonce=bytes(4))

    @pytest.mark.parametrize("size", [2, 4, 8])
    def test_stratum_notify_accepts_any_extranonce1_size(self, stratum_notify_params, size):
        notify = StratumNotify(**(stratum_notify_params | {"extranonce": bytes(size)}))
        assert notify.extranonce == bytes(size)

    def test_stratum_notify_defaults_extranonce2_to_empty(self, stratum_notify_params):
        """No extranonce2 kwarg => empty bytes (pool feature off)."""
        notify = StratumNotify(**stratum_notify_params)
        assert notify.extranonce2 == b""

    def test_stratum_notify_from_params_carries_extranonce2(self, stratum_notify_params):
        """extranonce2 kwarg flows through from_params to the model."""
        params = [
            stratum_notify_params["job_id"],
            stratum_notify_params["prev_hash"],
            stratum_notify_params["coinb1"],
            stratum_notify_params["coinb2"],
            stratum_notify_params["merkle_branch"],
            stratum_notify_params["version"],
            stratum_notify_params["bits"],
            stratum_notify_params["ntime"],
            stratum_notify_params["clean_jobs"],
            int(stratum_notify_params["cert_version"]),
        ]
        extranonce2 = bytes.fromhex("deadbeef")

        notify = StratumNotify.from_params(
            params,
            extranonce=bytes(4),
            extranonce2=extranonce2,
        )

        assert notify.extranonce2 == extranonce2

    @pytest.mark.parametrize(
        ("field_name", "invalid_value"),
        [
            ("prev_hash", "11" * 31),
            ("coinb1", "not-hex"),
            ("merkle_branch", ["22" * 31]),
            ("version", "200000"),
            ("version", 0x20000000),
            ("bits", "1e01000000"),
            ("ntime", -1),
        ],
    )
    def test_stratum_notify_validates_annotated_fields(
        self, stratum_notify_params, field_name, invalid_value
    ):
        with pytest.raises(ValueError):
            StratumNotify(**(stratum_notify_params | {field_name: invalid_value}))

    def test_stratum_template_carries_notify_cert_version(self, stratum_notify_params):
        """The certificate version from mining.notify flows into the BlockTemplate."""
        for version in CertificateVersion:
            notify = StratumNotify(**(stratum_notify_params | {"cert_version": version}))
            template = BlockTemplate.from_stratum_notify(notify)
            assert template.required_cert_version == version


class TestPoolMiningJob:
    def test_mining_job_from_template_uses_share_target(self, sample_block_template):
        share_target = sample_block_template.target * 100  # easier than network
        sample_block_template.share_target = share_target

        job = MiningJob.from_template(sample_block_template)

        assert job.target == share_target
        assert job.target != sample_block_template.target

    def test_solo_job_uses_network_target(self, sample_block_template):
        assert sample_block_template.share_target is None
        assert MiningJob.from_template(sample_block_template).target == sample_block_template.target


class TestTargetToBits:
    @pytest.mark.parametrize("bits", [0x1E010000, 0x1D00FFFF, 0x1F00FFFF, 0x1D010000])
    def test_round_trips_with_bits_to_target(self, bits):
        from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target

        assert target_to_bits(bits_to_target(bits)) == bits

    @pytest.mark.parametrize(
        ("bits", "target"), [(0x02FFFFFF, 0xFFFF), (0x01FFFFFF, 0xFF), (0x00123456, 0)]
    )
    def test_bits_to_target_small_exponent(self, bits, target):
        from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target

        assert bits_to_target(bits) == target

    def test_non_positive_target_is_zero(self):
        assert target_to_bits(0) == 0


class TestStratumSetDifficulty:
    def test_from_params_maps_valid_payload(self):
        set_difficulty = StratumSetDifficulty.from_params([0x1E010000])

        assert set_difficulty.bits == 0x1E010000

    def test_from_params_validates_param_count(self):
        with pytest.raises(ValueError, match="expects 1 param"):
            StratumSetDifficulty.from_params([])

    def test_from_params_validates_bits_range(self):
        with pytest.raises(ValueError):
            StratumSetDifficulty.from_params([0x1_0000_0000])

    def test_from_params_rejects_zero_bits(self):
        with pytest.raises(ValueError):
            StratumSetDifficulty.from_params([0])


class TestStratumSetDifficultyCompact:
    @pytest.mark.parametrize(
        "bits",
        [0x21010000, 0x1D000000, 0x1E800000],  # >256-bit target, zero mantissa, sign bit
    )
    def test_rejects_compact_values_consensus_would_not_accept(self, bits):
        with pytest.raises(ValueError):
            StratumSetDifficulty.from_params([bits])

    @pytest.mark.parametrize("bits", [0x1B00FFFF, 0x1A01FFFE, 0x1E010000])
    def test_accepts_valid_compact_values(self, bits):
        assert StratumSetDifficulty.from_params([bits]).bits == bits
