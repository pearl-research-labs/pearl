import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target
from pearl_gateway.comm.stratum import AsyncJSONRPCClient
from pearl_gateway.config import PoolConfig
from pearl_gateway.pool_stratum_client import PoolStratumClient
from pearl_gateway.work_cache import WorkCache

NOTIFY_COINB1 = (
    "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0704"
)
NOTIFY_COINB2 = (
    "0101ffffffff0100f2052a01000000232102deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    "deadbeefdeadbeefdeadbeefac00000000"
)


class FakeStreamWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        self._closing = True

    def is_closing(self) -> bool:
        return self._closing


class FakeStreamReader:
    def __init__(self) -> None:
        self._lines: asyncio.Queue[bytes] = asyncio.Queue()

    def feed_line(self, msg: dict[str, object] | str) -> None:
        if isinstance(msg, dict):
            line = json.dumps(msg) + "\n"
        else:
            line = msg if msg.endswith("\n") else msg + "\n"
        self._lines.put_nowait(line.encode())

    async def readline(self) -> bytes:
        return await self._lines.get()


def mining_notify_params(
    job_id: str = "job-1", clean_jobs: bool = False, cert_version: int = 1
) -> list[object]:
    return [
        job_id,
        "00" * 32,
        NOTIFY_COINB1,
        NOTIFY_COINB2,
        [],
        "20000000",
        "1e010000",
        "69df47a7",
        clean_jobs,
        cert_version,
    ]


@pytest.fixture
def pool_config() -> PoolConfig:
    return PoolConfig(stratum_url="stratum+tcp://127.0.0.1:3333")


@pytest.fixture
def work_cache() -> WorkCache:
    return WorkCache()


def make_connected_client(
    config: PoolConfig, work_cache: WorkCache
) -> tuple[PoolStratumClient, FakeStreamReader, FakeStreamWriter]:
    client = PoolStratumClient(config, "prl1miner1", work_cache)
    reader = FakeStreamReader()
    writer = FakeStreamWriter()
    client._conn._reader = reader
    client._conn._writer = writer
    client._conn._rpc_client = AsyncJSONRPCClient(writer)
    client._conn._running = True
    return client, reader, writer


def latest_request(writer: FakeStreamWriter) -> dict[str, object]:
    return json.loads(writer.writes[-1].decode())


async def wait_for_write_count(writer: FakeStreamWriter, count: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(writer.writes) >= count:
            return
        await asyncio.sleep(0)
    raise TimeoutError(f"Expected {count} writes, got {len(writer.writes)}")


def start_listener(client: PoolStratumClient) -> asyncio.Task:
    return asyncio.create_task(client._conn._listen())


async def stop_listener(listen_task: asyncio.Task) -> None:
    listen_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await listen_task


class TestPoolStratumClient:
    @pytest.mark.asyncio
    async def test_subscribe_and_authorize_success(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        assert subscribe_req["method"] == "mining.subscribe"
        assert subscribe_req["params"] == ["pearl-gateway/1.0"]
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef"]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        assert authorize_req["method"] == "mining.authorize"
        assert authorize_req["params"] == ["prl1miner1.gateway", "x"]
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert client._extranonce == bytes.fromhex("deadbeef")
        # Two-element subscribe result (no extranonce2_size) => feature off.
        assert client._extranonce2_size == 0

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_subscribe_captures_extranonce2_size_when_advertised(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """The third subscribe element (extranonce2_size) enables the feature when > 0."""
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef", 4]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert client._extranonce == bytes.fromhex("deadbeef")
        assert client._extranonce2_size == 4

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_subscribe_treats_zero_extranonce2_size_as_off(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """extranonce2_size=0 is the same as omitting it — feature off."""
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef", 0]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert client._extranonce2_size == 0

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_subscribe_rejects_oversized_extranonce2_size(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """A pool advertising an absurdly large extranonce2_size is clamped to
        0 (feature off) rather than letting os.urandom allocate huge buffers."""
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef", 999999]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert client._extranonce2_size == 0

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_subscribe_rejects_negative_extranonce2_size(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """A negative extranonce2_size is treated as feature off."""
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef", -1]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert client._extranonce2_size == 0

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_authorize_sends_password_when_configured(self, work_cache: WorkCache):
        config = PoolConfig(
            stratum_url="stratum+tcp://127.0.0.1:3333",
            password="hunter2",
        )
        client, reader, writer = make_connected_client(config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef"]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        assert authorize_req["method"] == "mining.authorize"
        assert authorize_req["params"] == ["prl1miner1.gateway", "hunter2"]
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_resubscribe_replaces_template_with_new_extranonce(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        async def handshake(extranonce1: str) -> None:
            task = asyncio.create_task(client._subscribe_and_authorize())
            start = len(writer.writes)
            await wait_for_write_count(writer, start + 1)
            reader.feed_line({"id": latest_request(writer)["id"], "result": [[], extranonce1]})
            await wait_for_write_count(writer, start + 2)
            reader.feed_line({"id": latest_request(writer)["id"], "result": True})
            await task

        await handshake("00000001")
        await client._on_mining_notify(mining_notify_params(job_id="job-1"))
        await client._on_set_difficulty([0x1E010000])
        first = work_cache.current_template

        # Disconnect: the dead session's job and share target are dropped.
        await client._on_disconnect()
        assert work_cache.current_template is None
        assert client.share_target is None and client.share_bits is None

        # Reconnect: new extranonce1, and the pool re-sends the same job without clean_jobs.
        await handshake("00000002")
        await client._on_mining_notify(mining_notify_params(job_id="job-1"))
        second = work_cache.current_template

        assert second is not first
        assert "00000002" in second.coinbase_tx.serialize()
        assert second.coinbase_tx.serialize() != first.coinbase_tx.serialize()

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_new_job_id_on_the_same_block_replaces_the_template(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, _, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)

        await client._on_mining_notify(mining_notify_params(job_id="job-1"))
        first = work_cache.current_template
        await client._on_mining_notify(mining_notify_params(job_id="job-2"))  # same block
        second = work_cache.current_template
        await client._on_mining_notify(mining_notify_params(job_id="job-2"))  # repeat

        assert second is not first and second.job_id == "job-2"
        assert work_cache.current_template is second  # a repeat keeps proofs in flight valid

    @pytest.mark.asyncio
    async def test_notify_sent_during_the_handshake_uses_the_new_extranonce(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        task = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)
        # The pool writes the subscribe reply and work in one burst, before authorize.
        reader.feed_line({"id": latest_request(writer)["id"], "result": [[], "00000003"]})
        reader.feed_line({"method": "mining.set_difficulty", "params": [0x1E010000]})
        reader.feed_line({"method": "mining.notify", "params": mining_notify_params(job_id="j")})
        await wait_for_write_count(writer, 2)
        assert work_cache.current_template is None  # held until the handshake finishes
        reader.feed_line({"id": latest_request(writer)["id"], "result": True})
        await task

        template = work_cache.current_template
        assert template is not None and template.job_id == "j"
        assert "00000003" in template.coinbase_tx.serialize()
        assert template.share_bits == 0x1E010000

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_handshake_holds_at_most_the_latest_notify_and_difficulty(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, _, _ = make_connected_client(pool_config, work_cache)

        for i in range(1000):  # a flood before the session is ready stays bounded
            await client._handle_notification("mining.notify", mining_notify_params(f"j{i}"))
            await client._handle_notification("mining.set_difficulty", [0x1E010000 + i])
            await client._handle_notification("mining.unknown", ["x" * 100])

        assert set(client._deferred) == {"mining.notify", "mining.set_difficulty"}
        assert client._deferred["mining.notify"][0] == "j999"
        assert client._deferred["mining.set_difficulty"] == [0x1E010000 + 999]

    @pytest.mark.asyncio
    async def test_subscribe_and_authorize_sends_difficulty_in_password(
        self, work_cache: WorkCache
    ):
        config = PoolConfig(stratum_url="stratum+tcp://127.0.0.1:3333", difficulty=128)
        client, reader, writer = make_connected_client(config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)
        reader.feed_line({"id": latest_request(writer)["id"], "result": [[], "deadbeef"]})

        await wait_for_write_count(writer, 2)
        authorize_req = latest_request(writer)
        assert authorize_req["method"] == "mining.authorize"
        assert authorize_req["params"][1] == "x;d=128"
        reader.feed_line({"id": authorize_req["id"], "result": True})

        await handshake
        assert len(writer.writes) == 2  # no mining.suggest_difficulty
        await stop_listener(listen_task)

    @pytest.mark.parametrize(
        ("password", "difficulty", "expected"),
        [
            ("x", None, "x"),
            ("x", 32768, "x;d=32768"),
            ("", 1, "d=1"),
        ],
    )
    def test_authorize_password(self, work_cache: WorkCache, password, difficulty, expected):
        config = PoolConfig(
            stratum_url="stratum+tcp://127.0.0.1:3333", password=password, difficulty=difficulty
        )
        client, _, _ = make_connected_client(config, work_cache)
        assert client.authorize_password == expected

    @pytest.mark.asyncio
    async def test_subscribe_and_authorize_failure(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        handshake = asyncio.create_task(client._subscribe_and_authorize())
        await wait_for_write_count(writer, 1)

        subscribe_req = latest_request(writer)
        reader.feed_line({"id": subscribe_req["id"], "result": [[], "deadbeef"]})
        await wait_for_write_count(writer, 2)

        authorize_req = latest_request(writer)
        reader.feed_line({"id": authorize_req["id"], "result": False})

        with pytest.raises(ConnectionError, match="Pool authorization failed"):
            await handshake

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_mining_notify_updates_work_cache(
        self, pool_config: PoolConfig, work_cache: WorkCache, sample_block_template
    ):
        client, _, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)
        client.share_target = bits_to_target(0x1E010000)

        await client._on_mining_notify(mining_notify_params(job_id="job-42"))

        template = work_cache.current_template
        assert template is not None
        assert template.job_id == "job-42"
        assert template.share_target == client.share_target

        await work_cache.update_template(sample_block_template)
        with patch.object(work_cache, "update_template", new_callable=AsyncMock) as mock_update:
            await client._on_mining_notify(mining_notify_params(clean_jobs=True))
            mock_update.assert_awaited_once()
            assert mock_update.await_args.kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_mining_notify_without_extranonce2_leaves_template_clean(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """Feature off => template.extranonce2 is None, coinbase is coinb1+e1+coinb2."""
        client, _, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)
        client.share_target = bits_to_target(0x1E010000)
        assert client._extranonce2_size == 0

        await client._on_mining_notify(mining_notify_params(job_id="job-1"))

        template = work_cache.current_template
        assert template is not None
        assert template.extranonce2 is None

    @pytest.mark.asyncio
    async def test_mining_notify_with_extranonce2_bakes_bytes_into_template(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """Feature on => one extranonce2 per job is baked into the coinbase, so
        the merkle root differs from the no-extranonce2 template."""
        from pearl_gateway.blockchain_utils.blockchain_utils import double_sha256
        from pearl_gateway.comm.dataclasses import BlockTemplate, StratumNotify

        client, _, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)
        client.share_target = bits_to_target(0x1E010000)
        client._extranonce2_size = 4

        # coinb1 scriptSig-length byte (41) bumped to 7+4=0x0b to cover e1+e2.
        coinb1_bytes = bytes.fromhex(NOTIFY_COINB1)
        coinb1_with_e2 = (coinb1_bytes[:41] + bytes([0x0B]) + coinb1_bytes[42:]).hex()
        params = mining_notify_params(job_id="job-1")
        params[2] = coinb1_with_e2

        await client._on_mining_notify(params)

        template = work_cache.current_template
        assert template is not None
        assert template.extranonce2 is not None
        assert len(template.extranonce2) == 4

        coinb1 = bytes.fromhex(params[2])
        coinb2 = bytes.fromhex(params[3])
        e1 = client._extranonce
        e2 = template.extranonce2

        with_e2 = double_sha256(coinb1 + e1 + e2 + coinb2)
        without_e2 = double_sha256(coinb1 + e1 + coinb2)
        assert with_e2 != without_e2

        baseline = BlockTemplate.from_stratum_notify(
            StratumNotify.from_params(params, extranonce=e1, extranonce2=e2)
        )
        assert (
            baseline.header.incomplete_header.merkle_root
            == template.header.incomplete_header.merkle_root
        )

    @pytest.mark.asyncio
    async def test_mining_set_difficulty_delegates_to_work_cache(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, _, _ = make_connected_client(pool_config, work_cache)
        expected = bits_to_target(0x1E010000)

        with patch.object(work_cache, "update_share_target", new_callable=AsyncMock) as mock_update:
            await client._on_set_difficulty([0x1E010000])

        assert client.share_target == expected
        assert client.share_bits == 0x1E010000
        mock_update.assert_called_once_with(expected, 0x1E010000)

    @pytest.mark.asyncio
    async def test_listen_dispatches_pool_notifications(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)
        client._session_ready = True
        listen_task = start_listener(client)

        reader.feed_line("not-json\n")
        reader.feed_line({"method": "mining.set_difficulty", "params": [0x1E010000]})
        reader.feed_line(
            {"method": "mining.notify", "params": mining_notify_params(job_id="job-99")}
        )
        await asyncio.sleep(0)

        assert client.share_target == bits_to_target(0x1E010000)
        assert work_cache.current_template is not None
        assert work_cache.current_template.job_id == "job-99"

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_listen_survives_bad_notifications(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, reader, _ = make_connected_client(pool_config, work_cache)
        client._extranonce = bytes(4)
        client._session_ready = True
        listen_task = start_listener(client)

        reader.feed_line({"method": "mining.notify", "params": ["too", "few"]})
        reader.feed_line({"method": "mining.set_difficulty", "params": [0x02FFFFFF]})  # sign bit
        reader.feed_line(
            {"method": "mining.notify", "params": mining_notify_params(job_id="job-ok")}
        )
        await asyncio.sleep(0)

        assert not listen_task.done()
        assert client.share_target is None  # invalid compact rejected, previous target kept
        assert work_cache.current_template is not None
        assert work_cache.current_template.job_id == "job-ok"

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_submit_timeout_drops_the_connection(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        client, _, _ = make_connected_client(pool_config, work_cache)

        with (
            patch.object(client._conn, "call", AsyncMock(side_effect=TimeoutError())),
            patch.object(client._conn, "disconnect") as disconnect,
        ):
            accepted = await client.submit_zk_cert("job-1", "deadbeef", "cert")

        assert accepted is False
        disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_zk_cert_accepted(self, pool_config: PoolConfig, work_cache: WorkCache):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        submit_task = asyncio.create_task(client.submit_zk_cert("job-1", "deadbeef", "certb64"))
        await asyncio.sleep(0)

        submit_req = latest_request(writer)
        assert submit_req["method"] == "mining.submit"
        assert submit_req["params"] == [
            "prl1miner1.gateway",
            "job-1",
            "deadbeef",
            "certb64",
        ]
        reader.feed_line({"id": submit_req["id"], "result": True})

        assert await submit_task is True

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_submit_zk_cert_with_extranonce2_uses_five_param_order(
        self, pool_config: PoolConfig, work_cache: WorkCache
    ):
        """When extranonce2_hex is supplied, submit carries the standard 5-param
        Stratum order: worker, job_id, extranonce2, ntime, zkcert_hex."""
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        submit_task = asyncio.create_task(
            client.submit_zk_cert("job-1", "deadbeef", "certb64", extranonce2_hex="aabbccdd")
        )
        await asyncio.sleep(0)

        submit_req = latest_request(writer)
        assert submit_req["method"] == "mining.submit"
        assert submit_req["params"] == [
            "prl1miner1.gateway",
            "job-1",
            "aabbccdd",
            "deadbeef",
            "certb64",
        ]
        reader.feed_line({"id": submit_req["id"], "result": True})

        assert await submit_task is True

        await stop_listener(listen_task)

    @pytest.mark.asyncio
    async def test_submit_zk_cert_rejected(self, pool_config: PoolConfig, work_cache: WorkCache):
        client, reader, writer = make_connected_client(pool_config, work_cache)
        listen_task = start_listener(client)

        submit_task = asyncio.create_task(client.submit_zk_cert("job-1", "deadbeef", "certb64"))
        await asyncio.sleep(0)

        submit_req = latest_request(writer)
        reader.feed_line({"id": submit_req["id"], "error": [23, "below share difficulty"]})

        assert await submit_task is False

        await stop_listener(listen_task)

    def test_stratum_user_joins_address_and_worker(self, work_cache: WorkCache):
        config = PoolConfig(stratum_url="stratum+tcp://127.0.0.1:3333", worker="myrig")
        assert (
            PoolStratumClient(config, "prl1miner1", work_cache).stratum_user == "prl1miner1.myrig"
        )
