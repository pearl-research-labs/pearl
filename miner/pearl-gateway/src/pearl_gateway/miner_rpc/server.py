import asyncio
import contextlib
import json
import os
from collections import deque
from typing import Any

import fastjsonschema
from miner_utils import get_logger
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    IncompleteBlockHeader,
    PlainProof,
    PlainProofV4,
    verify_plain_proof_for_cert_version,
)

from pearl_gateway.comm.dataclasses import MiningJob, MiningPausedError
from pearl_gateway.config import MinerRpcConfig
from pearl_gateway.miner_rpc.schemas import (
    validate_get_mining_info,
    validate_jsonrpc,
    validate_submit_plain_proof,
)
from pearl_gateway.submission_service import SubmissionService
from pearl_gateway.work_cache import WorkCache

logger = get_logger(__name__)

# V4 FP8 plain proofs exceed asyncio's default 64 KiB stream limit.
READER_BUFFER_LIMIT = 4 * 2**20


class MinerRpcServer:
    """Line-delimited JSON-RPC 2.0 server over Unix Domain Socket or loopback TCP."""

    def __init__(
        self,
        work_cache: WorkCache,
        submission_service: SubmissionService,
        config: MinerRpcConfig,
    ) -> None:
        self.work_cache = work_cache
        self.submission_service = submission_service
        self.config = config
        self.server: asyncio.Server | None = None
        self.clients: set[asyncio.StreamWriter] = set()
        # Bounded intake: excess submissions are rejected at the RPC boundary rather
        # than queued unboundedly while the single proving worker catches up. A deque
        # (not asyncio.Queue) so stale entries can be evicted when it fills.
        self._pending: deque[tuple[PlainProof | PlainProofV4, MiningJob]] = deque()
        self._max_pending = max(1, config.max_pending)
        self._pending_ready = asyncio.Event()
        self._submission_consumer: asyncio.Task[None] | None = None

    @staticmethod
    def _jsonrpc_success(result: Any, request_id: int | None) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "result": result, "id": request_id}

    @staticmethod
    def _jsonrpc_error(
        code: int, message: str, request_id: int | None, data: str | None = None
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "error": error, "id": request_id}

    async def start(self) -> None:
        if self.config.transport == "uds":
            assert self.config.socket_path is not None
            if os.path.exists(self.config.socket_path):
                os.unlink(self.config.socket_path)
            self.server = await asyncio.start_unix_server(
                self._handle_client, self.config.socket_path, limit=READER_BUFFER_LIMIT
            )
            os.chmod(self.config.socket_path, 0o600)
            logger.info(f"Miner RPC server listening on Unix socket: {self.config.socket_path}")
        else:
            self.server = await asyncio.start_server(
                self._handle_client, "127.0.0.1", self.config.port, limit=READER_BUFFER_LIMIT
            )
            logger.info(f"Miner RPC server listening on TCP: 127.0.0.1:{self.config.port}")

        self._submission_consumer = asyncio.create_task(self._consume_submissions())

    async def stop(self) -> None:
        if self.server is not None:
            for writer in list(self.clients):
                if not writer.is_closing():
                    writer.close()
                    with contextlib.suppress(asyncio.CancelledError):
                        await writer.wait_closed()
            self.clients.clear()
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Intake is closed above; cancel the consumer, dropping any queued submissions.
        if self._submission_consumer is not None:
            self._submission_consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._submission_consumer
            self._submission_consumer = None

        if (
            self.config.transport == "uds"
            and self.config.socket_path
            and os.path.exists(self.config.socket_path)
        ):
            os.unlink(self.config.socket_path)
        logger.info("Miner RPC server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.clients.add(writer)
        try:
            while True:
                try:
                    line = await reader.readline()
                    if not line:
                        break
                    response = await self._process_request(line.decode().strip())
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"Error processing request: {e}")
                    response = self._jsonrpc_error(-32000, str(e), request_id=None)
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
        except Exception as e:
            logger.exception(f"Client handler error: {e}")
        finally:
            self.clients.discard(writer)
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(asyncio.CancelledError):
                    await writer.wait_closed()

    def _validate_params(
        self, validator: Any, params: Any, request_id: int | None
    ) -> dict[str, Any] | None:
        try:
            validator(params)
        except fastjsonschema.exceptions.JsonSchemaException as e:
            return self._jsonrpc_error(-32602, "Invalid params", request_id, data=str(e))
        return None

    async def _process_request(self, request_line: str) -> dict[str, Any]:
        request_id = None
        try:
            body = json.loads(request_line)
            try:
                validate_jsonrpc(body)
            except fastjsonschema.exceptions.JsonSchemaException as e:
                envelope_id = body.get("id") if isinstance(body, dict) else None
                return self._jsonrpc_error(-32600, "Invalid Request", envelope_id, data=str(e))

            method = body["method"]
            params = body.get("params", {})
            request_id = body["id"]

            if method == "getMiningInfo":
                if error := self._validate_params(validate_get_mining_info, params, request_id):
                    return error
                job = await self.work_cache.get_mining_job()
                return self._jsonrpc_success(job.to_dict(), request_id)

            if method == "submitPlainProof":
                return self._accept_plain_proof(params, request_id)

            return self._jsonrpc_error(-32601, f"Method {method} not found", request_id)

        except MiningPausedError as e:
            logger.info(f"Mining paused: {e}")
            return self._jsonrpc_error(e.code, str(e), request_id)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in request: {e}")
            return self._jsonrpc_error(-32700, "Parse error", request_id=None, data=str(e))
        except Exception as e:
            logger.exception(f"Error handling RPC request: {e}")
            return self._jsonrpc_error(-32000, str(e), request_id)

    def _accept_plain_proof(self, params: dict[str, Any], request_id: int | None) -> dict[str, Any]:
        """Validate, apply backpressure, and enqueue a submitPlainProof request."""
        if error := self._validate_params(validate_submit_plain_proof, params, request_id):
            return error

        # The job is cheap to parse; the (miner-controlled) proof is parsed only after
        # the checks. No awaits until append, so the queue cannot fill in between.
        mining_job = MiningJob.from_dict(params["mining_job"])
        current = self._current_header()
        if mining_job.incomplete_header_bytes != current:
            logger.info("Rejecting submitPlainProof for a stale job")
            return self._jsonrpc_error(-32000, "stale job", request_id)

        if len(self._pending) >= self._max_pending:
            # Entries for a previous job would be dropped at dequeue anyway; evict them
            # so they cannot crowd out a proof for the current job.
            self._pending = deque(
                entry for entry in self._pending if entry[1].incomplete_header_bytes == current
            )
        if len(self._pending) >= self._max_pending:
            logger.warning("Proof submission queue full, rejecting submitPlainProof")
            return self._jsonrpc_error(-32000, "proof submission queue full", request_id)

        proof_type = (
            PlainProofV4 if int(mining_job.cert_version) == CERT_VERSION_PLAIN_FP8 else PlainProof
        )
        plain_proof = proof_type.from_base64(params["plain_proof"])
        self._pending.append((plain_proof, mining_job))
        self._pending_ready.set()
        return self._jsonrpc_success("submitted", request_id)

    def _current_header(self) -> bytes | None:
        template = self.work_cache.current_template
        return template.header.serialize_without_proof_commitment() if template else None

    async def _consume_submissions(self) -> None:
        """Process queued submissions one at a time (there is a single proving worker)."""
        while True:
            while not self._pending:
                self._pending_ready.clear()
                await self._pending_ready.wait()
            plain_proof, mining_job = self._pending.popleft()
            try:
                await self.handle_submit_plain_proof(plain_proof, mining_job)
            except Exception:
                # Keep consuming: one bad submission must not stall the pipeline.
                logger.exception("Error handling queued submission")

    async def handle_submit_plain_proof(
        self, plain_proof: PlainProof | PlainProofV4, mining_job: MiningJob
    ) -> None:
        """Handle a submission; re-checks the job is still current after the queue wait."""
        template = self.work_cache.current_template
        if template is None:
            logger.info("No current job; dropping queued submission")
            return

        current_header = template.header.serialize_without_proof_commitment()
        if mining_job.incomplete_header_bytes != current_header:
            logger.info("Job changed since submission was queued; dropping stale proof")
            return
        if template.share_bits is not None:
            # Pool mode: check what the proof achieved, not the target the miner
            # searched against, so a proof that still meets a raised share target
            # (or the network target) is kept. Cheap next to proving, and the
            # binding holds the GIL, so a thread hop would not unblock the loop.
            ok, msg = verify_plain_proof_for_cert_version(
                int(template.required_cert_version),
                IncompleteBlockHeader.from_bytes(current_header),
                plain_proof,
                nbits_override=template.share_bits,
            )
            if not ok:
                logger.info(f"Proof does not meet the current share target; dropping: {msg}")
                return

        result = await self.submission_service.submit_plain_proof(plain_proof, template)
        logger.info(f"Block submission result: {result}")
