import asyncio
import contextlib
import json
import os
from typing import Any

import fastjsonschema
from miner_utils import get_logger
from pearl_mining import CERT_VERSION_PLAIN_FP8, PlainProof, PlainProofV4

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
        self._background_tasks: set[asyncio.Task[None]] = set()

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

    async def stop(self) -> None:
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

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

    def _spawn_submission(
        self, plain_proof: PlainProof | PlainProofV4, mining_job: MiningJob
    ) -> None:
        task = asyncio.create_task(self.handle_submit_plain_proof(plain_proof, mining_job))
        self._background_tasks.add(task)

        def _done(done_task: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done_task)
            if not done_task.cancelled() and (exc := done_task.exception()) is not None:
                logger.error(f"Background proof submission failed: {exc}")

        task.add_done_callback(_done)

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
                if error := self._validate_params(validate_submit_plain_proof, params, request_id):
                    return error
                mining_job = MiningJob.from_dict(params["mining_job"])
                proof_type = (
                    PlainProofV4
                    if int(mining_job.cert_version) == CERT_VERSION_PLAIN_FP8
                    else PlainProof
                )
                plain_proof = proof_type.from_base64(params["plain_proof"])
                self._spawn_submission(plain_proof, mining_job)
                return self._jsonrpc_success("submitted", request_id)

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

    async def handle_submit_plain_proof(
        self, plain_proof: PlainProof | PlainProofV4, mining_job: MiningJob
    ) -> None:
        template = self.work_cache.current_template
        if template is None:
            logger.warning("Dropping submitted proof: no active block template.")
            return
        if (
            mining_job.incomplete_header_bytes
            != template.header.serialize_without_proof_commitment()
        ):
            logger.warning("Submitted block with old header. Skipping submission.")
            return
        result = await self.submission_service.submit_plain_proof(plain_proof, template)
        logger.info(f"Block submission result: {result}")
