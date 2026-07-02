import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any

import fastjsonschema
from miner_utils import get_logger
from pearl_mining import PlainProof

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

# max proof size should be smaller than this
READER_BUFFER_LIMIT = 2**20


@dataclass
class ClientInfo:
    """Information tracked for each connected miner client."""

    client_id: int
    writer: asyncio.StreamWriter


class MinerRpcServer:
    """
    Raw JSON-RPC server that handles requests from the Miner process over sockets.
    Supports line-delimited JSON-RPC over TCP and Unix Domain Sockets.

    Thread Safety:
        This server uses asyncio and runs entirely in a single thread. Multiple miner
        processes connect via sockets, but each connection is handled by a coroutine
        on the same event loop. All operations on shared state (self.clients, etc.)
        occur between await points and are therefore atomic. No locks are needed.
    """

    def __init__(
        self,
        work_cache: WorkCache,
        submission_service: SubmissionService,
        config: MinerRpcConfig,
        max_pending_submissions: int,
    ):
        self.work_cache = work_cache
        self.submission_service = submission_service
        self.config = config
        self.server: asyncio.Server | None = None
        self.clients: dict[int, ClientInfo] = {}
        self._next_client_id = 0
        # Bounded intake: excess submissions are rejected at the RPC boundary rather
        # than queued unboundedly while the single proving worker catches up.
        self._submission_queue: asyncio.Queue[tuple[PlainProof, MiningJob]] = asyncio.Queue(
            maxsize=max_pending_submissions
        )
        self._submission_consumer: asyncio.Task | None = None

    def _allocate_client_id(self) -> int:
        """Allocate a unique client ID."""
        client_id = self._next_client_id
        self._next_client_id += 1
        return client_id

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

    async def start(self):
        """Start the RPC server on the configured transport."""
        if self.config.transport == "uds":
            await self._start_uds()
        else:
            await self._start_tcp()

        self._submission_consumer = asyncio.create_task(self._consume_submissions())

    async def _start_uds(self):
        """Start the server on a Unix Domain Socket."""
        # Ensure the socket file doesn't already exist
        if os.path.exists(self.config.socket_path):
            os.unlink(self.config.socket_path)

        # Start the server
        self.server = await asyncio.start_unix_server(
            self._handle_client, self.config.socket_path, limit=READER_BUFFER_LIMIT
        )

        # Set permissions on the socket file
        os.chmod(self.config.socket_path, 0o600)

        logger.info(f"Miner RPC server listening on Unix socket: {self.config.socket_path}")

    async def _start_tcp(self):
        """Start the server on a TCP port."""
        self.server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", self.config.port, limit=READER_BUFFER_LIMIT
        )

        logger.info(f"Miner RPC server listening on TCP: 127.0.0.1:{self.config.port}")

    async def stop(self):
        """Stop the RPC server."""
        if self.server:
            # Close all client connections (use list() to avoid dict mutation during iteration)
            for client_info in list(self.clients.values()):
                if not client_info.writer.is_closing():
                    client_info.writer.close()
                    with contextlib.suppress(asyncio.CancelledError):
                        await client_info.writer.wait_closed()
            self.clients.clear()

            # Close the server
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Intake is closed above; cancel the consumer, dropping any queued submissions.
        if self._submission_consumer is not None:
            self._submission_consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._submission_consumer
            self._submission_consumer = None

        # Clean up UDS file if applicable
        if self.config.transport == "uds" and os.path.exists(self.config.socket_path):
            os.unlink(self.config.socket_path)

        logger.info("Miner RPC server stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection."""
        client_id = self._allocate_client_id()
        client_info = ClientInfo(client_id=client_id, writer=writer)
        self.clients[client_id] = client_info
        logger.debug(f"New client connected: {client_id}")

        try:
            # Process JSON-RPC requests
            while True:
                try:
                    # Read line-delimited JSON-RPC request
                    line = await reader.readline()
                    if not line:
                        break  # Client disconnected

                    # Process the request
                    response = await self._process_request(line.decode().strip(), client_info)

                    # Send response
                    response_line = json.dumps(response) + "\n"
                    writer.write(response_line.encode())
                    await writer.drain()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"Error processing request from {client_id}: {e}")
                    response = self._jsonrpc_error(-32000, str(e), request_id=None)
                    response_line = json.dumps(response) + "\n"
                    writer.write(response_line.encode())
                    await writer.drain()

        except Exception as e:
            logger.exception(f"Client handler error for {client_id}: {e}")
        finally:
            # Clean up client - single point of removal
            self.clients.pop(client_id, None)

            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(asyncio.CancelledError):
                    await writer.wait_closed()
            logger.debug(f"Client disconnected: {client_id}")

    def _validate_params(
        self, validator, params: Any, request_id: int | None
    ) -> dict[str, Any] | None:
        """Validate JSON-RPC params, returning an error response on failure or None on success."""
        try:
            validator(params)
        except fastjsonschema.exceptions.JsonSchemaException as e:
            return self._jsonrpc_error(-32602, "Invalid params", request_id, data=str(e))
        return None

    async def _process_request(self, request_line: str, client: ClientInfo) -> dict[str, Any]:
        """Process a single JSON-RPC request with validation."""
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

            logger.debug(f"Processing request: {method}")
            logger.trace(f"request params: {params}")

            if method == "getMiningInfo":
                if error := self._validate_params(validate_get_mining_info, params, request_id):
                    return error
                job = await self.work_cache.get_mining_job()
                return self._jsonrpc_success(job.to_dict(), request_id)

            elif method == "submitPlainProof":
                return self._accept_plain_proof(params, request_id)

            else:
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

        # Checked before parsing the (miner-controlled) proof; no awaits until
        # put_nowait, so the queue cannot fill in between.
        if self._submission_queue.full():
            logger.warning("Proof submission queue full, rejecting submitPlainProof")
            return self._jsonrpc_error(-32000, "proof submission queue full", request_id)

        plain_proof = PlainProof.from_base64(params["plain_proof"])
        mining_job = MiningJob.from_dict(params["mining_job"])
        self._submission_queue.put_nowait((plain_proof, mining_job))
        return self._jsonrpc_success("submitted", request_id)

    async def _consume_submissions(self) -> None:
        """Process queued submissions one at a time (there is a single proving worker)."""
        while True:
            plain_proof, mining_job = await self._submission_queue.get()
            try:
                await self.handle_submit_plain_proof(plain_proof, mining_job)
            except Exception:
                # Keep consuming: one bad submission must not stall the pipeline.
                logger.exception("Error handling queued submission")
            finally:
                self._submission_queue.task_done()

    async def handle_submit_plain_proof(
        self, plain_proof: PlainProof, mining_job: MiningJob
    ) -> None:
        """Handle a submission; re-checks the job is still current after the queue wait."""
        template = self.work_cache.current_template
        if template is None:
            logger.info("No current job; dropping queued submission")
            return

        logger.debug(f"Submitting plain proof for {mining_job.to_dict()=} and {plain_proof=}")

        current_header = template.header.serialize_without_proof_commitment()
        if mining_job.incomplete_header_bytes != current_header:
            logger.info("Job changed since submission was queued; dropping stale proof")
            return

        # Submit the block via the submission service
        result = await self.submission_service.submit_plain_proof(plain_proof, template)

        logger.info(f"Block submission result: {result}")
