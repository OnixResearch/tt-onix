"""Serialized OpenAI-compatible HTTP shell for Qwen3.8-27B on P150x2."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _blocks_for
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from .core import (
    CompletionRequest,
    Endpoint,
    RequestValidationError,
    build_completion_response,
    parse_completion_request,
    parse_endpoint,
)

EXPECTED_PCIE_DEVICE_IDS = [0, 1]
MESH_ROWS = 1
MESH_COLUMNS = 2
EXPECTED_LAYER_COUNT = 64
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 2
TRACE_REGION_SIZE_BYTES = 1_073_741_824
MAX_BATCH_SIZE = 1
PAGED_KV_BLOCK_SIZE = 64
ADMITTED_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_MAX_SEQUENCE_LENGTH = ADMITTED_MAX_SEQUENCE_LENGTH
DEFAULT_MAX_GENERATION_TOKENS = 64
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8000
MINIMUM_LISTEN_PORT = 1
MAXIMUM_LISTEN_PORT = 65_535
MAX_REQUEST_BODY_BYTES = 1_048_576
P300_CACHE_RELATIVE_PATH = Path("P300/tensor_cache_bfp8_mesh1x2")
MODEL_OBJECT_CREATION_TIME = 0


class ServiceStop(Exception):
    """A process signal requested an orderly device shutdown."""


def emit(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def render_prompt(request: CompletionRequest, tokenizer: Any) -> str:
    if request.endpoint is Endpoint.COMPLETIONS:
        assert request.prompt is not None
        return request.prompt
    messages = [
        {"content": message.content, "role": message.role}
        for message in request.messages
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


class QwenEngine:
    """Imperative adapter from admitted requests to the TTNN paged path."""

    def __init__(
        self,
        model: Qwen36Model,
        tokenizer: Any,
        model_alias: str,
        max_sequence_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_alias = model_alias
        self.max_sequence_length = max_sequence_length

    def _run_paged_generation(
        self, token_ids: torch.Tensor, maximum_generation_tokens: int
    ) -> tuple[list[int], dict[str, float]]:
        prompt_token_count = token_ids.shape[1]
        num_blocks = _blocks_for(prompt_token_count, maximum_generation_tokens)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        kv_cache_shape = [
            num_blocks,
            self.model.args.n_local_kv_heads,
            PAGED_KV_BLOCK_SIZE,
            self.model.args.head_dim,
        ]
        self.model.allocate_kv_caches(
            kv_cache_shape, ttnn.bfloat16, batch_size=MAX_BATCH_SIZE
        )

        prefill_started = time.monotonic()
        logits_device = self.model.prefill_traced_chunked(
            token_ids,
            page_table,
            actual_len=prompt_token_count,
        )
        ttnn.synchronize_device(self.model.device)
        logits = ttnn.to_torch(
            logits_device,
            mesh_composer=ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=0),
        ).reshape(-1, self.model.args.vocab_size)[0]
        next_token = int(torch.argmax(logits.float()).item())
        generated = [next_token]
        ttft_seconds = time.monotonic() - prefill_started

        decode_durations = []
        position = prompt_token_count
        while len(generated) < maximum_generation_tokens:
            decode_started = time.monotonic()
            device_inputs = self.model.prepare_inputs_decode(
                torch.tensor([[next_token]], dtype=torch.int32),
                torch.tensor([position], dtype=torch.int32),
                page_table,
            )
            output, _ = self.model.ttnn_decode_forward(
                device_inputs[0],
                device_inputs[1],
                rot_mat_idxs=device_inputs[2],
                page_table=device_inputs[3],
            )
            decode_logits = self.model.process_output_decode(output, MAX_BATCH_SIZE)
            next_token = int(
                torch.argmax(
                    decode_logits.reshape(-1)[: self.model.args.vocab_size].float()
                ).item()
            )
            generated.append(next_token)
            position += 1
            decode_durations.append(time.monotonic() - decode_started)

        steady_decode_durations = (
            decode_durations[1:] if len(decode_durations) > 1 else decode_durations
        )
        if steady_decode_durations:
            average_decode_seconds = sum(steady_decode_durations) / len(
                steady_decode_durations
            )
            decode_tokens_per_second = 1.0 / average_decode_seconds
        else:
            decode_tokens_per_second = 0.0
        return generated, {
            "decode_tok_s": decode_tokens_per_second,
            "ttft_s": ttft_seconds,
        }

    def generate(self, request: CompletionRequest) -> dict[str, Any]:
        prompt = render_prompt(request, self.tokenizer)
        token_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_token_count = token_ids.shape[1]
        total_token_count = prompt_token_count + request.max_tokens
        if total_token_count > self.max_sequence_length:
            raise RequestValidationError(
                f"prompt and generation require {total_token_count} tokens; limit is {self.max_sequence_length}"
            )

        started = time.monotonic()
        try:
            generated, perf = self._run_paged_generation(token_ids, request.max_tokens)
            generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            response = build_completion_response(
                request=request,
                request_id=f"cmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                generated_text=generated_text,
                prompt_token_count=prompt_token_count,
                generated_token_count=len(generated),
            )
            emit(
                "request-passed",
                decode_tokens_per_second=perf["decode_tok_s"],
                elapsed_seconds=time.monotonic() - started,
                generated_token_count=len(generated),
                prompt_token_count=prompt_token_count,
            )
            return response
        finally:
            self.model.free_kv_caches()
            gc.collect()


def make_handler(
    engine: QwenEngine, maximum_generation_tokens: int
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _write_error(self, status: HTTPStatus, message: str) -> None:
            self._write_json(
                status, {"error": {"message": message, "type": "invalid_request_error"}}
            )

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._write_json(
                    HTTPStatus.OK, {"model": engine.model_alias, "status": "ok"}
                )
                return
            if self.path == "/v1/models":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "data": [
                            {
                                "created": MODEL_OBJECT_CREATION_TIME,
                                "id": engine.model_alias,
                                "object": "model",
                                "owned_by": "OnixResearch",
                            }
                        ],
                        "object": "list",
                    },
                )
                return
            self._write_error(
                HTTPStatus.NOT_FOUND, f"unsupported endpoint: {self.path}"
            )

        def do_POST(self) -> None:
            try:
                endpoint = parse_endpoint(self.path)
                content_length_header = self.headers.get("Content-Length")
                if content_length_header is None:
                    raise RequestValidationError("Content-Length is required")
                try:
                    content_length = int(content_length_header)
                except ValueError as error:
                    raise RequestValidationError(
                        "Content-Length must be an integer"
                    ) from error
                if not 0 < content_length <= MAX_REQUEST_BODY_BYTES:
                    raise RequestValidationError(
                        f"request body must contain between 1 and {MAX_REQUEST_BODY_BYTES} bytes"
                    )
                try:
                    payload = json.loads(
                        self.rfile.read(content_length).decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RequestValidationError(
                        "request body must be valid UTF-8 JSON"
                    ) from error

                request = parse_completion_request(
                    payload=payload,
                    endpoint=endpoint,
                    admitted_model=engine.model_alias,
                    maximum_generation_tokens=maximum_generation_tokens,
                )
                self._write_json(HTTPStatus.OK, engine.generate(request))
            except RequestValidationError as error:
                self._write_error(HTTPStatus.BAD_REQUEST, str(error))
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                emit("request-failed", error_type=type(error).__name__)
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, "inference failed")

        def log_message(self, format: str, *args: Any) -> None:
            emit("http", client=self.client_address[0], message=format % args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-alias", default="Qwen3.8-27B")
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument(
        "--max-generation-tokens", type=int, default=DEFAULT_MAX_GENERATION_TOKENS
    )
    return parser.parse_args()


def stop_service(_signal_number: int, _frame: Any) -> None:
    raise ServiceStop


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute() or not model_path.is_dir():
        raise SystemExit("--model-path must be an existing absolute directory")
    if not (model_path / "config.json").is_file():
        raise SystemExit("--model-path does not contain config.json")
    if not (model_path / P300_CACHE_RELATIVE_PATH).is_dir():
        raise SystemExit("--model-path does not contain the P300 tensor cache")
    if not MINIMUM_LISTEN_PORT <= args.port <= MAXIMUM_LISTEN_PORT:
        raise SystemExit("--port is outside the TCP port range")
    if args.max_sequence_length <= 0:
        raise SystemExit("--max-sequence-length must be positive")
    if args.max_sequence_length > ADMITTED_MAX_SEQUENCE_LENGTH:
        raise SystemExit(
            f"--max-sequence-length exceeds the admitted limit {ADMITTED_MAX_SEQUENCE_LENGTH}"
        )
    if args.max_generation_tokens <= 0:
        raise SystemExit("--max-generation-tokens must be positive")
    if args.max_generation_tokens > args.max_sequence_length:
        raise SystemExit(
            "--max-generation-tokens must not exceed --max-sequence-length"
        )

    os.environ["HF_MODEL"] = str(model_path)
    os.environ["TT_CACHE_PATH"] = str(model_path)

    pcie_device_ids = list(ttnn.get_pcie_device_ids())
    if pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
        raise SystemExit(
            f"expected physical PCIe devices {EXPECTED_PCIE_DEVICE_IDS}, got {pcie_device_ids}"
        )

    signal.signal(signal.SIGINT, stop_service)
    signal.signal(signal.SIGTERM, stop_service)
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    mesh_device = None
    model = None
    server = None
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(MESH_ROWS, MESH_COLUMNS),
            l1_small_size=L1_SMALL_SIZE_BYTES,
            num_command_queues=COMMAND_QUEUE_COUNT,
            trace_region_size=TRACE_REGION_SIZE_BYTES,
        )
        if mesh_device.get_num_devices() != len(EXPECTED_PCIE_DEVICE_IDS):
            raise RuntimeError("P150x2 mesh did not open both admitted devices")
        mesh_device.enable_program_cache()
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=MAX_BATCH_SIZE,
            max_seq_len=args.max_sequence_length,
        )
        if len(model.layers) != EXPECTED_LAYER_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_LAYER_COUNT} layers, got {len(model.layers)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        engine = QwenEngine(
            model, tokenizer, args.model_alias, args.max_sequence_length
        )
        server = HTTPServer(
            (args.host, args.port), make_handler(engine, args.max_generation_tokens)
        )
        emit(
            "ready",
            host=args.host,
            layer_count=len(model.layers),
            model=args.model_alias,
            port=args.port,
            simulated=False,
        )
        try:
            server.serve_forever()
        except ServiceStop:
            emit("stopping")
    finally:
        if server is not None:
            server.server_close()
        if model is not None:
            model.free_kv_caches()
            del model
            gc.collect()
        try:
            if mesh_device is not None:
                for submesh in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
