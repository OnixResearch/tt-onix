"""Serialized OpenAI-compatible HTTP shell for Qwen3.8-27B on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence]

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import signal
import threading
import time
import uuid
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _blocks_for
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from .batch_admission_core import (
    BatchAdmissionPolicyError,
    BatchAdmissionProfile,
)
from .batch_decode_adapter import BatchedTracedDecodeAdapter
from .batch_service import BatchGenerationCoordinator
from .core import (
    CompletionRequest,
    Endpoint,
    RequestValidationError,
    build_completion_response,
    finish_generation,
    load_stop_token_ids,
    parse_completion_request,
    parse_endpoint,
)
from .decode_adapter import DecodeAdapterError, TracedDecodeAdapter
from .decode_core import (
    DecodePolicyError,
    DecodeProfile,
    request_deadline_expired,
)
from .mtp_drafter_core import MtpDrafterProfile
from .mtp_serving_core import MTP_PREFILL_BUCKET
from .mtp_trace_core import MtpTraceProfile
from .prefill_profile_adapter import RealtimePrefillProfiler
from .prefill_profile_core import select_measurement_mode
from .readiness_core import (
    StartupReadiness,
    begin_warmup,
    complete_warmup,
    fail_warmup,
    initial_readiness,
    plan_startup_warmup,
    record_warm_width,
)
from .stream_core import (
    SSE_DONE,
    IncrementalDetokenizer,
    response_text,
    sse_event,
    stream_chunk,
    usage_chunk,
)
from .structured_read_core import (
    DeviceRead,
    ReadPorts,
    SchemaError,
    decide,
    jev_response,
    parse_jev_request,
)
from .systemd_notify_adapter import notify_ready
from .trace_reuse_core import TraceReusePolicyError, TraceReuseProfile

EXPECTED_PCIE_DEVICE_IDS = [0, 1]
HOST_TIMELINE_PREFILL_START = "QWEN_PREFILL_START"
HOST_TIMELINE_PREFILL_END = "QWEN_PREFILL_END"
MESH_ROWS = 1
MESH_COLUMNS = 2
EXPECTED_LAYER_COUNT = 64
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 2
TRACE_REGION_SIZE_BYTES = 1_073_741_824
EAGER_BATCH_SIZE = 1
MAXIMUM_BATCH_WIDTH = 16
PAGED_KV_BLOCK_SIZE = 64
# One request's page-table row (decode-profile.ncl); rows come from the pooled KV cache.
ADMITTED_MAX_SEQUENCE_LENGTH = 65536
DEFAULT_MAX_SEQUENCE_LENGTH = ADMITTED_MAX_SEQUENCE_LENGTH
DEFAULT_MAX_GENERATION_TOKENS = 8192
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8000
MINIMUM_LISTEN_PORT = 1
MAXIMUM_LISTEN_PORT = 65_535
MAX_REQUEST_BODY_BYTES = 1_048_576
SYSTEMONE_PATH = "/v1/systemone"
# A thought before a structured read uses the chat template's brief reasoning mode.
STRUCTURED_THINK_REASONING_EFFORT = "low"
THOUGHT_CLOSE_TOKEN = "</think>"
THOUGHT_CLOSE_TEXT = "\n</think>\n\n"
P300_CACHE_RELATIVE_PATH = Path("P300/tensor_cache_bfp8_mesh1x2")
MODEL_OBJECT_CREATION_TIME = 0
DECODE_MODE_EAGER = "eager"
DECODE_MODE_CANDIDATE = "candidate"
DECODE_MODE_ADMITTED = "admitted"
DECODE_MODES = (DECODE_MODE_EAGER, DECODE_MODE_CANDIDATE, DECODE_MODE_ADMITTED)
STARTUP_WARMUP_WIDTHS = (1, 4)
STARTUP_WARMUP_PROMPT = "Warm up the production inference path."
STARTUP_WARMUP_GENERATION_TOKENS = 2
SYSTEMD_READY_STATUS = "Qwen startup warm-up complete"


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
    """Imperative shell for eager and bounded traced P150x2 decode."""

    def __init__(
        self,
        model: Qwen36Model,
        tokenizer: Any,
        model_alias: str,
        max_sequence_length: int,
        *,
        model_path: str,
        decode_mode: str,
        decode_profile: DecodeProfile,
        runtime_identity: str,
        package_identity: str,
        model_identity: str,
        enable_prefill_profile: bool,
        enable_host_timeline_profile: bool,
        batch_admission_profile: BatchAdmissionProfile | None = None,
        batch_admission_enabled: bool = False,
        trace_reuse_profile: TraceReuseProfile | None = None,
        trace_reuse_admission_enabled: bool = False,
        sampled_serving_enabled: bool = False,
        mtp_serving_enabled: bool = False,
        mtp_drafter_profile: Any | None = None,
        mtp_trace_profile: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_alias = model_alias
        self.max_sequence_length = max_sequence_length
        self.model_path = model_path
        # Generation ends at the first end-of-sequence token. Admission measurements keep
        # the fixed max_tokens length their receipts assume.
        self.stop_token_ids = (
            frozenset()
            if batch_admission_enabled or trace_reuse_admission_enabled
            else load_stop_token_ids(model_path, tokenizer)
        )
        self.decode_mode = decode_mode
        self.decode_profile = decode_profile
        self.sampled_serving_enabled = sampled_serving_enabled
        self.mtp_serving_enabled = mtp_serving_enabled
        self._mtp_plain_step_seconds: float | None = None
        if mtp_serving_enabled:
            if decode_mode != DECODE_MODE_EAGER:
                raise ValueError(
                    "MTP serving requires serialized eager mode, not batch admission"
                )
            if mtp_drafter_profile is None or mtp_trace_profile is None:
                raise ValueError("MTP serving requires drafter and trace profiles")
        self._mtp_drafter: Any | None = None
        self._mtp_drafter_profile = mtp_drafter_profile
        self._mtp_trace_profile = mtp_trace_profile
        self.runtime_identity = runtime_identity
        self.package_identity = package_identity
        self.model_identity = model_identity
        self._host_timeline_profile = enable_host_timeline_profile
        self._host_timeline_signpost = None
        if enable_host_timeline_profile:
            from tracy import signpost

            self._host_timeline_signpost = signpost
        self._prefill_profiler = (
            RealtimePrefillProfiler(
                ttnn.device,
                expected_chip_ids=EXPECTED_PCIE_DEVICE_IDS,
            )
            if enable_prefill_profile
            else None
        )
        candidate_requested = decode_mode == DECODE_MODE_CANDIDATE
        admitted_requested = (
            decode_mode == DECODE_MODE_ADMITTED and decode_profile.physically_admitted
        )
        self._decode_adapter: TracedDecodeAdapter | None = None
        self._batch_adapter = (
            BatchedTracedDecodeAdapter(
                model=model,
                profile=decode_profile,
                maximum_width=MAXIMUM_BATCH_WIDTH,
            )
            if candidate_requested or admitted_requested
            else None
        )
        self._startup_fallback_reason = (
            None
            if self._batch_adapter is not None
            else (
                "profile-not-physically-admitted"
                if decode_mode == DECODE_MODE_ADMITTED
                else "eager-selected"
            )
        )
        self._eager_lock = threading.Lock()
        self._startup_readiness: StartupReadiness | None = None
        self._batch_coordinator = (
            BatchGenerationCoordinator(
                adapter=self._batch_adapter,
                tokenizer=tokenizer,
                model=model,
                profile=decode_profile,
                model_alias=model_alias,
                model_identity=model_identity,
                package_identity=package_identity,
                runtime_identity=runtime_identity,
                maximum_sequence_length=max_sequence_length,
                physical_pcie_device_ids=tuple(EXPECTED_PCIE_DEVICE_IDS),
                emitter=emit,
                batch_admission_profile=batch_admission_profile,
                batch_admission_enabled=batch_admission_enabled,
                sampled_admission_enabled=os.environ.get("QWEN36_SAMPLED_FUSION", "0")
                == "1",
                trace_reuse_profile=trace_reuse_profile,
                trace_reuse_admission_enabled=trace_reuse_admission_enabled,
                stop_token_ids=self.stop_token_ids,
            )
            if self._batch_adapter is not None
            else None
        )

    @property
    def requires_startup_warmup(self) -> bool:
        return self._batch_coordinator is not None

    def warm_up(self) -> None:
        """Warm bounded production shapes before the service accepts traffic."""
        if self._batch_coordinator is None:
            return
        plan = plan_startup_warmup(
            widths=STARTUP_WARMUP_WIDTHS,
            prompt=STARTUP_WARMUP_PROMPT,
            generation_tokens=STARTUP_WARMUP_GENERATION_TOKENS,
            maximum_width=MAXIMUM_BATCH_WIDTH,
        )
        state = begin_warmup(initial_readiness(plan))
        self._startup_readiness = state
        token_ids = self.tokenizer(plan.prompt, return_tensors="pt").input_ids
        started = time.monotonic()
        emit("startup-warmup-started", widths=plan.widths)
        try:
            observations = self._batch_coordinator.warm_up(plan, token_ids)
            for observation in observations:
                state = record_warm_width(
                    state,
                    width=observation.width,
                    elapsed_seconds=observation.elapsed_seconds,
                )
            state = complete_warmup(
                state,
                total_elapsed_seconds=time.monotonic() - started,
            )
        except (
            AssertionError,
            DecodeAdapterError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            self._startup_readiness = fail_warmup(state)
            emit(
                "startup-warmup-failed",
                completed_widths=state.completed_widths,
            )
            raise
        self._startup_readiness = state
        emit(
            "startup-warmup-passed",
            elapsed_seconds=state.total_elapsed_seconds,
            widths=state.completed_widths,
        )
        self._emit_device_dram()

    def _emit_device_dram(self) -> None:
        """Record DRAM left after the KV pool, traces and weights, as the allocator sees it."""
        try:
            view = ttnn.get_memory_view(self.model.mesh_device, ttnn.BufferType.DRAM)
            banks = int(view.num_banks)
            fields = {
                "allocated_bytes": banks * int(view.total_bytes_allocated_per_bank),
                "free_bytes": banks * int(view.total_bytes_free_per_bank),
                "largest_free_bytes_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
                "banks": banks,
            }
        except (AttributeError, RuntimeError, TypeError) as error:
            emit("device-dram-unavailable", error_type=type(error).__name__)
            return
        emit("device-dram", **fields)

    def readiness_payload(self) -> dict[str, Any]:
        state = self._startup_readiness
        if state is None:
            speculative_status = (
                {
                    "speculative_decode": {
                        "enabled": True,
                        "candidate": True,
                        "plain_step_seconds": self._mtp_plain_step_seconds,
                    }
                }
                if self.mtp_serving_enabled
                else {}
            )
            return {
                **speculative_status,
                "ready": True,
                "startup_warmup": {
                    "elapsed_seconds": None,
                    "required": False,
                    "widths": [],
                },
            }
        if not state.ready or state.total_elapsed_seconds is None:
            raise RuntimeError("service readiness was queried before warm-up completed")
        return {
            "ready": True,
            "startup_warmup": {
                "elapsed_seconds": state.total_elapsed_seconds,
                "required": True,
                "widths": list(state.completed_widths),
            },
        }

    def _profiled_prefill_device(
        self,
        token_ids: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_token_count: int,
    ) -> Any:
        """Run the device prefill region exposed to partial Tracy profiling."""
        logits_device = self.model.prefill_traced_chunked(
            token_ids,
            page_table,
            actual_len=prompt_token_count,
        )
        ttnn.synchronize_device(self.model.device)
        return logits_device

    def _prefill_first_token(
        self, token_ids: torch.Tensor, page_table: torch.Tensor
    ) -> tuple[int, dict[str, Any]]:
        """Run prefill and return admission or diagnostic phase observations."""
        prompt_token_count = token_ids.shape[1]
        program_profile = None
        profile_drain_seconds = 0.0
        if self._prefill_profiler is not None:
            self._prefill_profiler.begin()
        if self._host_timeline_signpost is not None:
            self._host_timeline_signpost(HOST_TIMELINE_PREFILL_START)
        prefill_started = time.monotonic()
        try:
            logits_device = self._profiled_prefill_device(
                token_ids,
                page_table,
                prompt_token_count=prompt_token_count,
            )
            prefill_device_finished = time.monotonic()
            if self._host_timeline_signpost is not None:
                self._host_timeline_signpost(HOST_TIMELINE_PREFILL_END)
            if self._prefill_profiler is not None:
                profile_drain_started = time.monotonic()
                program_profile = self._prefill_profiler.finish()
                profile_drain_seconds = time.monotonic() - profile_drain_started
        except (AssertionError, OSError, RuntimeError, ValueError):
            if self._prefill_profiler is not None:
                self._prefill_profiler.cancel()
            raise
        logits = ttnn.to_torch(
            logits_device,
            mesh_composer=ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=0),
        ).reshape(-1, self.model.args.vocab_size)[0]
        next_token = int(torch.argmax(logits.float()).item())
        first_token_finished = time.monotonic()
        measurement_mode, admission_eligible = select_measurement_mode(
            prefill_program_profiled=program_profile is not None,
            host_timeline_profiled=self._host_timeline_profile,
        )
        return next_token, {
            "admission_eligible": admission_eligible,
            "first_token_readback_s": first_token_finished - prefill_device_finished,
            "measurement_mode": measurement_mode,
            "prefill_device_s": prefill_device_finished - prefill_started,
            "prefill_page_table_blocks": int(page_table.shape[1]),
            "prefill_profile_receiver_drain_s": profile_drain_seconds,
            "prefill_program_profile": program_profile,
            "ttft_s": first_token_finished - prefill_started,
        }

    def _run_paged_generation(
        self, token_ids: torch.Tensor, maximum_generation_tokens: int
    ) -> tuple[list[int], dict[str, Any]]:
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
            kv_cache_shape, ttnn.bfloat16, batch_size=EAGER_BATCH_SIZE
        )

        next_token, prefill_metrics = self._prefill_first_token(token_ids, page_table)
        generated = [next_token]

        decode_durations = []
        position = prompt_token_count
        while len(generated) < maximum_generation_tokens and generated[-1] not in self.stop_token_ids:
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
            decode_logits = self.model.process_output_decode(output, EAGER_BATCH_SIZE)
            next_token = int(
                torch.argmax(
                    decode_logits.reshape(-1)[: self.model.args.vocab_size].float()
                ).item()
            )
            generated.append(next_token)
            position += 1
            decode_durations.append(time.monotonic() - decode_started)

        return generated, self._decode_metrics(
            decode_durations,
            prefill_metrics,
            decode_path=DECODE_MODE_EAGER,
            decode_page_table_blocks=num_blocks,
            fallback_reason=self._startup_fallback_reason,
            trace_prepare_seconds=0.0,
        )

    def _run_persistent_generation(
        self,
        token_ids: torch.Tensor,
        maximum_generation_tokens: int,
        request_deadline: float,
    ) -> tuple[list[int], dict[str, Any]]:
        assert self._decode_adapter is not None
        prompt_token_count = token_ids.shape[1]
        prefill_page_table = self._decode_adapter.prefill_page_table(
            prompt_tokens=prompt_token_count,
            generation_tokens=maximum_generation_tokens,
        )
        next_token, prefill_metrics = self._prefill_first_token(
            token_ids, prefill_page_table
        )
        generated = [next_token]

        position = prompt_token_count
        trace_prepare_started = time.monotonic()
        eligibility = self._decode_adapter.ensure_trace(
            token=next_token,
            position=position,
        )
        trace_prepare_seconds = time.monotonic() - trace_prepare_started
        use_trace = eligibility.eligible
        fallback_reason = None if use_trace else eligibility.reason
        decode_path = "traced" if use_trace else DECODE_MODE_EAGER
        decode_durations = []
        while len(generated) < maximum_generation_tokens and generated[-1] not in self.stop_token_ids:
            if request_deadline_expired(time.monotonic(), request_deadline):
                raise DecodeAdapterError("request timeout exceeded")
            decode_started = time.monotonic()
            if use_trace:
                next_token = self._decode_adapter.decode_traced(
                    token=next_token,
                    position=position,
                )
            else:
                next_token = self._decode_adapter.decode_eager(
                    token=next_token,
                    position=position,
                )
            generated.append(next_token)
            position += 1
            decode_durations.append(time.monotonic() - decode_started)

        return generated, self._decode_metrics(
            decode_durations,
            prefill_metrics,
            decode_path=decode_path,
            decode_page_table_blocks=self._decode_adapter.decode_page_table_blocks,
            fallback_reason=fallback_reason,
            trace_prepare_seconds=trace_prepare_seconds,
        )

    def _ensure_mtp_drafter(self) -> Any:
        """Construct the MTP drafter once, gated and lazily, on first use."""
        if self._mtp_drafter is None:
            from .mtp_drafter_adapter import MtpDrafterAdapter

            if self._mtp_drafter_profile is None or self._mtp_trace_profile is None:
                raise RuntimeError(
                    "MTP serving requires the drafter and trace profiles"
                )
            if self._decode_adapter is None:
                self._decode_adapter = TracedDecodeAdapter(
                    model=self.model,
                    profile=self.decode_profile,
                    runtime_identity=self.runtime_identity,
                    package_identity=self.package_identity,
                    model_identity=self.model_identity,
                    slot_id="mtp-serving-slot",
                    buffer_generation=1,
                )
            self._mtp_drafter = MtpDrafterAdapter(
                model=self.model,
                decode_adapter=self._decode_adapter,
                model_path=self.model_path,
                profile=self._mtp_drafter_profile,
                trace_profile=self._mtp_trace_profile,
                runtime_identity=self.runtime_identity,
            )
        return self._mtp_drafter

    def _run_mtp_generation(
        self,
        token_ids: torch.Tensor,
        maximum_generation_tokens: int,
        request_deadline: float,
    ) -> tuple[list[int], dict[str, Any]]:
        import statistics

        drafter = self._ensure_mtp_drafter()
        if self._mtp_plain_step_seconds is None:
            calibration_tokens = 8
            _, calibration = self._run_persistent_generation(
                token_ids,
                calibration_tokens,
                request_deadline,
            )
            if calibration["decode_path"] != "traced":
                raise RuntimeError("MTP calibration requires the traced plain path")
            self._mtp_plain_step_seconds = statistics.median(
                calibration["token_latency_s"]
            )
        result = drafter.run_speculative_completion(
            prompt_tokens=token_ids,
            maximum_generation_tokens=maximum_generation_tokens,
            request_deadline=request_deadline,
            plain_step_seconds=self._mtp_plain_step_seconds,
        )
        generated_count = len(result["token_ids"])
        decode_seconds = max(result["elapsed_seconds"], 1.0e-9)
        perf = {
            "decode_path": "mtp-speculative",
            "decode_tok_s": (generated_count - 1) / decode_seconds,
            "elapsed_seconds": result["elapsed_seconds"],
            "fallback_reason": result["disengaged_reason"],
            "measurement_mode": "gated-mtp-speculative-greedy",
            "mtp_accepted_total": result["accepted_total"],
            "mtp_disengaged_reason": result["disengaged_reason"],
            "mtp_prefill_seconds": result["prefill_seconds"],
            "mtp_rejected_rounds": result["rejected_rounds"],
            "mtp_rounds": result["rounds"],
            "mtp_round_seconds": result.get("round_seconds", ()),
            "mtp_round_stage_seconds": result.get("round_stage_seconds", ()),
            "mtp_trace_capture_seconds": result.get("trace_capture_seconds", 0.0),
            "mtp_plain_step_seconds": self._mtp_plain_step_seconds,
            "mtp_minimum_verification_margin": result.get(
                "minimum_verification_margin"
            ),
            "profile_id": self.decode_profile.profile_id,
        }
        return list(result["token_ids"]), perf

    @staticmethod
    def _decode_metrics(
        decode_durations: list[float],
        prefill_metrics: dict[str, Any],
        *,
        decode_path: str,
        decode_page_table_blocks: int,
        fallback_reason: str | None,
        trace_prepare_seconds: float,
    ) -> dict[str, Any]:
        steady_durations = (
            decode_durations[1:] if len(decode_durations) > 1 else decode_durations
        )
        if steady_durations:
            average_decode_seconds = sum(steady_durations) / len(steady_durations)
            decode_tokens_per_second = 1.0 / average_decode_seconds
            tail_token_latency_seconds = max(steady_durations)
        else:
            decode_tokens_per_second = 0.0
            tail_token_latency_seconds = 0.0
        return {
            **prefill_metrics,
            "decode_page_table_blocks": decode_page_table_blocks,
            "decode_path": decode_path,
            "decode_tok_s": decode_tokens_per_second,
            "fallback_reason": fallback_reason,
            "tail_token_latency_s": tail_token_latency_seconds,
            "token_latency_s": steady_durations,
            "trace_prepare_s": trace_prepare_seconds,
        }

    def generate(
        self,
        request: CompletionRequest,
        token_sink: Callable[[int], None] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate one completion.

        `token_sink` receives each generated token as the batch worker decodes it, for
        streaming. The MTP and eager routes deliver no tokens early; their text arrives with
        the complete response."""
        prompt = render_prompt(request, self.tokenizer)
        token_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_token_count = token_ids.shape[1]
        total_token_count = prompt_token_count + request.max_tokens
        if total_token_count > self.max_sequence_length:
            raise RequestValidationError(
                f"prompt and generation require {total_token_count} tokens; limit is {self.max_sequence_length}"
            )

        request_id = request_id or f"cmpl-{uuid.uuid4().hex}"
        started = time.monotonic()
        request_deadline = started + self.decode_profile.request_timeout_seconds
        if (
            self.mtp_serving_enabled
            and request.temperature in (None, 0.0)
            and prompt_token_count < MTP_PREFILL_BUCKET
            and request.max_tokens > self._mtp_drafter_profile.draft_token_count + 1
        ):
            with self._eager_lock:
                generated, mtp_perf = self._run_mtp_generation(
                    token_ids,
                    request.max_tokens,
                    request_deadline,
                )
            generated, finish_reason = finish_generation(generated, self.stop_token_ids)
            generated_text = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            response = build_completion_response(
                request=request,
                request_id=request_id,
                created=int(time.time()),
                generated_text=generated_text,
                prompt_token_count=prompt_token_count,
                generated_token_count=len(generated),
                finish_reason=finish_reason,
            )
            response["x_onix_decode_evidence"] = {
                "decode_path": mtp_perf["decode_path"],
                "decode_tokens_per_second": mtp_perf["decode_tok_s"],
                "elapsed_seconds": mtp_perf["elapsed_seconds"],
                "fallback_reason": mtp_perf["fallback_reason"],
                "generated_token_ids": generated,
                "measurement_mode": mtp_perf["measurement_mode"],
                "model_identity": self.model_identity,
                "package_identity": self.package_identity,
                "profile_id": self.decode_profile.profile_id,
                "runtime_identity": self.runtime_identity,
                "mtp_accepted_total": mtp_perf["mtp_accepted_total"],
                "mtp_prefill_seconds": mtp_perf["mtp_prefill_seconds"],
                "mtp_rejected_rounds": mtp_perf["mtp_rejected_rounds"],
                "mtp_rounds": mtp_perf["mtp_rounds"],
                "mtp_round_seconds": mtp_perf["mtp_round_seconds"],
                "mtp_round_stage_seconds": mtp_perf["mtp_round_stage_seconds"],
                "mtp_trace_capture_seconds": mtp_perf["mtp_trace_capture_seconds"],
                "mtp_plain_step_seconds": mtp_perf["mtp_plain_step_seconds"],
                "mtp_minimum_verification_margin": mtp_perf[
                    "mtp_minimum_verification_margin"
                ],
                "simulated": False,
            }
            emit(
                "request-passed",
                decode_path=mtp_perf["decode_path"],
                generated_token_count=len(generated),
                mtp_rounds=mtp_perf["mtp_rounds"],
                request_id=request_id,
            )
            return response
        if self._batch_coordinator is not None:
            return self._batch_coordinator.generate(
                request=request,
                request_id=request_id,
                token_ids=token_ids,
                prompt_token_count=prompt_token_count,
                started=started,
                deadline=request_deadline,
                token_sink=token_sink,
            )

        with self._eager_lock:
            response: dict[str, Any] | None = None
            cleanup_started = None
            try:
                generated, perf = self._run_paged_generation(
                    token_ids,
                    request.max_tokens,
                )
                generated, finish_reason = finish_generation(generated, self.stop_token_ids)
                generated_text = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                )
                response = build_completion_response(
                    request=request,
                    request_id=request_id,
                    created=int(time.time()),
                    generated_text=generated_text,
                    prompt_token_count=prompt_token_count,
                    generated_token_count=len(generated),
                    finish_reason=finish_reason,
                )
                response["x_onix_decode_evidence"] = {
                    "admission_eligible": perf["admission_eligible"],
                    "clean_reset": False,
                    "decode_page_table_blocks": perf["decode_page_table_blocks"],
                    "decode_path": perf["decode_path"],
                    "decode_tokens_per_second": perf["decode_tok_s"],
                    "fallback_reason": perf["fallback_reason"],
                    "first_token_readback_seconds": perf["first_token_readback_s"],
                    "generated_token_ids": generated,
                    "measurement_mode": perf["measurement_mode"],
                    "model_identity": self.model_identity,
                    "package_identity": self.package_identity,
                    "physical_pcie_device_ids": EXPECTED_PCIE_DEVICE_IDS,
                    "prefill_device_seconds": perf["prefill_device_s"],
                    "prefill_page_table_blocks": perf["prefill_page_table_blocks"],
                    "prefill_profile_receiver_drain_seconds": perf[
                        "prefill_profile_receiver_drain_s"
                    ],
                    "prefill_program_profile": perf["prefill_program_profile"],
                    "profile_id": self.decode_profile.profile_id,
                    "request_reset_seconds": 0.0,
                    "runtime_identity": self.runtime_identity,
                    "simulated": False,
                    "tail_token_latency_seconds": perf["tail_token_latency_s"],
                    "token_latency_seconds": perf["token_latency_s"],
                    "trace_capture_seconds": None,
                    "trace_prepare_seconds": perf["trace_prepare_s"],
                    "ttft_seconds": perf["ttft_s"],
                }
                emit(
                    "request-passed",
                    decode_path=perf["decode_path"],
                    decode_tokens_per_second=perf["decode_tok_s"],
                    elapsed_seconds=time.monotonic() - started,
                    fallback_reason=perf["fallback_reason"],
                    generated_token_count=len(generated),
                    prompt_token_count=prompt_token_count,
                    request_id=request_id,
                    tail_token_latency_seconds=perf["tail_token_latency_s"],
                    ttft_seconds=perf["ttft_s"],
                )
                return response
            finally:
                cleanup_started = time.monotonic()
                self.model.free_kv_caches()
                gc.collect()
                if response is not None:
                    response["x_onix_decode_evidence"]["clean_reset"] = True
                    response["x_onix_decode_evidence"]["cleanup_seconds"] = (
                        time.monotonic() - cleanup_started
                    )

    @property
    def structured_reads_available(self) -> bool:
        """Structured reads run on the batch worker; serialized eager mode has none."""
        return self._batch_coordinator is not None

    def _structured_read_ports(self, deadline: float, maximum_think_tokens: int) -> ReadPorts:
        coordinator = self._batch_coordinator
        assert coordinator is not None
        tokenizer = self.tokenizer

        def encode(text: str) -> list[int]:
            return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]

        def chat_prompt(system: str, user: str, thinking: bool) -> list[int]:
            template_arguments: dict[str, Any] = {"enable_thinking": thinking}
            if thinking:
                template_arguments["reasoning_effort"] = STRUCTURED_THINK_REASONING_EFFORT
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                add_generation_prompt=True,
                tokenize=True,
                **template_arguments,
            )
            ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
            return [int(token) for token in ids]

        def read(requests: tuple[Any, ...]) -> tuple[DeviceRead, ...]:
            rows = coordinator.read(
                requests=tuple(
                    (torch.tensor([request.token_ids], dtype=torch.int32), request.positions)
                    for request in requests
                ),
                deadline=deadline,
            )
            return tuple(
                DeviceRead(logits=row.logits, seconds=row.elapsed_seconds, route=row.route)
                for row in rows
            )

        def think(prompt_ids: list[int], budget: int) -> list[int]:
            if budget > maximum_think_tokens:
                raise SchemaError(f"think: at most {maximum_think_tokens} tokens on this service")
            if len(prompt_ids) + budget > self.max_sequence_length:
                raise SchemaError(
                    f"the thought needs {len(prompt_ids) + budget} tokens; the service holds "
                    f"{self.max_sequence_length}"
                )
            response = coordinator.generate(
                request=CompletionRequest(
                    endpoint=Endpoint.COMPLETIONS,
                    model=self.model_alias,
                    max_tokens=budget,
                    prompt=None,
                    messages=(),
                    temperature=0.0,
                ),
                request_id=f"think-{uuid.uuid4().hex}",
                token_ids=torch.tensor([prompt_ids], dtype=torch.int64),
                prompt_token_count=len(prompt_ids),
                started=time.monotonic(),
                deadline=deadline,
            )
            return [int(token) for token in response["x_onix_decode_evidence"]["generated_token_ids"]]

        return ReadPorts(
            encode=encode,
            decode=lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
            chat_prompt=chat_prompt,
            read=read,
            think=think,
            noise_upper=min(tokenizer.added_tokens_decoder),
            thought_close_id=int(tokenizer.convert_tokens_to_ids(THOUGHT_CLOSE_TOKEN)),
            thought_close_ids=tuple(encode(THOUGHT_CLOSE_TEXT)),
            maximum_sequence_length=self.max_sequence_length,
        )

    def structured_decide(self, payload: Any, maximum_think_tokens: int) -> dict[str, Any]:
        """One Jev ``/v1/systemone`` decision from structured reads."""
        schema, state, seed = parse_jev_request(payload, maximum_think_tokens=maximum_think_tokens)
        started = time.monotonic()
        ports = self._structured_read_ports(
            started + self.decode_profile.request_timeout_seconds, maximum_think_tokens
        )
        result, output_rows = decide(ports, schema, state, seed)
        timing = result["diagnostics"]["timing"]
        emit(
            "systemone-passed",
            device_reads=timing["device_reads"],
            elapsed_seconds=time.monotonic() - started,
            question_count=len(result["answers"]),
            reads=timing["reads"],
            routes=timing["routes"],
        )
        return jev_response(schema, result, output_rows, self.model_alias)

    def close(self) -> None:
        if self._mtp_drafter is not None:
            self._mtp_drafter.close()
            self._mtp_drafter = None
        if self._batch_coordinator is not None:
            self._batch_coordinator.close()
            self._batch_coordinator = None
        if self._prefill_profiler is not None:
            self._prefill_profiler.close()
        if self._decode_adapter is not None:
            self._decode_adapter.close()
        if self._batch_adapter is not None:
            self._batch_adapter.close()


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
            if self.path in ("/healthz", "/readyz"):
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "model": engine.model_alias,
                        "status": "ok",
                        **engine.readiness_payload(),
                    },
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

        def _read_json_body(self) -> Any:
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
                return json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestValidationError(
                    "request body must be valid UTF-8 JSON"
                ) from error

        def _systemone(self) -> None:
            if not engine.structured_reads_available:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": {
                            "message": "structured reads need the batch decode worker (candidate mode)",
                            "type": "server_error",
                        }
                    },
                )
                return
            try:
                payload = self._read_json_body()
                result = engine.structured_decide(payload, maximum_generation_tokens)
            except RequestValidationError as error:
                self._write_error(HTTPStatus.BAD_REQUEST, str(error))
            except SchemaError as error:
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": {"message": str(error), "type": "validation_error"}},
                )
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                emit("request-failed", endpoint=SYSTEMONE_PATH, error_type=type(error).__name__)
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": "inference failed", "type": "server_error"}},
                )
            else:
                self._write_json(HTTPStatus.OK, result)

        def do_POST(self) -> None:
            if self.path == SYSTEMONE_PATH:
                self._systemone()
                return
            try:
                endpoint = parse_endpoint(self.path)
                payload = self._read_json_body()
                request = parse_completion_request(
                    payload=payload,
                    endpoint=endpoint,
                    admitted_model=engine.model_alias,
                    maximum_generation_tokens=maximum_generation_tokens,
                    sampled_serving_enabled=engine.sampled_serving_enabled,
                )
                if request.stream:
                    self._stream(request)
                    return
                self._write_json(HTTPStatus.OK, engine.generate(request))
            except RequestValidationError as error:
                self._write_error(HTTPStatus.BAD_REQUEST, str(error))
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                emit("request-failed", error_type=type(error).__name__)
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, "inference failed")

        def _stream(self, request: CompletionRequest) -> None:
            """Answer with server-sent events: text deltas as the batch worker decodes them."""
            request_id = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

            def generate() -> None:
                try:
                    response = engine.generate(
                        request,
                        token_sink=lambda token: events.put(("token", token)),
                        request_id=request_id,
                    )
                except Exception as error:  # noqa: BLE001 - reported on the HTTP thread
                    events.put(("error", error))
                else:
                    events.put(("done", response))

            threading.Thread(target=generate, name="qwen36-stream", daemon=True).start()
            kind, value = events.get()
            if kind == "error":
                # Nothing has been sent yet, so the failure stays a plain JSON error.
                if isinstance(value, RequestValidationError):
                    self._write_error(HTTPStatus.BAD_REQUEST, str(value))
                else:
                    emit("request-failed", error_type=type(value).__name__)
                    self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, "inference failed")
                return
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            detokenizer = IncrementalDetokenizer(
                lambda token_ids: engine.tokenizer.decode(token_ids, skip_special_tokens=True),
                engine.stop_token_ids,
            )

            def send(chunk: dict[str, Any]) -> None:
                self.wfile.write(sse_event(chunk))

            try:
                if request.endpoint is Endpoint.CHAT_COMPLETIONS:
                    send(stream_chunk(request, request_id, created, text="", role=True))
                while kind == "token":
                    text = detokenizer.push(value)
                    if text:
                        send(stream_chunk(request, request_id, created, text=text))
                    kind, value = events.get()
                if kind == "error":
                    emit("request-failed", error_type=type(value).__name__, streamed=True)
                    send({"error": {"message": "inference failed", "type": "server_error"}})
                    return
                response = value
                tail = detokenizer.finish(response_text(request, response))
                if detokenizer.diverged:
                    emit("stream-text-diverged", request_id=request_id)
                if tail:
                    send(stream_chunk(request, request_id, created, text=tail))
                final = stream_chunk(
                    request,
                    request_id,
                    created,
                    finish_reason=response["choices"][0]["finish_reason"],
                )
                if "x_onix_decode_evidence" in response:
                    final["x_onix_decode_evidence"] = response["x_onix_decode_evidence"]
                send(final)
                if request.include_usage:
                    send(usage_chunk(request, request_id, created, response["usage"]))
                self.wfile.write(SSE_DONE)
            except (BrokenPipeError, ConnectionResetError):
                emit("stream-client-disconnected", request_id=request_id)

        def log_message(self, format: str, *args: Any) -> None:
            emit("http", client=self.client_address[0], message=format % args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-alias", default="Qwen3.8-27B")
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--batch-admission-profile")
    parser.add_argument("--batch-admission", action="store_true")
    parser.add_argument("--trace-reuse-profile")
    parser.add_argument("--trace-reuse-admission", action="store_true")
    parser.add_argument("--mtp-drafter-profile")
    parser.add_argument("--mtp-trace-profile")
    parser.add_argument(
        "--decode-mode", choices=DECODE_MODES, default=DECODE_MODE_EAGER
    )
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--package-identity", required=True)
    parser.add_argument("--prefill-profile", action="store_true")
    parser.add_argument("--host-timeline-profile", action="store_true")
    parser.add_argument("--require-systemd-notify", action="store_true")
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
    decode_profile_path = Path(args.decode_profile).expanduser()
    if not decode_profile_path.is_absolute() or not decode_profile_path.is_file():
        raise SystemExit("--decode-profile must be an existing absolute file")
    try:
        with decode_profile_path.open(encoding="utf-8") as profile_file:
            decode_profile = DecodeProfile.from_mapping(json.load(profile_file))
    except (DecodePolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"decode profile is invalid: {error}") from error
    batch_admission_profile = None
    if args.batch_admission:
        if args.decode_mode != DECODE_MODE_CANDIDATE:
            raise SystemExit("--batch-admission requires candidate decode mode")
        if args.batch_admission_profile is None:
            raise SystemExit("--batch-admission requires --batch-admission-profile")
        batch_admission_profile_path = Path(args.batch_admission_profile).expanduser()
        if (
            not batch_admission_profile_path.is_absolute()
            or not batch_admission_profile_path.is_file()
        ):
            raise SystemExit(
                "--batch-admission-profile must be an existing absolute file"
            )
        try:
            with batch_admission_profile_path.open(encoding="utf-8") as profile_file:
                batch_admission_profile = BatchAdmissionProfile.from_mapping(
                    json.load(profile_file)
                )
        except (
            BatchAdmissionPolicyError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise SystemExit(f"batch admission profile is invalid: {error}") from error
    elif args.batch_admission_profile is not None:
        raise SystemExit("--batch-admission-profile requires --batch-admission")
    trace_reuse_profile = None
    if args.trace_reuse_profile is not None:
        trace_reuse_profile_path = Path(args.trace_reuse_profile).expanduser()
        if (
            not trace_reuse_profile_path.is_absolute()
            or not trace_reuse_profile_path.is_file()
        ):
            raise SystemExit("--trace-reuse-profile must be an existing absolute file")
        try:
            with trace_reuse_profile_path.open(encoding="utf-8") as profile_file:
                trace_reuse_profile = TraceReuseProfile.from_mapping(
                    json.load(profile_file)
                )
        except (
            TraceReusePolicyError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise SystemExit(f"trace reuse profile is invalid: {error}") from error
    if args.trace_reuse_admission:
        if args.decode_mode != DECODE_MODE_CANDIDATE:
            raise SystemExit("--trace-reuse-admission requires candidate decode mode")
        if trace_reuse_profile is None:
            raise SystemExit("--trace-reuse-admission requires --trace-reuse-profile")
        if args.batch_admission:
            raise SystemExit("--trace-reuse-admission conflicts with --batch-admission")
    if not model_path.is_absolute() or not model_path.is_dir():
        raise SystemExit("--model-path must be an existing absolute directory")
    if not (model_path / "config.json").is_file():
        raise SystemExit("--model-path does not contain config.json")
    if not (model_path / P300_CACHE_RELATIVE_PATH).is_dir():
        raise SystemExit("--model-path does not contain the P300 tensor cache")
    if not MINIMUM_LISTEN_PORT <= args.port <= MAXIMUM_LISTEN_PORT:
        raise SystemExit("--port is outside the TCP port range")
    if args.prefill_profile and args.host_timeline_profile:
        raise SystemExit("--prefill-profile conflicts with --host-timeline-profile")
    if args.batch_admission and (args.prefill_profile or args.host_timeline_profile):
        raise SystemExit("--batch-admission conflicts with profiling modes")
    if args.trace_reuse_admission and (
        args.prefill_profile or args.host_timeline_profile
    ):
        raise SystemExit("--trace-reuse-admission conflicts with profiling modes")
    if args.max_sequence_length <= 0:
        raise SystemExit("--max-sequence-length must be positive")
    if args.max_sequence_length > ADMITTED_MAX_SEQUENCE_LENGTH:
        raise SystemExit(
            f"--max-sequence-length exceeds the admitted limit {ADMITTED_MAX_SEQUENCE_LENGTH}"
        )
    if args.max_sequence_length > decode_profile.maximum_sequence_length:
        raise SystemExit("--max-sequence-length exceeds the decode profile limit")
    if decode_profile.trace_region_size_bytes != TRACE_REGION_SIZE_BYTES:
        raise SystemExit("decode profile trace-region size does not match the service")
    if model_path.name != decode_profile.checkpoint_revision:
        raise SystemExit("model path does not match the decode profile checkpoint")
    if (
        batch_admission_profile is not None
        and model_path.name != batch_admission_profile.checkpoint_revision
    ):
        raise SystemExit("model path does not match the batch admission checkpoint")
    if (
        trace_reuse_profile is not None
        and model_path.name != trace_reuse_profile.checkpoint_revision
    ):
        raise SystemExit("model path does not match the trace reuse checkpoint")
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
    engine = None
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
        persistent_requested = args.decode_mode == DECODE_MODE_CANDIDATE or (
            args.decode_mode == DECODE_MODE_ADMITTED
            and decode_profile.physically_admitted
        )
        model_batch_size = (
            MAXIMUM_BATCH_WIDTH if persistent_requested else EAGER_BATCH_SIZE
        )
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=model_batch_size,
            max_seq_len=args.max_sequence_length,
        )
        if len(model.layers) != EXPECTED_LAYER_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_LAYER_COUNT} layers, got {len(model.layers)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        engine = QwenEngine(
            model,
            tokenizer,
            args.model_alias,
            args.max_sequence_length,
            model_path=args.model_path,
            decode_mode=args.decode_mode,
            decode_profile=decode_profile,
            runtime_identity=args.runtime_identity,
            package_identity=args.package_identity,
            model_identity=f"{args.model_alias}:{decode_profile.checkpoint_revision}",
            enable_prefill_profile=args.prefill_profile,
            enable_host_timeline_profile=args.host_timeline_profile,
            batch_admission_profile=batch_admission_profile,
            batch_admission_enabled=args.batch_admission,
            trace_reuse_profile=trace_reuse_profile,
            trace_reuse_admission_enabled=args.trace_reuse_admission,
            sampled_serving_enabled=os.environ.get("QWEN36_SAMPLED_FUSION", "0") == "1",
            mtp_serving_enabled=(
                os.environ.get("QWEN36_MTP_SERVING", "0") == "1"
                and args.mtp_drafter_profile is not None
                and args.mtp_trace_profile is not None
            ),
            mtp_drafter_profile=(
                MtpDrafterProfile.from_mapping(
                    json.loads(
                        Path(args.mtp_drafter_profile).read_text(encoding="utf-8")
                    )
                )
                if args.mtp_drafter_profile is not None
                else None
            ),
            mtp_trace_profile=(
                MtpTraceProfile.from_mapping(
                    json.loads(Path(args.mtp_trace_profile).read_text(encoding="utf-8"))
                )
                if args.mtp_trace_profile is not None
                else None
            ),
        )
        engine.warm_up()
        server = ThreadingHTTPServer(
            (args.host, args.port), make_handler(engine, args.max_generation_tokens)
        )
        server.daemon_threads = True
        readiness = engine.readiness_payload()
        systemd_notified = notify_ready(
            SYSTEMD_READY_STATUS,
            required=args.require_systemd_notify,
        )
        emit(
            "ready",
            decode_mode=args.decode_mode,
            decode_profile=decode_profile.profile_id,
            host=args.host,
            layer_count=len(model.layers),
            model=args.model_alias,
            port=args.port,
            batch_admission=args.batch_admission,
            trace_reuse_admission=args.trace_reuse_admission,
            trace_reuse_profile=(
                trace_reuse_profile.profile_id
                if trace_reuse_profile is not None
                else None
            ),
            host_timeline_profile=args.host_timeline_profile,
            prefill_profile=args.prefill_profile,
            simulated=False,
            startup_warmup=readiness["startup_warmup"],
            systemd_notified=systemd_notified,
        )
        try:
            server.serve_forever()
        except ServiceStop:
            emit("stopping")
    finally:
        if server is not None:
            server.server_close()
            server = None
        if engine is not None:
            engine.close()
            engine = None
        if model is not None:
            model.free_kv_caches()
            model = None
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
