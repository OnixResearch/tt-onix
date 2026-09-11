"""TTNN adapter for the Qwen same-process TTFT control."""

# r[impl tenstorrent.qwen38_p150x2.traced_ttft_control]

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn

from .decode_adapter import TracedDecodeAdapter
from .decode_core import required_page_table_blocks
from .ttft_control_core import (
    CAPACITY_CACHE_TRACED,
    REQUEST_CACHE_UNTRACED,
    TtftControlEvidence,
    TtftControlIdentity,
    TtftControlProfile,
    TtftControlSample,
)

CACHE_PAIR_WIDTH = 2
PROMPT_RANK = 2


class TtftControlAdapterError(RuntimeError):
    """The TTFT control adapter cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class TtftControlRun:
    evidence: TtftControlEvidence
    prompt_token_count: int
    trace_capture_seconds: float


class TtftControlAdapter:
    """Own two stable KV sets and run the balanced TTFT route schedule."""

    def __init__(
        self,
        *,
        model: Any,
        decode_adapter: TracedDecodeAdapter,
        profile: TtftControlProfile,
        identity: TtftControlIdentity,
        physical_pcie_device_ids: tuple[int, ...],
        host_cpu_affinity: tuple[int, ...],
    ) -> None:
        self.model = model
        self.decode_adapter = decode_adapter
        self.profile = profile
        self.identity = identity
        self.physical_pcie_device_ids = physical_pcie_device_ids
        self.host_cpu_affinity = host_cpu_affinity
        self.mesh = model.mesh_device
        self._capacity_cache = model._paged_kv_caches
        if self._capacity_cache is None:
            raise TtftControlAdapterError("trace-capacity KV cache is missing")
        self._validate_cache_set(
            self._capacity_cache,
            profile.trace_capacity_blocks,
            "trace-capacity",
        )
        self._request_cache = self._allocate_cache_set(profile.request_cache_blocks)
        self._validate_cache_set(
            self._request_cache,
            profile.request_cache_blocks,
            "request-sized",
        )
        self._closed = False

    def run(self, prompt_tokens: torch.Tensor) -> TtftControlRun:
        """Run all roots and return raw evidence for pure evaluation."""
        self._require_open()
        if prompt_tokens.ndim != PROMPT_RANK or prompt_tokens.shape[0] != 1:
            raise TtftControlAdapterError(
                "TTFT control prompt must have batch size one"
            )
        prompt_token_count = int(prompt_tokens.shape[1])
        required_blocks = required_page_table_blocks(
            prompt_tokens=prompt_token_count,
            generation_tokens=self.profile.generated_tokens,
            kv_block_size=self.profile.kv_block_size,
            capacity_blocks=self.profile.trace_capacity_blocks,
        )
        if required_blocks != self.profile.page_table_blocks:
            raise TtftControlAdapterError(
                "prompt and generation do not match the configured page-table width"
            )
        page_table = self.decode_adapter.prefill_page_table(
            prompt_tokens=prompt_token_count,
            generation_tokens=self.profile.generated_tokens,
        )
        if int(page_table.shape[1]) != self.profile.page_table_blocks:
            raise TtftControlAdapterError("bounded page-table width changed")

        samples: list[TtftControlSample] = []
        last_first_token: int | None = None
        trace_capture_seconds = 0.0
        trace_seen = False
        for root_index, route in enumerate(self.profile.route_schedule, start=1):
            wants_trace = route == CAPACITY_CACHE_TRACED
            if wants_trace and not self.decode_adapter.trace_parked:
                if last_first_token is None:
                    raise TtftControlAdapterError(
                        "trace parking lacks a prompt-bound first token"
                    )
                self._bind_cache(self._capacity_cache)
                eligibility = self.decode_adapter.ensure_trace(
                    token=last_first_token,
                    position=prompt_token_count,
                )
                if not eligibility.eligible or not self.decode_adapter.trace_parked:
                    raise TtftControlAdapterError(
                        f"decode trace parking failed: {eligibility.reason}"
                    )
                capture = self.decode_adapter.trace_capture_seconds
                if capture is None or capture <= 0:
                    raise TtftControlAdapterError("trace capture duration is invalid")
                trace_capture_seconds += capture
                trace_seen = True
            elif not wants_trace and self.decode_adapter.trace_parked:
                self.decode_adapter.release_trace()
                if self.decode_adapter.trace_parked:
                    raise TtftControlAdapterError("decode trace release failed")

            cache = (
                self._request_cache
                if route == REQUEST_CACHE_UNTRACED
                else self._capacity_cache
            )
            self._bind_cache(cache)
            for _warmup_index in range(self.profile.warmup_samples):
                last_first_token = self._measure_first_token(
                    prompt_tokens=prompt_tokens,
                    page_table=page_table,
                ).first_token_id
            for sample_index in range(1, self.profile.retained_samples + 1):
                observation = self._measure_first_token(
                    prompt_tokens=prompt_tokens,
                    page_table=page_table,
                )
                last_first_token = observation.first_token_id
                samples.append(
                    TtftControlSample(
                        root_index=root_index,
                        sample_index=sample_index,
                        route=route,
                        cache_blocks=self._cache_blocks(cache),
                        page_table_blocks=int(page_table.shape[1]),
                        trace_parked=self.decode_adapter.trace_parked,
                        first_token_id=observation.first_token_id,
                        prefill_device_seconds=observation.prefill_device_seconds,
                        first_token_readback_seconds=(
                            observation.first_token_readback_seconds
                        ),
                        ttft_seconds=observation.ttft_seconds,
                    )
                )

        if not trace_seen:
            raise TtftControlAdapterError("TTFT control never parked a decode trace")
        if self.decode_adapter.trace_parked:
            self.decode_adapter.release_trace()
        self._bind_cache(self._capacity_cache)
        return TtftControlRun(
            evidence=TtftControlEvidence(
                identity=self.identity,
                physical_pcie_device_ids=self.physical_pcie_device_ids,
                mesh_shape=self.profile.mesh_shape,
                host_cpu_affinity=self.host_cpu_affinity,
                samples=tuple(samples),
                cleanup_complete=False,
            ),
            prompt_token_count=prompt_token_count,
            trace_capture_seconds=trace_capture_seconds,
        )

    def close(self) -> None:
        """Release request-sized buffers and restore the model cache binding."""
        if self._closed:
            return
        self._closed = True
        if self.decode_adapter.trace_parked:
            self.decode_adapter.release_trace()
        self._bind_cache(self._capacity_cache)
        self._deallocate_cache_set(self._request_cache)

    def _measure_first_token(
        self,
        *,
        prompt_tokens: torch.Tensor,
        page_table: torch.Tensor,
    ) -> _FirstTokenObservation:
        self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(self.mesh)
        prefill_started = time.monotonic()
        logits_device = self.model.prefill_traced_chunked(
            prompt_tokens,
            page_table,
            actual_len=int(prompt_tokens.shape[1]),
        )
        ttnn.synchronize_device(self.mesh)
        prefill_finished = time.monotonic()
        try:
            logits = ttnn.to_torch(
                logits_device,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0),
            ).reshape(-1, int(self.model.args.vocab_size))[0]
            first_token_id = int(torch.argmax(logits.float()).item())
            first_token_finished = time.monotonic()
        finally:
            ttnn.deallocate(logits_device)
        return _FirstTokenObservation(
            first_token_id=first_token_id,
            prefill_device_seconds=prefill_finished - prefill_started,
            first_token_readback_seconds=first_token_finished - prefill_finished,
            ttft_seconds=first_token_finished - prefill_started,
        )

    def _allocate_cache_set(self, block_count: int) -> list[list[Any]]:
        cache_shape = (
            block_count,
            int(self.model.args.n_local_kv_heads),
            self.profile.kv_block_size,
            int(self.model.args.head_dim),
        )
        cache_set: list[list[Any]] = []
        try:
            for _layer_index in self.model._attention_layer_indices:
                pair: list[Any] = []
                cache_set.append(pair)
                pair.append(self._allocate_cache_tensor(cache_shape))
                pair.append(self._allocate_cache_tensor(cache_shape))
        except (
            AssertionError,
            MemoryError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            try:
                self._deallocate_cache_set(cache_set)
            except TtftControlAdapterError as cleanup_error:
                raise TtftControlAdapterError(
                    "request cache allocation and cleanup failed"
                ) from cleanup_error
            raise TtftControlAdapterError("request cache allocation failed") from error
        if not cache_set:
            raise TtftControlAdapterError("model has no full-attention cache layers")
        return cache_set

    def _allocate_cache_tensor(self, cache_shape: tuple[int, ...]) -> Any:
        return ttnn.as_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _bind_cache(self, cache_set: list[list[Any]]) -> None:
        self.model.set_paged_kv_caches(cache_set)
        if self.model._paged_kv_caches is not cache_set:
            raise TtftControlAdapterError("model did not retain the selected cache set")
        for cache_index, layer_index in enumerate(self.model._attention_layer_indices):
            attention = self.model.layers[layer_index].attention
            expected_k, expected_v = cache_set[cache_index]
            if (
                attention.paged_k is not expected_k
                or attention.paged_v is not expected_v
            ):
                raise TtftControlAdapterError(
                    "full-attention layer retained a stale cache binding"
                )

    @staticmethod
    def _cache_blocks(cache_set: list[list[Any]]) -> int:
        return int(cache_set[0][0].shape[0])

    def _validate_cache_set(
        self,
        cache_set: list[list[Any]],
        expected_blocks: int,
        label: str,
    ) -> None:
        if len(cache_set) != len(self.model._attention_layer_indices):
            raise TtftControlAdapterError(f"{label} cache layer count changed")
        for pair in cache_set:
            if len(pair) != CACHE_PAIR_WIDTH:
                raise TtftControlAdapterError(f"{label} cache pair is malformed")
            if any(int(tensor.shape[0]) != expected_blocks for tensor in pair):
                raise TtftControlAdapterError(f"{label} cache width changed")

    @staticmethod
    def _deallocate_cache_set(cache_set: list[list[Any]]) -> None:
        errors: list[str] = []
        for pair in cache_set:
            for tensor in pair:
                try:
                    ttnn.deallocate(tensor)
                except (AssertionError, OSError, RuntimeError, ValueError) as error:
                    errors.append(type(error).__name__)
        if errors:
            raise TtftControlAdapterError(
                "request cache deallocation failed: " + ",".join(errors)
            )

    def _require_open(self) -> None:
        if self._closed:
            raise TtftControlAdapterError("TTFT control adapter is closed")


@dataclass(frozen=True)
class _FirstTokenObservation:
    first_token_id: int
    prefill_device_seconds: float
    first_token_readback_seconds: float
    ttft_seconds: float
