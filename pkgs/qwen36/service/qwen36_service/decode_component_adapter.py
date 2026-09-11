"""TTNN adapter for diagnostic Qwen decode component replay."""

# r[impl tenstorrent.qwen38_p150x2.decode_component_attribution]

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.tt_transformers.tt.common import Mode

from .decode_adapter import TracedDecodeAdapter
from .decode_component_core import (
    EMBEDDING_COMPONENT,
    FULL_ATTENTION_MIXER_COMPONENT,
    FULL_ATTENTION_TYPE,
    GDN_MIXER_COMPONENT,
    LINEAR_ATTENTION_TYPE,
    LM_HEAD_ARGMAX_COMPONENT,
    MLP_COMPONENT,
    ComponentReplaySample,
    DecodeComponentEvidence,
    DecodeComponentIdentity,
    DecodeComponentProfile,
    FullReplaySample,
    HostBoundarySample,
    LayerTypeObservation,
)

TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False
MINIMUM_TOKEN_ID = 0
PROMPT_TOKEN_DIMENSION = 1
PROBE_GENERATION_TOKENS = 1


class DecodeComponentAdapterError(RuntimeError):
    """The physical component replay adapter cannot continue safely."""


@dataclass(frozen=True)
class TraceCaptureObservation:
    component: str
    layer_index: int | None
    capture_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "layer_index": self.layer_index,
            "capture_seconds": self.capture_seconds,
        }


@dataclass(frozen=True)
class DecodeComponentProbeRun:
    evidence: DecodeComponentEvidence
    trace_captures: tuple[TraceCaptureObservation, ...]
    prompt_token_count: int
    first_decode_token: int


class DecodeComponentProbeAdapter:
    """Measure coherent component traces on one prompt-bound full model."""

    def __init__(
        self,
        *,
        model: Any,
        decode_adapter: TracedDecodeAdapter,
        profile: DecodeComponentProfile,
        identity: DecodeComponentIdentity,
        physical_pcie_device_ids: tuple[int, ...],
    ) -> None:
        if len(model.layers) != profile.layer_count:
            raise DecodeComponentAdapterError(
                "component probe model layer count differs from policy"
            )
        if model.num_devices != len(profile.expected_pcie_device_ids):
            raise DecodeComponentAdapterError(
                "component probe device count differs from policy"
            )
        if physical_pcie_device_ids != profile.expected_pcie_device_ids:
            raise DecodeComponentAdapterError(
                "component probe physical device IDs differ from policy"
            )
        self.model = model
        self.decode_adapter = decode_adapter
        self.profile = profile
        self.identity = identity
        self.physical_pcie_device_ids = physical_pcie_device_ids
        self.mesh = model.mesh_device
        self._mesh_reader = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self._layers = {int(layer.layer_num): layer for layer in model.layers}
        if len(self._layers) != profile.layer_count:
            raise DecodeComponentAdapterError(
                "component probe layer indices are not unique"
            )
        self._validate_representatives()
        self._prompt_state: (
            tuple[tuple[torch.Tensor, tuple[torch.Tensor, ...]], ...] | None
        ) = None
        self._trace_captures: list[TraceCaptureObservation] = []

    def run(self, prompt_tokens: torch.Tensor) -> DecodeComponentProbeRun:
        prompt_token_count, first_token = self._prefill(prompt_tokens)
        self._prompt_state = self.decode_adapter._snapshot_tp_gdn_state()
        component_samples: list[ComponentReplaySample] = []
        full_samples, host_samples = self._measure_full_trace(
            token=first_token,
            position=prompt_token_count,
        )
        self._release_model_prefill_trace()
        self._restore_prompt_state()
        self.decode_adapter._update_inputs(
            token=first_token, position=prompt_token_count
        )
        ttnn.synchronize_device(self.mesh)
        hidden = self._embedding_forward()
        ttnn.synchronize_device(self.mesh)
        try:
            component_samples.extend(
                self._measure_component(
                    component=EMBEDDING_COMPONENT,
                    layer_index=None,
                    forward=self._embedding_forward,
                    restore_gdn=False,
                )
            )
            for layer_index in self.profile.gdn_representatives:
                layer = self._layers[layer_index]
                component_samples.extend(
                    self._measure_component(
                        component=GDN_MIXER_COMPONENT,
                        layer_index=layer_index,
                        forward=lambda layer=layer: self._mixer_forward(layer, hidden),
                        restore_gdn=True,
                    )
                )
            for layer_index in self.profile.mlp_representatives:
                layer = self._layers[layer_index]
                component_samples.extend(
                    self._measure_component(
                        component=MLP_COMPONENT,
                        layer_index=layer_index,
                        forward=lambda layer=layer: self._mlp_forward(layer, hidden),
                        restore_gdn=False,
                    )
                )
            for layer_index in self.profile.full_attention_representatives:
                layer = self._layers[layer_index]
                component_samples.extend(
                    self._measure_component(
                        component=FULL_ATTENTION_MIXER_COMPONENT,
                        layer_index=layer_index,
                        forward=lambda layer=layer: self._mixer_forward(layer, hidden),
                        restore_gdn=False,
                    )
                )
            component_samples.extend(
                self._measure_component(
                    component=LM_HEAD_ARGMAX_COMPONENT,
                    layer_index=None,
                    forward=lambda: self._lm_head_argmax_forward(hidden),
                    restore_gdn=False,
                )
            )
        finally:
            ttnn.deallocate(hidden)

        evidence = DecodeComponentEvidence(
            identity=self.identity,
            physical_pcie_device_ids=self.physical_pcie_device_ids,
            mesh_shape=tuple(int(dimension) for dimension in self.mesh.shape),
            host_cpu_affinity=tuple(sorted(os.sched_getaffinity(0))),
            layer_types=self._layer_type_observations(),
            component_samples=tuple(component_samples),
            full_samples=full_samples,
            host_samples=host_samples,
            cleanup_complete=False,
        )
        return DecodeComponentProbeRun(
            evidence=evidence,
            trace_captures=tuple(self._trace_captures),
            prompt_token_count=prompt_token_count,
            first_decode_token=first_token,
        )

    def run_sequence(
        self,
        prompt_tokens: torch.Tensor,
        *,
        generation_tokens: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> dict[str, Any]:
        """Exercise an unadmitted candidate without passing through a serving router."""
        if (
            isinstance(generation_tokens, bool)
            or not isinstance(generation_tokens, int)
            or generation_tokens < PROBE_GENERATION_TOKENS
        ):
            raise DecodeComponentAdapterError(
                "sequence length must be a positive integer"
            )
        modes = {
            layer.attention._decode_fusion_mode
            for layer in self.model.layers
            if not layer.is_full_attention
        }
        if len(modes) != 1:
            raise DecodeComponentAdapterError(
                "sequence requires one consistent GDN mode"
            )
        prompt_count, token = self._prefill(
            prompt_tokens, generation_tokens=generation_tokens
        )
        eligibility = self.decode_adapter.ensure_trace(
            token=token, position=prompt_count
        )
        if not eligibility.eligible:
            raise DecodeComponentAdapterError(
                f"sequence trace unavailable: {eligibility.reason}"
            )
        tokens = [token]
        decode_step_seconds = []
        for offset in range(1, generation_tokens):
            # Prefill and trace preparation stay outside each timing window.
            # decode_traced includes input transfer, mesh sync, and winner readback.
            started = clock()
            token = self.decode_adapter.decode_traced(
                token=token, position=prompt_count + offset - 1
            )
            elapsed = clock() - started
            if not _valid_duration(elapsed):
                raise DecodeComponentAdapterError("sequence decode duration is invalid")
            decode_step_seconds.append(elapsed)
            tokens.append(token)
        return {
            "fusion_mode": modes.pop(),
            "prompt_token_count": prompt_count,
            "generated_token_ids": tokens,
            "decode_step_count": len(decode_step_seconds),
            "decode_step_seconds": decode_step_seconds,
        }

    def _validate_representatives(self) -> None:
        for layer_index in self.profile.gdn_representatives:
            if self._layers[layer_index].is_full_attention:
                raise DecodeComponentAdapterError(
                    f"GDN representative {layer_index} is a full-attention layer"
                )
        for layer_index in self.profile.full_attention_representatives:
            if not self._layers[layer_index].is_full_attention:
                raise DecodeComponentAdapterError(
                    f"full-attention representative {layer_index} is a GDN layer"
                )
        for layer_index in self.profile.mlp_representatives:
            if layer_index not in self._layers:
                raise DecodeComponentAdapterError(
                    f"MLP representative {layer_index} is missing"
                )

    def _prefill(
        self,
        prompt_tokens: torch.Tensor,
        *,
        generation_tokens: int = PROBE_GENERATION_TOKENS,
    ) -> tuple[int, int]:
        if prompt_tokens.dtype not in (torch.int32, torch.int64):
            raise DecodeComponentAdapterError(
                "component prompt must contain integer token IDs"
            )
        if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
            raise DecodeComponentAdapterError(
                "component prompt must have shape [1, tokens]"
            )
        prompt_token_count = int(prompt_tokens.shape[PROMPT_TOKEN_DIMENSION])
        if prompt_token_count < PROBE_GENERATION_TOKENS:
            raise DecodeComponentAdapterError("component prompt is empty")
        if (
            prompt_token_count + generation_tokens
            > self.profile.maximum_sequence_length
        ):
            raise DecodeComponentAdapterError(
                "component prompt exceeds the sequence bound"
            )
        if int(torch.min(prompt_tokens).item()) < MINIMUM_TOKEN_ID:
            raise DecodeComponentAdapterError(
                "component prompt contains a negative token ID"
            )
        page_table = self.decode_adapter.prefill_page_table(
            prompt_tokens=prompt_token_count,
            generation_tokens=generation_tokens,
        )
        self.decode_adapter.reset_for_request()
        logits_device = self.model.prefill_traced_chunked(
            prompt_tokens.to(torch.int32),
            page_table,
            actual_len=prompt_token_count,
        )
        ttnn.synchronize_device(self.mesh)
        logits = ttnn.to_torch(
            logits_device,
            mesh_composer=self._mesh_reader,
        ).reshape(-1, self.model.args.vocab_size)[0]
        ttnn.deallocate(logits_device)
        first_token = int(torch.argmax(logits.float()).item())
        if not MINIMUM_TOKEN_ID <= first_token < self.model.args.vocab_size:
            raise DecodeComponentAdapterError(
                "component prefill returned an invalid token"
            )
        return prompt_token_count, first_token

    def _measure_full_trace(
        self,
        *,
        token: int,
        position: int,
    ) -> tuple[tuple[FullReplaySample, ...], tuple[HostBoundarySample, ...]]:
        if self._prompt_state is None:
            raise DecodeComponentAdapterError("prompt state is not available")
        eligibility = self.decode_adapter.ensure_trace(token=token, position=position)
        if not eligibility.eligible:
            raise DecodeComponentAdapterError(
                f"full decode trace is unavailable: {eligibility.reason}"
            )
        trace_id = self.decode_adapter._trace_id
        trace_indices = self.decode_adapter._trace_indices
        trace_values = self.decode_adapter._trace_values
        capture_seconds = self.decode_adapter.trace_capture_seconds
        if trace_id is None or trace_indices is None or trace_values is None:
            raise DecodeComponentAdapterError("full decode trace outputs are missing")
        if capture_seconds is None or not _valid_duration(capture_seconds):
            raise DecodeComponentAdapterError(
                "full decode trace capture time is invalid"
            )
        self._trace_captures.append(
            TraceCaptureObservation(
                component="full_decode",
                layer_index=None,
                capture_seconds=capture_seconds,
            )
        )

        full_samples: list[FullReplaySample] = []
        host_samples: list[HostBoundarySample] = []
        try:
            for _warm_index in range(self.profile.warmup_replays):
                self._restore_prompt_state()
                self.decode_adapter._update_inputs(token=token, position=position)
                ttnn.synchronize_device(self.mesh)
                ttnn.execute_trace(
                    self.mesh,
                    trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                ttnn.synchronize_device(self.mesh)
                self.decode_adapter._read_global_winner(trace_indices, trace_values)
            for sample_index in range(1, self.profile.samples_per_group + 1):
                self._restore_prompt_state()
                update_started = time.monotonic()
                self.decode_adapter._update_inputs(token=token, position=position)
                ttnn.synchronize_device(self.mesh)
                input_update_seconds = time.monotonic() - update_started

                replay_started = time.monotonic()
                ttnn.execute_trace(
                    self.mesh,
                    trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                ttnn.synchronize_device(self.mesh)
                device_seconds = time.monotonic() - replay_started

                readback_started = time.monotonic()
                token_id = self.decode_adapter._read_global_winner(
                    trace_indices,
                    trace_values,
                )
                winner_readback_seconds = time.monotonic() - readback_started
                for label, duration in (
                    ("input update", input_update_seconds),
                    ("full replay", device_seconds),
                    ("winner readback", winner_readback_seconds),
                ):
                    if not _valid_duration(duration):
                        raise DecodeComponentAdapterError(
                            f"component {label} duration is invalid"
                        )
                full_samples.append(
                    FullReplaySample(
                        sample_index=sample_index,
                        device_seconds=device_seconds,
                        token_id=token_id,
                    )
                )
                host_samples.append(
                    HostBoundarySample(
                        sample_index=sample_index,
                        input_update_seconds=input_update_seconds,
                        winner_readback_seconds=winner_readback_seconds,
                    )
                )
        finally:
            self.decode_adapter._release_trace()
        return tuple(full_samples), tuple(host_samples)

    def _measure_component(
        self,
        *,
        component: str,
        layer_index: int | None,
        forward: Callable[[], Any],
        restore_gdn: bool,
    ) -> tuple[ComponentReplaySample, ...]:
        trace_id: Any | None = None
        trace_output: Any | None = None
        try:
            if restore_gdn:
                self._restore_prompt_state()
            warm_output = forward()
            ttnn.synchronize_device(self.mesh)
            _deallocate_nested(warm_output)
            if restore_gdn:
                self._restore_prompt_state()

            capture_started = time.monotonic()
            trace_id = ttnn.begin_trace_capture(
                self.mesh,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            trace_output = forward()
            ttnn.end_trace_capture(
                self.mesh,
                trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.mesh)
            capture_seconds = time.monotonic() - capture_started
            if (
                not _valid_duration(capture_seconds)
                or capture_seconds > self.profile.trace_capture_timeout_seconds
            ):
                raise DecodeComponentAdapterError(
                    f"{component} trace capture exceeded its bound"
                )
            self._trace_captures.append(
                TraceCaptureObservation(
                    component=component,
                    layer_index=layer_index,
                    capture_seconds=capture_seconds,
                )
            )

            for _warm_index in range(self.profile.warmup_replays):
                if restore_gdn:
                    self._restore_prompt_state()
                ttnn.execute_trace(
                    self.mesh,
                    trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                ttnn.synchronize_device(self.mesh)

            samples = []
            for sample_index in range(1, self.profile.samples_per_group + 1):
                if restore_gdn:
                    self._restore_prompt_state()
                replay_started = time.monotonic()
                ttnn.execute_trace(
                    self.mesh,
                    trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                ttnn.synchronize_device(self.mesh)
                seconds = time.monotonic() - replay_started
                if not _valid_duration(seconds):
                    raise DecodeComponentAdapterError(
                        f"{component} trace replay duration is invalid"
                    )
                samples.append(
                    ComponentReplaySample(
                        component=component,
                        layer_index=layer_index,
                        sample_index=sample_index,
                        seconds=seconds,
                    )
                )
            return tuple(samples)
        finally:
            if trace_id is not None:
                ttnn.release_trace(self.mesh, trace_id)
            if trace_output is not None:
                _deallocate_nested(trace_output)

    def _embedding_forward(self) -> Any:
        hidden = self.model.embd(self.decode_adapter.decode_inputs[0])
        return ttnn.reshape(
            hidden,
            (1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[-1]),
        )

    def _mixer_forward(self, layer: Any, hidden: Any) -> Any:
        norm_config = layer.args.get_norm_config("attn", Mode.DECODE)
        attention_input = layer.attention_norm(
            hidden,
            mode=Mode.DECODE,
            norm_config=norm_config,
        )
        if layer.is_full_attention:
            cos, sin = unpack_rope(self.decode_adapter.decode_inputs[2])
            attention_output = layer.attention.forward_decode(
                attention_input,
                self.decode_adapter.decode_inputs[1],
                cos,
                sin,
                page_table=self.decode_adapter.decode_inputs[3],
            )
        else:
            attention_output = layer.attention.forward_decode(attention_input)
        ttnn.deallocate(attention_input)
        output = ttnn.add(hidden, attention_output)
        ttnn.deallocate(attention_output)
        return output

    def _mlp_forward(self, layer: Any, hidden: Any) -> Any:
        norm_config = layer.args.get_norm_config("attn", Mode.DECODE)
        feed_forward_input = layer.ffn_norm(
            hidden,
            mode=Mode.DECODE,
            norm_config=norm_config,
        )
        feed_forward_output = layer.feed_forward.forward(feed_forward_input)
        ttnn.deallocate(feed_forward_input)
        output = ttnn.add(hidden, feed_forward_output)
        ttnn.deallocate(feed_forward_output)
        return output

    def _lm_head_argmax_forward(self, hidden: Any) -> tuple[Any, Any]:
        normalized = self.model._final_norm_decode(hidden)
        sharded_logits = ttnn.linear(normalized, self.model.lm_head_weight)
        ttnn.deallocate(normalized)
        indices, values = self.decode_adapter._reduce_shards(sharded_logits)
        ttnn.deallocate(sharded_logits)
        return indices, values

    def _restore_prompt_state(self) -> None:
        if self._prompt_state is None:
            raise DecodeComponentAdapterError("prompt state is not available")
        self.decode_adapter._restore_tp_gdn_state(self._prompt_state)
        ttnn.synchronize_device(self.mesh)

    def _release_model_prefill_trace(self) -> None:
        trace_id = getattr(self.model, "_chunked_trace_id", None)
        if trace_id is not None:
            ttnn.release_trace(self.mesh, trace_id)
            self.model._chunked_trace_id = None

    def _layer_type_observations(self) -> tuple[LayerTypeObservation, ...]:
        indices = set(self.profile.gdn_representatives)
        indices.update(self.profile.full_attention_representatives)
        indices.update(self.profile.mlp_representatives)
        return tuple(
            LayerTypeObservation(
                layer_index=layer_index,
                mixer_type=(
                    FULL_ATTENTION_TYPE
                    if self._layers[layer_index].is_full_attention
                    else LINEAR_ATTENTION_TYPE
                ),
            )
            for layer_index in sorted(indices)
        )


def _deallocate_nested(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _deallocate_nested(item)
        return
    ttnn.deallocate(value)


def _valid_duration(value: float) -> bool:
    return math.isfinite(value) and value > 0
