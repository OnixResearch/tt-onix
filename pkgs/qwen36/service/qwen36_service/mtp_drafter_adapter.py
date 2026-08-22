"""TTNN adapter for the diagnostic Qwen3.8 MTP drafter probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_drafter_probe]

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.ccl import tt_all_gather
from models.tt_transformers.tt.common import Mode
from safetensors import safe_open

from .decode_adapter import TracedDecodeAdapter
from .mtp_drafter_core import (
    EXPECTED_MTP_WEIGHT_SHAPES,
    MtpDrafterProfile,
    accepted_prefix_length,
    validate_mtp_weight_shapes,
)
from .mtp_probe_adapter import (
    StateComparison,
    compare_tensor_collections,
    compare_tp_gdn_states,
    snapshot_tp_paged_kv,
)
from .native_mtp_verifier_adapter import NativeMtpVerifierAdapter

PROBE_BATCH_SIZE = 1
TOKEN_DIMENSION = 1
LAST_DIMENSION = -1
MTP_PREFILL_BUCKET = 128
MTP_REFERENCE_FULL_ATTENTION_LAYER = 63
TARGET_VERIFICATION_WIDTH = 4
QK_STREAM_COUNT = 2
RMS_NORM_EPSILON = 1.0e-6
MINIMUM_TOKEN_ID = 0
EXACT_STATE_PCC = 1.0
EXACT_STATE_NMSE = 0.0
ZERO_DURATION_SECONDS = 0.0


class MtpDrafterAdapterError(RuntimeError):
    """The physical MTP drafter cannot continue safely."""


@dataclass(frozen=True)
class MtpDrafterSmokeResult:
    anchor_token_id: int
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    mtp_prefill_seconds: float
    draft_decode_seconds: float
    target_sequential_seconds: float
    weight_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class MtpDrafterPilotResult:
    anchor_token_id: int
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    target_continuation_token_id: int
    target_verification_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    mtp_prefill_seconds: float
    draft_decode_seconds: float
    target_sequential_seconds: float
    target_verification_trace_capture_seconds: float
    target_verification_seconds: float
    target_verification_readback_seconds: float
    rollback_seconds: float
    mtp_cache_reprefill_seconds: float
    target_output_valid: bool
    rollback_applied: bool
    rollback_valid: bool
    rollback_recurrent_state_pcc: float
    rollback_recurrent_state_nmse: float
    rollback_convolution_state_pcc: float
    rollback_convolution_state_nmse: float
    rollback_paged_kv_state_pcc: float
    rollback_paged_kv_state_nmse: float
    weight_shapes: dict[str, tuple[int, ...]]


class MtpDrafterAdapter:
    """Load the pinned MTP layer and run one aligned three-draft smoke."""

    def __init__(
        self,
        *,
        model: Any,
        decode_adapter: TracedDecodeAdapter,
        model_path: Path,
        profile: MtpDrafterProfile,
    ) -> None:
        if model.num_devices != len(profile.expected_pcie_device_ids):
            raise MtpDrafterAdapterError("MTP drafter device count differs from policy")
        self.model = model
        self.decode_adapter = decode_adapter
        self.model_path = model_path
        self.profile = profile
        self.mesh = model.mesh_device
        self._replicate = ttnn.ReplicateTensorToMesh(self.mesh)
        self._weight_shapes: dict[str, tuple[int, ...]] = {}
        self._native_verifier = NativeMtpVerifierAdapter(
            model=model,
            maximum_width=TARGET_VERIFICATION_WIDTH,
            trace_capture_timeout_seconds=(
                decode_adapter.profile.trace_capture_timeout_seconds
            ),
        )
        self._load_mtp()

    def run_smoke(self, prompt_tokens: torch.Tensor) -> MtpDrafterSmokeResult:
        if (
            prompt_tokens.ndim != QK_STREAM_COUNT
            or prompt_tokens.shape[0] != PROBE_BATCH_SIZE
        ):
            raise MtpDrafterAdapterError("MTP smoke prompt must have shape [1, tokens]")
        prompt_tokens = prompt_tokens.to(torch.int32)
        prompt_length = int(prompt_tokens.shape[TOKEN_DIMENSION])
        if not MINIMUM_TOKEN_ID < prompt_length < MTP_PREFILL_BUCKET:
            raise MtpDrafterAdapterError(
                "MTP smoke prompt length is outside the bucket"
            )
        page_table = self.decode_adapter.prefill_page_table(
            prompt_tokens=prompt_length,
            generation_tokens=self.profile.draft_token_count + PROBE_BATCH_SIZE,
        )
        target_hidden, anchor = self._target_prefill_hidden(
            prompt_tokens,
            page_table,
            prompt_length,
        )
        try:
            draft_tokens, mtp_prefill_seconds, draft_decode_seconds = self._draft(
                prompt_tokens,
                target_hidden,
                anchor,
                prompt_length,
            )
        finally:
            ttnn.deallocate(target_hidden)

        target_started = time.monotonic()
        current = anchor
        target_tokens = []
        for offset in range(self.profile.draft_token_count):
            current = self.decode_adapter.decode_traced(
                token=current,
                position=prompt_length + offset,
            )
            target_tokens.append(current)
        target_sequential_seconds = time.monotonic() - target_started
        self.decode_adapter._release_trace()
        if (
            not math.isfinite(target_sequential_seconds)
            or target_sequential_seconds <= 0
        ):
            raise MtpDrafterAdapterError("target sequential duration is invalid")
        accepted = accepted_prefix_length(draft_tokens, target_tokens)
        return MtpDrafterSmokeResult(
            anchor_token_id=anchor,
            draft_token_ids=draft_tokens,
            target_token_ids=tuple(target_tokens),
            accepted_prefix_length=accepted,
            mtp_prefill_seconds=mtp_prefill_seconds,
            draft_decode_seconds=draft_decode_seconds,
            target_sequential_seconds=target_sequential_seconds,
            weight_shapes=dict(self._weight_shapes),
        )

    def run_pilot(self, prompt_tokens: torch.Tensor) -> MtpDrafterPilotResult:
        """Measure one complete draft and native target-verification round."""
        if (
            prompt_tokens.ndim != QK_STREAM_COUNT
            or prompt_tokens.shape[0] != PROBE_BATCH_SIZE
        ):
            raise MtpDrafterAdapterError("MTP pilot prompt must have shape [1, tokens]")
        prompt_tokens = prompt_tokens.to(torch.int32)
        prompt_length = int(prompt_tokens.shape[TOKEN_DIMENSION])
        if not MINIMUM_TOKEN_ID < prompt_length < MTP_PREFILL_BUCKET:
            raise MtpDrafterAdapterError(
                "MTP pilot prompt length is outside the bucket"
            )
        page_table = self.decode_adapter.prefill_page_table(
            prompt_tokens=prompt_length,
            generation_tokens=TARGET_VERIFICATION_WIDTH,
        )
        target_hidden, anchor = self._target_prefill_hidden(
            prompt_tokens,
            page_table,
            prompt_length,
        )
        try:
            draft_tokens, mtp_prefill_seconds, draft_decode_seconds = self._draft(
                prompt_tokens,
                target_hidden,
                anchor,
                prompt_length,
            )
        finally:
            ttnn.deallocate(target_hidden)

        eligibility = self.decode_adapter.ensure_trace(
            token=anchor,
            position=prompt_length,
        )
        if not eligibility.eligible:
            raise MtpDrafterAdapterError(
                "target sequential trace is unavailable: " + eligibility.reason
            )
        sequential_inputs = []
        sequential_outputs = []
        current = anchor
        target_started = time.monotonic()
        try:
            for offset in range(TARGET_VERIFICATION_WIDTH):
                sequential_inputs.append(current)
                current = self.decode_adapter.decode_traced(
                    token=current,
                    position=prompt_length + offset,
                )
                sequential_outputs.append(current)
            target_sequential_seconds = time.monotonic() - target_started
        finally:
            # The native verifier allocates width-specific inputs and snapshots.
            # Release the parked sequential trace before those allocations.
            self.decode_adapter._release_trace()
        if (
            not math.isfinite(target_sequential_seconds)
            or target_sequential_seconds <= 0
        ):
            raise MtpDrafterAdapterError("target sequential duration is invalid")

        target_outputs = tuple(sequential_outputs)
        target_tokens = target_outputs[: self.profile.draft_token_count]
        accepted = accepted_prefix_length(draft_tokens, target_tokens)
        committed_width = accepted + PROBE_BATCH_SIZE
        rollback_applied = accepted < self.profile.draft_token_count
        expected_state = None
        expected_kv = None
        if rollback_applied:
            expected_state, expected_kv = self._sequential_prefix_snapshots(
                prompt_tokens=prompt_tokens,
                page_table=page_table,
                prompt_length=prompt_length,
                expected_anchor=anchor,
                input_token_ids=tuple(sequential_inputs),
                committed_width=committed_width,
            )

        verification_hidden, verification_anchor = self._target_prefill_hidden(
            prompt_tokens,
            page_table,
            prompt_length,
        )
        try:
            if verification_anchor != anchor:
                raise MtpDrafterAdapterError(
                    "matched target prefill changed the anchor"
                )
            verification = self._native_verifier.run(
                tuple(sequential_inputs),
                start_position=prompt_length,
                page_table=page_table,
                committed_width=committed_width,
            )
            target_output_valid = verification.token_ids == target_outputs
            mtp_cache_reprefill_seconds = ZERO_DURATION_SECONDS
            if rollback_applied:
                assert expected_state is not None
                assert expected_kv is not None
                actual_state = self.decode_adapter._snapshot_tp_gdn_state()
                positions = tuple(
                    prompt_length + offset for offset in range(committed_width)
                )
                actual_kv = snapshot_tp_paged_kv(
                    self.model,
                    page_table,
                    positions,
                )
                recurrent, convolution = compare_tp_gdn_states(
                    expected_state,
                    actual_state,
                )
                paged_kv = compare_tensor_collections(expected_kv, actual_kv)
                continuation = self.decode_adapter.decode_eager(
                    token=target_outputs[accepted],
                    position=prompt_length + committed_width,
                )
                target_output_valid = (
                    target_output_valid
                    and continuation == target_outputs[accepted + PROBE_BATCH_SIZE]
                )
                (
                    mtp_cache_reprefill_seconds,
                    mtp_cache_valid,
                ) = self._rebuild_mtp_cache(
                    prompt_tokens=prompt_tokens,
                    target_hidden=verification_hidden,
                    anchor=anchor,
                    prompt_length=prompt_length,
                    accepted_prefix_length=accepted,
                    expected_draft_tokens=draft_tokens,
                )
                rollback_valid = (
                    recurrent.pcc == EXACT_STATE_PCC
                    and recurrent.nmse == EXACT_STATE_NMSE
                    and convolution.pcc == EXACT_STATE_PCC
                    and convolution.nmse == EXACT_STATE_NMSE
                    and paged_kv.pcc == EXACT_STATE_PCC
                    and paged_kv.nmse == EXACT_STATE_NMSE
                    and mtp_cache_valid
                    and target_output_valid
                )
            else:
                recurrent = convolution = paged_kv = _exact_state_comparison()
                rollback_valid = target_output_valid
        finally:
            ttnn.deallocate(verification_hidden)

        return MtpDrafterPilotResult(
            anchor_token_id=anchor,
            draft_token_ids=draft_tokens,
            target_token_ids=target_tokens,
            target_continuation_token_id=target_outputs[-1],
            target_verification_token_ids=verification.token_ids,
            accepted_prefix_length=accepted,
            mtp_prefill_seconds=mtp_prefill_seconds,
            draft_decode_seconds=draft_decode_seconds,
            target_sequential_seconds=target_sequential_seconds,
            target_verification_trace_capture_seconds=(
                verification.trace_capture_seconds
            ),
            target_verification_seconds=verification.device_seconds,
            target_verification_readback_seconds=verification.readback_seconds,
            rollback_seconds=verification.rollback_seconds,
            mtp_cache_reprefill_seconds=mtp_cache_reprefill_seconds,
            target_output_valid=target_output_valid,
            rollback_applied=rollback_applied,
            rollback_valid=rollback_valid,
            rollback_recurrent_state_pcc=recurrent.pcc,
            rollback_recurrent_state_nmse=recurrent.nmse,
            rollback_convolution_state_pcc=convolution.pcc,
            rollback_convolution_state_nmse=convolution.nmse,
            rollback_paged_kv_state_pcc=paged_kv.pcc,
            rollback_paged_kv_state_nmse=paged_kv.nmse,
            weight_shapes=dict(self._weight_shapes),
        )

    def _load_mtp(self) -> None:
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise MtpDrafterAdapterError("MTP checkpoint index is missing")
        try:
            with index_path.open(encoding="utf-8") as index_file:
                weight_map = json.load(index_file)["weight_map"]
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            raise MtpDrafterAdapterError("MTP checkpoint index is invalid") from error
        mtp_files = {
            weight_map[name]
            for name in EXPECTED_MTP_WEIGHT_SHAPES
            if name in weight_map
        }
        if len(mtp_files) != PROBE_BATCH_SIZE:
            raise MtpDrafterAdapterError(
                "MTP tensors do not share one checkpoint shard"
            )
        shard_path = self.model_path / next(iter(mtp_files))
        if not shard_path.is_file():
            raise MtpDrafterAdapterError("MTP checkpoint shard is missing")
        with safe_open(str(shard_path), framework="pt") as shard:
            observed = {
                name: tuple(
                    int(dimension) for dimension in shard.get_slice(name).get_shape()
                )
                for name in list(shard.keys())
                if name.startswith("mtp.")
            }
            validate_mtp_weight_shapes(observed)
            weights = {
                name: shard.get_tensor(name) for name in EXPECTED_MTP_WEIGHT_SHAPES
            }
        self._weight_shapes = observed

        cache_root = (
            self.model_path / "P300" / "tensor_cache_bfp8_mesh1x2" / "mtp_drafter_probe"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        layer_prefix = f"layers.{MTP_REFERENCE_FULL_ATTENTION_LAYER}"
        layer_weights = {
            f"{layer_prefix}.{name.removeprefix('mtp.layers.0.')}": tensor
            for name, tensor in weights.items()
            if name.startswith("mtp.layers.0.")
        }
        self.mtp_layer = Qwen36DecoderLayer(
            self.mesh,
            self.model.args,
            layer_weights,
            MTP_REFERENCE_FULL_ATTENTION_LAYER,
            cache_root / "layer",
            tt_ccl=self.model.tt_ccl,
        )
        self.fc_weight = tpc.shard_w(
            weights["mtp.fc.weight"],
            self.mesh,
            dim=LAST_DIMENSION,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=str(cache_root / "fc.bf16"),
            dtype=ttnn.bfloat16,
        )
        self.pre_embedding_norm = tpc.replicate(
            weights["mtp.pre_fc_norm_embedding.weight"].float() + 1.0,
            self.mesh,
            str(cache_root / "pre_embedding_norm"),
        )
        self.pre_hidden_norm = tpc.replicate(
            weights["mtp.pre_fc_norm_hidden.weight"].float() + 1.0,
            self.mesh,
            str(cache_root / "pre_hidden_norm"),
        )
        self.final_norm = tpc.replicate(
            weights["mtp.norm.weight"].float() + 1.0,
            self.mesh,
            str(cache_root / "final_norm"),
        )

    def _target_prefill_hidden(
        self,
        prompt_tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_length: int,
    ) -> tuple[Any, int]:
        self.decode_adapter.reset_for_request()
        padding = torch.zeros(
            (PROBE_BATCH_SIZE, MTP_PREFILL_BUCKET - prompt_length),
            dtype=torch.int32,
        )
        token_buffer = torch.cat([prompt_tokens, padding], dim=TOKEN_DIMENSION)
        hidden = self.model._forward_prefill_chunk_masked(
            token_buffer,
            prompt_length,
            0,
            page_table,
            MTP_PREFILL_BUCKET,
            flex_sdpa=True,
        )
        target_hidden = self.model.norm(hidden, mode=Mode.PREFILL)
        ttnn.deallocate(hidden)
        selected = self._select_row(target_hidden, prompt_length - PROBE_BATCH_SIZE)
        logits = self.model._lm_head(selected)
        ttnn.deallocate(selected)
        ttnn.synchronize_device(self.mesh)
        anchor = self._read_token(logits)
        ttnn.deallocate(logits)
        return target_hidden, anchor

    def _sequential_prefix_snapshots(
        self,
        *,
        prompt_tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_length: int,
        expected_anchor: int,
        input_token_ids: tuple[int, ...],
        committed_width: int,
    ) -> tuple[Any, Any]:
        hidden, anchor = self._target_prefill_hidden(
            prompt_tokens,
            page_table,
            prompt_length,
        )
        ttnn.deallocate(hidden)
        if anchor != expected_anchor:
            raise MtpDrafterAdapterError(
                "rollback reference prefill changed the anchor"
            )
        eligibility = self.decode_adapter.ensure_trace(
            token=anchor,
            position=prompt_length,
        )
        if not eligibility.eligible:
            raise MtpDrafterAdapterError(
                "rollback reference trace is unavailable: " + eligibility.reason
            )
        current = anchor
        try:
            for offset in range(committed_width):
                if current != input_token_ids[offset]:
                    raise MtpDrafterAdapterError(
                        "rollback reference input changed before the commit point"
                    )
                current = self.decode_adapter.decode_traced(
                    token=current,
                    position=prompt_length + offset,
                )
        finally:
            self.decode_adapter._release_trace()
        state = self.decode_adapter._snapshot_tp_gdn_state()
        positions = tuple(prompt_length + offset for offset in range(committed_width))
        paged_kv = snapshot_tp_paged_kv(self.model, page_table, positions)
        return state, paged_kv

    def _draft(
        self,
        prompt_tokens: torch.Tensor,
        target_hidden: Any,
        anchor: int,
        prompt_length: int,
    ) -> tuple[tuple[int, ...], float, float]:
        prefill_started = time.monotonic()
        first_draft, feedback = self._mtp_prefill(
            prompt_tokens,
            target_hidden,
            anchor,
            prompt_length,
        )
        mtp_prefill_seconds = time.monotonic() - prefill_started

        drafts = [first_draft]
        draft_started = time.monotonic()
        current = first_draft
        for step in range(self.profile.draft_token_count - PROBE_BATCH_SIZE):
            current, next_feedback = self._draft_step(
                current,
                feedback,
                prompt_length + step,
            )
            ttnn.deallocate(feedback)
            feedback = next_feedback
            drafts.append(current)
        ttnn.deallocate(feedback)
        draft_decode_seconds = time.monotonic() - draft_started
        if any(
            not math.isfinite(value) or value <= 0
            for value in (mtp_prefill_seconds, draft_decode_seconds)
        ):
            raise MtpDrafterAdapterError("MTP drafter duration is invalid")
        return tuple(drafts), mtp_prefill_seconds, draft_decode_seconds

    def _rebuild_mtp_cache(
        self,
        *,
        prompt_tokens: torch.Tensor,
        target_hidden: Any,
        anchor: int,
        prompt_length: int,
        accepted_prefix_length: int,
        expected_draft_tokens: tuple[int, ...],
    ) -> tuple[float, bool]:
        rebuild_started = time.monotonic()
        first_draft, feedback = self._mtp_prefill(
            prompt_tokens,
            target_hidden,
            anchor,
            prompt_length,
        )
        reproduced = [first_draft]
        current = first_draft
        for step in range(accepted_prefix_length):
            current, next_feedback = self._draft_step(
                current,
                feedback,
                prompt_length + step,
            )
            ttnn.deallocate(feedback)
            feedback = next_feedback
            reproduced.append(current)
        ttnn.deallocate(feedback)
        rebuild_seconds = time.monotonic() - rebuild_started
        if not math.isfinite(rebuild_seconds) or rebuild_seconds <= 0:
            raise MtpDrafterAdapterError("MTP cache re-prefill duration is invalid")
        expected_count = accepted_prefix_length + PROBE_BATCH_SIZE
        valid = tuple(reproduced) == expected_draft_tokens[:expected_count]
        return rebuild_seconds, valid

    def _mtp_prefill(
        self,
        prompt_tokens: torch.Tensor,
        target_hidden: Any,
        anchor: int,
        prompt_length: int,
    ) -> tuple[int, Any]:
        shifted = torch.cat(
            [
                prompt_tokens[:, PROBE_BATCH_SIZE:],
                torch.tensor([[anchor]], dtype=torch.int32),
            ],
            dim=TOKEN_DIMENSION,
        )
        shifted = torch.cat(
            [
                shifted,
                torch.zeros(
                    (PROBE_BATCH_SIZE, MTP_PREFILL_BUCKET - prompt_length),
                    dtype=torch.int32,
                ),
            ],
            dim=TOKEN_DIMENSION,
        )
        self.mtp_layer.attention.reset_state()
        embedding = self._embed_full(shifted)
        fused = self._fuse_inputs(embedding, target_hidden)
        ttnn.deallocate(embedding)
        cos_t, sin_t = self.model._rope_tp_cos_sin_torch(0, MTP_PREFILL_BUCKET)
        cos = ttnn.from_torch(
            cos_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=self._replicate,
        )
        sin = ttnn.from_torch(
            sin_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=self._replicate,
        )
        output = self.mtp_layer.forward(
            fused,
            cos=cos,
            sin=sin,
            mode="prefill",
        )
        ttnn.deallocate(fused)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        normalized = self._final_normalize(output)
        ttnn.deallocate(output)
        feedback = self._select_row(normalized, prompt_length - PROBE_BATCH_SIZE)
        ttnn.deallocate(normalized)
        logits = self.model._lm_head(feedback)
        ttnn.synchronize_device(self.mesh)
        first_draft = self._read_token(logits)
        ttnn.deallocate(logits)
        return first_draft, feedback

    def _draft_step(self, token: int, feedback: Any, position: int) -> tuple[int, Any]:
        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=None,
        )
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
        from models.tt_transformers.tt.common import copy_host_to_device

        token_buffer, position_buffer, rope_buffer, _page_table = copy_host_to_device(
            host_inputs,
            mesh_device=self.mesh,
        )
        embedding = self._embed_full_device(token_buffer)
        fused = self._fuse_inputs(embedding, feedback)
        ttnn.deallocate(embedding)
        cos, sin = unpack_rope(rope_buffer)
        output = self.mtp_layer.forward(
            fused,
            cos=cos,
            sin=sin,
            mode="decode",
            position_tensor=position_buffer,
        )
        ttnn.deallocate(fused)
        normalized = self._final_normalize(output)
        ttnn.deallocate(output)
        logits = self.model._lm_head(normalized)
        ttnn.synchronize_device(self.mesh)
        next_token = self._read_token(logits)
        ttnn.deallocate(logits)
        for tensor in (token_buffer, position_buffer, rope_buffer):
            ttnn.deallocate(tensor)
        return next_token, normalized

    def _embed_full(self, tokens: torch.Tensor) -> Any:
        token_device = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            device=self.mesh,
            mesh_mapper=self._replicate,
        )
        try:
            return self._embed_full_device(token_device)
        finally:
            ttnn.deallocate(token_device)

    def _embed_full_device(self, token_device: Any) -> Any:
        embedding = self.model.embd(token_device)
        embedding = ttnn.reshape(
            embedding,
            (
                PROBE_BATCH_SIZE,
                PROBE_BATCH_SIZE,
                embedding.shape[0] * embedding.shape[TOKEN_DIMENSION],
                embedding.shape[LAST_DIMENSION],
            ),
        )
        full = self._all_gather(embedding)
        ttnn.deallocate(embedding)
        normalized = ttnn.rms_norm(
            full,
            weight=self.pre_embedding_norm,
            epsilon=RMS_NORM_EPSILON,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(full)
        return normalized

    def _fuse_inputs(self, embedding: Any, hidden: Any) -> Any:
        hidden_normalized = ttnn.rms_norm(
            hidden,
            weight=self.pre_hidden_norm,
            epsilon=RMS_NORM_EPSILON,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_input = ttnn.concat([embedding, hidden_normalized], dim=LAST_DIMENSION)
        ttnn.deallocate(hidden_normalized)
        fused = ttnn.linear(fused_input, self.fc_weight)
        ttnn.deallocate(fused_input)
        return fused

    def _final_normalize(self, hidden: Any) -> Any:
        full = self._all_gather(hidden)
        normalized = ttnn.rms_norm(
            full,
            weight=self.final_norm,
            epsilon=RMS_NORM_EPSILON,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(full)
        return normalized

    def _all_gather(self, hidden: Any) -> Any:
        return tt_all_gather(
            hidden,
            self.mesh,
            self.model.tt_ccl,
            cluster_axis=None,
            dim=len(hidden.shape) - PROBE_BATCH_SIZE,
            topology=self.model.args.ccl_topology(),
        )

    def _select_row(self, hidden: Any, row: int) -> Any:
        selector = torch.zeros(
            PROBE_BATCH_SIZE,
            PROBE_BATCH_SIZE,
            PROBE_BATCH_SIZE,
            MTP_PREFILL_BUCKET,
            dtype=torch.float32,
        )
        selector[0, 0, 0, row] = 1.0
        selector_device = ttnn.from_torch(
            selector,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=self._replicate,
        )
        selected = ttnn.matmul(selector_device, hidden)
        ttnn.deallocate(selector_device)
        return ttnn.to_memory_config(selected, ttnn.DRAM_MEMORY_CONFIG)

    def _read_token(self, logits: Any) -> int:
        device_logits = ttnn.get_device_tensors(logits)
        if not device_logits:
            raise MtpDrafterAdapterError("MTP drafter produced no logit replica")
        row = ttnn.to_torch(device_logits[0]).float().reshape(-1)
        row = row[: self.model.args.vocab_size]
        if not torch.isfinite(row).all():
            raise MtpDrafterAdapterError("MTP drafter logits are not finite")
        token = int(torch.argmax(row).item())
        if not MINIMUM_TOKEN_ID <= token < self.model.args.vocab_size:
            raise MtpDrafterAdapterError("MTP drafter token is outside the vocabulary")
        return token


def _exact_state_comparison() -> StateComparison:
    return StateComparison(pcc=EXACT_STATE_PCC, nmse=EXACT_STATE_NMSE)
