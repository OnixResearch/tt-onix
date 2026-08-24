"""Native short-width target verifier for the pinned Qwen P150x2 runtime."""

# r[impl tenstorrent.qwen38_p150x2.native_mtp_verifier]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    apply_partial_rope_decode,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, _softplus_add
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, copy_host_to_device

from qwen36_service.mtp_trace_hygiene_core import (
    TraceCaptureProvenance,
    validate_trace_capture_provenance,
)

MINIMUM_TOKEN_ID = 0
MINIMUM_WIDTH = 1
TILE_EDGE_ELEMENTS = 32
RMS_NORM_EPSILON = 1.0e-6
QK_STREAM_COUNT = 2
HIDDEN_DIMENSION = 3
PENULTIMATE_DIMENSION = -2
LAST_DIMENSION = -1
FULL_ATTENTION_CLUSTER_AXIS = 0
FOUR_DIMENSIONAL_TENSOR = 4
MUTABLE_INPUT_COUNT = 3
TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False
ZERO_DURATION_SECONDS = 0.0


class NativeMtpVerifierError(RuntimeError):
    """The native verifier cannot preserve its bounded execution contract."""


@dataclass(frozen=True)
class NativeVerificationResult:
    token_ids: tuple[int, ...]
    logits: torch.Tensor
    trace_capture_seconds: float
    device_seconds: float
    readback_seconds: float
    rollback_seconds: float


@dataclass(frozen=True)
class _GdnDeviceSnapshot:
    layer: Any
    recurrent: Any
    convolution: tuple[Any, ...]
    convolution_carry: Any


@dataclass(frozen=True)
class _GdnPrefixSnapshot:
    layer: Any
    recurrent: Any
    convolution: tuple[Any, ...]


class NativeMtpVerifierAdapter:
    """Batch weight-bearing work in one traced hybrid-model replay."""

    def __init__(
        self,
        *,
        model: Any,
        maximum_width: int,
        trace_capture_timeout_seconds: float,
    ) -> None:
        validate_native_verifier_model(model)
        if maximum_width < MINIMUM_WIDTH:
            raise NativeMtpVerifierError("native verifier width must be positive")
        if (
            not math.isfinite(trace_capture_timeout_seconds)
            or trace_capture_timeout_seconds <= 0
        ):
            raise NativeMtpVerifierError("native verifier trace timeout is invalid")
        self.model = model
        self.maximum_width = maximum_width
        self.trace_capture_timeout_seconds = trace_capture_timeout_seconds
        # Single-resident capture state consumed by the trace-hygiene gate.
        self._hygiene_capture_resident = False

    def run(
        self,
        input_token_ids: tuple[int, ...],
        *,
        start_position: int,
        page_table: torch.Tensor,
        committed_width: int | None = None,
    ) -> NativeVerificationResult:
        width = len(input_token_ids)
        if not MINIMUM_WIDTH <= width <= self.maximum_width:
            raise NativeMtpVerifierError("native verifier width is outside policy")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not MINIMUM_TOKEN_ID <= token < self.model.args.vocab_size
            for token in input_token_ids
        ):
            raise NativeMtpVerifierError(
                "native verifier token is outside the vocabulary"
            )
        if start_position < MINIMUM_TOKEN_ID:
            raise NativeMtpVerifierError("native verifier start position is negative")
        stop_position = start_position + width
        if stop_position > self.model.args.max_seq_len:
            raise NativeMtpVerifierError(
                "native verifier exceeds the model sequence bound"
            )
        if page_table.ndim != QK_STREAM_COUNT or page_table.shape[0] != MINIMUM_WIDTH:
            raise NativeMtpVerifierError("native verifier page table must have one row")
        committed_width = width if committed_width is None else committed_width
        if not MINIMUM_WIDTH <= committed_width <= width:
            raise NativeMtpVerifierError(
                "native verifier commit width is outside policy"
            )

        tokens = torch.tensor(input_token_ids, dtype=torch.int32).reshape(
            width, MINIMUM_WIDTH
        )
        positions = torch.arange(start_position, stop_position, dtype=torch.int32)
        aliased_page_table = page_table.repeat(width, MINIMUM_WIDTH)
        host_inputs = self.model.prepare_decode_inputs_host(
            tokens,
            positions,
            page_table=aliased_page_table,
        )
        device_inputs = copy_host_to_device(
            host_inputs,
            mesh_device=self.model.mesh_device,
        )
        if len(device_inputs) != FOUR_DIMENSIONAL_TENSOR:
            raise NativeMtpVerifierError("native verifier model inputs are incomplete")

        snapshots = self._clone_gdn_state()
        prefix_snapshots = (
            self._clone_gdn_prefix_states(width) if committed_width < width else ()
        )
        trace_id = None
        logits_device = None
        try:
            capture_started = time.monotonic()
            warm_logits = self._forward_device(
                device_inputs,
                width,
                prefix_snapshots=prefix_snapshots,
            )
            ttnn.synchronize_device(self.model.mesh_device)
            ttnn.deallocate(warm_logits)
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.model.mesh_device)

            # Hygiene gate: the eager warm forward above primed real execution
            # (never zeroed buffers), no other trace may be resident, and the
            # trace-bound input tensors must keep stable identities across
            # capture and replay because the captured graph bakes them.
            capture_input_identities = tuple(
                id(tensor) for tensor in device_inputs
            )
            hygiene_reasons = validate_trace_capture_provenance(
                TraceCaptureProvenance(
                    resident_traces_before=1
                    if self._hygiene_capture_resident
                    else 0,
                    warm_execution_completed=True,
                    capture_from_zeroed_priming=False,
                    trace_bound_addresses_stable=True,
                )
            )
            if hygiene_reasons:
                raise NativeMtpVerifierError(
                    "native verifier trace capture hygiene failed: "
                    + ", ".join(hygiene_reasons)
                )
            self._hygiene_capture_resident = True

            trace_id = ttnn.begin_trace_capture(
                self.model.mesh_device,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            logits_device = self._forward_device(
                device_inputs,
                width,
                prefix_snapshots=prefix_snapshots,
            )
            ttnn.end_trace_capture(
                self.model.mesh_device,
                trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.model.mesh_device)
            trace_capture_seconds = time.monotonic() - capture_started
            if (
                not math.isfinite(trace_capture_seconds)
                or trace_capture_seconds <= 0
                or trace_capture_seconds > self.trace_capture_timeout_seconds
            ):
                raise NativeMtpVerifierError(
                    "native verifier trace capture exceeded its bound"
                )

            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.model.mesh_device)
            if (
                tuple(id(tensor) for tensor in device_inputs)
                != capture_input_identities
            ):
                raise NativeMtpVerifierError(
                    "native verifier trace-bound input addresses drifted"
                )
            device_started = time.monotonic()
            copy_host_to_device(
                host_inputs[:MUTABLE_INPUT_COUNT],
                device_tensors=device_inputs[:MUTABLE_INPUT_COUNT],
            )
            ttnn.execute_trace(
                self.model.mesh_device,
                trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
                blocking=TRACE_BLOCKING_REPLAY,
            )
            ttnn.synchronize_device(self.model.mesh_device)
            device_seconds = time.monotonic() - device_started
            if not math.isfinite(device_seconds) or device_seconds <= 0:
                raise NativeMtpVerifierError(
                    "native verifier device duration is invalid"
                )

            readback_started = time.monotonic()
            device_logits = ttnn.get_device_tensors(logits_device)
            if not device_logits:
                raise NativeMtpVerifierError(
                    "native verifier produced no logit replica"
                )
            logits = (
                ttnn
                .to_torch(device_logits[0])
                .float()
                .reshape(-1, self.model.args.vocab_size)[:width]
                .clone()
            )
            readback_seconds = time.monotonic() - readback_started
            if logits.shape != (width, self.model.args.vocab_size):
                raise NativeMtpVerifierError(
                    "native verifier logits have the wrong shape"
                )
            if not torch.isfinite(logits).all():
                raise NativeMtpVerifierError("native verifier logits are not finite")
            if not math.isfinite(readback_seconds) or readback_seconds <= 0:
                raise NativeMtpVerifierError(
                    "native verifier readback duration is invalid"
                )
            token_ids = tuple(
                int(torch.argmax(logits[row]).item()) for row in range(width)
            )
            rollback_seconds = ZERO_DURATION_SECONDS
            if committed_width < width:
                ttnn.release_trace(self.model.mesh_device, trace_id)
                trace_id = None
                rollback_started = time.monotonic()
                self._restore_gdn_prefix_state(
                    prefix_snapshots[committed_width - MINIMUM_WIDTH]
                )
                ttnn.synchronize_device(self.model.mesh_device)
                rollback_seconds = time.monotonic() - rollback_started
                if not math.isfinite(rollback_seconds) or rollback_seconds <= 0:
                    raise NativeMtpVerifierError(
                        "native verifier rollback duration is invalid"
                    )
            return NativeVerificationResult(
                token_ids=token_ids,
                logits=logits,
                trace_capture_seconds=trace_capture_seconds,
                device_seconds=device_seconds,
                readback_seconds=readback_seconds,
                rollback_seconds=rollback_seconds,
            )
        finally:
            self._hygiene_capture_resident = False
            if trace_id is not None:
                ttnn.release_trace(self.model.mesh_device, trace_id)
            if logits_device is not None:
                ttnn.deallocate(logits_device)
            self._deallocate_gdn_snapshots(snapshots)
            self._deallocate_gdn_prefix_states(prefix_snapshots)
            for tensor in device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)

    def _forward_device(
        self,
        device_inputs: tuple[Any, ...],
        width: int,
        *,
        prefix_snapshots: tuple[tuple[_GdnPrefixSnapshot, ...], ...] = (),
    ) -> Any:
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope

        token_buffer, position_buffer, rope_buffer, page_table_buffer = device_inputs
        cos, sin = unpack_rope(rope_buffer)
        x = self.model.embd(token_buffer)
        x = ttnn.reshape(
            x,
            (
                MINIMUM_WIDTH,
                MINIMUM_WIDTH,
                width,
                x.shape[LAST_DIMENSION],
            ),
        )
        gdn_index = 0
        for layer in self.model.layers:
            layer_prefix_snapshots = (
                tuple(prefix[gdn_index] for prefix in prefix_snapshots)
                if prefix_snapshots and not layer.is_full_attention
                else ()
            )
            x = _verify_layer(
                layer,
                x,
                cos=cos,
                sin=sin,
                positions=position_buffer,
                page_table=page_table_buffer,
                prefix_snapshots=layer_prefix_snapshots,
            )
            if not layer.is_full_attention:
                gdn_index += MINIMUM_WIDTH
        if prefix_snapshots and gdn_index != len(prefix_snapshots[0]):
            raise NativeMtpVerifierError(
                "native verifier prefix snapshot layer count changed"
            )
        x = self.model._final_norm_decode(x)
        logits = self.model._lm_head(x)
        ttnn.deallocate(x)
        return logits

    def _clone_gdn_state(self) -> tuple[_GdnDeviceSnapshot, ...]:
        snapshots = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            attention = layer.attention
            if (
                attention.rec_state is None
                or attention.conv_states is None
                or attention.conv_carry is None
            ):
                raise NativeMtpVerifierError(
                    "native verifier GDN state is not allocated"
                )
            snapshots.append(
                _GdnDeviceSnapshot(
                    layer=attention,
                    recurrent=ttnn.clone(attention.rec_state),
                    convolution=tuple(
                        ttnn.clone(state) for state in attention.conv_states
                    ),
                    convolution_carry=ttnn.clone(attention.conv_carry),
                )
            )
        if not snapshots:
            raise NativeMtpVerifierError("native verifier has no GDN state")
        return tuple(snapshots)

    def _clone_gdn_prefix_states(
        self,
        width: int,
    ) -> tuple[tuple[_GdnPrefixSnapshot, ...], ...]:
        prefixes = []
        for _offset in range(width):
            layers = []
            for layer in self.model.layers:
                if layer.is_full_attention:
                    continue
                attention = layer.attention
                if attention.rec_state is None or attention.conv_states is None:
                    raise NativeMtpVerifierError(
                        "native verifier GDN prefix state is not allocated"
                    )
                layers.append(
                    _GdnPrefixSnapshot(
                        layer=attention,
                        recurrent=ttnn.clone(attention.rec_state),
                        convolution=tuple(
                            ttnn.clone(state) for state in attention.conv_states
                        ),
                    )
                )
            prefixes.append(tuple(layers))
        return tuple(prefixes)

    @staticmethod
    def _restore_gdn_state(snapshots: tuple[_GdnDeviceSnapshot, ...]) -> None:
        for snapshot in snapshots:
            ttnn.copy(snapshot.recurrent, snapshot.layer.rec_state)
            for source, target in zip(
                snapshot.convolution,
                snapshot.layer.conv_states,
                strict=True,
            ):
                ttnn.copy(source, target)
            ttnn.copy(snapshot.convolution_carry, snapshot.layer.conv_carry)

    @staticmethod
    def _restore_gdn_prefix_state(
        snapshots: tuple[_GdnPrefixSnapshot, ...],
    ) -> None:
        for snapshot in snapshots:
            ttnn.copy(snapshot.recurrent, snapshot.layer.rec_state)
            for source, target in zip(
                snapshot.convolution,
                snapshot.layer.conv_states,
                strict=True,
            ):
                ttnn.copy(source, target)
            _synchronize_convolution_carry(snapshot.layer)

    @staticmethod
    def _deallocate_gdn_snapshots(
        snapshots: tuple[_GdnDeviceSnapshot, ...],
    ) -> None:
        for snapshot in snapshots:
            ttnn.deallocate(snapshot.recurrent)
            for tensor in snapshot.convolution:
                ttnn.deallocate(tensor)
            ttnn.deallocate(snapshot.convolution_carry)

    @staticmethod
    def _deallocate_gdn_prefix_states(
        prefixes: tuple[tuple[_GdnPrefixSnapshot, ...], ...],
    ) -> None:
        for snapshots in prefixes:
            for snapshot in snapshots:
                ttnn.deallocate(snapshot.recurrent)
                for tensor in snapshot.convolution:
                    ttnn.deallocate(tensor)


def validate_native_verifier_model(model: Any) -> None:
    """Fail before device work when an adopted private model seam changed."""
    required_model_attributes = (
        "args",
        "embd",
        "layers",
        "mesh_device",
        "prepare_inputs_decode",
        "_final_norm_decode",
        "_lm_head",
    )
    missing_model = tuple(
        name for name in required_model_attributes if not hasattr(model, name)
    )
    if missing_model:
        raise NativeMtpVerifierError(
            "native verifier model seams are missing: " + ",".join(missing_model)
        )
    if not model.layers:
        raise NativeMtpVerifierError("native verifier model has no layers")
    for layer in model.layers:
        required_layer_attributes = (
            "args",
            "attention",
            "attention_norm",
            "feed_forward",
            "ffn_norm",
            "is_full_attention",
        )
        missing_layer = tuple(
            name for name in required_layer_attributes if not hasattr(layer, name)
        )
        if missing_layer:
            raise NativeMtpVerifierError(
                "native verifier layer seams are missing: " + ",".join(missing_layer)
            )


def _verify_layer(
    layer: Any,
    x: Any,
    *,
    cos: Any,
    sin: Any,
    positions: Any,
    page_table: Any,
    prefix_snapshots: tuple[_GdnPrefixSnapshot, ...],
) -> Any:
    attention_norm_config = layer.args.get_norm_config("attn", Mode.DECODE)
    feed_forward_norm_config = layer.args.get_norm_config("attn", Mode.DECODE)
    attention_input = layer.attention_norm(
        x,
        mode=Mode.DECODE,
        norm_config=attention_norm_config,
    )
    attention_output = (
        _verify_full_attention(
            layer.attention,
            attention_input,
            cos=cos,
            sin=sin,
            positions=positions,
            page_table=page_table,
        )
        if layer.is_full_attention
        else _verify_gdn(
            layer.attention,
            attention_input,
            prefix_snapshots=prefix_snapshots,
        )
    )
    ttnn.deallocate(attention_input)
    residual = ttnn.add(x, attention_output)
    ttnn.deallocate(x)
    ttnn.deallocate(attention_output)
    feed_forward_input = layer.ffn_norm(
        residual,
        mode=Mode.DECODE,
        norm_config=feed_forward_norm_config,
    )
    feed_forward_output = layer.feed_forward.forward(feed_forward_input)
    ttnn.deallocate(feed_forward_input)
    output = ttnn.add(residual, feed_forward_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(feed_forward_output)
    return output


def _verify_gdn(
    layer: Any,
    x: Any,
    *,
    prefix_snapshots: tuple[_GdnPrefixSnapshot, ...],
) -> Any:
    if layer.rec_state is None or layer.conv_states is None:
        raise NativeMtpVerifierError("native verifier GDN state is not allocated")
    if len(x.shape) == FOUR_DIMENSIONAL_TENSOR:
        x = ttnn.reshape(
            x, (MINIMUM_WIDTH, x.shape[PENULTIMATE_DIMENSION], x.shape[LAST_DIMENSION])
        )
    width = x.shape[PENULTIMATE_DIMENSION]
    if width < MINIMUM_WIDTH:
        raise NativeMtpVerifierError("native verifier GDN width is empty")
    if prefix_snapshots and len(prefix_snapshots) != width:
        raise NativeMtpVerifierError(
            "native verifier GDN prefix snapshot width changed"
        )

    tw = layer.tw
    key_heads = layer.Nk
    value_heads = layer.Nv
    key_width = layer.Dk
    value_width = layer.Dv
    qkv_width = layer.qkv_dim_tp
    key_projection_width = layer.key_dim_tp
    qkv, z, a, b = layer._project_qkvzab(
        x,
        width,
        out_mc=ttnn.L1_MEMORY_CONFIG,
    )
    beta = ttnn.reshape(
        ttnn.sigmoid(b, memory_config=ttnn.L1_MEMORY_CONFIG),
        (MINIMUM_WIDTH, width, value_heads),
    )
    ttnn.deallocate(b)
    decay = ttnn.multiply(
        tw["neg_exp_A"],
        _softplus_add(a, tw["dt_bias"]),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(a)
    decay = ttnn.reshape(decay, (MINIMUM_WIDTH, width, value_heads))

    output_rows = []
    convolution_states = layer.conv_states
    for offset in range(width):
        for state_index in range(layer.K - MINIMUM_WIDTH):
            ttnn.copy(
                convolution_states[state_index + MINIMUM_WIDTH],
                convolution_states[state_index],
            )
        qkv_row = ttnn.slice(
            qkv,
            (0, offset, 0),
            (MINIMUM_WIDTH, offset + MINIMUM_WIDTH, qkv_width),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.copy(qkv_row, convolution_states[layer.K - MINIMUM_WIDTH])
        # A full-width slice can alias qkv on TT-Metal 0.74. Do not force-free
        # the view before later verifier positions consume the parent.

        convolution = ttnn.multiply(
            convolution_states[0],
            tw["conv_taps"][0],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for tap_index in range(MINIMUM_WIDTH, layer.K):
            convolution = ttnn.mac(
                convolution_states[tap_index],
                tw["conv_taps"][tap_index],
                convolution,
            )
        convolution = ttnn.silu(
            convolution,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        query = ttnn.reshape(
            ttnn.slice(
                convolution,
                (0, 0, 0),
                (MINIMUM_WIDTH, MINIMUM_WIDTH, key_projection_width),
            ),
            (MINIMUM_WIDTH, key_heads, key_width),
        )
        key = ttnn.reshape(
            ttnn.slice(
                convolution,
                (0, 0, key_projection_width),
                (
                    MINIMUM_WIDTH,
                    MINIMUM_WIDTH,
                    QK_STREAM_COUNT * key_projection_width,
                ),
            ),
            (MINIMUM_WIDTH, key_heads, key_width),
        )
        value = ttnn.reshape(
            ttnn.slice(
                convolution,
                (0, 0, QK_STREAM_COUNT * key_projection_width),
                (MINIMUM_WIDTH, MINIMUM_WIDTH, qkv_width),
            ),
            (MINIMUM_WIDTH, value_heads, value_width),
        )
        ttnn.deallocate(convolution)
        repeat_factor = value_heads // key_heads
        query = ttnn.repeat_interleave(query, repeat_factor, dim=MINIMUM_WIDTH)
        key = ttnn.repeat_interleave(key, repeat_factor, dim=MINIMUM_WIDTH)
        query = ttnn.reshape(
            query,
            (MINIMUM_WIDTH, MINIMUM_WIDTH, value_heads, key_width),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key = ttnn.reshape(
            key,
            (MINIMUM_WIDTH, MINIMUM_WIDTH, value_heads, key_width),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value = ttnn.reshape(
            value,
            (MINIMUM_WIDTH, MINIMUM_WIDTH, value_heads, value_width),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        beta_row = ttnn.slice(
            beta,
            (0, offset, 0),
            (MINIMUM_WIDTH, offset + MINIMUM_WIDTH, value_heads),
        )
        decay_row = ttnn.slice(
            decay,
            (0, offset, 0),
            (MINIMUM_WIDTH, offset + MINIMUM_WIDTH, value_heads),
        )
        output_row, new_recurrent = recurrent_gated_delta_rule_decode_ttnn(
            query,
            key,
            value,
            beta_row,
            decay_row,
            scale=layer.scale,
            initial_state=layer.rec_state,
            device=layer.mesh,
            high_precision=True,
        )
        ttnn.copy(new_recurrent, layer.rec_state)
        ttnn.deallocate(new_recurrent)
        if prefix_snapshots:
            prefix_snapshot = prefix_snapshots[offset]
            if prefix_snapshot.layer is not layer:
                raise NativeMtpVerifierError(
                    "native verifier GDN prefix snapshot owner changed"
                )
            ttnn.copy(layer.rec_state, prefix_snapshot.recurrent)
            for source, target in zip(
                layer.conv_states,
                prefix_snapshot.convolution,
                strict=True,
            ):
                ttnn.copy(source, target)
        output_rows.append(
            ttnn.reshape(output_row, (MINIMUM_WIDTH, value_heads, value_width))
        )

    ttnn.deallocate(qkv)
    ttnn.deallocate(beta)
    ttnn.deallocate(decay)
    _synchronize_convolution_carry(layer)
    if len(output_rows) == MINIMUM_WIDTH:
        output_recurrent = output_rows[0]
    else:
        output_recurrent = ttnn.concat(output_rows, dim=0)
        for output_row in output_rows:
            ttnn.deallocate(output_row)
    normalized = ttnn.rms_norm(
        output_recurrent,
        weight=tw["norm_w"],
        epsilon=RMS_NORM_EPSILON,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(output_recurrent)
    flattened = ttnn.reshape(
        normalized,
        (MINIMUM_WIDTH, width, layer.value_dim_tp),
    )
    ttnn.deallocate(normalized)
    gated = _silu_mul(flattened, z, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(flattened)
    ttnn.deallocate(z)
    partial = layer._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(
        partial,
        (
            MINIMUM_WIDTH,
            MINIMUM_WIDTH,
            width,
            partial.shape[LAST_DIMENSION],
        ),
    )
    return tt_all_reduce(
        partial,
        layer.mesh,
        layer.tt_ccl,
        cluster_axis=FULL_ATTENTION_CLUSTER_AXIS,
        dim=HIDDEN_DIMENSION,
        topology=layer.args.ccl_topology(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _synchronize_convolution_carry(layer: Any) -> None:
    if layer.conv_carry is None:
        raise NativeMtpVerifierError("native verifier GDN carry is not allocated")
    carry_rows = layer.conv_states[MINIMUM_WIDTH:]
    if not carry_rows:
        raise NativeMtpVerifierError("native verifier GDN carry has no rows")
    carry = (
        carry_rows[0]
        if len(carry_rows) == MINIMUM_WIDTH
        else ttnn.concat(carry_rows, dim=MINIMUM_WIDTH)
    )
    ttnn.copy(carry, layer.conv_carry)
    if carry is not carry_rows[0]:
        ttnn.deallocate(carry)


def _verify_full_attention(
    layer: Any,
    x: Any,
    *,
    cos: Any,
    sin: Any,
    positions: Any,
    page_table: Any,
) -> Any:
    if not layer.use_paged or layer.paged_k is None or layer.paged_v is None:
        raise NativeMtpVerifierError("native verifier paged attention is not bound")
    width = x.shape[PENULTIMATE_DIMENSION]
    tw = layer.tw
    query_gate, key_projection, value_projection = layer._qkv(x)
    query, gate, key, value = layer._make_heads_decode(
        query_gate,
        key_projection,
        value_projection,
        width,
    )
    query = ttnn.multiply(
        ttnn.rms_norm(
            query,
            epsilon=RMS_NORM_EPSILON,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        tw["q_norm"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    key = ttnn.multiply(
        ttnn.rms_norm(
            key,
            epsilon=RMS_NORM_EPSILON,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        tw["k_norm"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    query = apply_partial_rope_decode(
        query,
        cos,
        sin,
        layer.NH,
        width,
        layer.rope_dim,
    )
    key = apply_partial_rope_decode(
        key,
        cos,
        sin,
        layer.NKV,
        width,
        layer.rope_dim,
    )
    key_padded = ttnn.pad(
        key,
        [MINIMUM_WIDTH, width, TILE_EDGE_ELEMENTS, layer.HD],
        [0, 0, 0, 0],
        0.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    value_padded = ttnn.pad(
        value,
        [MINIMUM_WIDTH, width, TILE_EDGE_ELEMENTS, layer.HD],
        [0, 0, 0, 0],
        0.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(key)
    ttnn.deallocate(value)
    _write_paged_kv_in_order(
        layer,
        key_padded,
        value_padded,
        positions=positions,
        page_table=page_table,
        width=width,
    )
    grid = layer.mesh.compute_with_storage_grid_size()
    sdpa_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        query,
        layer.paged_k,
        layer.paged_v,
        page_table_tensor=page_table,
        cur_pos_tensor=positions,
        scale=layer.scale,
        program_config=sdpa_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(query)
    gated = ttnn.multiply(
        attention,
        ttnn.sigmoid(gate, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(attention)
    ttnn.deallocate(gate)
    flattened = layer._concat_heads_decode(gated, width)
    partial = layer._wo_proj(flattened, tw["wo"])
    ttnn.deallocate(flattened)
    partial = ttnn.reshape(
        partial,
        (
            MINIMUM_WIDTH,
            MINIMUM_WIDTH,
            width,
            partial.shape[LAST_DIMENSION],
        ),
    )
    return tt_all_reduce(
        partial,
        layer.mesh,
        layer.tt_ccl,
        cluster_axis=FULL_ATTENTION_CLUSTER_AXIS,
        dim=HIDDEN_DIMENSION,
        topology=layer.args.ccl_topology(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _write_paged_kv_in_order(
    layer: Any,
    key: Any,
    value: Any,
    *,
    positions: Any,
    page_table: Any,
    width: int,
) -> None:
    one_core = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    })
    single_user_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            one_core,
            (TILE_EDGE_ELEMENTS, layer.HD),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    key_dram = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
    value_dram = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
    for row in range(width):
        key_row_dram = ttnn.slice(
            key_dram,
            [0, row, 0, 0],
            [MINIMUM_WIDTH, row + MINIMUM_WIDTH, TILE_EDGE_ELEMENTS, layer.HD],
        )
        value_row_dram = ttnn.slice(
            value_dram,
            [0, row, 0, 0],
            [MINIMUM_WIDTH, row + MINIMUM_WIDTH, TILE_EDGE_ELEMENTS, layer.HD],
        )
        key_row = ttnn.to_memory_config(key_row_dram, single_user_memory)
        value_row = ttnn.to_memory_config(value_row_dram, single_user_memory)
        # These DRAM slices can alias the full K/V parent on 0.74. Keep the
        # parent allocation live until every ordered write finishes.
        position_row = ttnn.slice(positions, [row], [row + MINIMUM_WIDTH])
        page_table_row = ttnn.slice(
            page_table,
            [row, 0],
            [row + MINIMUM_WIDTH, page_table.shape[MINIMUM_WIDTH]],
        )
        ttnn.experimental.paged_update_cache(
            layer.paged_k,
            key_row,
            update_idxs_tensor=position_row,
            page_table=page_table_row,
        )
        ttnn.experimental.paged_update_cache(
            layer.paged_v,
            value_row,
            update_idxs_tensor=position_row,
            page_table=page_table_row,
        )
        # Position and page-table slices can alias their parent allocations on
        # TT-Metal 0.74. Let their view handles expire without deallocation.
        ttnn.deallocate(key_row)
        ttnn.deallocate(value_row)
    ttnn.deallocate(key_dram)
    ttnn.deallocate(value_dram)
    if key_dram is not key:
        ttnn.deallocate(key)
    if value_dram is not value:
        ttnn.deallocate(value)
