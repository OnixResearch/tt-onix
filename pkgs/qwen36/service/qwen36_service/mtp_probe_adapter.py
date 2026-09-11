"""TTNN adapter for the diagnostic Qwen3.8 target-verification probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_verification_probe]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn

from .decode_adapter import TracedDecodeAdapter
from .mtp_probe_core import (
    MtpProbeIdentity,
    MtpProbeProfile,
    MtpProbeSample,
)
from .native_mtp_verifier_adapter import NativeMtpVerifierAdapter

PROBE_BATCH_SIZE = 1
TOKEN_DIMENSION = 1
CACHE_BLOCK_DIMENSION = 2
MINIMUM_TOKEN_ID = 0
EXACT_STATE_PCC = 1.0
EXACT_STATE_NMSE = 0.0
ZERO_VARIANCE = 0.0


class MtpProbeAdapterError(RuntimeError):
    """The physical target probe cannot continue safely."""


@dataclass(frozen=True)
class StateComparison:
    pcc: float
    nmse: float


@dataclass
class _ComparisonTotals:
    count: int = 0
    left_sum: float = 0.0
    right_sum: float = 0.0
    left_square_sum: float = 0.0
    right_square_sum: float = 0.0
    product_sum: float = 0.0
    error_square_sum: float = 0.0


class MtpVerificationProbeAdapter:
    """Run matched sequential and teacher-forced paths on one loaded model."""

    def __init__(
        self,
        *,
        model: Any,
        decode_adapter: TracedDecodeAdapter,
        profile: MtpProbeProfile,
        identity: MtpProbeIdentity,
    ) -> None:
        if model.num_devices != len(profile.expected_pcie_device_ids):
            raise MtpProbeAdapterError(
                "target probe device count does not match policy"
            )
        self.model = model
        self.decode_adapter = decode_adapter
        self.profile = profile
        self.identity = identity
        self._mesh_reader = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        self._native_verifier = NativeMtpVerifierAdapter(
            model=model,
            maximum_width=profile.native_width_limit,
            trace_capture_timeout_seconds=(
                decode_adapter.profile.trace_capture_timeout_seconds
            ),
        )
        self._page_table: torch.Tensor | None = None
        self._prompt_tokens: torch.Tensor | None = None
        self._prompt_token_count = 0

    def run(self, prompt_tokens: torch.Tensor) -> tuple[MtpProbeSample, ...]:
        """Warm the closed width set, then collect matched physical samples."""
        self._bind_prompt(prompt_tokens)
        teacher_inputs: dict[int, tuple[int, ...]] = {}
        for width in self.profile.verification_widths:
            first_token = self._prefill()
            inputs, _outputs, _logits, _elapsed = self._run_sequential(
                first_token, width, traced=False
            )
            teacher_inputs[width] = inputs
            self._prefill()
            verification = self._run_teacher_forced(inputs)
            if len(verification.token_ids) != width:
                raise MtpProbeAdapterError(
                    "warm target verification returned the wrong width"
                )

        samples = []
        for width in self.profile.verification_widths:
            for sample_index in range(1, self.profile.minimum_samples_per_width + 1):
                sequential_first_token = self._prefill()
                eligibility = self.decode_adapter.ensure_trace(
                    token=sequential_first_token,
                    position=self._prompt_token_count,
                )
                if not eligibility.eligible:
                    raise MtpProbeAdapterError(
                        "decode trace is unavailable for the probe: "
                        f"{eligibility.reason}"
                    )
                (
                    sequential_inputs,
                    sequential_outputs,
                    sequential_logits,
                    sequential_seconds,
                ) = self._run_sequential(
                    sequential_first_token,
                    width,
                    traced=True,
                )
                # State and KV readback can allocate conversion buffers. Release the
                # parked decode trace first so those diagnostic reads cannot corrupt it.
                self.decode_adapter._release_trace()
                sequential_state = self.decode_adapter._snapshot_tp_gdn_state()
                positions = tuple(
                    self._prompt_token_count + offset for offset in range(width)
                )
                sequential_kv = snapshot_tp_paged_kv(
                    self.model,
                    self._page_table,
                    positions,
                )
                verification_first_token = self._prefill()
                if verification_first_token != sequential_first_token:
                    raise MtpProbeAdapterError(
                        "matched prefill produced different first tokens"
                    )
                if teacher_inputs[width] != sequential_inputs:
                    raise MtpProbeAdapterError(
                        "teacher input changed across deterministic replays"
                    )
                verification = self._run_teacher_forced(sequential_inputs)
                verification_state = self.decode_adapter._snapshot_tp_gdn_state()
                verification_kv = snapshot_tp_paged_kv(
                    self.model,
                    self._page_table,
                    positions,
                )
                logits = compare_tensor_collections(
                    (sequential_logits,),
                    (verification.logits,),
                )
                recurrent, convolution = compare_tp_gdn_states(
                    sequential_state,
                    verification_state,
                )
                paged_kv = compare_tensor_collections(
                    sequential_kv,
                    verification_kv,
                )
                samples.append(
                    MtpProbeSample(
                        width=width,
                        sample_index=sample_index,
                        identity=self.identity,
                        physical_pcie_device_ids=self.profile.expected_pcie_device_ids,
                        sequential_token_ids=sequential_outputs,
                        verification_token_ids=verification.token_ids,
                        sequential_seconds=sequential_seconds,
                        verification_trace_capture_seconds=(
                            verification.trace_capture_seconds
                        ),
                        verification_device_seconds=verification.device_seconds,
                        verification_readback_seconds=verification.readback_seconds,
                        logits_pcc=logits.pcc,
                        logits_nmse=logits.nmse,
                        recurrent_state_pcc=recurrent.pcc,
                        recurrent_state_nmse=recurrent.nmse,
                        convolution_state_pcc=convolution.pcc,
                        convolution_state_nmse=convolution.nmse,
                        paged_kv_state_pcc=paged_kv.pcc,
                        paged_kv_state_nmse=paged_kv.nmse,
                    )
                )
        return tuple(samples)

    def _bind_prompt(self, prompt_tokens: torch.Tensor) -> None:
        if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != PROBE_BATCH_SIZE:
            raise MtpProbeAdapterError(
                "target probe prompt must have shape [1, tokens]"
            )
        if prompt_tokens.shape[1] <= 0:
            raise MtpProbeAdapterError("target probe prompt must not be empty")
        if prompt_tokens.dtype not in (torch.int32, torch.int64):
            raise MtpProbeAdapterError(
                "target probe prompt must contain integer token IDs"
            )
        if int(torch.min(prompt_tokens).item()) < MINIMUM_TOKEN_ID:
            raise MtpProbeAdapterError(
                "target probe prompt contains a negative token ID"
            )
        self._prompt_tokens = prompt_tokens.to(torch.int32)
        self._prompt_token_count = int(prompt_tokens.shape[TOKEN_DIMENSION])
        maximum_width = max(self.profile.verification_widths)
        self._page_table = self.decode_adapter.prefill_page_table(
            prompt_tokens=self._prompt_token_count,
            generation_tokens=maximum_width,
        )
        if self._prompt_token_count + self.profile.native_width_limit > (
            self.decode_adapter.profile.maximum_sequence_length
        ):
            raise MtpProbeAdapterError(
                "prompt leaves no room for the native verifier width"
            )

    def _prefill(self) -> int:
        if self._prompt_tokens is None or self._page_table is None:
            raise MtpProbeAdapterError("target probe prompt is not bound")
        self.decode_adapter.reset_for_request()
        logits_device = self.model.prefill_traced_chunked(
            self._prompt_tokens,
            self._page_table,
            actual_len=self._prompt_token_count,
        )
        ttnn.synchronize_device(self.model.mesh_device)
        logits = ttnn.to_torch(
            logits_device,
            mesh_composer=self._mesh_reader,
        ).reshape(-1, self.model.args.vocab_size)[0]
        ttnn.deallocate(logits_device)
        token = int(torch.argmax(logits.float()).item())
        if not MINIMUM_TOKEN_ID <= token < self.model.args.vocab_size:
            raise MtpProbeAdapterError("prefill produced an out-of-range token")
        return token

    def _run_sequential(
        self, first_token: int, width: int, *, traced: bool
    ) -> tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, float]:
        current_token = first_token
        inputs = []
        outputs = []
        logits_rows = []
        elapsed = 0.0
        for offset in range(width):
            inputs.append(current_token)
            position = self._prompt_token_count + offset
            started = time.monotonic()
            current_token = (
                self.decode_adapter.decode_traced(
                    token=current_token,
                    position=position,
                )
                if traced
                else self.decode_adapter.decode_eager(
                    token=current_token,
                    position=position,
                )
            )
            elapsed += time.monotonic() - started
            outputs.append(current_token)
            if traced:
                logits_rows.append(self.decode_adapter.read_trace_logits())
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise MtpProbeAdapterError("sequential decode duration is invalid")
        logits = (
            torch.stack(logits_rows)
            if logits_rows
            else torch.empty((0, self.model.args.vocab_size), dtype=torch.float32)
        )
        return tuple(inputs), tuple(outputs), logits, elapsed

    def _run_teacher_forced(self, input_token_ids: tuple[int, ...]):
        width = len(input_token_ids)
        if width not in self.profile.verification_widths:
            raise MtpProbeAdapterError("teacher-forced width is outside policy")
        if self._page_table is None:
            raise MtpProbeAdapterError("target probe page table is not bound")
        return self._native_verifier.run(
            input_token_ids,
            start_position=self._prompt_token_count,
            page_table=self._page_table,
        )


def snapshot_tp_paged_kv(
    model: Any,
    page_table: torch.Tensor | None,
    positions: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """Read only the full-attention K/V positions written by one verifier."""
    if (
        page_table is None
        or page_table.ndim != 2
        or page_table.shape[0] != PROBE_BATCH_SIZE
    ):
        raise MtpProbeAdapterError("paged-KV snapshot requires one page-table row")
    if not positions or any(position < MINIMUM_TOKEN_ID for position in positions):
        raise MtpProbeAdapterError("paged-KV snapshot positions are invalid")
    snapshots = []
    full_attention_layers = tuple(
        layer.attention for layer in model.layers if layer.is_full_attention
    )
    if not full_attention_layers:
        raise MtpProbeAdapterError("the model has no full-attention layers")
    for layer in full_attention_layers:
        for cache in (layer.paged_k, layer.paged_v):
            if cache is None or len(cache.shape) <= CACHE_BLOCK_DIMENSION:
                raise MtpProbeAdapterError("paged-KV cache is not bound")
            block_size = int(cache.shape[CACHE_BLOCK_DIMENSION])
            if block_size <= 0:
                raise MtpProbeAdapterError("paged-KV block size is invalid")
            for device_cache in ttnn.get_device_tensors(cache):
                host_cache = ttnn.to_torch(device_cache)
                selected = []
                for position in positions:
                    logical_block = position // block_size
                    block_offset = position % block_size
                    if logical_block >= page_table.shape[TOKEN_DIMENSION]:
                        raise MtpProbeAdapterError(
                            "paged-KV snapshot exceeds the page table"
                        )
                    physical_block = int(page_table[0, logical_block].item())
                    if not 0 <= physical_block < host_cache.shape[0]:
                        raise MtpProbeAdapterError(
                            "paged-KV snapshot block is outside the cache"
                        )
                    selected.append(
                        host_cache[physical_block, :, block_offset, :].clone()
                    )
                snapshots.append(torch.stack(selected))
    if not snapshots:
        raise MtpProbeAdapterError("paged-KV snapshot is empty")
    return tuple(snapshots)


def compare_tensor_collections(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> StateComparison:
    """Compare one non-empty tensor collection with aggregate PCC and NMSE."""
    if len(left) != len(right) or not left:
        raise MtpProbeAdapterError("tensor collection counts do not match")
    totals = _ComparisonTotals()
    for left_tensor, right_tensor in zip(left, right, strict=True):
        _accumulate_tensor(totals, left_tensor, right_tensor)
    return _finish_comparison(totals)


def compare_tp_gdn_states(
    left: tuple[tuple[torch.Tensor, tuple[torch.Tensor, ...]], ...],
    right: tuple[tuple[torch.Tensor, tuple[torch.Tensor, ...]], ...],
) -> tuple[StateComparison, StateComparison]:
    """Compare full TP recurrent and convolution state with aggregate metrics."""
    if len(left) != len(right) or not left:
        raise MtpProbeAdapterError("GDN state layer counts do not match")
    recurrent_totals = _ComparisonTotals()
    convolution_totals = _ComparisonTotals()
    for left_layer, right_layer in zip(left, right, strict=True):
        left_recurrent, left_convolution = left_layer
        right_recurrent, right_convolution = right_layer
        _accumulate_tensor(recurrent_totals, left_recurrent, right_recurrent)
        if len(left_convolution) != len(right_convolution):
            raise MtpProbeAdapterError("GDN convolution state counts do not match")
        for left_state, right_state in zip(
            left_convolution, right_convolution, strict=True
        ):
            _accumulate_tensor(convolution_totals, left_state, right_state)
    return (
        _finish_comparison(recurrent_totals),
        _finish_comparison(convolution_totals),
    )


def _accumulate_tensor(
    totals: _ComparisonTotals, left: torch.Tensor, right: torch.Tensor
) -> None:
    if left.shape != right.shape or left.numel() == 0:
        raise MtpProbeAdapterError("GDN state tensor shapes do not match")
    left_values = left.detach().reshape(-1).to(torch.float64)
    right_values = right.detach().reshape(-1).to(torch.float64)
    if not torch.isfinite(left_values).all() or not torch.isfinite(right_values).all():
        raise MtpProbeAdapterError("GDN state contains a non-finite value")
    difference = left_values - right_values
    totals.count += int(left_values.numel())
    totals.left_sum += float(torch.sum(left_values).item())
    totals.right_sum += float(torch.sum(right_values).item())
    totals.left_square_sum += float(torch.sum(left_values * left_values).item())
    totals.right_square_sum += float(torch.sum(right_values * right_values).item())
    totals.product_sum += float(torch.sum(left_values * right_values).item())
    totals.error_square_sum += float(torch.sum(difference * difference).item())


def _finish_comparison(totals: _ComparisonTotals) -> StateComparison:
    if totals.count <= 0:
        raise MtpProbeAdapterError("GDN state comparison is empty")
    count = float(totals.count)
    left_centered = totals.left_square_sum - (totals.left_sum**2 / count)
    right_centered = totals.right_square_sum - (totals.right_sum**2 / count)
    covariance = totals.product_sum - (totals.left_sum * totals.right_sum / count)
    denominator = math.sqrt(max(ZERO_VARIANCE, left_centered * right_centered))
    if denominator == ZERO_VARIANCE:
        pcc = (
            EXACT_STATE_PCC
            if totals.error_square_sum == ZERO_VARIANCE
            else ZERO_VARIANCE
        )
    else:
        pcc = covariance / denominator
    nmse = (
        totals.error_square_sum / totals.right_square_sum
        if totals.right_square_sum > ZERO_VARIANCE
        else (
            EXACT_STATE_NMSE if totals.error_square_sum == ZERO_VARIANCE else math.inf
        )
    )
    return StateComparison(pcc=pcc, nmse=nmse)
