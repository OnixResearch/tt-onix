"""TTNN adapter for the diagnostic Qwen3.8 target-verification probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_verification_probe]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.tt_transformers.tt.common import Mode

from .decode_adapter import TracedDecodeAdapter
from .mtp_probe_core import (
    MtpProbeIdentity,
    MtpProbeProfile,
    MtpProbeSample,
)

PROBE_BATCH_SIZE = 1
TOKEN_DIMENSION = 1
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
        self._page_table: torch.Tensor | None = None
        self._prompt_tokens: torch.Tensor | None = None
        self._prompt_token_count = 0

    def run(self, prompt_tokens: torch.Tensor) -> tuple[MtpProbeSample, ...]:
        """Warm the closed width set, then collect matched physical samples."""
        self._bind_prompt(prompt_tokens)
        teacher_inputs: dict[int, tuple[int, ...]] = {}
        for width in self.profile.verification_widths:
            first_token = self._prefill()
            inputs, _outputs, _elapsed = self._run_sequential(
                first_token, width, traced=False
            )
            teacher_inputs[width] = inputs
            self._prefill()
            verification_logits, _device_seconds, _readback_seconds = (
                self._run_teacher_forced(inputs)
            )
            if len(verification_logits) != width:
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
                sequential_inputs, sequential_outputs, sequential_seconds = (
                    self._run_sequential(
                        sequential_first_token,
                        width,
                        traced=True,
                    )
                )
                sequential_state = self.decode_adapter._snapshot_tp_gdn_state()

                # The masked verifier allocates temporary buffers. Release the parked
                # trace first. Recapture it for the next sequential sample.
                self.decode_adapter._release_trace()
                verification_first_token = self._prefill()
                if verification_first_token != sequential_first_token:
                    raise MtpProbeAdapterError(
                        "matched prefill produced different first tokens"
                    )
                if teacher_inputs[width] != sequential_inputs:
                    raise MtpProbeAdapterError(
                        "teacher input changed across deterministic replays"
                    )
                (
                    verification_outputs,
                    verification_device_seconds,
                    verification_readback_seconds,
                ) = self._run_teacher_forced(sequential_inputs)
                verification_state = self.decode_adapter._snapshot_tp_gdn_state()
                recurrent, convolution = compare_tp_gdn_states(
                    sequential_state,
                    verification_state,
                )
                samples.append(
                    MtpProbeSample(
                        width=width,
                        sample_index=sample_index,
                        identity=self.identity,
                        physical_pcie_device_ids=self.profile.expected_pcie_device_ids,
                        sequential_token_ids=sequential_outputs,
                        verification_token_ids=verification_outputs,
                        sequential_seconds=sequential_seconds,
                        verification_device_seconds=verification_device_seconds,
                        verification_readback_seconds=verification_readback_seconds,
                        recurrent_state_pcc=recurrent.pcc,
                        recurrent_state_nmse=recurrent.nmse,
                        convolution_state_pcc=convolution.pcc,
                        convolution_state_nmse=convolution.nmse,
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
        if self._prompt_token_count + self.profile.masked_bucket_width > (
            self.decode_adapter.profile.maximum_sequence_length
        ):
            raise MtpProbeAdapterError(
                "prompt leaves no room for the pinned verification bucket"
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
    ) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        current_token = first_token
        inputs = []
        outputs = []
        started = time.monotonic()
        for offset in range(width):
            inputs.append(current_token)
            position = self._prompt_token_count + offset
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
            outputs.append(current_token)
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise MtpProbeAdapterError("sequential decode duration is invalid")
        return tuple(inputs), tuple(outputs), elapsed

    def _run_teacher_forced(
        self, input_token_ids: tuple[int, ...]
    ) -> tuple[tuple[int, ...], float, float]:
        width = len(input_token_ids)
        if width not in self.profile.verification_widths:
            raise MtpProbeAdapterError("teacher-forced width is outside policy")
        if self._page_table is None:
            raise MtpProbeAdapterError("target probe page table is not bound")
        bucket = self.profile.masked_bucket_width
        real = torch.tensor([input_token_ids], dtype=torch.int32)
        padding = torch.zeros(
            (PROBE_BATCH_SIZE, bucket - width),
            dtype=torch.int32,
        )
        token_buffer = torch.cat([real, padding], dim=TOKEN_DIMENSION)

        device_started = time.monotonic()
        hidden = self.model._forward_prefill_chunk_masked(
            token_buffer,
            width,
            self._prompt_token_count,
            self._page_table,
            bucket,
            flex_sdpa=True,
        )
        selector = torch.zeros(
            PROBE_BATCH_SIZE,
            PROBE_BATCH_SIZE,
            width,
            bucket,
            dtype=torch.float32,
        )
        for row in range(width):
            selector[0, 0, row, row] = 1.0
        selector_device = ttnn.from_torch(
            selector,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.model.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh_device),
        )
        selected = ttnn.matmul(selector_device, hidden)
        ttnn.deallocate(selector_device)
        ttnn.deallocate(hidden)
        selected = ttnn.to_memory_config(selected, ttnn.DRAM_MEMORY_CONFIG)
        normalized = self.model.norm(selected, mode=Mode.PREFILL)
        ttnn.deallocate(selected)
        logits_device = self.model._lm_head(normalized)
        ttnn.deallocate(normalized)
        ttnn.synchronize_device(self.model.mesh_device)
        device_seconds = time.monotonic() - device_started

        readback_started = time.monotonic()
        logits = ttnn.to_torch(ttnn.get_device_tensors(logits_device)[0]).reshape(
            -1, self.model.args.vocab_size
        )
        outputs = tuple(
            int(torch.argmax(logits[row].float()).item()) for row in range(width)
        )
        readback_seconds = time.monotonic() - readback_started
        ttnn.deallocate(logits_device)
        if any(
            token < MINIMUM_TOKEN_ID or token >= self.model.args.vocab_size
            for token in outputs
        ):
            raise MtpProbeAdapterError(
                "teacher-forced verification produced an out-of-range token"
            )
        return outputs, device_seconds, readback_seconds


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
