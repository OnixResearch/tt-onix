"""Device-free checks for admission before batch effects and worker recovery."""

import json
import sys
from collections import deque
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from qwen36_service.batch_admission_core import BatchAdmissionProfile
from qwen36_service.batch_decode_adapter import BatchDecodeAdapterError
from qwen36_service.batch_service import (
    MAXIMUM_BATCH_WIDTH,
    BatchGenerationCoordinator,
    _PendingBatchRequest,
)
from qwen36_service.core import CompletionRequest, Endpoint
from qwen36_service.sampled_mode_router_core import ROUTE_DETERMINISTIC

GENERATION_TOKENS = 2
TOKEN_ID = 1
VOCABULARY_SIZE = 128
MAXIMUM_SEQUENCE_LENGTH = 128
DEADLINE_SECONDS = 100.0
FIXTURE_SECONDS = 0.01
PHYSICAL_DEVICE_IDS = (0, 1)
PROFILE_NAMES = (
    "batch-admission-profile.json",
    "batch-admission-b8-profile.json",
    "batch-admission-b16-profile.json",
)
POLICY_DIAGNOSTIC = "batch admission measurement is outside policy"


class BatchAdapter:
    """Record effect order without a device or trace compiler."""

    fusion_mode = ROUTE_DETERMINISTIC
    blocks_per_user = 1

    def __init__(self, *, prefill_error=False, reset_error=False):
        self.calls = []
        self.prefill_error = prefill_error
        self.reset_error = reset_error

    def prefill_users(self, prompts, **kwargs):
        self.calls.append(("prefill", len(prompts)))
        if self.prefill_error:
            raise BatchDecodeAdapterError("fixture prefill failed")
        return SimpleNamespace(
            first_token_ids=(TOKEN_ID,) * len(prompts),
            first_token_ready_seconds=FIXTURE_SECONDS,
            state_install_seconds=FIXTURE_SECONDS,
            page_table_blocks=1,
        )

    def prepare_width(self, *, token_ids, positions):
        assert len(token_ids) == len(positions)
        self.calls.append(("prepare", len(token_ids)))
        return SimpleNamespace(
            resident_trace_count=1,
            binding_stable=True,
            capture_count=1,
            capture_seconds=FIXTURE_SECONDS,
            reused=False,
            transition_reason=None,
        )

    def decode_traced(self, *, token_ids, positions):
        assert len(token_ids) == len(positions)
        self.calls.append(("decode", len(token_ids)))
        return token_ids

    def reset_all(self):
        self.calls.append(("reset", 0))
        if self.reset_error:
            raise BatchDecodeAdapterError("fixture reset failed")


def pending_requests(width):
    request = CompletionRequest(
        endpoint=Endpoint.COMPLETIONS,
        model="model-test",
        max_tokens=GENERATION_TOKENS,
        prompt="fixture",
        messages=(),
    )
    return tuple(
        _PendingBatchRequest(
            request=request,
            request_id=f"request-{index}",
            token_ids=torch.tensor([[TOKEN_ID]], dtype=torch.int32),
            prompt_token_count=1,
            started=0.0,
            deadline=DEADLINE_SECONDS,
        )
        for index in range(width)
    )


def run_worker(profile, groups, *, enabled=True, adapter=None):
    coordinator = object.__new__(BatchGenerationCoordinator)
    coordinator.adapter = adapter if adapter is not None else BatchAdapter()
    coordinator.batch_admission_profile = profile
    coordinator.batch_admission_enabled = enabled
    coordinator.trace_reuse_profile = None
    coordinator.trace_reuse_admission_enabled = False
    coordinator.sampled_admission_enabled = False
    coordinator.profile = SimpleNamespace(profile_id="production-profile")
    coordinator.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "fixture")
    coordinator.model = SimpleNamespace(
        args=SimpleNamespace(vocab_size=VOCABULARY_SIZE)
    )
    coordinator.maximum_sequence_length = MAXIMUM_SEQUENCE_LENGTH
    coordinator.model_identity = "fixture-model"
    coordinator.package_identity = "fixture-package"
    coordinator.runtime_identity = "fixture-runtime"
    coordinator.physical_pcie_device_ids = PHYSICAL_DEVICE_IDS
    emitted = []
    coordinator.emitter = lambda stage, **fields: emitted.append((stage, fields))
    remaining = deque((*groups, None))
    coordinator._take_batch = remaining.popleft
    ticks = count(start=0.0, step=FIXTURE_SECONDS)
    with patch("qwen36_service.batch_service.time.monotonic", side_effect=ticks):
        coordinator._worker_loop()
    assert not remaining
    assert all(pending.completed.is_set() for group in groups for pending in group)
    return coordinator.adapter, emitted


def check(profile_directory):
    case_count = 0
    for profile_name in PROFILE_NAMES:
        profile = BatchAdmissionProfile.from_mapping(
            json.loads((profile_directory / profile_name).read_text())
        )
        accepted_widths = (profile.baseline_width, profile.candidate_width)
        for width in accepted_widths:
            group = pending_requests(width)
            adapter, emitted = run_worker(profile, (group,))
            assert adapter.calls == [
                ("prefill", width),
                ("prepare", width),
                ("decode", width),
                ("reset", 0),
            ]
            assert len(emitted) == width
            for pending in group:
                assert pending.error is None
                evidence = pending.response["x_onix_decode_evidence"]
                assert evidence["batch_width"] == width
                assert evidence["measurement_mode"] == "batch-admission"
                assert evidence["admission_eligible"]
                assert evidence["profile_id"] == profile.profile_id
            case_count += 1

        for width in range(1, MAXIMUM_BATCH_WIDTH + 1):
            if width in accepted_widths:
                continue
            rejected = pending_requests(width)
            accepted = pending_requests(profile.baseline_width)
            adapter, emitted = run_worker(profile, (rejected, accepted))
            assert adapter.calls == [
                ("prefill", profile.baseline_width),
                ("prepare", profile.baseline_width),
                ("decode", profile.baseline_width),
                ("reset", 0),
            ], f"width {width} reached a device effect: {adapter.calls}"
            for pending in rejected:
                assert pending.response is None
                assert isinstance(pending.error, BatchDecodeAdapterError)
                assert str(pending.error) == POLICY_DIAGNOSTIC
                assert str(pending.error.__cause__) == (
                    "batch admission width is outside the profile"
                )
            assert emitted[0] == (
                "batch-failed",
                {
                    "batch_width": width,
                    "error_type": "BatchMeasurementPolicyError",
                    "failure_stage": "admission",
                    "error_reason": "batch-width-outside-admission-profile",
                },
            )
            assert accepted[0].error is None
            assert accepted[0].response is not None
            case_count += 1

        # Disabled measurement must not restrict ordinary production widths.
        partial_width = profile.candidate_width - 1
        group = pending_requests(partial_width)
        adapter, _ = run_worker(profile, (group,), enabled=False)
        assert adapter.calls[0] == ("prefill", partial_width)
        for pending in group:
            assert pending.error is None
            evidence = pending.response["x_onix_decode_evidence"]
            assert evidence["measurement_mode"] == "physical-batched-service"
            assert not evidence["admission_eligible"]
            assert evidence["profile_id"] == "production-profile"
        case_count += 1

        for reset_error in (False, True):
            group = pending_requests(profile.baseline_width)
            adapter = BatchAdapter(prefill_error=True, reset_error=reset_error)
            adapter, emitted = run_worker(profile, (group,), adapter=adapter)
            assert adapter.calls == [("prefill", profile.baseline_width), ("reset", 0)]
            assert group[0].response is None
            assert emitted[0][1]["failure_stage"] == "execution"
            expected = (
                "batch reset failed after request error"
                if reset_error
                else "fixture prefill failed"
            )
            assert str(group[0].error) == expected
            case_count += 1

        # Expired work retains its existing rejection and cleanup behavior.
        expired = tuple(
            replace(pending, deadline=0.0)
            for pending in pending_requests(profile.baseline_width)
        )
        adapter, _ = run_worker(profile, (expired,))
        assert adapter.calls == [("reset", 0)]
        assert str(expired[0].error) == "request timeout exceeded before batch start"
        case_count += 1

    print(f"PASS batch admission effect ordering and recovery: {case_count} cases")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
