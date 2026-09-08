"""Device-free regression checks for greedy and sampled startup trace tails."""

from collections import deque
from threading import Condition
from types import SimpleNamespace
from unittest.mock import patch

import torch
from qwen36_service.batch_decode_adapter import (
    SAMPLED_CANDIDATE_COUNT,
    BatchDecodeAdapterError,
    BatchedTracedDecodeAdapter,
)
from qwen36_service.batch_service import BatchGenerationCoordinator
from qwen36_service.sampled_mode_router_core import ROUTE_DETERMINISTIC, ROUTE_SAMPLED

DEVICE_COUNT = 2
MODEL_CAPACITY = 4
VOCABULARY_PER_SHARD = 128
GENERATION_TOKENS = 2
TOKEN_ID = 1
FIXTURE_SECONDS = 0.01


class TailAdapter:
    """Use the real tail selector and readers with host tensor fixtures."""

    _reduce_for_mode = BatchedTracedDecodeAdapter._reduce_for_mode
    _read_global_winners = BatchedTracedDecodeAdapter._read_global_winners
    _read_global_candidates = BatchedTracedDecodeAdapter._read_global_candidates

    def __init__(self, mode, *, wrong_width=False, read_error=False):
        self.fusion_mode = mode
        self.device_count = DEVICE_COUNT
        self.vocabulary_per_shard = VOCABULARY_PER_SHARD
        self.vocabulary_size = DEVICE_COUNT * VOCABULARY_PER_SHARD
        self._mesh_reader = object()
        self.readers = []
        self.reset_count = 0
        self.wrong_width = wrong_width
        self.read_error = read_error

    def _reduce_shards(self, logits, width):
        return self._tail(width, 1)

    def _reduce_shards_topk(self, logits, width):
        return self._tail(width, SAMPLED_CANDIDATE_COUNT)

    @staticmethod
    def _tail(width, candidate_count):
        shape = (DEVICE_COUNT, width, candidate_count)
        return torch.zeros(shape, dtype=torch.int32), torch.ones(shape)

    def prefill_users(self, prompts, **kwargs):
        return SimpleNamespace(
            first_token_ids=(TOKEN_ID,) * len(prompts), elapsed_seconds=FIXTURE_SECONDS
        )

    def prepare_width(self, *, token_ids, positions):
        self.indices, self.values = self._reduce_for_mode(None, len(token_ids))
        return SimpleNamespace(
            capture_seconds=FIXTURE_SECONDS, reused=False, resident_trace_count=1
        )

    def decode_traced(self, *, token_ids, positions):
        self.readers.append(("greedy", len(token_ids)))
        return self._read_global_winners(self.indices, self.values, len(token_ids))

    def decode_traced_sampled(self, *, token_ids, positions):
        self.readers.append(("sampled", len(token_ids)))
        if self.read_error:
            raise OSError("fixture readback failed")
        rows = self._read_global_candidates(self.indices, self.values, len(token_ids))
        return rows[:-1] if self.wrong_width else rows

    def reset_all(self):
        self.reset_count += 1


def coordinator_for(adapter):
    coordinator = object.__new__(BatchGenerationCoordinator)
    coordinator.adapter = adapter
    coordinator._condition = Condition()
    coordinator._closing = False
    coordinator._accepting_requests = False
    coordinator._queue = deque()
    coordinator.emitter = lambda *args, **kwargs: None
    return coordinator


def check():
    widths = (1, MODEL_CAPACITY)
    plan = SimpleNamespace(widths=widths, generation_tokens=GENERATION_TOKENS)
    prompt = torch.tensor([[TOKEN_ID]], dtype=torch.int32)
    with patch("ttnn.to_torch", side_effect=lambda tensor, **kwargs: tensor):
        # The old path reproduces the physical error without opening a device.
        mismatched = TailAdapter(ROUTE_SAMPLED)
        mismatched.prepare_width(token_ids=(TOKEN_ID,), positions=(1,))
        try:
            mismatched.decode_traced(token_ids=(TOKEN_ID,), positions=(1,))
        except RuntimeError as error:
            element_count = DEVICE_COUNT * SAMPLED_CANDIDATE_COUNT
            assert f"shape '[{DEVICE_COUNT}, 1]'" in str(error)
            assert f"input of size {element_count}" in str(error)
        else:
            raise AssertionError("greedy reader accepted a sampled trace tail")

        for mode, reader in (
            (ROUTE_DETERMINISTIC, "greedy"),
            (ROUTE_SAMPLED, "sampled"),
        ):
            adapter = TailAdapter(mode)
            coordinator = coordinator_for(adapter)
            observations = coordinator.warm_up(plan, prompt)
            assert tuple(item.width for item in observations) == widths
            assert adapter.readers == [(reader, width) for width in widths]
            assert adapter.reset_count == len(widths)
            assert coordinator._accepting_requests

        for failure in ({"wrong_width": True}, {"read_error": True}):
            adapter = TailAdapter(ROUTE_SAMPLED, **failure)
            coordinator = coordinator_for(adapter)
            try:
                coordinator.warm_up(plan, prompt)
            except BatchDecodeAdapterError as error:
                assert "startup warm-up failed" in str(error)
                assert adapter.reset_count == 1
                assert not coordinator._accepting_requests
            else:
                raise AssertionError("failed sampled warm-up admitted requests")
    print("PASS startup tail selection and rejection")


if __name__ == "__main__":
    check()
