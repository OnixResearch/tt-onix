"""Device-free request timing checks with explicit fake backend effects.

These tests do not validate tensors, traces, recurrent state, or silicon.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from qwen36_service import mtp_drafter_adapter as adapter_module
from qwen36_service.mtp_drafter_core import MINIMUM_DRAFT_TOKEN_COUNT

CLOCK_EPOCH_SECONDS = 100.0
EFFECT_SECONDS = 1.0
ANCHOR_TOKEN = 11
DRAFT_TOKEN = 12
TAIL_TOKEN = 13
GENERATION_TOKENS = 2
REPEATED_REQUEST_COUNT = 2


class FakeClock:
    def __init__(self):
        self.now = CLOCK_EPOCH_SECONDS
        self.events = []

    def monotonic(self):
        return self.now

    def effect(self, name):
        self.events.append(name)
        self.now += EFFECT_SECONDS


class MtpServingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.decode_error = None
        self.feedback = object()
        self.prompt = torch.tensor([[ANCHOR_TOKEN]], dtype=torch.int32)
        self.adapter = adapter_module.MtpDrafterAdapter.__new__(
            adapter_module.MtpDrafterAdapter
        )
        self.adapter._serving_quarantine = None
        self.adapter.profile = SimpleNamespace(
            draft_token_count=MINIMUM_DRAFT_TOKEN_COUNT
        )
        self.adapter.decode_adapter = SimpleNamespace(
            prefill_page_table=self.page_table,
            decode_eager=self.decode,
        )
        self.adapter._target_prefill_hidden = self.prefill
        self.adapter._mtp_prefill = self.seed_head
        self.adapter._native_verifier = SimpleNamespace(
            capture_persistent=self.capture,
            close_persistent=self.close_verifier,
            trace_capture_timeout_seconds=EFFECT_SECONDS,
        )
        patches = patch.multiple(
            adapter_module,
            time=self.clock,
            ttnn=SimpleNamespace(deallocate=self.deallocate),
            _FeedbackRecorder=self.recorder,
        )
        patches.start()
        self.addCleanup(patches.stop)

    def page_table(self, **_kwargs):
        self.clock.effect("page-table")
        return object()

    def prefill(self, *_args):
        self.clock.effect("target-prefill")
        return object(), ANCHOR_TOKEN

    def seed_head(self, *_args, **_kwargs):
        self.clock.effect("head-seed")
        return DRAFT_TOKEN, self.feedback, EFFECT_SECONDS

    def capture(self, *_args, **_kwargs):
        self.clock.effect("verifier-capture")

    def close_verifier(self):
        self.clock.effect("verifier-close")

    def deallocate(self, tensor):
        self.clock.effect(
            "feedback-release" if tensor is self.feedback else "hidden-release"
        )

    def recorder(self, _feedback, _width):
        self.clock.effect("recorder-allocate")
        return SimpleNamespace(close=lambda: self.clock.effect("recorder-close"))

    def decode(self, **_kwargs):
        self.clock.effect("plain-step")
        if self.decode_error is not None:
            raise self.decode_error
        return TAIL_TOKEN

    def complete(self):
        return self.adapter.run_speculative_completion(
            prompt_tokens=self.prompt,
            maximum_generation_tokens=GENERATION_TOKENS,
        )

    def test_elapsed_includes_setup_and_cleanup_for_each_request(self):
        for _request in range(REPEATED_REQUEST_COUNT):
            with self.subTest(request=_request):
                self.clock.events.clear()
                result = self.complete()
                self.assertEqual(result["token_ids"], (ANCHOR_TOKEN, TAIL_TOKEN))
                self.assertEqual(result["rounds"], 0)
                self.assertEqual(result["prefill_seconds"], EFFECT_SECONDS)
                self.assertEqual(
                    result["elapsed_seconds"],
                    len(self.clock.events) * EFFECT_SECONDS,
                )
                self.assertGreater(result["elapsed_seconds"], result["prefill_seconds"])
                self.assertEqual(self.clock.events[0], "page-table")
                self.assertEqual(self.clock.events[-1], "feedback-release")

    def test_invalid_prompt_has_no_backend_effects(self):
        invalid_prompts = (
            torch.tensor([ANCHOR_TOKEN]),
            torch.empty((1, 0), dtype=torch.int32),
        )
        for prompt in invalid_prompts:
            with self.subTest(shape=prompt.shape):
                self.prompt = prompt
                with self.assertRaises(adapter_module.MtpDrafterAdapterError):
                    self.complete()
                self.assertEqual(self.clock.events, [])

    def test_decode_error_propagates_after_cleanup_without_a_receipt(self):
        self.decode_error = RuntimeError("fake decode failed")
        with self.assertRaises(RuntimeError) as raised:
            self.complete()
        self.assertIs(raised.exception, self.decode_error)
        self.assertIn("verifier-close", self.clock.events)
        self.assertEqual(self.clock.events[-1], "feedback-release")


if __name__ == "__main__":
    unittest.main()
