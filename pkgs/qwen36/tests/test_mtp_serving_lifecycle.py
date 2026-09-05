"""Host orchestration histories, not tensor or physical-serving evidence.

The fake target predicts the input token plus one. Feedback values identify
host snapshots. They do not establish MTP cache alignment or numerical parity.
"""

import threading
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from qwen36_service import mtp_drafter_adapter as adapter_module
from qwen36_service import mtp_serving_core as core
from qwen36_service import mtp_watchdog_adapter as watchdog_module
from qwen36_service.mtp_drafter_core import MINIMUM_DRAFT_TOKEN_COUNT
from qwen36_service.mtp_watchdog_adapter import MtpServingRoundTimeout

DRAFT_WIDTH = MINIMUM_DRAFT_TOKEN_COUNT
GENERATION_TOKENS = 12
ANCHOR_TOKEN = 10
WRONG_TOKEN = 9000
CLOCK_EPOCH_SECONDS = 100.0
FAST_EFFECT_SECONDS = 0.01
SLOW_EFFECT_SECONDS = 1.0
FUTURE_DEADLINE_SECONDS = 1000.0
PREFIX_CLONES_PER_ROUND = DRAFT_WIDTH + 1
PARTIAL_POOL_FAILURE = PREFIX_CLONES_PER_ROUND + 1
ANCHOR_AND_BONUS_TOKENS = 2
REPEATED_REQUESTS = 2
HOST_WATCHDOG_SECONDS = 0.05
HOST_TEST_WAIT_SECONDS = 5.0


@dataclass
class Tensor:
    value: object
    closed: bool = False


class Scenario:
    def __init__(self, acceptance=(DRAFT_WIDTH,)):
        self.acceptance = acceptance
        self.now = CLOCK_EPOCH_SECONDS
        self.effect_seconds = FAST_EFFECT_SECONDS
        self.events = []
        self.clones = []
        self.clone_failure = None
        self.failure_stage = None
        self.timeout_stage = None
        self.error = RuntimeError("fixture stage failed")
        self.timeout = MtpServingRoundTimeout("fixture worker remains active")
        self.worker_active = False
        self.expire_after_verify = False
        self.deadline = None
        self.request = 0
        self.rounds = 0
        self.rollbacks = []
        self.adapter = adapter_module.MtpDrafterAdapter.__new__(
            adapter_module.MtpDrafterAdapter
        )
        self.adapter.profile = SimpleNamespace(draft_token_count=DRAFT_WIDTH)
        self.adapter.trace_profile = SimpleNamespace(
            maximum_trace_capture_seconds=FUTURE_DEADLINE_SECONDS
        )
        self.adapter._serving_trace_adapter = None
        self.adapter._serving_quarantine = None
        self.adapter._recovery_trace_adapter = None
        self.adapter._mtp_zero_caches = ()
        self.adapter.decode_adapter = SimpleNamespace(
            prefill_page_table=self.page_table, decode_eager=self.plain
        )
        self.adapter._target_prefill_hidden = self.prefill
        self.adapter._mtp_prefill = self.seed
        self.adapter._native_verifier = SimpleNamespace(
            capture_persistent=self.capture,
            replay_round=self.verify,
            rollback_to_prefix=self.rollback,
            close_persistent=lambda: self.effect("verifier-close"),
            trace_capture_timeout_seconds=FUTURE_DEADLINE_SECONDS,
        )

    def monotonic(self):
        return self.now

    def effect(self, name):
        if self.worker_active:
            raise AssertionError(f"unsafe effect with live worker: {name}")
        self.events.append(name)
        self.now += self.effect_seconds

    def install(self):
        return patch.multiple(
            adapter_module,
            time=self,
            ttnn=SimpleNamespace(
                clone=self.clone, copy=self.copy, deallocate=self.deallocate
            ),
            TracedMtpDraftAdapter=lambda **_kwargs: SimpleNamespace(
                run=self.draft,
                close=lambda: self.effect("draft-close"),
                _trace_feedback=Tensor(None),
            ),
            run_with_watchdog=self.watchdog,
        )

    def watchdog(self, work, *, seconds, stage):
        assert seconds > 0
        if stage == self.timeout_stage:
            self.worker_active = True
            raise self.timeout
        if stage == self.failure_stage:
            raise self.error
        return work()

    def page_table(self, **_kwargs):
        self.effect("page-table")
        return object()

    def prefill(self, *_args):
        self.effect("prefill")
        self.request += 1
        self.rounds = 0
        self.current = ANCHOR_TOKEN
        self.position = 1
        self.expected_feedback = (self.request, "seed")
        return Tensor("hidden"), ANCHOR_TOKEN

    def seed(self, *_args, **_kwargs):
        self.effect("seed")
        return self.drafts()[0], Tensor(self.expected_feedback), FAST_EFFECT_SECONDS

    def wanted_prefix(self):
        return self.acceptance[min(self.rounds, len(self.acceptance) - 1)]

    def drafts(self):
        drafts = [self.current + offset + 1 for offset in range(DRAFT_WIDTH)]
        accepted = self.wanted_prefix()
        if accepted < DRAFT_WIDTH:
            drafts[accepted] = WRONG_TOKEN
        return tuple(drafts)

    def capture(self, *_args, **_kwargs):
        self.effect("capture")

    def clone(self, tensor):
        self.effect("clone")
        if self.clone_failure == len(self.clones):
            raise self.error
        assert not tensor.closed
        copied = Tensor(tensor.value)
        self.clones.append(copied)
        return copied

    def copy(self, source, destination):
        self.effect("copy")
        assert not source.closed and not destination.closed
        destination.value = source.value

    def deallocate(self, tensor):
        self.effect("release")
        assert not tensor.closed, "double release"
        tensor.closed = True

    def draft(
        self,
        *,
        initial_token,
        initial_feedback,
        start_position,
        step_count,
        rollback_context,
    ):
        self.effect("draft")
        assert start_position == self.position
        assert initial_feedback.value == self.expected_feedback, "stale feedback"
        drafts = self.drafts()
        first_round = self.rounds == 0
        assert initial_token == (drafts[0] if first_round else self.current)
        outputs = drafts[1:] if first_round else drafts
        assert len(outputs) == step_count
        self.prefix_feedback = [initial_feedback.value]
        feedback = self.adapter._serving_trace_adapter._trace_feedback
        for step in range(step_count):
            feedback.value = (self.request, self.rounds, step + 1)
            self.prefix_feedback.append(feedback.value)
            rollback_context.record_prefix_feedback(step + 1, feedback)
        return SimpleNamespace(token_ids=outputs)

    def verify(self, inputs, *, start_position):
        self.effect("verify")
        assert start_position == self.position
        assert inputs[0] == self.current
        continuations = tuple(token + 1 for token in inputs)
        accepted = DRAFT_WIDTH
        for index, (draft, target) in enumerate(
            zip(inputs[1:], continuations, strict=False)
        ):
            if draft != target:
                accepted = index
                break
        assert accepted == self.wanted_prefix()
        self.pending_prefix = accepted
        self.committed_position = start_position + accepted + 1
        self.position = start_position + len(inputs)
        self.current = continuations[accepted]
        self.expected_feedback = (
            self.prefix_feedback[accepted]
            if accepted < DRAFT_WIDTH
            else self.adapter._serving_trace_adapter._trace_feedback.value
        )
        self.rounds += 1
        if self.expire_after_verify:
            self.now = self.deadline + FAST_EFFECT_SECONDS
        return SimpleNamespace(token_ids=continuations)

    def rollback(self, accepted):
        self.effect("rollback")
        assert accepted == self.pending_prefix
        self.rollbacks.append(accepted)
        self.position = self.committed_position

    def plain(self, *, token, position):
        self.effect("plain")
        assert position == self.position
        assert token == self.current
        assert self.deadline is None or self.now <= self.deadline, (
            "decode after deadline"
        )
        self.position += 1
        self.current = token + 1
        return self.current

    def complete(self, maximum=GENERATION_TOKENS):
        with self.install():
            return self.adapter.run_speculative_completion(
                prompt_tokens=torch.tensor([[ANCHOR_TOKEN]], dtype=torch.int32),
                maximum_generation_tokens=maximum,
                request_deadline=self.deadline,
            )


class MtpServingLifecycleTests(unittest.TestCase):
    def test_acceptance_spectrum_commits_the_fake_greedy_stream(self):
        for prefix in range(DRAFT_WIDTH + 1):
            with self.subTest(prefix=prefix):
                scenario = Scenario((prefix,))
                result = scenario.complete()
                self.assertEqual(
                    result["token_ids"],
                    tuple(range(ANCHOR_TOKEN, ANCHOR_TOKEN + GENERATION_TOKENS)),
                )
                self.assertEqual(
                    result["rejected_rounds"],
                    0 if prefix == DRAFT_WIDTH else result["rounds"],
                )
                self.assertTrue(all(tensor.closed for tensor in scenario.clones))
                self.assertEqual(
                    scenario.rollbacks, [prefix] * result["rejected_rounds"]
                )

    def test_zero_acceptance_cannot_exhaust_the_planned_pool(self):
        scenario = Scenario((0,))
        result = scenario.complete()
        self.assertIsNone(result["disengaged_reason"])
        self.assertEqual(result["rounds"], GENERATION_TOKENS - DRAFT_WIDTH - 1)

    def test_current_feedback_survives_acceptance_then_zero_rejection(self):
        scenario = Scenario((DRAFT_WIDTH, 0, 0))
        result = scenario.complete()
        self.assertEqual(
            result["token_ids"],
            tuple(range(ANCHOR_TOKEN, ANCHOR_TOKEN + GENERATION_TOKENS)),
        )
        self.assertGreater(result["rejected_rounds"], 1)

    def test_repeated_requests_reset_host_state(self):
        scenario = Scenario()
        for _request in range(REPEATED_REQUESTS):
            result = scenario.complete()
            self.assertEqual(
                result["token_ids"],
                tuple(range(ANCHOR_TOKEN, ANCHOR_TOKEN + GENERATION_TOKENS)),
            )
            self.assertTrue(all(tensor.closed for tensor in scenario.clones))

    def test_partial_pool_allocation_and_capture_errors_release_completed_resources(
        self,
    ):
        for failure in ("allocation", "verify-capture"):
            with self.subTest(failure=failure):
                scenario = Scenario()
                if failure == "allocation":
                    scenario.clone_failure = PARTIAL_POOL_FAILURE
                else:
                    scenario.failure_stage = failure
                with self.assertRaises(RuntimeError) as raised:
                    scenario.complete()
                self.assertIs(raised.exception, scenario.error)
                self.assertTrue(scenario.clones)
                self.assertTrue(all(tensor.closed for tensor in scenario.clones))

    def test_watchdog_timeout_retains_resources_and_blocks_reuse(self):
        for stage in (
            "prefill",
            "head-seed",
            "verify-capture",
            "round-0-draft",
            "round-0-verify",
            "round-0-rollback",
            "plain-step",
        ):
            with self.subTest(stage=stage):
                scenario = Scenario((0,))
                scenario.timeout_stage = stage
                with self.assertRaises(MtpServingRoundTimeout) as raised:
                    scenario.complete()
                self.assertIs(raised.exception, scenario.timeout)
                self.assertTrue(scenario.worker_active)
                self.assertTrue(all(not tensor.closed for tensor in scenario.clones))
                events = tuple(scenario.events)
                with self.assertRaisesRegex(
                    adapter_module.MtpDrafterAdapterError, "process recovery"
                ):
                    scenario.complete()
                with (
                    scenario.install(),
                    self.assertRaisesRegex(
                        adapter_module.MtpDrafterAdapterError, "process recovery"
                    ),
                ):
                    scenario.adapter.close()
                self.assertEqual(tuple(scenario.events), events)

    def test_expired_deadline_and_invalid_limits_have_no_backend_effects(self):
        for maximum in (0, -1, True, FAST_EFFECT_SECONDS):
            with self.subTest(maximum=maximum):
                scenario = Scenario()
                with self.assertRaises((
                    adapter_module.MtpDrafterAdapterError,
                    core.MtpServingPolicyError,
                )):
                    scenario.complete(maximum)
                self.assertEqual(scenario.events, [])
        scenario = Scenario()
        scenario.deadline = CLOCK_EPOCH_SECONDS - FAST_EFFECT_SECONDS
        with self.assertRaisesRegex(adapter_module.MtpDrafterAdapterError, "deadline"):
            scenario.complete()
        self.assertEqual(scenario.events, [])

    def test_deadline_does_not_fall_through_to_plain_decode(self):
        scenario = Scenario()
        scenario.deadline = FUTURE_DEADLINE_SECONDS
        scenario.expire_after_verify = True
        result = scenario.complete()
        self.assertEqual(result["disengaged_reason"], "request-deadline")
        self.assertNotIn("plain", scenario.events)
        self.assertEqual(
            len(result["token_ids"]), DRAFT_WIDTH + ANCHOR_AND_BONUS_TOKENS
        )

    def test_consecutive_slow_rounds_disengage(self):
        scenario = Scenario()
        scenario.effect_seconds = SLOW_EFFECT_SECONDS
        result = scenario.complete()
        self.assertEqual(result["rounds"], core.ECONOMICS_DISENGAGE_STREAK)
        self.assertEqual(result["disengaged_reason"], "round-economics")
        self.assertEqual(len(result["token_ids"]), GENERATION_TOKENS)


class PendingWorkerTests(unittest.TestCase):
    def test_real_timeout_keeps_its_result_and_adapter_quarantined(self):
        scenario = Scenario()
        started = threading.Event()
        release = threading.Event()
        ordinary_watchdog = scenario.watchdog
        pending = None

        def bounded_watchdog(work, *, seconds, stage):
            if stage != "round-0-verify":
                return ordinary_watchdog(work, seconds=seconds, stage=stage)

            def blocked_work():
                scenario.worker_active = True
                started.set()
                if not release.wait(HOST_TEST_WAIT_SECONDS):
                    raise RuntimeError("fixture release deadline expired")
                scenario.worker_active = False
                return work()

            return watchdog_module.run_with_watchdog(
                blocked_work, seconds=HOST_WATCHDOG_SECONDS, stage=stage
            )

        scenario.watchdog = bounded_watchdog
        with scenario.install():
            try:
                with self.assertRaises(MtpServingRoundTimeout) as raised:
                    scenario.complete()
                self.assertTrue(started.wait(HOST_TEST_WAIT_SECONDS))
                pending = watchdog_module.pending_mtp_work(raised.exception)
                self.assertIsNotNone(pending)
                self.assertTrue(pending.worker.is_alive())
                self.assertEqual(pending.result, {})
                self.assertTrue(all(not tensor.closed for tensor in scenario.clones))
                release.set()
                pending.worker.join(HOST_TEST_WAIT_SECONDS)
                self.assertFalse(pending.worker.is_alive())
                self.assertIn("value", pending.result)
                events = tuple(scenario.events)
                with self.assertRaisesRegex(
                    adapter_module.MtpDrafterAdapterError, "process recovery"
                ):
                    scenario.complete()
                self.assertEqual(tuple(scenario.events), events)
            finally:
                release.set()
                if pending is not None:
                    pending.worker.join(HOST_TEST_WAIT_SECONDS)

    def test_interrupted_wait_preserves_the_signal_and_retains_pending_work(self):
        for signal in (
            KeyboardInterrupt("fixture interrupt"),
            SystemExit("fixture exit"),
        ):
            with self.subTest(signal=type(signal).__name__):
                worker = SimpleNamespace(
                    start=lambda: None,
                    join=lambda **_kwargs: self.raise_signal(signal),
                    is_alive=lambda: True,
                )
                with patch.object(
                    watchdog_module.threading, "Thread", return_value=worker
                ):
                    with self.assertRaises(type(signal)) as raised:
                        watchdog_module.run_with_watchdog(
                            lambda: None,
                            seconds=HOST_TEST_WAIT_SECONDS,
                            stage="interrupted-wait",
                        )
                self.assertIs(raised.exception, signal)
                pending = watchdog_module.pending_mtp_work(signal)
                self.assertIsNotNone(pending)
                self.assertIs(pending.worker, worker)
                scenario = Scenario()
                ordinary_watchdog = scenario.watchdog

                def interrupted(work, *, seconds, stage):
                    if stage == "round-0-verify":
                        scenario.worker_active = True
                        raise signal
                    return ordinary_watchdog(work, seconds=seconds, stage=stage)

                scenario.watchdog = interrupted
                with self.assertRaises(type(signal)) as propagated:
                    scenario.complete()
                self.assertIs(propagated.exception, signal)
                self.assertTrue(all(not tensor.closed for tensor in scenario.clones))
                events = tuple(scenario.events)
                with self.assertRaisesRegex(
                    adapter_module.MtpDrafterAdapterError, "process recovery"
                ):
                    scenario.complete()
                self.assertEqual(tuple(scenario.events), events)

    @staticmethod
    def raise_signal(signal):
        raise signal

    def test_interrupted_start_and_liveness_check_retain_pending_work(self):
        for phase in ("start", "liveness"):
            with self.subTest(phase=phase):
                signal = KeyboardInterrupt(f"fixture {phase} interrupted")
                worker = SimpleNamespace(
                    start=(lambda: self.raise_signal(signal))
                    if phase == "start"
                    else lambda: None,
                    join=lambda **_kwargs: None,
                    is_alive=(lambda: self.raise_signal(signal))
                    if phase == "liveness"
                    else lambda: True,
                )
                with patch.object(
                    watchdog_module.threading, "Thread", return_value=worker
                ):
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        watchdog_module.run_with_watchdog(
                            lambda: None, seconds=HOST_TEST_WAIT_SECONDS, stage=phase
                        )
                self.assertIs(raised.exception, signal)
                pending = watchdog_module.pending_mtp_work(signal)
                self.assertIsNotNone(pending)
                self.assertIs(pending.worker, worker)

    def test_completed_worker_errors_do_not_claim_pending_work(self):
        error = RuntimeError("completed worker error")
        with self.assertRaises(RuntimeError) as raised:
            watchdog_module.run_with_watchdog(
                lambda: self.raise_signal(error),
                seconds=HOST_TEST_WAIT_SECONDS,
                stage="completed-error",
            )
        self.assertIs(raised.exception, error)
        self.assertIsNone(watchdog_module.pending_mtp_work(error))


class FeedbackRecorderTests(unittest.TestCase):
    def test_snapshot_copy_and_closed_or_invalid_prefix_rejection(self):
        scenario = Scenario()
        with scenario.install():
            initial = Tensor("initial")
            recorder = adapter_module._FeedbackRecorder(initial, DRAFT_WIDTH)
            updated = Tensor("updated")
            recorder.start_round(updated)
            self.assertEqual(recorder.feedback_for_prefix(0).value, "updated")
            recorder.record_prefix_feedback(1, initial)
            self.assertEqual(recorder.feedback_for_prefix(1).value, "initial")
            for prefix in (-1, DRAFT_WIDTH + 1):
                with self.assertRaises(adapter_module.MtpDrafterAdapterError):
                    recorder.feedback_for_prefix(prefix)
            with self.assertRaises(adapter_module.MtpDrafterAdapterError):
                recorder.record_prefix_feedback(0, initial)
            recorder.close()
            recorder.close()
            with self.assertRaises(adapter_module.MtpDrafterAdapterError):
                recorder.start_round(initial)
            self.assertTrue(all(tensor.closed for tensor in scenario.clones))
            before = tuple(scenario.events)
            for width in (0, -1, True, "invalid"):
                with self.assertRaises(adapter_module.MtpDrafterAdapterError):
                    adapter_module._FeedbackRecorder(initial, width)
            self.assertEqual(tuple(scenario.events), before)


class ServingPolicyHistoryTests(unittest.TestCase):
    def test_economic_streak_advances_and_resets(self):
        first = core.assess_round_economics(
            round_seconds=SLOW_EFFECT_SECONDS, emitted_tokens=1, consecutive_losses=0
        )
        self.assertEqual(first.consecutive_losses, 1)
        self.assertFalse(first.disengaged)
        second = core.assess_round_economics(
            round_seconds=SLOW_EFFECT_SECONDS,
            emitted_tokens=1,
            consecutive_losses=first.consecutive_losses,
        )
        self.assertTrue(second.disengaged)
        reset = core.assess_round_economics(
            round_seconds=FAST_EFFECT_SECONDS,
            emitted_tokens=1,
            consecutive_losses=second.consecutive_losses,
        )
        self.assertEqual(reset.consecutive_losses, 0)
        self.assertFalse(reset.disengaged)
        with self.assertRaises(core.MtpServingPolicyError):
            core.assess_round_economics(
                round_seconds=SLOW_EFFECT_SECONDS,
                emitted_tokens=1,
                consecutive_losses=-1,
            )

    def test_pool_bound_matches_the_worst_accepted_prefix_history(self):
        for maximum in range(1, GENERATION_TOKENS + 1):
            generated = 1
            rounds = 0
            while (
                core.plan_round(
                    tokens_generated=generated,
                    maximum_generation_tokens=maximum,
                    draft_token_count=DRAFT_WIDTH,
                ).mode
                == core.ROUND_MODE_SPECULATIVE
            ):
                generated += 1
                rounds += 1
            self.assertEqual(
                core.feedback_pool_capacity(
                    maximum_generation_tokens=maximum, draft_token_count=DRAFT_WIDTH
                ),
                rounds,
            )
        with self.assertRaises(core.MtpServingPolicyError):
            core.feedback_pool_capacity(
                maximum_generation_tokens=0, draft_token_count=DRAFT_WIDTH
            )


if __name__ == "__main__":
    unittest.main()
