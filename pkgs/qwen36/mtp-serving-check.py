"""Device-free checks that execute the real speculative serving method."""

import ast
import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_SOURCE = Path(__file__).parent / "service"
if REPOSITORY_SOURCE.is_dir():
    sys.path.insert(0, str(REPOSITORY_SOURCE))
core = importlib.import_module("qwen36_service.mtp_serving_core")

SOURCE_ROOT = Path(core.__file__).parent

DRAFT_WIDTH = 3
PROMPT_RANK = 2
PROMPT_LENGTH = 5
GENERATION_LIMIT = 8
ANCHOR_TOKEN = 10
PREFILL_BUCKET = 128
EXPIRED_DEADLINE = 1.0
FIXED_CLOCK = 2.0
SLOW_ROUND_SECONDS = 10.0


def load_completion():
    tree = ast.parse((SOURCE_ROOT / "mtp_drafter_adapter.py").read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MtpDrafterAdapter"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_speculative_completion"
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    namespace = {
        name: getattr(core, name) for name in dir(core) if not name.startswith("__")
    }
    namespace.update({
        "MtpDrafterAdapterError": RuntimeError,
        "PROBE_BATCH_SIZE": 1,
        "QK_STREAM_COUNT": PROMPT_RANK,
        "TOKEN_DIMENSION": 1,
        "MINIMUM_TOKEN_ID": 0,
        "MTP_PREFILL_BUCKET": PREFILL_BUCKET,
        "SHARD_LOCAL_TOP1": "shard-local-top1",
        "torch": SimpleNamespace(int32=object()),
        "math": math,
        "ttnn": SimpleNamespace(deallocate=lambda value: None),
        "time": SimpleNamespace(monotonic=lambda: FIXED_CLOCK),
        "run_with_watchdog": lambda work, **kwargs: work(),
        "_FeedbackRecorder": lambda *args: SimpleNamespace(close=lambda: None),
    })
    exec(compile(tree, "mtp_drafter_adapter.py", "exec"), namespace)
    return namespace["run_speculative_completion"], namespace


class Prompt:
    ndim = PROMPT_RANK
    shape = (1, PROMPT_LENGTH)

    def to(self, dtype):
        return self


class ServingLoopTests(unittest.TestCase):
    def owner(self):
        effects = []
        owner = SimpleNamespace(
            profile=SimpleNamespace(draft_token_count=DRAFT_WIDTH),
            decode_adapter=SimpleNamespace(
                prefill_page_table=lambda **kwargs: effects.append("page-table"),
                decode_eager=lambda **kwargs: (
                    effects.append("plain-step") or ANCHOR_TOKEN
                ),
            ),
            _target_prefill_hidden=lambda *args: (object(), ANCHOR_TOKEN),
            _mtp_prefill=lambda *args, **kwargs: (ANCHOR_TOKEN, object(), None),
            _native_verifier=SimpleNamespace(
                trace_capture_timeout_seconds=1.0,
                capture_persistent=lambda *args, **kwargs: effects.append("capture"),
                close_persistent=lambda: effects.append("close"),
            ),
        )
        return owner, effects

    def test_one_token_has_valid_timing_without_speculation(self):
        run, namespace = load_completion()
        namespace["_FeedbackRecorder"] = lambda *args: SimpleNamespace(
            close=lambda: None
        )
        owner, effects = self.owner()
        result = run(owner, prompt_tokens=Prompt(), maximum_generation_tokens=1)
        self.assertEqual(result["token_ids"], (ANCHOR_TOKEN,))
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertNotIn("capture", effects)

    def test_invalid_limit_precedes_device_effects(self):
        run, _ = load_completion()
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                owner, effects = self.owner()
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "maximum_generation_tokens"
                ):
                    run(owner, prompt_tokens=Prompt(), maximum_generation_tokens=limit)
                self.assertEqual(effects, [])

    def test_expired_request_never_starts_device_work(self):
        run, _ = load_completion()
        owner, effects = self.owner()
        with self.assertRaisesRegex((RuntimeError, ValueError), "deadline"):
            run(
                owner,
                prompt_tokens=Prompt(),
                maximum_generation_tokens=GENERATION_LIMIT,
                request_deadline=EXPIRED_DEADLINE,
            )
        self.assertEqual(effects, [])


class EconomicsTests(unittest.TestCase):
    def advance(self, state, duration):
        return core.advance_round_economics(
            state,
            round_seconds=duration,
            emitted_tokens=1,
            plain_step_seconds=1.0,
        )

    def test_success_resets_loss_streak(self):
        loss = self.advance(core.RoundEconomics(), SLOW_ROUND_SECONDS)
        self.assertEqual(loss.consecutive_losses, 1)
        self.assertFalse(loss.disengaged)
        win = self.advance(loss, 1.0 / SLOW_ROUND_SECONDS)
        self.assertEqual(win, core.RoundEconomics())

    def test_repeated_losses_disengage_and_remain_sticky(self):
        state = core.RoundEconomics()
        for _ in range(core.ECONOMICS_DISENGAGE_STREAK):
            state = self.advance(state, SLOW_ROUND_SECONDS)
        self.assertTrue(state.disengaged)
        self.assertEqual(self.advance(state, 1.0 / SLOW_ROUND_SECONDS), state)

    def test_invalid_measurements_fail_closed(self):
        for duration in (0, -1, True, None, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaises(core.MtpServingPolicyError):
                    self.advance(core.RoundEconomics(), duration)
        with self.assertRaises(core.MtpServingPolicyError):
            core.advance_round_economics(
                core.RoundEconomics(),
                round_seconds=1.0,
                emitted_tokens=0,
                plain_step_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
