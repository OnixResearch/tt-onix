"""Execute serving and snapshot code against a stateful device-free adapter."""

import ast
import copy
import importlib
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_SOURCE = Path(__file__).parent / "service"
if REPOSITORY_SOURCE.is_dir():
    sys.path.insert(0, str(REPOSITORY_SOURCE))
core = importlib.import_module("qwen36_service.mtp_serving_core")

SOURCE_ROOT = Path(core.__file__).parent
DRAFT_WIDTH = 3
PROMPT_LENGTH = 5
GENERATION_LIMIT = 24
ANCHOR_TOKEN = 10
WRONG_TOKEN = 1000
CLOCK_STEP = 0.01
FAST_PLAIN_SECONDS = CLOCK_STEP / GENERATION_LIMIT
SLOW_PLAIN_SECONDS = 1.0
FUTURE_DEADLINE_SECONDS = 100.0


class Tensor:
    def __init__(self, value):
        self.value = copy.deepcopy(value)
        self.closed = False


def tensor_copy(source, target):
    if source.closed or target.closed:
        raise RuntimeError("closed tensor access")
    target.value = copy.deepcopy(source.value)


def deallocate(tensor):
    if tensor.closed:
        raise RuntimeError("double tensor release")
    tensor.closed = True


def load_source(name, namespace):
    tree = ast.parse((SOURCE_ROOT / name).read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(tree, name, "exec"), namespace)


class RoundFixture:
    """Model device rows with rewrite-before-read cache semantics.

    Committed rows are positions below ``next``. Verification writes rows
    speculatively; a rejected round's ``rollback_to_prefix`` un-commits the
    rejected tail, and any later write at the same position must match the
    committed value.
    """

    def __init__(self, accepted, *, failed_capture=False, failed_head=False):
        self.accepted = accepted
        self.failed_capture = failed_capture
        self.failed_head = failed_head
        self.clock = 0.0
        self.events = []
        self.target = Tensor({"next": PROMPT_LENGTH, "rows": {}})
        self.head = Tensor({
            "next": PROMPT_LENGTH,
            "rows": {PROMPT_LENGTH - 1: ANCHOR_TOKEN},
        })
        self.feedback = Tensor(ANCHOR_TOKEN)
        self.trace = None
        self.verify_resident = False
        self.verify_start = None
        self.namespace = {
            name: getattr(core, name) for name in dir(core) if not name.startswith("__")
        }
        self.namespace.update({
            "ExitStack": ExitStack,
            "time": SimpleNamespace(monotonic=self.tick),
            "ttnn": SimpleNamespace(
                copy=tensor_copy,
                clone=lambda tensor: Tensor(tensor.value),
                deallocate=deallocate,
                synchronize_device=lambda mesh: None,
            ),
            "tensor_identity": lambda tensor: id(tensor),
            "run_device_stage": lambda work, **kwargs: work(),
            "TracedMtpDraftAdapter": self.make_trace,
        })
        load_source("mtp_state_snapshot_adapter.py", self.namespace)
        self.namespace["target_state_tensors"] = lambda model: (self.target,)
        load_source("mtp_serving_adapter.py", self.namespace)
        verifier = SimpleNamespace(
            capture_persistent=self.capture,
            replay_round=self.verify,
            rollback_to_prefix=self.rollback,
            close_persistent=self.close_verify,
            trace_capture_timeout_seconds=SLOW_PLAIN_SECONDS,
        )
        self.owner = SimpleNamespace(
            model=object(),
            mesh=object(),
            profile=SimpleNamespace(draft_token_count=DRAFT_WIDTH),
            trace_profile=SimpleNamespace(
                maximum_trace_capture_seconds=SLOW_PLAIN_SECONDS
            ),
            _native_verifier=verifier,
            _mtp_cache_tensors=lambda: (self.head,),
            decode_adapter=SimpleNamespace(
                decode_eager=self.plain,
                decode_traced=self.plain,
                ensure_trace=self.capture_plain,
            ),
        )

    def tick(self):
        self.clock += CLOCK_STEP
        return self.clock

    def sequential_token(self, position):
        """The committed stream value at an absolute position."""
        return ANCHOR_TOKEN + (position - PROMPT_LENGTH) + 1

    def commit_row(self, position, token):
        state = self.target.value
        if position == state["next"]:
            state["rows"][position] = token
            state["next"] = position + 1
        elif position < state["next"]:
            if state["rows"].get(position) != token:
                raise RuntimeError("committed row rewrite changed the value")
        else:
            raise RuntimeError("committed write skipped a position")

    def plain(self, *, token, position):
        self.events.append("plain")
        emitted = self.sequential_token(position)
        self.commit_row(position, emitted)
        return emitted

    def capture(self, tokens, *, start_position, page_table):
        self.events.append("verify-capture")
        self.verify_resident = True
        if self.failed_capture:
            raise RuntimeError("capture failed")
        return CLOCK_STEP

    def verify(self, tokens, *, start_position):
        self.events.append("verify")
        self.verify_start = start_position
        # The verify replay writes the whole window. Values for the accepted
        # prefix are exactly the sequential rows; the rejected tail is
        # un-committed by rollback_to_prefix.
        state = self.target.value
        for offset in range(len(tokens)):
            position = start_position + offset
            if position == state["next"]:
                state["rows"][position] = self.sequential_token(position)
                state["next"] = position + 1
        token_ids = tuple(
            self.sequential_token(start_position + offset)
            for offset in range(len(tokens))
        )
        return SimpleNamespace(
            token_ids=token_ids,
            minimum_top_two_margin=1.0,
        )

    def rollback(self, accepted):
        state = self.target.value
        state["next"] = self.verify_start + accepted + 1

    def close_verify(self):
        self.verify_resident = False
        self.events.append("verify-close")

    def capture_plain(self, **kwargs):
        if self.verify_resident or (self.trace is not None and not self.trace.closed):
            raise RuntimeError("plain capture with speculative trace resident")
        self.events.append("plain-capture")
        return SimpleNamespace(eligible=True)

    def make_trace(self, **kwargs):
        fixture = self

        class DraftTrace:
            closed = False
            _trace_feedback = Tensor(ANCHOR_TOKEN)

            def consume(self, token, position, feedback):
                state = fixture.head.value
                if feedback.value != state["rows"].get(position - 1):
                    raise RuntimeError("draft feedback drift")
                if position > state["next"]:
                    raise RuntimeError("draft position skipped ahead")
                # A rebuild may rewrite speculative rows below the pointer.
                state["next"] = max(state["next"], position + 1)
                state["rows"][position] = token
                self._trace_feedback.value = token
                return token + 1

            def run(
                self, *, initial_token, initial_feedback, start_position, step_count
            ):
                fixture.events.append("draft")
                tokens = []
                token = initial_token
                feedback = initial_feedback
                for offset in range(step_count):
                    token = self.consume(token, start_position + offset, feedback)
                    if fixture.accepted is not None and offset + 1 == fixture.accepted:
                        token = WRONG_TOKEN
                    tokens.append(token)
                    feedback = self._trace_feedback
                return SimpleNamespace(
                    token_ids=tuple(tokens),
                    trace_binding_valid=True,
                    capture_idempotence_valid=True,
                    mtp_kv_prefix_valid=True,
                    trace_capture_seconds=0.0,
                )

            def replay_token_inputs(
                self, *, token_inputs, initial_feedback, start_position
            ):
                if fixture.failed_head:
                    raise RuntimeError("head commit failed")
                fixture.events.append("head-commit")
                token = self.consume(token_inputs[0], start_position, initial_feedback)
                if fixture.accepted == 0:
                    token = WRONG_TOKEN
                return SimpleNamespace(
                    token_ids=(token,), feedback=self._trace_feedback
                )

            def close(self):
                self.closed = True
                fixture.events.append("draft-close")

        self.trace = DraftTrace()
        return self.trace

    def run(self, *, plain_seconds=SLOW_PLAIN_SECONDS, deadline=None):
        return self.namespace["run_rounds"](
            self.owner,
            anchor=ANCHOR_TOKEN,
            first_draft=WRONG_TOKEN if self.accepted == 0 else ANCHOR_TOKEN + 1,
            feedback=self.feedback,
            page_table=object(),
            prompt_length=PROMPT_LENGTH,
            maximum_generation_tokens=GENERATION_LIMIT,
            request_deadline=deadline,
            plain_step_seconds=plain_seconds,
        )


class RoundTests(unittest.TestCase):
    EXPECTED = tuple(range(ANCHOR_TOKEN, ANCHOR_TOKEN + GENERATION_LIMIT))

    def sequential_token(self, position):
        """The committed stream value at an absolute position."""
        return ANCHOR_TOKEN + (position - PROMPT_LENGTH) + 1

    def assert_exact_stream(self, fixture, result):
        self.assertEqual(result["token_ids"], self.EXPECTED)
        committed = fixture.target.value["next"]
        self.assertEqual(committed, PROMPT_LENGTH + GENERATION_LIMIT - 1)
        for position in range(PROMPT_LENGTH, committed):
            self.assertEqual(
                fixture.target.value["rows"].get(position),
                self.sequential_token(position),
            )

    def test_every_acceptance_length_preserves_tokens_and_target_state(self):
        for accepted in (None, *range(DRAFT_WIDTH)):
            with self.subTest(accepted=accepted):
                fixture = RoundFixture(accepted)
                result = fixture.run()
                self.assert_exact_stream(fixture, result)
                self.assertGreater(result["rounds"], 0)
                self.assertFalse(fixture.verify_resident)
                self.assertTrue(fixture.trace.closed)
                self.assertEqual(fixture.events.count("verify-capture"), 1)

    def test_slow_rounds_switch_once_and_preserve_exact_output(self):
        fixture = RoundFixture(0)
        result = fixture.run(plain_seconds=FAST_PLAIN_SECONDS)
        self.assertEqual(result["rounds"], core.ECONOMICS_DISENGAGE_STREAK)
        self.assertEqual(result["disengaged_reason"], "round-economics")
        self.assertEqual(fixture.events.count("plain-capture"), 1)
        self.assertEqual(result["token_ids"], self.EXPECTED)

    def test_capture_failure_releases_trace(self):
        fixture = RoundFixture(None, failed_capture=True)
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            fixture.run()
        self.assertFalse(fixture.verify_resident)
        self.assertNotIn("plain", fixture.events)

    def test_head_failure_never_resumes_plain_decode(self):
        fixture = RoundFixture(None, failed_head=True)
        with self.assertRaisesRegex(RuntimeError, "head commit failed"):
            fixture.run()
        self.assertFalse(fixture.verify_resident)
        self.assertTrue(fixture.trace.closed)
        self.assertNotIn("plain-capture", fixture.events)

    def test_deadline_does_not_trigger_plain_fallback(self):
        fixture = RoundFixture(None)
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            fixture.run(deadline=0)
        self.assertNotIn("plain", fixture.events)
        self.assertNotIn("verify-capture", fixture.events)

    def test_deadline_after_final_step_rejects_the_response(self):
        fixture = RoundFixture(None)
        original = fixture.owner.decode_adapter.decode_traced

        def expires(**kwargs):
            token = original(**kwargs)
            if fixture.target.value["next"] == PROMPT_LENGTH + GENERATION_LIMIT - 1:
                fixture.clock = FUTURE_DEADLINE_SECONDS
            return token

        fixture.owner.decode_adapter.decode_traced = expires
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            fixture.run(deadline=FUTURE_DEADLINE_SECONDS)
        self.assertFalse(fixture.verify_resident)
        self.assertTrue(fixture.trace.closed)

    def test_changed_state_binding_is_rejected(self):
        fixture = RoundFixture(None)
        state = [Tensor(ANCHOR_TOKEN)]
        snapshot = fixture.namespace["MtpStateSnapshot"](lambda: tuple(state))
        state[0] = Tensor(ANCHOR_TOKEN)
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            snapshot.restore()
        snapshot.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            snapshot.refresh()


if __name__ == "__main__":
    unittest.main()
