"""Exercise prefix trace capture, live-buffer replay, and failure cleanup."""

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))
SOURCE = Path(importlib.util.find_spec("qwen36_service").origin).parent
PREFIX_COUNT = 3
UPDATED_PREFIX_BASE = 10
FAILED_PREFIX_INDEX = 1
INPUT_BUFFER_COUNT = 4
CAPTURE_TIMEOUT_SECONDS = 10.0


class Device:
    def __init__(self):
        self.traces = {}
        self.active = None
        self.next_id = 0
        self.events = []

    def begin_trace_capture(self, mesh, **kwargs):
        if self.active is not None:
            raise RuntimeError("nested capture")
        trace_id = self.next_id
        self.next_id += 1
        self.active = trace_id
        self.traces[trace_id] = []
        self.events.append(("begin", trace_id))
        return trace_id

    def end_trace_capture(self, mesh, trace_id, **kwargs):
        if trace_id != self.active:
            raise RuntimeError("wrong capture id")
        self.active = None
        self.events.append(("end", trace_id))

    def copy(self, source, target):
        target.value = source.value
        if self.active is not None:
            self.traces[self.active].append((source, target))
        self.events.append(("copy", source.value))

    def execute_trace(self, mesh, trace_id, **kwargs):
        if self.active is not None:
            raise RuntimeError("replay during capture")
        for source, target in self.traces[trace_id]:
            target.value = source.value
        self.events.append(("replay", trace_id))

    def synchronize_device(self, mesh):
        self.events.append(("sync", None))

    def deallocate(self, tensor):
        self.events.append(("free", id(tensor)))

    def release_trace(self, mesh, trace_id):
        del self.traces[trace_id]
        self.events.append(("release", trace_id))


class PrefixTraceTests(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        spec = importlib.util.spec_from_file_location(
            "prefix_trace_under_test", SOURCE / "mtp_prefix_trace_adapter.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"ttnn": self.device}):
            spec.loader.exec_module(self.module)
        self.prefixes = tuple(
            SimpleNamespace(value=index) for index in range(PREFIX_COUNT)
        )
        self.target = SimpleNamespace(value=0)
        self.fail_capture = False

        def restore(prefix):
            if (
                self.fail_capture
                and self.device.active is not None
                and prefix is self.prefixes[FAILED_PREFIX_INDEX]
            ):
                raise RuntimeError("injected capture failure")
            self.device.copy(prefix, self.target)

        self.adapter = self.module.MtpPrefixTraces(
            mesh=object(),
            prefixes=self.prefixes,
            restore_prefix=restore,
            binding_ids=lambda: (*map(id, self.prefixes), id(self.target)),
        )

    def verifier(self):
        tree = ast.parse((SOURCE / "native_mtp_verifier_adapter.py").read_text())
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "NativeMtpVerifierAdapter"
        )
        methods = {"capture_persistent", "close_persistent", "rollback_to_prefix"}
        tree.body = [
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]

        class Input:
            def reshape(self, *args):
                return self

            def repeat(self, *args):
                return self

        self.clock = 0.0

        def tick():
            self.clock += 1.0
            return self.clock

        namespace = {
            "torch": SimpleNamespace(
                Tensor=object,
                tensor=lambda *args, **kwargs: Input(),
                arange=lambda *args, **kwargs: Input(),
                int32=object(),
            ),
            "ttnn": self.device,
            "math": math,
            "time": SimpleNamespace(monotonic=tick),
            "copy_host_to_device": lambda inputs, **kwargs: inputs,
            "MINIMUM_WIDTH": 1,
            "FOUR_DIMENSIONAL_TENSOR": INPUT_BUFFER_COUNT,
            "TRACE_COMMAND_QUEUE_ID": 0,
            "TraceCaptureProvenance": SimpleNamespace,
            "validate_trace_capture_provenance": lambda _: (),
            "NativeMtpVerifierError": RuntimeError,
            "MtpPrefixTraces": self.module.MtpPrefixTraces,
        }
        exec(compile(tree, "native_mtp_verifier_adapter.py", "exec"), namespace)

        def restore(value):
            self.target.value = value
            self.device.events.append(("restore-full", value))

        def forward(*args, **kwargs):
            self.target.value = UPDATED_PREFIX_BASE
            return object()

        verifier = SimpleNamespace(
            model=SimpleNamespace(
                mesh_device=object(),
                prepare_decode_inputs_host=lambda *args, **kwargs: tuple(
                    object() for _ in range(INPUT_BUFFER_COUNT)
                ),
            ),
            maximum_width=PREFIX_COUNT,
            trace_capture_timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
            _persistent_trace_id=None,
            _persistent_prefix_traces=None,
            _persistent_inputs=(),
            _persistent_logits=None,
            _persistent_prefix_snapshots=(),
            _persistent_width=0,
            _hygiene_capture_resident=False,
            _validate_round_inputs=lambda *args: None,
            _clone_gdn_state=lambda: self.target.value,
            _clone_gdn_prefix_states=lambda _: self.prefixes,
            _restore_gdn_state=restore,
            _restore_gdn_prefix_state=self.adapter._restore_prefix,
            _prefix_trace_binding_ids=lambda _: (
                *map(id, self.prefixes),
                id(self.target),
            ),
            _forward_device=forward,
            _deallocate_gdn_snapshots=lambda _: self.device.events.append((
                "free-snapshots",
                None,
            )),
            _deallocate_gdn_prefix_states=lambda _: self.device.events.append((
                "free-prefixes",
                None,
            )),
        )
        for name in methods:
            setattr(verifier, name, MethodType(namespace[name], verifier))
        return verifier, Input()

    def test_verifier_restores_capture_state_and_releases_dependencies_first(self):
        verifier, page_table = self.verifier()
        verifier.capture_persistent(
            tuple(range(PREFIX_COUNT)), start_position=0, page_table=page_table
        )
        self.assertEqual(self.target.value, 0)
        self.prefixes[0].value = UPDATED_PREFIX_BASE
        verifier.rollback_to_prefix(0)
        self.assertEqual(self.target.value, UPDATED_PREFIX_BASE)
        verifier.close_persistent()
        releases = [value for event, value in self.device.events if event == "release"]
        self.assertEqual(releases[-1], 0)
        self.assertEqual(self.device.traces, {})
        last_release = max(
            index
            for index, event in enumerate(self.device.events)
            if event[0] == "release"
        )
        prefix_free = next(
            index
            for index, event in enumerate(self.device.events)
            if event[0] == "free-prefixes"
        )
        self.assertLess(last_release, prefix_free)

    def test_verifier_capture_failure_restores_state_and_clears_ownership(self):
        verifier, page_table = self.verifier()
        self.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
            verifier.capture_persistent(
                tuple(range(PREFIX_COUNT)), start_position=0, page_table=page_table
            )
        self.assertEqual(self.target.value, 0)
        self.assertEqual(self.device.traces, {})
        self.assertIsNone(verifier._persistent_prefix_traces)
        self.assertIsNone(verifier._persistent_trace_id)
        self.assertFalse(verifier._hygiene_capture_resident)

    def test_replay_reads_updated_prefix_buffers_without_host_restore_dispatch(self):
        self.adapter.capture()
        for index, prefix in enumerate(self.prefixes):
            prefix.value = UPDATED_PREFIX_BASE + index
        copies = sum(event[0] == "copy" for event in self.device.events)
        for index, prefix in enumerate(self.prefixes):
            self.adapter.replay(index)
            self.assertEqual(self.target.value, prefix.value)
        self.assertEqual(
            sum(event[0] == "copy" for event in self.device.events), copies
        )
        self.adapter.close()
        self.adapter.close()
        self.assertEqual(self.device.traces, {})

    def test_invalid_indices_do_not_execute_any_device_effect(self):
        self.adapter.capture()
        for index in (-1, PREFIX_COUNT, True, None):
            with self.subTest(index=index):
                before = list(self.device.events)
                with self.assertRaises(self.module.MtpPrefixTraceError):
                    self.adapter.replay(index)
                self.assertEqual(self.device.events, before)
        self.adapter.close()

    def test_replay_before_capture_and_after_close_is_rejected(self):
        with self.assertRaises(self.module.MtpPrefixTraceError):
            self.adapter.replay(0)
        self.adapter.capture()
        self.adapter.close()
        before = list(self.device.events)
        with self.assertRaises(self.module.MtpPrefixTraceError):
            self.adapter.replay(0)
        self.assertEqual(self.device.events, before)

    def test_capture_failure_ends_capture_and_releases_every_created_trace(self):
        self.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
            self.adapter.capture()
        self.assertIsNone(self.device.active)
        self.assertEqual(self.device.traces, {})
        self.fail_capture = False
        self.adapter.capture()
        self.adapter.replay(0)
        self.assertEqual(self.target.value, self.prefixes[0].value)
        self.adapter.close()

    def test_changed_destination_binding_is_rejected_before_replay(self):
        self.adapter.capture()
        self.target = SimpleNamespace(value=0)
        before = list(self.device.events)
        with self.assertRaisesRegex(
            self.module.MtpPrefixTraceError, "bindings changed"
        ):
            self.adapter.replay(0)
        self.assertEqual(self.device.events, before)
        self.adapter.close()

    def test_duplicate_capture_is_rejected_without_effects(self):
        self.adapter.capture()
        before = list(self.device.events)
        with self.assertRaises(self.module.MtpPrefixTraceError):
            self.adapter.capture()
        self.assertEqual(self.device.events, before)
        self.adapter.close()

    def test_empty_prefixes_are_rejected_without_effects(self):
        with self.assertRaises(self.module.MtpPrefixTraceError):
            self.module.MtpPrefixTraces(
                mesh=object(),
                prefixes=(),
                restore_prefix=lambda _: None,
                binding_ids=lambda: (),
            )
        self.assertEqual(self.device.events, [])


if __name__ == "__main__":
    unittest.main()
