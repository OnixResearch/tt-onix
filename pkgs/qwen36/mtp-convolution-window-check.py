"""Check ordered convolution windows without a device or model import."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))
SOURCE = Path(importlib.util.find_spec("qwen36_service").origin).parent
TAP_COUNT = 4
MAXIMUM_TEST_WIDTH = TAP_COUNT + 1
ROW_CHANNELS = 1
INITIAL_ROWS = tuple(range(TAP_COUNT))
NEW_ROW_BASE = 10
TEST_DTYPE = "bf16"


class Tensor:
    def __init__(self, rows, *, dtype=TEST_DTYPE):
        self.rows = list(rows)
        self.shape = (1, len(rows), ROW_CHANNELS)
        self.dtype = dtype


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.copy_count = 0
        self.slice_count = 0
        source = ast.parse((SOURCE / "native_mtp_verifier_adapter.py").read_text())
        names = {"_verification_convolution_windows", "_commit_convolution_window"}
        source.body = [
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.namespace = {
            "Any": object,
            "MINIMUM_WIDTH": 1,
            "NativeMtpVerifierError": ValueError,
            "ttnn": SimpleNamespace(
                slice=self.slice, copy=self.copy, L1_MEMORY_CONFIG="l1"
            ),
        }
        exec(compile(source, "native_mtp_verifier_adapter.py", "exec"), self.namespace)
        self.windows = self.namespace["_verification_convolution_windows"]
        self.commit = self.namespace["_commit_convolution_window"]

    def slice(self, tensor, start, end, **kwargs):
        self.slice_count += 1
        _, first, _ = start
        _, last, _ = end
        return Tensor(tensor.rows[first:last], dtype=tensor.dtype)

    def copy(self, source, target):
        self.copy_count += 1
        target.rows[:] = source.rows

    def layer(self):
        return SimpleNamespace(
            K=TAP_COUNT,
            qkv_dim_tp=ROW_CHANNELS,
            conv_states=[Tensor((value,)) for value in INITIAL_ROWS],
        )

    def test_every_prefix_matches_sequential_shift_without_intermediate_writes(self):
        for width in range(1, MAXIMUM_TEST_WIDTH + 1):
            with self.subTest(width=width):
                layer = self.layer()
                bindings = tuple(map(id, layer.conv_states))
                rows = tuple(range(NEW_ROW_BASE, NEW_ROW_BASE + width))
                expected = list(INITIAL_ROWS)
                copies_before = self.copy_count
                for row, window in zip(
                    rows, self.windows(layer, Tensor(rows), width), strict=True
                ):
                    expected = expected[1:] + [row]
                    self.assertEqual([state.rows[0] for state in window], expected)
                    self.assertEqual(
                        [state.rows[0] for state in layer.conv_states],
                        list(INITIAL_ROWS),
                    )
                    self.assertEqual(self.copy_count, copies_before)
                self.commit(layer, window)
                self.assertEqual(
                    [state.rows[0] for state in layer.conv_states], expected
                )
                self.assertEqual(tuple(map(id, layer.conv_states)), bindings)
                self.assertEqual(self.copy_count - copies_before, TAP_COUNT)

    def test_second_round_uses_committed_rows_not_stale_window(self):
        layer = self.layer()
        expected = list(INITIAL_ROWS)
        for row in range(NEW_ROW_BASE, NEW_ROW_BASE + MAXIMUM_TEST_WIDTH):
            window = tuple(self.windows(layer, Tensor((row,)), 1))[0]
            expected = expected[1:] + [row]
            self.commit(layer, window)
            self.assertEqual([state.rows[0] for state in layer.conv_states], expected)

    def test_invalid_window_inputs_fail_before_slicing_or_copying(self):
        for defect in ("empty", "count", "dtype", "width"):
            with self.subTest(defect=defect):
                layer = self.layer()
                qkv = Tensor((NEW_ROW_BASE,))
                width = 1
                if defect == "empty":
                    layer.conv_states = []
                elif defect == "count":
                    layer.conv_states.pop()
                elif defect == "dtype":
                    qkv.dtype = "fp32"
                else:
                    width = 0
                before = (self.slice_count, self.copy_count)
                with self.assertRaises(ValueError):
                    tuple(self.windows(layer, qkv, width))
                self.assertEqual((self.slice_count, self.copy_count), before)

    def test_late_dtype_mismatch_rejects_before_any_commit(self):
        layer = self.layer()
        window = [Tensor((NEW_ROW_BASE,)) for _ in range(TAP_COUNT)]
        window[-1].dtype = "fp32"
        with self.assertRaises(ValueError):
            self.commit(layer, tuple(window))
        self.assertEqual(self.copy_count, 0)
        self.assertEqual(
            [state.rows[0] for state in layer.conv_states], list(INITIAL_ROWS)
        )

    def test_short_commit_rejects_without_partial_writes(self):
        layer = self.layer()
        with self.assertRaises(ValueError):
            self.commit(layer, (Tensor((NEW_ROW_BASE,)),))
        self.assertEqual(self.copy_count, 0)
        self.assertEqual(
            [state.rows[0] for state in layer.conv_states], list(INITIAL_ROWS)
        )


if __name__ == "__main__":
    unittest.main()
