"""Host checks for paged target-KV recovery, including preserved neighboring rows."""

import ast
import importlib
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))
core = importlib.import_module("qwen36_service.mtp_kv_tail_core")
SOURCE = Path(core.__file__).parent
PAGE_COUNT = 4
PAGE_SIZE = 64
HEAD_COUNT = 2
CHANNELS = 32
TILE_SIZE = 32
DRAFT_COUNT = 3
PAGES = (2, 0, 3, 1)
START = PAGE_SIZE - 1
SENTINEL_MODULUS = 97
WRITE_VALUE = 200


def load(name, namespace):
    tree = ast.parse((SOURCE / name).read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(tree, name, "exec"), namespace)


class Tensor:
    def __init__(self, value, dtype="bf16"):
        self.value = np.array(value, dtype=np.float32, copy=True)
        self.dtype = dtype
        self.closed = False
        self.address = id(self)

    @property
    def shape(self):
        return self.value.shape


class FakeDevice:
    def __init__(self):
        self.writes = 0
        self.fail_update = False
        self.allocated = []
        self.bfloat16 = "bf16"
        self.int32 = "int32"
        self.TILE_SIZE = TILE_SIZE
        self.ROW_MAJOR_LAYOUT = "row-major"
        self.experimental = SimpleNamespace(paged_update_cache=self.update)

    def allocate(self, value):
        tensor = Tensor(value)
        self.allocated.append(tensor)
        return tensor

    def clone(self, tensor):
        return self.allocate(tensor.value)

    def copy(self, source, destination):
        destination.value[:] = source.value

    def deallocate(self, tensor):
        if tensor.closed:
            raise RuntimeError("double release")
        tensor.closed = True

    def from_torch(self, value, **kwargs):
        return self.allocate(value)

    def ReplicateTensorToMesh(self, mesh):
        return mesh

    def synchronize_device(self, mesh):
        pass

    def slice(self, value, start, stop):
        indices = tuple(
            slice(begin, end) for begin, end in zip(start, stop, strict=True)
        )
        return self.allocate(value.value[indices])

    def permute(self, value, order):
        return self.allocate(np.transpose(value.value, order))

    def pad(self, value, shape, start, fill):
        result = np.full(shape, fill)
        indices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(start, value.shape, strict=True)
        )
        result[indices] = value.value
        return self.allocate(result)

    def to_memory_config(self, value, config):
        return self.clone(value)

    def update(self, live, value, *, update_idxs_tensor, page_table):
        if self.fail_update:
            raise RuntimeError("injected update error")
        position = int(update_idxs_tensor.value[0])
        page = int(page_table.value[0, position // PAGE_SIZE])
        offset = position % PAGE_SIZE
        live.value[page, :, offset, :] = value.value[0, 0, :HEAD_COUNT, :]
        self.writes += 1


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        shape = (PAGE_COUNT, HEAD_COUNT, PAGE_SIZE, CHANNELS)
        values = np.arange(np.prod(shape)).reshape(shape) % SENTINEL_MODULUS
        values = values.astype(np.float32)
        self.caches = (Tensor(values), Tensor(-values))
        self.entry = tuple(cache.value.copy() for cache in self.caches)
        attention = SimpleNamespace(
            paged_k=self.caches[0],
            paged_v=self.caches[1],
            _kv_shard_cfg=lambda _: "height-sharded",
        )
        self.model = SimpleNamespace(
            layers=(SimpleNamespace(is_full_attention=True, attention=attention),)
        )
        self.namespace = {
            "ttnn": self.device,
            "torch": SimpleNamespace(
                Tensor=np.ndarray, tensor=np.array, int32=np.int32
            ),
            "ExitStack": ExitStack,
            "plan_kv_tail": core.plan_kv_tail,
            "tensor_identity": lambda value: (
                value.address if not value.closed else None
            ),
        }
        load("mtp_state_snapshot_adapter.py", self.namespace)
        self.snapshot = self.namespace["MtpStateSnapshot"](lambda: self.caches)
        load("mtp_kv_tail_adapter.py", self.namespace)
        self.recovery = self.namespace["MtpTargetKvRecovery"](
            model=self.model,
            snapshot=self.snapshot,
            mesh=object(),
            page_table=np.array([PAGES]),
            position=START,
            draft_count=DRAFT_COUNT,
        )

    def tearDown(self):
        self.recovery.close()
        self.snapshot.close()
        self.assertTrue(all(value.closed for value in self.device.allocated))
        self.assertTrue(all(not cache.closed for cache in self.caches))

    def speculate(self, position):
        for cache in self.caches:
            for index in range(position, position + DRAFT_COUNT + 1):
                cache.value[PAGES[index // PAGE_SIZE], :, index % PAGE_SIZE, :] = (
                    WRITE_VALUE + index % HEAD_COUNT
                )

    def assert_round(self, position, accepted):
        self.speculate(position)
        expected = [cache.value.copy() for cache in self.caches]
        for index in range(position + accepted + 1, position + DRAFT_COUNT + 1):
            page, row = PAGES[index // PAGE_SIZE], index % PAGE_SIZE
            for result, entry in zip(expected, self.entry, strict=True):
                result[page, :, row, :] = entry[page, :, row, :]
        self.recovery.finish_round(position=position, accepted=accepted)
        for cache, result in zip(self.caches, expected, strict=True):
            np.testing.assert_array_equal(
                cache.value.view(np.uint32), result.view(np.uint32)
            )
        return position + accepted + 1

    def test_every_acceptance_prefix_and_cross_page_recovery(self):
        position = START
        for accepted in range(DRAFT_COUNT + 1):
            position = self.assert_round(position, accepted)

    def test_repeated_rounds_preserve_committed_rows_and_entry_tail(self):
        position = START
        for accepted in (0, DRAFT_COUNT, 1, 0, HEAD_COUNT):
            position = self.assert_round(position, accepted)

    def test_bitwise_oracle_rejects_signed_zero_normalization(self):
        original_permute = self.device.permute

        def normalize_zero(value, order):
            result = original_permute(value, order)
            result.value[result.value == 0] = 0.0
            return result

        # This mutation models the observed loss, not its unidentified cause.
        self.device.permute = normalize_zero
        with self.assertRaises(AssertionError):
            self.assert_round(START, 0)

    def test_full_acceptance_has_no_restore_writes(self):
        self.assert_round(START, DRAFT_COUNT)
        self.assertEqual(self.device.writes, 0)

    def test_late_dtype_change_rejects_before_any_cache_write(self):
        self.caches[1].dtype = "bf8"
        with self.assertRaisesRegex(ValueError, "bf16"):
            self.recovery.finish_round(position=START, accepted=0)
        self.assertEqual(self.device.writes, 0)

    def test_changed_allocation_rejects_before_any_cache_write(self):
        self.caches[1].address += 1
        with self.assertRaises(RuntimeError):
            self.recovery.finish_round(position=START, accepted=0)
        self.assertEqual(self.device.writes, 0)

    def test_backward_or_skipped_boundary_rejects_before_writes(self):
        for position in (START - 1, START + 1):
            with self.assertRaisesRegex(ValueError, "boundary"):
                self.recovery.finish_round(position=position, accepted=0)
        self.assertEqual(self.device.writes, 0)

    def test_device_error_poison_prevents_a_second_attempt(self):
        self.device.fail_update = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.recovery.finish_round(position=START, accepted=0)
        self.device.fail_update = False
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.recovery.finish_round(position=START, accepted=0)
        self.assertEqual(self.device.writes, 0)

    def test_snapshot_rejects_unknown_and_closed_borrows(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.snapshot.saved_tensor(Tensor([0]))
        self.snapshot.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.snapshot.saved_tensor(self.caches[0])

    def test_invalid_plan_fields_and_capacity(self):
        valid = dict(
            position=START,
            expected_position=START,
            draft_count=DRAFT_COUNT,
            accepted=0,
            page_size=PAGE_SIZE,
            page_count=PAGE_COUNT,
            pages=PAGES,
        )
        for changes in (
            {"accepted": True},
            {"accepted": -1},
            {"accepted": DRAFT_COUNT + 1},
            {"draft_count": 0},
            {"page_size": 0},
            {"pages": ()},
            {"pages": list(PAGES)},
            {"pages": None},
            {"pages": START},
            {"pages": (0, 0)},
            {"pages": (True,)},
            {"pages": (PAGE_COUNT,)},
            {"pages": (0,)},
            {"position": -1},
            {"position": float(START)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                core.plan_kv_tail(**(valid | changes))


if __name__ == "__main__":
    unittest.main()
