"""Diagnostic target-KV recovery; signed-zero bit preservation remains blocked."""

from contextlib import ExitStack

import torch
import ttnn

from .mtp_kv_tail_core import plan_kv_tail
from .mtp_state_snapshot_adapter import tensor_identity

CACHE_RANK = 4
PAGE_TABLE_RANK = 2
PAGE_AXIS = 0
HEAD_AXIS = 1
ROW_AXIS = 2
CHANNEL_AXIS = 3
SINGLE_USER = 1
ROW_TO_UPDATE_ORDER = (PAGE_AXIS, ROW_AXIS, HEAD_AXIS, CHANNEL_AXIS)


class MtpTargetKvRecovery:
    """Own a page table, not the borrowed live caches or entry snapshots.

    The caller must restore capture writes before the first round. Each finish
    advances one contiguous committed boundary. No other writer may touch the
    uncommitted tail between rounds. A device error poisons this request owner.
    """

    def __init__(self, *, model, snapshot, mesh, page_table, position, draft_count):
        if (
            not isinstance(page_table, torch.Tensor)
            or page_table.ndim != PAGE_TABLE_RANK
        ):
            raise ValueError("KV recovery requires a host page table")
        if page_table.shape[PAGE_AXIS] != SINGLE_USER:
            raise ValueError("KV recovery requires exactly one user")
        self._pages = tuple(page_table.tolist()[PAGE_AXIS])
        self._snapshot = snapshot
        self._mesh = mesh
        self._position = position
        self._draft_count = draft_count
        self._closed = False
        self._failed = False
        self._page_table = None
        self._pairs = tuple(
            (
                cache,
                snapshot.saved_tensor(cache),
                layer.attention._kv_shard_cfg(SINGLE_USER),
            )
            for layer in model.layers
            if layer.is_full_attention
            for cache in (layer.attention.paged_k, layer.attention.paged_v)
        )
        if not self._pairs:
            raise ValueError("KV recovery has no target caches")
        self._shape = tuple(self._pairs[PAGE_AXIS][PAGE_AXIS].shape)
        if len(self._shape) != CACHE_RANK or any(size < 1 for size in self._shape):
            raise ValueError("KV recovery cache geometry is invalid")
        if (
            self._shape[ROW_AXIS] % ttnn.TILE_SIZE
            or self._shape[CHANNEL_AXIS] % ttnn.TILE_SIZE
            or self._shape[HEAD_AXIS] > ttnn.TILE_SIZE
        ):
            raise ValueError("KV recovery requires tile-aligned pages and padded heads")
        self._bindings = tuple(
            (tensor_identity(live), tensor_identity(saved))
            for live, saved, _ in self._pairs
        )
        self._require_bindings()
        self._plan(position, draft_count)
        self._page_table = self._integer_tensor([self._pages])

    def _integer_tensor(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            device=self._mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._mesh),
        )

    def _plan(self, position, accepted):
        return plan_kv_tail(
            position=position,
            expected_position=self._position,
            draft_count=self._draft_count,
            accepted=accepted,
            page_size=self._shape[ROW_AXIS],
            page_count=self._shape[PAGE_AXIS],
            pages=self._pages,
        )

    def _require_bindings(self):
        if self._closed or self._failed:
            raise RuntimeError("KV recovery owner is closed or failed")
        # Validate every source and destination before the first cache write.
        for (live, saved, _), binding in zip(self._pairs, self._bindings, strict=True):
            if (
                tuple(live.shape) != self._shape
                or tuple(saved.shape) != self._shape
                or live.dtype != ttnn.bfloat16
                or saved.dtype != ttnn.bfloat16
            ):
                raise ValueError("KV recovery requires matching bf16 cache tensors")
            if (tensor_identity(live), tensor_identity(saved)) != binding:
                raise RuntimeError("KV recovery allocation changed")
            if self._snapshot.saved_tensor(live) is not saved:
                raise RuntimeError("KV recovery snapshot binding changed")

    def _row_input(self, saved, row, shard_config, cleanup):
        def keep(tensor):
            cleanup.callback(ttnn.deallocate, tensor)
            return tensor

        tile_start = row.offset // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        heads = self._shape[HEAD_AXIS]
        channels = self._shape[CHANNEL_AXIS]
        # Cut one aligned tile first. An unaligned slice of the entire cache
        # would force a full-cache layout conversion on the pinned runtime.
        tile = keep(
            ttnn.slice(
                saved,
                (row.page, 0, tile_start, 0),
                (row.page + 1, heads, tile_start + ttnn.TILE_SIZE, channels),
            )
        )
        offset = row.offset - tile_start
        value = keep(
            ttnn.slice(tile, (0, 0, offset, 0), (1, heads, offset + 1, channels))
        )
        transposed = keep(ttnn.permute(value, ROW_TO_UPDATE_ORDER))
        padded = keep(
            ttnn.pad(transposed, [1, 1, ttnn.TILE_SIZE, channels], [0, 0, 0, 0], 0.0)
        )
        return keep(ttnn.to_memory_config(padded, shard_config))

    def finish_round(self, *, position, accepted):
        plan = self._plan(position, accepted)
        self._require_bindings()
        try:
            with ExitStack() as cleanup:
                updates = []
                for row in plan.rows:
                    index = self._integer_tensor([row.position])
                    cleanup.callback(ttnn.deallocate, index)
                    for live, saved, shard_config in self._pairs:
                        value = self._row_input(saved, row, shard_config, cleanup)
                        updates.append((live, value, index))
                for live, value, index in updates:
                    ttnn.experimental.paged_update_cache(
                        live,
                        value,
                        update_idxs_tensor=index,
                        page_table=self._page_table,
                    )
                if updates:
                    ttnn.synchronize_device(self._mesh)
            self._position = plan.next_position
        except BaseException:
            self._failed = True
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._page_table is not None:
            ttnn.deallocate(self._page_table)
            self._page_table = None
