"""Pooled paged-KV layout for the batched P150x2 decode service.

Every generation group owns the whole pool while it runs: the batch worker runs one group
at a time and resets every slot before the next prefill. Each active row therefore takes a
contiguous run of blocks sized to its own request, so one long request can use a long row
while short requests share the rest, instead of every slot reserving the longest row.

Pool layout (block indices):

- ``[0, reserved_prefix_blocks)`` hold the KV of a saved structured-read prefix. Generation
  never assigns them, so a later read that shares the prefix still replays only its tail.
- ``[reserved_prefix_blocks, generation_end_block)`` are assigned to generation rows.
- ``[generation_end_block, scratch_block)`` hold four independent captured read rows,
  each with enough blocks for a 256-token bucket. Neither generation nor single-read
  warm-up rows use them.
- ``scratch_block`` (the last block) fills page-table entries past each row's run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Prompts are prefilled in 1,024-token segments (full masked-trace or eager chunks plus a
# tail bucket of at most 1,024 tokens), and each segment writes whole buckets of KV.
PREFILL_SEGMENT_TOKENS = 1024
# A structured read saves the state after its first prefill segment.
READ_PREFIX_TOKENS = PREFILL_SEGMENT_TOKENS

# The model captures four 128/256-token batched structured-read rows. Even unused
# lanes write their entire padded bucket, so reserve the largest bucket for every lane.
READ_BATCH_WIDTH = 4
READ_BATCH_MIN_BUCKET_TOKENS = 128
READ_BATCH_MAX_BUCKET_TOKENS = 256


class KvPoolError(ValueError):
    """A request or group does not fit the KV pool."""


def _require_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KvPoolError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class KvPoolLayout:
    """Block geometry of one pooled KV cache."""

    pool_blocks: int
    row_blocks: int
    kv_block_size: int
    # Zero exists only to model pre-reservation pool geometry in historical comparisons.
    # The serving adapter requires all four model lanes.
    batch_read_width: int = READ_BATCH_WIDTH

    def __post_init__(self) -> None:
        _require_positive("KV pool block count", self.pool_blocks)
        _require_positive("page-table row width", self.row_blocks)
        _require_positive("KV block size", self.kv_block_size)
        if (
            isinstance(self.batch_read_width, bool)
            or not isinstance(self.batch_read_width, int)
            or self.batch_read_width not in (0, READ_BATCH_WIDTH)
        ):
            raise KvPoolError("batched read width must be zero or four")
        if READ_PREFIX_TOKENS % self.kv_block_size:
            raise KvPoolError("the KV block size must divide the read prefix")
        if READ_BATCH_MIN_BUCKET_TOKENS % self.kv_block_size:
            raise KvPoolError("the KV block size must divide the batched read buckets")
        if self.batch_read_blocks_per_row > self.row_blocks:
            raise KvPoolError("the page-table row cannot hold a batched read bucket")
        if self.group_capacity_blocks < self.row_blocks:
            raise KvPoolError("the KV pool cannot hold one full-length row")

    @property
    def reserved_prefix_blocks(self) -> int:
        return READ_PREFIX_TOKENS // self.kv_block_size

    @property
    def scratch_block(self) -> int:
        return self.pool_blocks - 1

    @property
    def batch_read_blocks_per_row(self) -> int:
        return READ_BATCH_MAX_BUCKET_TOKENS // self.kv_block_size

    @property
    def generation_end_block(self) -> int:
        """Exclusive end of generation KV; batched read lanes follow this block."""
        return (
            self.scratch_block - self.batch_read_width * self.batch_read_blocks_per_row
        )

    @property
    def group_capacity_blocks(self) -> int:
        """Blocks one generation group may assign across all of its rows."""
        return self.generation_end_block - self.reserved_prefix_blocks

    def request_blocks(self, *, prompt_tokens: int, generation_tokens: int) -> int:
        """Blocks one request's row needs: its prefill buckets and every decoded position."""
        _require_positive("prompt token count", prompt_tokens)
        _require_positive("generation token count", generation_tokens)
        prefill_extent = (
            -(-prompt_tokens // PREFILL_SEGMENT_TOKENS) * PREFILL_SEGMENT_TOKENS
        )
        written_tokens = max(prefill_extent, prompt_tokens + generation_tokens)
        blocks = -(-written_tokens // self.kv_block_size)
        if blocks > self.row_blocks:
            raise KvPoolError(
                f"request requires {blocks} KV blocks; a row holds {self.row_blocks}"
            )
        return blocks

    def assign_group(
        self, block_counts: Sequence[int], *, width: int
    ) -> list[list[int]]:
        """Page-table rows for one group: row i gets a contiguous run of block_counts[i].

        Rows past the group, and entries past each run, point at the scratch block.
        """
        _require_positive("page-table width", width)
        if not block_counts or len(block_counts) > width:
            raise KvPoolError("a group needs between one row and the page-table width")
        for count in block_counts:
            _require_positive("row block count", count)
            if count > self.row_blocks:
                raise KvPoolError(
                    f"a row needs {count} KV blocks; a row holds {self.row_blocks}"
                )
        total = sum(block_counts)
        if total > self.group_capacity_blocks:
            raise KvPoolError(
                f"the group needs {total} KV blocks; the pool holds {self.group_capacity_blocks}"
            )
        rows = []
        start = self.reserved_prefix_blocks
        for count in block_counts:
            rows.append(
                list(range(start, start + count))
                + [self.scratch_block] * (self.row_blocks - count)
            )
            start += count
        rows.extend(
            [self.scratch_block] * self.row_blocks for _ in range(width - len(rows))
        )
        return rows

    def batch_read_rows(self) -> list[list[int]]:
        """Persistent capture rows: disjoint 256-token lanes, then scratch entries."""
        blocks = self.batch_read_blocks_per_row
        return [
            list(range(start, start + blocks))
            + [self.scratch_block] * (self.row_blocks - blocks)
            for start in range(self.generation_end_block, self.scratch_block, blocks)
        ]

    def startup_rows(self, *, width: int) -> list[list[int]]:
        """Rows for warm-up and trace capture, before any group is assigned.

        The last row is the structured-read row: a full-length run from block 0, so its
        first blocks are the reserved read prefix. The other rows split generation
        blocks only; batched reads capture their separate reserved rows.
        """
        _require_positive("page-table width", width)
        if width < 2:
            raise KvPoolError(
                "the startup layout needs a read row and a generation row"
            )
        read_row = list(range(self.row_blocks))
        first_free = self.row_blocks
        share = min(
            self.row_blocks, (self.generation_end_block - first_free) // (width - 1)
        )
        if share < PREFILL_SEGMENT_TOKENS // self.kv_block_size:
            raise KvPoolError("the KV pool is too small for the startup rows")
        rows = []
        for row in range(width - 1):
            start = first_free + row * share
            rows.append(
                list(range(start, start + share))
                + [self.scratch_block] * (self.row_blocks - share)
            )
        rows.append(read_row)
        return rows
