# SPDX-License-Identifier: LGPL-2.1-only
"""Device-free 32-token prefix-cache index for the Nanbeige4.3-33B runner.

The index is pure host state: it owns token identity, block refcounts, LRU
eviction and per-request token accounting, while the runner owns the KV writes.
Nothing here imports ttnn, torch, or device code, so the whole policy is checked
without cards (see `check.self_test_prefix_cache`).

Scope and non-claims:

- Identity is the exact token prefix (a digest chain over 32-token blocks, rooted
  at the configuration identity) plus the checkpoint, package/runtime, preset,
  mesh and execution keys. A different token at any position, or a different
  configuration, yields a different chain and therefore a miss.
- Blocks are the residency unit; entries mark exact prefix boundaries with a
  generation. Only whole 32-token blocks are shared, so a prompt can hit a longer
  entry's resident blocks at a shorter boundary. The block that would contain a
  partial tail must be copied before it is written (`tail_copy_required`), and a
  write into a leased prefix block raises `SharedBlockWriteError`.
- The cache is process-local. It makes no claim about cross-process or
  cross-tenant sharing, and it does not allocate device KV: the runner's page
  table and `--page-blocks` stay the authority for capacity.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# The paged KV pool uses 32-token pages (`--page-block 32`), so prefix blocks use
# the same granularity. A different page size would make every entry a lie.
BLOCK_TOKENS = 32

# Exact identity keys. `execution` carries the runner knobs that change the KV
# layout or the numerical path (prompt mode, pool size, trace); batch is not a
# key because rows do not change what the prefix tokens are.
IDENTITY_KEYS = (
    "checkpoint_digest",
    "checkpoint_revision",
    "package_identity",
    "runtime_identity",
    "preset",
    "mesh",
    "execution",
)


class PrefixCacheError(ValueError):
    """Invalid cache request or corrupt index state."""


class PrefixCacheBudgetError(PrefixCacheError):
    """A prefix does not fit the block budget even after evicting every idle entry."""


class SharedBlockWriteError(PrefixCacheError):
    """A write would target a read-only shared prefix block instead of a copy."""


def cache_identity(keys: Mapping[str, str]) -> str:
    """Digest the exact identity key set; missing or extra keys are errors.

    A silently missing key would let two different configurations share a cache
    entry, so the key set is fixed rather than merged.
    """
    missing = [key for key in IDENTITY_KEYS if not str(keys.get(key, "")).strip()]
    if missing:
        raise PrefixCacheError(f"prefix identity is missing {missing}")
    extra = sorted(set(keys) - set(IDENTITY_KEYS))
    if extra:
        raise PrefixCacheError(f"prefix identity has unbound keys {extra}")
    payload = json.dumps(
        {key: str(keys[key]) for key in IDENTITY_KEYS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aligned_tokens(tokens: int) -> int:
    """Round a token count down to whole 32-token blocks."""
    if tokens < 0:
        raise PrefixCacheError(f"token count {tokens} is negative")
    return tokens - tokens % BLOCK_TOKENS


def tail_copy_required(cached_tokens: int) -> bool:
    """Whether the block holding `cached_tokens` ends mid-block and must be copied.

    False for the block-aligned prefixes this cache stores; True is the marker a
    caller must honour (copy or recompute that block) before writing past a
    partially cached block.
    """
    return cached_tokens % BLOCK_TOKENS != 0


def block_chain(
    identity: str, tokens: Sequence[int]
) -> list[tuple[str, tuple[int, ...]]]:
    """Digest chain over the full 32-token blocks of `tokens`.

    The chain is rooted at the configuration identity and each block digest binds
    its parent, so a block key includes the exact tokens of every earlier block
    and the block's absolute position. The trailing partial block is not cached.
    """
    chain: list[tuple[str, tuple[int, ...]]] = []
    parent = identity
    for start in range(0, len(tokens) - BLOCK_TOKENS + 1, BLOCK_TOKENS):
        block = tuple(int(token) for token in tokens[start : start + BLOCK_TOKENS])
        digest = hashlib.sha256(
            parent.encode("utf-8") + struct.pack(f"<{BLOCK_TOKENS}q", *block)
        ).hexdigest()
        chain.append((digest, block))
        parent = digest
    return chain


@dataclass
class _Block:
    tokens: tuple[int, ...]
    entries: int = 0  # live prefix entries that include this block
    leases: int = 0  # active leases reading through this block


@dataclass
class _Entry:
    identity: str
    tokens: int
    keys: tuple[str, ...]
    generation: int
    references: int = 0  # active leases that resolved to this exact boundary
    last_used: int = 0


@dataclass(frozen=True)
class EntryView:
    """Immutable snapshot of one prefix entry."""

    identity: str
    tokens: int
    blocks: int
    generation: int
    references: int

    def as_dict(self) -> dict:
        return {
            "identity_digest": self.identity,
            "tokens": self.tokens,
            "blocks": self.blocks,
            "generation": self.generation,
            "references": self.references,
        }


@dataclass(frozen=True)
class Lease:
    """Result of resolving a request against the index.

    `cached_tokens` is always block aligned and no larger than the prompt.
    `copy_tail_block` is False for these aligned prefixes and True only when a
    caller asks to reuse a partial tail; `write_start_block` is the first block
    the request may write without disturbing shared K/V.
    """

    identity: str
    hit: bool
    cached_tokens: int
    blocks: int
    generation: int | None
    copy_tail_block: bool
    evicted_blocks: int
    keys: tuple[str, ...] = ()

    @property
    def write_start_block(self) -> int:
        return self.cached_tokens // BLOCK_TOKENS

    def as_dict(self) -> dict:
        return {
            "hit": self.hit,
            "cached_tokens": self.cached_tokens,
            "blocks": self.blocks,
            "generation": self.generation,
            "copy_tail_block": self.copy_tail_block,
            "write_start_block": self.write_start_block,
            "evicted_blocks": self.evicted_blocks,
        }


@dataclass(frozen=True)
class RequestAccounting:
    """Per-request token accounting for one prefix-cache run."""

    prompt_tokens: int
    cached_input_tokens: int
    new_input_tokens: int
    output_tokens: int

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "new_input_tokens": self.new_input_tokens,
            "output_tokens": self.output_tokens,
        }


def request_accounting(
    prompt_tokens: int, cached_input_tokens: int, output_tokens: int
) -> RequestAccounting:
    """Check and record one request's cached/new/output token counts.

    Cached tokens must be whole blocks of the prompt: a count that is not would
    bill recomputed tokens as cached.
    """
    if prompt_tokens < 0 or output_tokens < 0:
        raise PrefixCacheError(
            f"prompt {prompt_tokens} and output {output_tokens} token counts must be non-negative"
        )
    if not 0 <= cached_input_tokens <= prompt_tokens:
        raise PrefixCacheError(
            f"cached {cached_input_tokens} tokens is outside the {prompt_tokens}-token prompt"
        )
    if cached_input_tokens % BLOCK_TOKENS:
        raise PrefixCacheError(
            f"cached {cached_input_tokens} tokens is not a multiple of {BLOCK_TOKENS}"
        )
    return RequestAccounting(
        prompt_tokens=prompt_tokens,
        cached_input_tokens=cached_input_tokens,
        new_input_tokens=prompt_tokens - cached_input_tokens,
        output_tokens=output_tokens,
    )


class PrefixCache:
    """LRU prefix index over shared read-only 32-token blocks.

    `budget_blocks` bounds the blocks the index holds. A lease pins the blocks it
    reads and the entry at its exact boundary; a pinned entry or block cannot be
    evicted or freed, so a request's cached prefix stays readable for its life.
    """

    def __init__(self, budget_blocks: int):
        if budget_blocks < 1:
            raise PrefixCacheBudgetError(
                f"prefix cache budget {budget_blocks} blocks must be at least 1"
            )
        self.budget_blocks = int(budget_blocks)
        self._blocks: dict[str, _Block] = {}
        self._entries: dict[str, _Entry] = {}  # terminal block digest -> entry
        self._clock = 0
        self.evicted_blocks = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def resident_blocks(self) -> int:
        return len(self._blocks)

    def entries(self) -> list[EntryView]:
        return [
            self._view(entry)
            for entry in sorted(self._entries.values(), key=lambda item: item.last_used)
        ]

    def block_references(self) -> dict[str, int]:
        """Block digest -> number of entries holding it (unit-test evidence)."""
        return {key: block.entries for key, block in self._blocks.items()}

    def lookup(self, identity: str, tokens: Sequence[int]) -> Lease:
        """Longest whole-block prefix of `tokens` resident under `identity`."""
        chain = block_chain(identity, tokens)
        present: list[tuple[str, tuple[int, ...]]] = []
        for key, block in chain:
            resident = self._blocks.get(key)
            if resident is None or resident.tokens != block:
                break
            present.append((key, block))
        if not present:
            return Lease(
                identity=identity,
                hit=False,
                cached_tokens=0,
                blocks=0,
                generation=None,
                copy_tail_block=False,
                evicted_blocks=self.evicted_blocks,
            )
        cached = len(present) * BLOCK_TOKENS
        keys = tuple(key for key, _ in present)
        last_key = keys[-1]
        entry = self._entries.get(last_key)
        touching = [item for item in self._entries.values() if last_key in item.keys]
        for item in touching:
            item.last_used = self._tick()
        if entry is not None:
            generation = entry.generation
        else:
            generation = max((item.generation for item in touching), default=None)
        return Lease(
            identity=identity,
            hit=True,
            cached_tokens=cached,
            blocks=len(present),
            generation=generation,
            copy_tail_block=tail_copy_required(cached),
            evicted_blocks=self.evicted_blocks,
            keys=keys,
        )

    def retain(self, identity: str, tokens: Sequence[int]) -> Lease:
        """Resolve and pin a hit for the caller's lifetime (`release` to unpin)."""
        lease = self.lookup(identity, tokens)
        if not lease.hit:
            return lease
        boundary = self._entries.get(lease.keys[-1])
        if boundary is not None:
            boundary.references += 1
        for key in lease.keys:
            self._blocks[key].leases += 1
        return lease

    def release(self, lease: Lease) -> None:
        if not lease.hit:
            return
        boundary = self._entries.get(lease.keys[-1])
        if boundary is not None:
            if boundary.references <= 0:
                raise PrefixCacheError(
                    f"prefix {lease.cached_tokens} of {lease.identity} has no outstanding lease"
                )
            boundary.references -= 1
            boundary.last_used = self._tick()
        for key in lease.keys:
            block = self._blocks.get(key)
            if block is None or block.leases <= 0:
                # Leased blocks are never deleted, so this is index corruption.
                raise PrefixCacheError(
                    f"leased prefix block {key[:12]} of {lease.identity} is missing"
                )
            block.leases -= 1
            if block.entries == 0 and block.leases == 0:
                del self._blocks[key]

    def insert(
        self, identity: str, tokens: Sequence[int], keep_tokens: int | None = None
    ) -> EntryView:
        """Record `tokens` (or its first `keep_tokens`) as a reusable prefix.

        Evicts least-recently-used unpinned entries first. Raises
        `PrefixCacheBudgetError` without mutating anything when the prefix cannot
        fit even after evicting every unpinned entry.
        """
        requested = len(tokens) if keep_tokens is None else int(keep_tokens)
        cached = aligned_tokens(requested)
        if cached <= 0:
            raise PrefixCacheError(
                f"prefix of {requested} tokens has no whole {BLOCK_TOKENS}-token block"
            )
        if cached > len(tokens):
            raise PrefixCacheError(
                f"prefix of {cached} tokens exceeds the {len(tokens)}-token sequence"
            )
        chain = block_chain(identity, tokens[:cached])
        if len(chain) * BLOCK_TOKENS != cached:
            raise PrefixCacheError(
                f"prefix {cached} did not yield {cached // BLOCK_TOKENS} whole blocks"
            )
        keys = tuple(key for key, _ in chain)

        previous = self._entries.get(keys[-1])
        if previous is not None:
            # Same identity and same block chain: the entry is already resident.
            # A new generation marks recomputed KV behind the same prefix.
            previous.generation += 1
            previous.last_used = self._tick()
            return self._view(previous)

        fresh = [key for key in keys if key not in self._blocks]
        pinned = {
            key
            for entry in self._entries.values()
            if entry.references
            for key in entry.keys
        }
        leased = {key for key, block in self._blocks.items() if block.leases}
        reclaimable = (
            {
                key
                for entry in self._entries.values()
                if not entry.references
                for key in entry.keys
            }
            - pinned
            - leased
        )
        if self.resident_blocks - len(reclaimable) + len(fresh) > self.budget_blocks:
            raise PrefixCacheBudgetError(
                f"prefix of {cached} tokens needs {len(keys)} block(s) ({len(fresh)} new) "
                f"with {self.resident_blocks} resident and {len(reclaimable)} reclaimable "
                f"against a {self.budget_blocks}-block budget"
            )
        self.evict(target=self.budget_blocks - len(fresh))
        if self.resident_blocks + len(fresh) > self.budget_blocks:
            raise PrefixCacheBudgetError(
                f"prefix of {cached} tokens did not fit after eviction: "
                f"{self.resident_blocks} resident + {len(fresh)} new > {self.budget_blocks}"
            )

        for key, block in chain:
            if key not in self._blocks:
                self._blocks[key] = _Block(block)
            self._blocks[key].entries += 1
        entry = _Entry(
            identity=identity,
            tokens=cached,
            keys=keys,
            generation=1,
            last_used=self._tick(),
        )
        self._entries[keys[-1]] = entry
        return self._view(entry)

    def evict(self, target: int | None = None) -> int:
        """Evict LRU unpinned entries until at most `target` blocks remain.

        Returns the number of blocks freed; `self.evicted_blocks` accumulates.
        Entries whose blocks are leased elsewhere may be evicted without freeing a
        block; the walk continues in LRU order until the target or no candidate.
        """
        limit = self.budget_blocks if target is None else int(target)
        freed = 0
        while self.resident_blocks > limit:
            candidates = sorted(
                (entry for entry in self._entries.values() if not entry.references),
                key=lambda entry: entry.last_used,
            )
            if not candidates:
                break
            victim = candidates[0]
            del self._entries[victim.keys[-1]]
            for key in victim.keys:
                block = self._blocks[key]
                block.entries -= 1
                if block.entries == 0 and block.leases == 0:
                    del self._blocks[key]
                    freed += 1
        self.evicted_blocks += freed
        return freed

    def writable(self, lease: Lease, position: int) -> bool:
        """Whether a KV write at `position` is outside every shared prefix block."""
        if not lease.hit:
            return True
        return position // BLOCK_TOKENS >= lease.write_start_block

    def assert_writable(self, lease: Lease, position: int) -> None:
        if not self.writable(lease, position):
            raise SharedBlockWriteError(
                f"position {position} writes shared prefix block "
                f"{position // BLOCK_TOKENS} of the {lease.cached_tokens}-token cached "
                f"prefix; copy the tail block or resume from a fresh block"
            )

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _view(entry: _Entry) -> EntryView:
        return EntryView(
            identity=entry.identity,
            tokens=entry.tokens,
            blocks=len(entry.keys),
            generation=entry.generation,
            references=entry.references,
        )


def block_statistics(tokens: int) -> dict:
    """Device-free block accounting for a prompt/prefix pair."""
    if tokens < 0:
        raise PrefixCacheError(f"token count {tokens} is negative")
    return {
        "tokens": tokens,
        "blocks": math.ceil(tokens / BLOCK_TOKENS),
        "partial_tail_tokens": tokens % BLOCK_TOKENS,
    }
