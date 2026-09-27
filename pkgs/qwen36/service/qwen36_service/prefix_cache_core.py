"""Pure identity, policy, and bookkeeping for the hybrid generation prefix cache.

r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_identity]

A Qwen3.8 prefix spans both attention families: full-attention KV in the paged pool and
the carried GDN state (recurrent state and convolution carry) after the prefix. An entry
therefore names both: the KV blocks that hold positions [0, prefix_tokens), and one saved
carried-state buffer on the device. A KV-only entry cannot restore this model.

Keys are content-bound. Digest i of a prompt chains BLAKE3 over a header binding the
serving identity (execution closure, checkpoint, tokenizer, template, precision flags)
and the tenant scope, then over each 1,024-token chunk of token ids, so it covers
exactly tokens [0, (i + 1) * 1024). Equal text with different token ids, another
identity, or another tenant yields other digests: a miss, never another tenant's entry.
Only prefixes strictly shorter than the prompt get a digest, so a hit always leaves at
least one token to prefill for the first-token logits.

Entries are immutable once ready. A request that hits one reads its KV blocks through
its own page-table row and never writes them: the continuation starts at the chunk
boundary after the prefix, and every later write lands in the request's own blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Prefill segment length of the masked prefill traces (EAGER_PREFILL_CHUNK_TOKENS).
CHUNK_TOKENS = 1024
PROFILE_SCHEMA_VERSION = 1
MAXIMUM_STATE_SLOTS = 16
DIGEST_DOMAIN = b"qwen38-hybrid-prefix-v1"
# BLAKE3's default 32-byte output, in hex.
DIGEST_HEX_LENGTH = 64
TOKEN_BYTES = 4
POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "state_slots",
        "minimum_prefix_tokens",
        "maximum_cached_blocks",
        "tenant",
    }
)


class PrefixCachePolicyError(ValueError):
    """The prefix-cache policy or an entry is outside the closed contract."""


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PrefixCachePolicyError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class PrefixCachePolicy:
    """Closed prefix-cache bounds projected from prefix-cache-policy.ncl."""

    profile_id: str
    state_slots: int
    minimum_prefix_tokens: int
    maximum_cached_blocks: int
    tenant: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise PrefixCachePolicyError("profile_id must be a non-empty string")
        if not isinstance(self.tenant, str) or not self.tenant:
            raise PrefixCachePolicyError("tenant must be a non-empty string")
        _require_int(self.state_slots, "state_slots", 1, MAXIMUM_STATE_SLOTS)
        _require_int(self.minimum_prefix_tokens, "minimum_prefix_tokens", CHUNK_TOKENS, 2**20)
        if self.minimum_prefix_tokens % CHUNK_TOKENS:
            raise PrefixCachePolicyError("minimum_prefix_tokens must be a whole number of chunks")
        _require_int(self.maximum_cached_blocks, "maximum_cached_blocks", 1, 2**20)

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> PrefixCachePolicy:
        if not isinstance(root, Mapping) or set(root) != POLICY_FIELDS:
            raise PrefixCachePolicyError("policy fields differ from the closed schema")
        if _require_int(root["schema_version"], "schema_version", 1, 1) != PROFILE_SCHEMA_VERSION:
            raise PrefixCachePolicyError("schema_version is unsupported")
        return cls(
            profile_id=root["profile_id"],
            state_slots=root["state_slots"],
            minimum_prefix_tokens=root["minimum_prefix_tokens"],
            maximum_cached_blocks=root["maximum_cached_blocks"],
            tenant=root["tenant"],
        )


def prefix_digests(
    token_ids: Sequence[int],
    *,
    identity: str,
    tenant: str,
    blake3_digest: Callable[[bytes], bytes],
) -> tuple[str, ...]:
    """Hex digests of every chunk-aligned prefix strictly shorter than the prompt.

    Digest i covers tokens [0, (i + 1) * CHUNK_TOKENS) under this identity and tenant."""
    if not identity or not tenant:
        raise PrefixCachePolicyError("a prefix digest needs an identity and a tenant")
    header = b"\0".join((DIGEST_DOMAIN, identity.encode(), tenant.encode()))
    state = blake3_digest(header)
    digests = []
    for chunk in range((len(token_ids) - 1) // CHUNK_TOKENS):
        tokens = token_ids[chunk * CHUNK_TOKENS : (chunk + 1) * CHUNK_TOKENS]
        payload = b"".join(int(token).to_bytes(TOKEN_BYTES, "little", signed=False) for token in tokens)
        state = blake3_digest(state + payload)
        digests.append(state.hex())
    return tuple(digests)


@dataclass(frozen=True)
class PrefixEntry:
    """One cached prefix: its KV block ranges and the saved carried-state buffer."""

    entry_id: int
    digest: str
    prefix_tokens: int
    block_ranges: tuple[tuple[int, int], ...]
    state_index: int
    ready: bool
    pins: int
    last_used: float

    def validate(self, block_tokens: int, state_slots: int) -> None:
        if self.prefix_tokens % CHUNK_TOKENS or self.prefix_tokens <= 0:
            raise PrefixCachePolicyError("an entry must cover whole chunks")
        if not self.block_ranges:
            raise PrefixCachePolicyError("an entry must name its KV blocks; a KV-less entry is invalid")
        if sum(count for _start, count in self.block_ranges) * block_tokens != self.prefix_tokens:
            raise PrefixCachePolicyError("an entry's KV blocks must cover exactly its prefix")
        if not 0 <= self.state_index < state_slots:
            raise PrefixCachePolicyError("an entry must name an allocated carried-state buffer")
        if self.pins < 0:
            raise PrefixCachePolicyError("an entry cannot be pinned a negative number of times")


def expand_blocks(ranges: Iterable[tuple[int, int]]) -> list[int]:
    return [block for start, count in ranges for block in range(start, start + count)]


def cached_blocks(entries: Iterable[PrefixEntry]) -> frozenset[int]:
    """Every KV block some entry holds; entries in a chain share their common blocks."""
    return frozenset(block for entry in entries for block in expand_blocks(entry.block_ranges))


def lookup(entries: Sequence[PrefixEntry], digests: Sequence[str]) -> PrefixEntry | None:
    """The ready entry for the longest prefix of these digests, or None."""
    ready = {entry.digest: entry for entry in entries if entry.ready}
    for digest in reversed(digests):
        entry = ready.get(digest)
        if entry is not None:
            return entry
    return None


def eviction_order(entries: Sequence[PrefixEntry], keep: frozenset[int] = frozenset()) -> list[PrefixEntry]:
    """Entries that may be evicted, least recently used first: ready, unpinned, not kept."""
    candidates = [
        entry for entry in entries if entry.ready and entry.pins == 0 and entry.entry_id not in keep
    ]
    return sorted(candidates, key=lambda entry: (entry.last_used, entry.entry_id))


def free_state_index(entries: Sequence[PrefixEntry], state_slots: int) -> int | None:
    used = {entry.state_index for entry in entries}
    return next((index for index in range(state_slots) if index not in used), None)
