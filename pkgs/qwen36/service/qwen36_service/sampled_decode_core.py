"""Pure host-side sampling core for sampled-mode decode.

r[impl tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

The sampled decode tail reads back each device shard's top-K candidate
values and token ids from the traced step, merges the shards, and draws one
token per user on the host with a seeded generator. This core owns the
merge, the temperature-scaled distribution, the top-k and nucleus cuts, and
the draw. The shell owns the device readback and the RNG construction.

Greedy requests never reach this core: they route to the exact-parity argmax
path. Every input fails closed, and the draw is a pure function of the
candidates and the injected generator state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Qwen3's recommended sampling configuration: temperature 0.6, top_k 20,
# top_p 0.95. The candidate count matches the trace tail's per-shard top-K.
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.95
PROBABILITY_FLOOR = 1e-12


class SampledDecodeError(ValueError):
    """The sampled decode input violates its closed policy."""


@dataclass(frozen=True)
class ShardCandidates:
    """One device shard's top-K candidates for one decode step."""

    values: tuple[float, ...]
    token_ids: tuple[int, ...]


def _require_candidates(candidates: ShardCandidates, shard: int) -> None:
    if not candidates.values or not candidates.token_ids:
        raise SampledDecodeError(f"shard {shard} has no candidates")
    if len(candidates.values) != len(candidates.token_ids):
        raise SampledDecodeError(f"shard {shard} value and token counts differ")
    for token in candidates.token_ids:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise SampledDecodeError(
                f"shard {shard} token ids must be non-negative integers"
            )
    for value in candidates.values:
        if not math.isfinite(value):
            raise SampledDecodeError(f"shard {shard} candidate values must be finite")


def merge_shard_candidates(
    shards: tuple[ShardCandidates, ...],
    *,
    top_k: int,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Merge shard candidates into one descending list cut to top_k."""
    if not shards:
        raise SampledDecodeError("at least one shard is required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise SampledDecodeError("top_k must be a positive integer")
    for index, shard in enumerate(shards):
        _require_candidates(shard, index)
    merged = sorted(
        (
            (value, token)
            for shard in shards
            for value, token in zip(shard.values, shard.token_ids, strict=True)
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )
    cut = merged[:top_k]
    tokens = [token for _, token in cut]
    if len(set(tokens)) != len(tokens):
        raise SampledDecodeError("duplicate token ids across shards")
    return (
        tuple(value for value, _ in cut),
        tuple(tokens),
    )


def temperature_probabilities(
    values: tuple[float, ...], temperature: float
) -> tuple[float, ...]:
    """Temperature-scaled softmax over candidate logit values."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise SampledDecodeError("temperature must be a number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise SampledDecodeError("temperature must be finite and positive")
    peak = max(values)
    weights = [math.exp((value - peak) / temperature) for value in values]
    total = math.fsum(weights)
    if total <= 0.0:
        raise SampledDecodeError("candidate weights collapsed to zero")
    return tuple(weight / total for weight in weights)


def apply_nucleus_cut(
    probabilities: tuple[float, ...],
    *,
    top_p: float,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Keep the smallest prefix of candidates whose mass reaches top_p.

    Returns the kept probabilities (unnormalized) and their positions.
    """
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise SampledDecodeError("top_p must be a number")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise SampledDecodeError("top_p must lie in (0, 1]")
    cumulative = 0.0
    kept_positions: list[int] = []
    for position, probability in enumerate(probabilities):
        kept_positions.append(position)
        cumulative += probability
        if cumulative >= top_p:
            break
    return (tuple(probabilities[p] for p in kept_positions), tuple(kept_positions))


def sample_candidates(
    *,
    values: tuple[float, ...],
    token_ids: tuple[int, ...],
    temperature: float,
    top_k: int,
    top_p: float,
    draw: float,
) -> int:
    """Draw one token from the candidate distribution.

    ``draw`` is a uniform sample in [0, 1) from the caller's seeded
    generator, keeping this function pure and deterministic.
    """
    if len(values) != len(token_ids) or not values:
        raise SampledDecodeError("candidate values and tokens must align")
    if isinstance(draw, bool) or not isinstance(draw, (int, float)):
        raise SampledDecodeError("draw must be a number")
    draw = float(draw)
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise SampledDecodeError("draw must lie in [0, 1)")
    probabilities = temperature_probabilities(values, temperature)
    kept, _positions = apply_nucleus_cut(probabilities, top_p=top_p)
    kept_count = min(len(kept), top_k)
    kept = kept[:kept_count]
    total = math.fsum(kept)
    threshold = draw * total
    cumulative = 0.0
    for rank, probability in enumerate(kept):
        cumulative += probability
        if threshold < cumulative:
            return token_ids[rank]
    raise SampledDecodeError("draw did not land inside the kept mass")
