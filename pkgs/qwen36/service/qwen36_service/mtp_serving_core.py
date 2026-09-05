"""Pure serving policy for gated MTP speculative decode rounds.

r[impl tenstorrent.qwen38_p150x2.mtp_serving_rounds]

The core owns round scheduling, token commitment, the feedback-pool bound,
and the economic-loss transition. Each decision uses explicit inputs and
returns a closed result or a specific policy error.

`commit_verified_round` defines greedy commitment for supplied target rows.
The rows must match the sequential target for each accepted prefix. The
adapter and its device evidence must establish that precondition. This
core does not prove tensor parity, cache alignment, or device recovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .mtp_drafter_core import (
    MAXIMUM_DRAFT_TOKEN_COUNT,
    MINIMUM_DRAFT_TOKEN_COUNT,
)

ROUND_MODE_SPECULATIVE = "speculative"
ROUND_MODE_PLAIN = "plain"
ROUND_MODES = (ROUND_MODE_SPECULATIVE, ROUND_MODE_PLAIN)

MINIMUM_TOKEN_ID = 0

# A speculative round must beat the plain decode path it replaces. The plain
# step budget is the measured eager/traced step cost on P150x2 (~59-63 ms);
# a round may cost up to twice its emitted-token budget before it counts as
# a loss, and two consecutive losses disengage speculation for the request.
PLAIN_STEP_SECONDS_BUDGET = 0.12
ECONOMICS_SLACK = 2.0
ECONOMICS_DISENGAGE_STREAK = 2


@dataclass(frozen=True)
class RoundEconomics:
    """The next loss count and its speculation decision."""

    consecutive_losses: int
    disengaged: bool


def assess_round_economics(
    *,
    round_seconds: float,
    emitted_tokens: int,
    consecutive_losses: int,
) -> RoundEconomics:
    """Advance the loss count, or reset it after a round within budget."""
    if isinstance(round_seconds, bool) or not isinstance(round_seconds, (int, float)):
        raise MtpServingPolicyError("round_seconds must be a number")
    if not math.isfinite(round_seconds) or round_seconds <= 0.0:
        raise MtpServingPolicyError("round_seconds must be positive and finite")
    if isinstance(emitted_tokens, bool) or not isinstance(emitted_tokens, int):
        raise MtpServingPolicyError("emitted_tokens must be an integer")
    if emitted_tokens < 1:
        raise MtpServingPolicyError("emitted_tokens must be at least one")
    if isinstance(consecutive_losses, bool) or not isinstance(consecutive_losses, int):
        raise MtpServingPolicyError("consecutive_losses must be an integer")
    if consecutive_losses < 0:
        raise MtpServingPolicyError("consecutive_losses must not be negative")
    budget = PLAIN_STEP_SECONDS_BUDGET * emitted_tokens * ECONOMICS_SLACK
    losses = consecutive_losses + 1 if round_seconds > budget else 0
    return RoundEconomics(
        consecutive_losses=losses,
        disengaged=losses >= ECONOMICS_DISENGAGE_STREAK,
    )


def round_economics_disengaged(
    *,
    round_seconds: float,
    emitted_tokens: int,
    consecutive_losses: int,
) -> bool:
    """Return the decision without discarding the historical public API."""
    return assess_round_economics(
        round_seconds=round_seconds,
        emitted_tokens=emitted_tokens,
        consecutive_losses=consecutive_losses,
    ).disengaged


class MtpServingPolicyError(ValueError):
    """The serving round input does not satisfy its closed policy."""


@dataclass(frozen=True)
class RoundPlan:
    """One scheduling decision for the next serving step."""

    mode: str
    draft_token_count: int


@dataclass(frozen=True)
class RoundCommit:
    """The exact greedy result of one verified speculative round."""

    accepted: int
    rejected: bool
    committed_token_ids: tuple[int, ...]


def _require_counter(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MtpServingPolicyError(f"{label} must be an integer")
    if value < 0:
        raise MtpServingPolicyError(f"{label} must not be negative")
    return value


def _require_draft_width(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MtpServingPolicyError("draft_token_count must be an integer")
    if not MINIMUM_DRAFT_TOKEN_COUNT <= value <= MAXIMUM_DRAFT_TOKEN_COUNT:
        raise MtpServingPolicyError(
            "draft width must lie between "
            f"{MINIMUM_DRAFT_TOKEN_COUNT} and {MAXIMUM_DRAFT_TOKEN_COUNT}"
        )
    return value


def plan_round(
    *,
    tokens_generated: int,
    maximum_generation_tokens: int,
    draft_token_count: int,
) -> RoundPlan:
    """Schedule one speculative round or plain steps for the next position.

    A speculative round emits between 1 and `draft_token_count + 1` tokens,
    so it is scheduled only while more than `draft_token_count` tokens
    remain. The tail of a request always finishes on plain steps, which
    keeps the generated count inside the request bound without partial
    rounds.
    """
    generated = _require_counter(tokens_generated, "tokens_generated")
    maximum = _require_counter(maximum_generation_tokens, "maximum_generation_tokens")
    width = _require_draft_width(draft_token_count)
    if generated > maximum:
        raise MtpServingPolicyError(
            "tokens_generated exceeds maximum_generation_tokens"
        )
    remaining = maximum - generated
    if remaining >= width + 1:
        return RoundPlan(mode=ROUND_MODE_SPECULATIVE, draft_token_count=width)
    return RoundPlan(mode=ROUND_MODE_PLAIN, draft_token_count=0)


def feedback_pool_capacity(
    *, maximum_generation_tokens: int, draft_token_count: int
) -> int:
    """Bound rounds after one anchor, including rejection at every first draft."""
    plan_round(
        tokens_generated=1,
        maximum_generation_tokens=maximum_generation_tokens,
        draft_token_count=draft_token_count,
    )
    # A round needs width + 1 remaining slots and commits at least one token.
    return max(0, maximum_generation_tokens - draft_token_count - 1)


def commit_verified_round(
    *,
    draft_token_ids: tuple[int, ...],
    continuation_token_ids: tuple[int, ...],
) -> RoundCommit:
    """Commit one verified round exactly as sequential greedy would.

    `continuation_token_ids` holds the target model's own greedy outputs
    for the round window plus one extra token: `k + 1` entries for `k`
    drafts. The first disagreement between a draft and the continuation
    rejects the chain; the committed tokens are the accepted drafts plus
    the continuation token at that position. On full acceptance the extra
    continuation token is the bonus.
    """
    for label, sequence in (
        ("draft_token_ids", draft_token_ids),
        ("continuation_token_ids", continuation_token_ids),
    ):
        if not isinstance(sequence, tuple):
            raise MtpServingPolicyError(f"{label} must be a tuple")
        for token in sequence:
            if isinstance(token, bool) or not isinstance(token, int):
                raise MtpServingPolicyError(f"{label} contains a non-integer token")
            if token < MINIMUM_TOKEN_ID:
                raise MtpServingPolicyError(f"{label} contains a negative token")
    if not (
        MINIMUM_DRAFT_TOKEN_COUNT <= len(draft_token_ids) <= MAXIMUM_DRAFT_TOKEN_COUNT
    ):
        raise MtpServingPolicyError("draft chain length is outside the policy range")
    if len(continuation_token_ids) != len(draft_token_ids) + 1:
        raise MtpServingPolicyError(
            "continuation count differs from draft count plus one"
        )
    accepted = len(draft_token_ids)
    for index, (draft, target) in enumerate(
        zip(draft_token_ids, continuation_token_ids, strict=False)
    ):
        if draft != target:
            accepted = index
            break
    rejected = accepted < len(draft_token_ids)
    committed = draft_token_ids[:accepted] + (continuation_token_ids[accepted],)
    return RoundCommit(
        accepted=accepted,
        rejected=rejected,
        committed_token_ids=committed,
    )
