"""Pure serving policy for gated MTP speculative decode rounds.

r[impl tenstorrent.qwen38_p150x2.mtp_serving_rounds]

The core owns two decisions and nothing else: whether a request position
runs a speculative round or plain steps, and which tokens a verified round
commits. Both are device-free and total: every input either yields a
closed result or a specific policy error.

The exactness contract for greedy requests lives in
`commit_verified_round`. The verification argmaxes ARE the sequential
greedy outputs for the round's rows, so appending the argmax at the
rejection position (or the final argmax on full acceptance) makes the
committed stream equal sequential greedy output for every acceptance
pattern.
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
MTP_PREFILL_BUCKET = 128

ECONOMICS_DISENGAGE_STREAK = 2


@dataclass(frozen=True)
class RoundEconomics:
    consecutive_losses: int = 0
    disengaged: bool = False


def advance_round_economics(
    state: RoundEconomics,
    *,
    round_seconds: float,
    emitted_tokens: int,
    plain_step_seconds: float,
    maximum_consecutive_losses: int = ECONOMICS_DISENGAGE_STREAK,
) -> RoundEconomics:
    """Compare observed costs and retain a sticky disengagement decision."""
    if not isinstance(state, RoundEconomics) or not isinstance(state.disengaged, bool):
        raise MtpServingPolicyError("invalid round economics state")
    losses = _require_counter(state.consecutive_losses, "consecutive_losses")
    emitted = _require_counter(emitted_tokens, "emitted_tokens")
    maximum_losses = _require_counter(
        maximum_consecutive_losses, "maximum_consecutive_losses"
    )
    if emitted == 0 or maximum_losses == 0:
        raise MtpServingPolicyError(
            "emitted_tokens and maximum_consecutive_losses must be positive"
        )
    for label, duration in (
        ("round_seconds", round_seconds),
        ("plain_step_seconds", plain_step_seconds),
    ):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise MtpServingPolicyError(f"{label} must be a number")
        if not math.isfinite(duration) or duration <= 0:
            raise MtpServingPolicyError(f"{label} must be positive and finite")
    if state.disengaged:
        return state
    loses = round_seconds / emitted >= plain_step_seconds
    next_losses = losses + 1 if loses else 0
    return RoundEconomics(next_losses, next_losses >= maximum_losses)


class MtpServingPolicyError(ValueError):
    """The serving round input does not satisfy its closed policy."""


ROUND_WATCHDOG_SECONDS = 60.0


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
