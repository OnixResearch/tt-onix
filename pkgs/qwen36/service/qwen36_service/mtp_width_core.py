"""Expected-value selection among prepared MTP draft widths.

The serving adapter runs one draft width because a large verification trace is
resident for the request; capturing a second width while it resides corrupts
the first trace. This module therefore prices width choices from measured round
costs and per-position acceptance, for an offline or startup profile choice.
It does not capture or switch traces.

A speculative round always emits the target's continuation token. A draft at
position j is emitted only when every draft through j was accepted, so the
expected emitted tokens are

    E(w) = 1 + sum_{j=1..w} product_{m=1..j} q[m],

where `q[m]` is the measured probability that draft m is accepted given the
previous ones. The round is worth scheduling when its expected tokens per
second exceed a plain step's one token per second. The tail keeps the
existing `plan_round` rule: a width is eligible only while the request has at
least `w + 1` tokens left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .mtp_drafter_core import (
    MAXIMUM_DRAFT_TOKEN_COUNT,
    MINIMUM_DRAFT_TOKEN_COUNT,
)

MODE_SPECULATIVE = "speculative"
MODE_PLAIN = "plain"


class MtpWidthPolicyError(ValueError):
    """A width measurement is outside the closed policy."""


@dataclass(frozen=True)
class WidthMeasurement:
    """One prepared width's measured round cost and acceptance probabilities."""

    draft_token_count: int
    round_seconds: float
    acceptance: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.draft_token_count, bool)
            or not isinstance(self.draft_token_count, int)
            or not MINIMUM_DRAFT_TOKEN_COUNT
            <= self.draft_token_count
            <= MAXIMUM_DRAFT_TOKEN_COUNT
        ):
            raise MtpWidthPolicyError(
                "draft_token_count must be between "
                f"{MINIMUM_DRAFT_TOKEN_COUNT} and {MAXIMUM_DRAFT_TOKEN_COUNT}"
            )
        if (
            isinstance(self.round_seconds, bool)
            or not isinstance(self.round_seconds, (int, float))
            or not math.isfinite(self.round_seconds)
            or self.round_seconds <= 0
        ):
            raise MtpWidthPolicyError("round_seconds must be positive and finite")
        if not isinstance(self.acceptance, tuple) or not self.acceptance:
            raise MtpWidthPolicyError("acceptance must be a non-empty tuple")
        if len(self.acceptance) != self.draft_token_count:
            raise MtpWidthPolicyError(
                "acceptance must hold one probability per draft position"
            )
        for probability in self.acceptance:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise MtpWidthPolicyError(
                    "acceptance probabilities must be finite values in [0, 1]"
                )


@dataclass(frozen=True)
class WidthPlan:
    """The highest-value prepared width, or plain stepping when none wins."""

    mode: str
    draft_token_count: int
    expected_emitted_tokens: float
    expected_tokens_per_second: float


def expected_emitted_tokens(measurement: WidthMeasurement) -> float:
    """Tokens a round of this width is expected to commit, bonus included."""
    if not isinstance(measurement, WidthMeasurement):
        raise MtpWidthPolicyError("expected tokens needs a WidthMeasurement")
    expected = 1.0
    prefix = 1.0
    for probability in measurement.acceptance:
        prefix *= probability
        expected += prefix
    return expected


def select_round_width(
    *,
    remaining_tokens: int,
    plain_step_seconds: float,
    measurements: tuple[WidthMeasurement, ...],
) -> WidthPlan:
    """Pick the width with the highest expected tokens per second.

    Widths that cannot fit their full acceptance window in the remaining
    tokens are ineligible, matching `mtp_serving_core.plan_round`. A plain
    step wins ties and wins whenever no speculative width beats it.
    """
    if (
        isinstance(remaining_tokens, bool)
        or not isinstance(remaining_tokens, int)
        or remaining_tokens < 1
    ):
        raise MtpWidthPolicyError("remaining_tokens must be a positive integer")
    if (
        isinstance(plain_step_seconds, bool)
        or not isinstance(plain_step_seconds, (int, float))
        or not math.isfinite(plain_step_seconds)
        or plain_step_seconds <= 0
    ):
        raise MtpWidthPolicyError("plain_step_seconds must be positive and finite")
    if not isinstance(measurements, tuple):
        raise MtpWidthPolicyError("measurements must be a tuple")
    seen: set[int] = set()
    for measurement in measurements:
        if not isinstance(measurement, WidthMeasurement):
            raise MtpWidthPolicyError("measurements must hold WidthMeasurement values")
        if measurement.draft_token_count in seen:
            raise MtpWidthPolicyError("a width is measured twice")
        seen.add(measurement.draft_token_count)

    plain_score = 1.0 / plain_step_seconds
    best = WidthPlan(MODE_PLAIN, 0, 1.0, plain_score)
    for measurement in measurements:
        if remaining_tokens < measurement.draft_token_count + 1:
            continue
        expected = expected_emitted_tokens(measurement)
        score = expected / measurement.round_seconds
        if score > best.expected_tokens_per_second:
            best = WidthPlan(
                MODE_SPECULATIVE,
                measurement.draft_token_count,
                expected,
                score,
            )
    return best
