"""Pure distribution-exact MTP acceptance rules for sampled speculative decode."""

# r[impl tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling]

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

RULE_EXACT = "exact"
RULE_LENIENT = "lenient"
RULE_RELAXED = "relaxed"
RULE_GREEDY = "greedy"
ACCEPTANCE_RULES = (RULE_EXACT, RULE_LENIENT, RULE_RELAXED, RULE_GREEDY)

# The shipped production rule. Sampled rules stay diagnostic until a validated
# change admits them with matched physical evidence.
PRODUCTION_RULE = RULE_GREEDY

DEFAULT_LENIENCE = 0.5
DEFAULT_TYPICAL_EPSILON = 0.09
DEFAULT_TYPICAL_DELTA = 0.3

_PROBABILITY_FLOOR = 1e-12


class MtpSamplerPolicyError(ValueError):
    """The sampled-acceptance input violates its closed policy."""


def _require_finite_probability(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpSamplerPolicyError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise MtpSamplerPolicyError(f"{label} must be finite")
    if not 0.0 <= value <= 1.0:
        raise MtpSamplerPolicyError(f"{label} must lie in [0, 1]")
    return value


def admitted_production_rule() -> str:
    """Return the only rule eligible for production admission."""
    return PRODUCTION_RULE


def resolve_rule(rule: str, *, diagnostic_only: bool) -> str:
    """Resolve one acceptance-rule identifier against its admission policy.

    Greedy is always eligible. Every sampled rule stays diagnostic-only until
    a validated Cairn change admits it with matched physical evidence.
    """
    if rule not in ACCEPTANCE_RULES:
        joined = ", ".join(ACCEPTANCE_RULES)
        raise MtpSamplerPolicyError(
            f"acceptance rule must be one of {joined}, got {rule!r}"
        )
    if rule != PRODUCTION_RULE and not diagnostic_only:
        raise MtpSamplerPolicyError(
            "sampled acceptance rules are diagnostic-only; production admits "
            f"only {PRODUCTION_RULE!r}"
        )
    return rule


def realized_total_variation(
    acceptance_probability: float,
    draft_probability: float,
) -> float:
    """Return the exact per-position total-variation divergence ``|a - p(x̂)|``.

    For a point-mass draft the realized distribution is
    ``a·δ_x̂ + (1-a)·norm(p minus x̂)``, whose distance from ``p`` collapses to
    the absolute difference between the acceptance probability and ``p(x̂)``.
    """
    accepted = _require_finite_probability(acceptance_probability, "a")
    drafted = _require_finite_probability(draft_probability, "p(x̂)")
    return abs(accepted - drafted)


def truncated_target_distribution(
    candidate_values: Sequence[float],
    candidate_token_ids: Sequence[int],
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    penalised_token_ids: Sequence[int] = (),
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Build the truncated sampling distribution from one verify row.

    Applies the presence penalty before truncation. With ``ncand`` candidates
    and a kept ``top_k`` this equals penalising the whole vocabulary first
    whenever at most ``ncand - top_k`` candidates are penalised, so the
    restriction is exact rather than approximate.
    """
    if len(candidate_values) != len(candidate_token_ids):
        raise MtpSamplerPolicyError("candidate values and token ids differ in length")
    if not candidate_values:
        raise MtpSamplerPolicyError("candidate list is empty")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise MtpSamplerPolicyError("temperature must be a number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise MtpSamplerPolicyError("temperature must be finite and positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise MtpSamplerPolicyError("top_k must be a non-negative integer")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise MtpSamplerPolicyError("top_p must be a number")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise MtpSamplerPolicyError("top_p must lie in (0, 1]")
    if (
        isinstance(presence_penalty, bool)
        or not isinstance(presence_penalty, (int, float))
        or not math.isfinite(float(presence_penalty))
        or float(presence_penalty) < 0.0
    ):
        raise MtpSamplerPolicyError("presence penalty must be finite and non-negative")

    penalty_ids = {int(token_id) for token_id in penalised_token_ids}
    scaled: list[tuple[float, int]] = []
    for raw_value, raw_token_id in zip(
        candidate_values,
        candidate_token_ids,
        strict=True,
    ):
        if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
            raise MtpSamplerPolicyError("candidate token ids must be integers")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise MtpSamplerPolicyError("candidate values must be numbers")
        value = float(raw_value)
        if not math.isfinite(value):
            raise MtpSamplerPolicyError("candidate values must be finite")
        token_id = int(raw_token_id)
        if token_id in penalty_ids:
            value -= float(presence_penalty)
        scaled.append((value / temperature, token_id))

    # Deterministic ordering: descending logit, ties by lower token id.
    scaled.sort(key=lambda item: (-item[0], item[1]))
    if 0 < top_k < len(scaled):
        scaled = scaled[:top_k]

    largest = scaled[0][0]
    exponentials = [math.exp(value - largest) for value, _ in scaled]
    total = sum(exponentials)
    probabilities = [weight / total for weight in exponentials]
    token_ids = [token_id for _, token_id in scaled]

    if top_p < 1.0:
        cumulative = 0.0
        kept = len(probabilities)
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if cumulative >= top_p:
                kept = index + 1
                break
        probabilities = probabilities[:kept]
        token_ids = token_ids[:kept]
        remainder = sum(probabilities)
        probabilities = [probability / remainder for probability in probabilities]

    return tuple(probabilities), tuple(token_ids)


def acceptance_probability(
    rule: str,
    distribution_probabilities: Sequence[float],
    distribution_token_ids: Sequence[int],
    draft_token_id: int,
    *,
    lenience: float = DEFAULT_LENIENCE,
    typical_epsilon: float = DEFAULT_TYPICAL_EPSILON,
    typical_delta: float = DEFAULT_TYPICAL_DELTA,
) -> tuple[float, float]:
    """Return ``(p(x̂), a)`` — the draft's target mass and the rule's accept
    probability for it.

    Splitting the two makes divergence measurable: ``realized_total_variation``
    needs the probability the rule uses, not the coin's outcome.
    """
    if rule not in ACCEPTANCE_RULES:
        joined = ", ".join(ACCEPTANCE_RULES)
        raise MtpSamplerPolicyError(
            f"acceptance rule must be one of {joined}, got {rule!r}"
        )
    if not distribution_probabilities:
        raise MtpSamplerPolicyError("distribution is empty")
    if len(distribution_probabilities) != len(distribution_token_ids):
        raise MtpSamplerPolicyError("distribution probabilities and ids differ")

    draft_mass = 0.0
    for probability, token_id in zip(
        distribution_probabilities,
        distribution_token_ids,
        strict=True,
    ):
        _require_finite_probability(probability, "distribution probability")
        if int(token_id) == int(draft_token_id):
            draft_mass = float(probability)

    if rule == RULE_GREEDY:
        return draft_mass, float(int(draft_token_id) == int(distribution_token_ids[0]))
    if rule == RULE_EXACT:
        # Point-mass draft: standard speculative sampling collapses to
        # rejection sampling, so the emitted stream is exactly p-distributed.
        return draft_mass, draft_mass
    if rule == RULE_LENIENT:
        if (
            isinstance(lenience, bool)
            or not isinstance(lenience, (int, float))
            or not math.isfinite(float(lenience))
            or float(lenience) <= 0.0
        ):
            raise MtpSamplerPolicyError("lenience must be finite and positive")
        return draft_mass, min(1.0, draft_mass / float(lenience))

    # RULE_RELAXED: Medusa-style typical acceptance.
    entropy = 0.0
    for probability in distribution_probabilities:
        clamped = max(float(probability), _PROBABILITY_FLOOR)
        entropy -= clamped * math.log(clamped)
    threshold = min(
        float(typical_epsilon),
        float(typical_delta) * math.exp(-entropy),
    )
    return draft_mass, float(draft_mass >= threshold)


def sample_from_distribution(
    distribution_probabilities: Sequence[float],
    distribution_token_ids: Sequence[int],
    uniform: float,
    *,
    exclude_token_id: int | None = None,
) -> int:
    """Inverse-CDF sample deterministic in ``uniform``, optionally from the
    renormalised residual obtained by removing ``exclude_token_id``."""
    _require_finite_probability(uniform, "uniform")
    probabilities = list(distribution_probabilities)
    token_ids = list(distribution_token_ids)
    if exclude_token_id is not None:
        kept = [
            (probability, token_id)
            for probability, token_id in zip(probabilities, token_ids, strict=True)
            if int(token_id) != int(exclude_token_id)
        ]
        if not kept:
            # Degenerate support after exclusion must not raise; fall back to
            # the best candidate that is not the excluded draft.
            fallback = [
                token_id
                for token_id in token_ids
                if int(token_id) != int(exclude_token_id)
            ]
            if not fallback:
                return int(distribution_token_ids[0])
            return int(fallback[0])
        probabilities = [probability for probability, _ in kept]
        token_ids = [token_id for _, token_id in kept]
        remainder = sum(probabilities)
        if remainder <= 0.0:
            return int(token_ids[0])
        probabilities = [probability / remainder for probability in probabilities]

    cumulative = 0.0
    last_index = len(probabilities) - 1
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if uniform * sum(probabilities) <= cumulative:
            return int(token_ids[index])
    return int(token_ids[last_index])


@dataclass(frozen=True)
class SpeculativeRoundOutcome:
    """One resolved speculative round.

    ``emitted_token_ids`` length is ``accepted_draft_count + 1``: accepting m
    drafts emits them plus one residual or bonus token, matching the greedy
    contract consumed by commit-style rollback.
    """

    emitted_token_ids: tuple[int, ...]
    accepted_draft_count: int
    bonus_emitted: bool
    draft_probabilities: tuple[float, ...]
    acceptance_probabilities: tuple[float, ...]
    realized_deviations: tuple[float, ...]


def resolve_speculative_round(
    candidate_values_per_row: Sequence[Sequence[float]],
    candidate_token_ids_per_row: Sequence[Sequence[int]],
    draft_token_ids: Sequence[int],
    uniforms: Callable[[], float],
    *,
    rule: str = RULE_EXACT,
    diagnostic_only: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    generated_token_ids: Sequence[int] = (),
    lenience: float = DEFAULT_LENIENCE,
    typical_epsilon: float = DEFAULT_TYPICAL_EPSILON,
    typical_delta: float = DEFAULT_TYPICAL_DELTA,
) -> SpeculativeRoundOutcome:
    """Resolve one whole speculative round. Pure: no device, no RNG state.

    Row ``i`` holds the target candidates for the token following verify input
    ``i+1``; row ``gamma`` is the bonus row. Drafts are point-mass argmax
    choices, so every rule shares one captured verify shape.
    """
    resolved_rule = resolve_rule(rule, diagnostic_only=diagnostic_only)
    gamma = len(draft_token_ids)
    if gamma == 0:
        raise MtpSamplerPolicyError("speculative rounds require at least one draft")
    if len(candidate_values_per_row) != gamma + 1:
        raise MtpSamplerPolicyError("verify row count must equal draft count plus one")
    if len(candidate_token_ids_per_row) != gamma + 1:
        raise MtpSamplerPolicyError("verify id row count must equal draft count plus one")

    emitted: list[int] = []
    seen: list[int] = [*generated_token_ids]
    draft_probabilities: list[float] = []
    acceptance_probabilities: list[float] = []
    deviations: list[float] = []

    def draw_uniform() -> float:
        try:
            value = uniforms()
        except StopIteration:
            raise MtpSamplerPolicyError(
                "uniform stream exhausted before the round resolved"
            ) from None
        _require_finite_probability(value, "uniform")
        return float(value)

    for index in range(gamma):
        probabilities, token_ids = truncated_target_distribution(
            candidate_values_per_row[index],
            candidate_token_ids_per_row[index],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            penalised_token_ids=seen + emitted,
        )
        draft_token_id = int(draft_token_ids[index])
        draft_mass, accepted = acceptance_probability(
            resolved_rule,
            probabilities,
            token_ids,
            draft_token_id,
            lenience=lenience,
            typical_epsilon=typical_epsilon,
            typical_delta=typical_delta,
        )
        draft_probabilities.append(draft_mass)
        acceptance_probabilities.append(accepted)
        deviations.append(realized_total_variation(accepted, draft_mass))
        if draw_uniform() < accepted:
            emitted.append(draft_token_id)
            continue
        emitted.append(
            sample_from_distribution(
                probabilities,
                token_ids,
                draw_uniform(),
                exclude_token_id=draft_token_id,
            )
        )
        return SpeculativeRoundOutcome(
            emitted_token_ids=tuple(emitted),
            accepted_draft_count=len(emitted) - 1,
            bonus_emitted=False,
            draft_probabilities=tuple(draft_probabilities),
            acceptance_probabilities=tuple(acceptance_probabilities),
            realized_deviations=tuple(deviations),
        )

    bonus_probabilities, bonus_token_ids = truncated_target_distribution(
        candidate_values_per_row[gamma],
        candidate_token_ids_per_row[gamma],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        penalised_token_ids=seen + emitted,
    )
    emitted.append(
        sample_from_distribution(
            bonus_probabilities,
            bonus_token_ids,
            draw_uniform(),
        )
    )
    return SpeculativeRoundOutcome(
        emitted_token_ids=tuple(emitted),
        accepted_draft_count=gamma,
        bonus_emitted=True,
        draft_probabilities=tuple(draft_probabilities),
        acceptance_probabilities=tuple(acceptance_probabilities),
        realized_deviations=tuple(deviations),
    )
