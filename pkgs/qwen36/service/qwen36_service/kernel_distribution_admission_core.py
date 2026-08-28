"""Pure distributional admission core for sampled-mode decode kernel changes.

r[impl tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

A decode execution-identity change (for example the fused GDN kernel) can keep
its speed while flipping late greedy tokens, because small fp32 recurrence
drift crosses near-tie argmax margins. Exact token parity stays the admission
gate for deterministic serving. This core owns the weaker, bounded gate that
may admit such a candidate for SAMPLED serving only: the emitted stream of a
temperature-sampled endpoint is stochastic, so its contract is closeness of
the token distribution, not token equality.

Every bound lives here as a named constant and again in the Nickel admission
contract. The core stays pure and device-free. It computes every metric from
per-position paired probe samples and fails closed on missing, short,
mismatched, or non-finite evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# The only serving mode a distributional receipt can make eligible. Greedy and
# deterministic requests keep the exact-parity path under the existing decode
# performance admission requirement.
ADMITTED_SERVING_MODE = "sampled"

# Probe corpus floor. Five prompts of at least forty-eight decode positions
# match the teacher-forced drift-probe discipline the field uses for KV and
# kernel precision changes.
MINIMUM_PROMPT_COUNT = 5
MINIMUM_POSITION_COUNT = 48

# Each side of one position supplies its top-K (token, probability, logit)
# triples. K=20 matches the model's recommended sampling top_k.
TOP_K_CANDIDATES = 20

# Teacher-forced logit closeness over the shared tokens of both top-K
# sets. Fewer shared tokens than this floor means the distributions are too
# far apart for correlation to be meaningful, so the probe is invalid.
MINIMUM_MEAN_LOGIT_PCC = 0.995
MINIMUM_MIN_LOGIT_PCC = 0.99
MINIMUM_PAIRED_LOGIT_SAMPLES = 8

# Truncated top-K total-variation bounds. TV is 0.5 * sum |p_control - p_cand|
# over the union of the two top-K token sets.
MAXIMUM_MEAN_TOPK_TV = 0.10
MAXIMUM_MAX_TOPK_TV = 0.25

# A side whose top-K carries less than half the probability mass makes the
# truncated comparison meaningless, so the probe is invalid.
MAXIMUM_MEAN_TOPK_RESIDUAL_MASS = 0.50

# Margin stratification. A position is a near tie when the control top1-top2
# logit margin is below this epsilon; near-tie flips are reported but never
# gated. Stable positions must keep their argmax.
NEAR_TIE_MARGIN_LOGITS = 0.5
MINIMUM_STABLE_AGREEMENT = 0.95
MINIMUM_STABLE_POSITION_COUNT = 120

# Matched-physical-root order for probe evidence, matching the house style of
# control/candidate pairing used by every other admission receipt.
REQUIRED_ROOT_ORDER = ("control", "candidate", "candidate", "control")

REQUIRED_PROBE_MODE = "teacher_forced"


class KernelAdmissionPolicyError(ValueError):
    """The distributional admission input violates its closed policy."""


@dataclass(frozen=True)
class PositionSample:
    """One teacher-forced decode position from one prompt."""

    control: tuple[tuple[int, float, float], ...]
    candidate: tuple[tuple[int, float, float], ...]

    def validate(self) -> None:
        if len(self.control) != TOP_K_CANDIDATES:
            raise KernelAdmissionPolicyError(
                f"control side must carry {TOP_K_CANDIDATES} triples, "
                f"got {len(self.control)}"
            )
        if len(self.candidate) != TOP_K_CANDIDATES:
            raise KernelAdmissionPolicyError(
                f"candidate side must carry {TOP_K_CANDIDATES} triples, "
                f"got {len(self.candidate)}"
            )
        for side in (self.control, self.candidate):
            for token, probability, logit in side:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise KernelAdmissionPolicyError("token ids must be integers")
                if not math.isfinite(probability) or not math.isfinite(logit):
                    raise KernelAdmissionPolicyError(
                        "probabilities and logits must be finite"
                    )
                if probability < 0.0:
                    raise KernelAdmissionPolicyError(
                        "probabilities must be non-negative"
                    )


@dataclass(frozen=True)
class PromptProbe:
    """Teacher-forced probe positions for one prompt."""

    positions: tuple[PositionSample, ...]


@dataclass(frozen=True)
class DistributionAdmissionEvidence:
    """Complete probe evidence for one execution-identity candidate."""

    probe_mode: str
    root_order: tuple[str, ...]
    prompts: tuple[PromptProbe, ...]

    def validate(self) -> None:
        if self.probe_mode != REQUIRED_PROBE_MODE:
            raise KernelAdmissionPolicyError(
                f"probe mode must be {REQUIRED_PROBE_MODE!r}, "
                f"got {self.probe_mode!r}"
            )
        if tuple(self.root_order) != REQUIRED_ROOT_ORDER:
            raise KernelAdmissionPolicyError(
                f"root order must be {REQUIRED_ROOT_ORDER}, "
                f"got {tuple(self.root_order)}"
            )
        if len(self.prompts) < MINIMUM_PROMPT_COUNT:
            raise KernelAdmissionPolicyError(
                f"probe needs at least {MINIMUM_PROMPT_COUNT} prompts, "
                f"got {len(self.prompts)}"
            )
        for prompt in self.prompts:
            if len(prompt.positions) < MINIMUM_POSITION_COUNT:
                raise KernelAdmissionPolicyError(
                    f"each prompt needs at least {MINIMUM_POSITION_COUNT} "
                    f"positions, got {len(prompt.positions)}"
                )
            for position in prompt.positions:
                position.validate()


@dataclass(frozen=True)
class AdmissionVerdict:
    """Fail-closed evaluation result for one candidate."""

    admitted: bool
    eligible_modes: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    mean_logit_pcc: float | None = None
    min_logit_pcc: float | None = None
    mean_topk_tv: float | None = None
    max_topk_tv: float | None = None
    mean_residual_mass: float | None = None
    near_tie_positions: int = 0
    stable_positions: int = 0
    stable_agreement: float | None = None
    near_tie_agreement: float | None = None
    detail: tuple[str, ...] = field(default_factory=tuple)


def _union_pcc(sample: PositionSample) -> float:
    """Pearson correlation of paired logits over shared top-K tokens.

    Only tokens present in BOTH top-K sets contribute a pair. A shared set
    smaller than the pairing floor means the two distributions differ too
    much for correlation to measure, so the probe is rejected as invalid.
    """
    control_logits = {token: logit for token, _, logit in sample.control}
    candidate_logits = {token: logit for token, _, logit in sample.candidate}
    shared = sorted(control_logits.keys() & candidate_logits.keys())
    if len(shared) < MINIMUM_PAIRED_LOGIT_SAMPLES:
        raise KernelAdmissionPolicyError(
            f"top-K sets share only {len(shared)} tokens; the pairing floor "
            f"is {MINIMUM_PAIRED_LOGIT_SAMPLES}"
        )
    control_values = [control_logits[token] for token in shared]
    candidate_values = [candidate_logits[token] for token in shared]
    return _pearson(control_values, candidate_values)


def _pearson(left: list[float], right: list[float]) -> float:
    count = len(left)
    mean_left = math.fsum(left) / count
    mean_right = math.fsum(right) / count
    covariance = math.fsum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    )
    variance_left = math.fsum((a - mean_left) ** 2 for a in left)
    variance_right = math.fsum((b - mean_right) ** 2 for b in right)
    if variance_left <= 0.0 or variance_right <= 0.0:
        raise KernelAdmissionPolicyError(
            "logit variance is zero; the probe cannot measure correlation"
        )
    return covariance / math.sqrt(variance_left * variance_right)


def _topk_tv(sample: PositionSample) -> float:
    """Total variation between the two truncated top-K distributions."""
    control_mass = {token: probability for token, probability, _ in sample.control}
    candidate_mass = {token: probability for token, probability, _ in sample.candidate}
    union = sorted(control_mass.keys() | candidate_mass.keys())
    return 0.5 * math.fsum(
        abs(control_mass.get(token, 0.0) - candidate_mass.get(token, 0.0))
        for token in union
    )


def _residual_mass(sample: PositionSample) -> float:
    """Mean top-K residual mass across both sides of one position."""
    control_residual = 1.0 - math.fsum(p for _, p, _ in sample.control)
    candidate_residual = 1.0 - math.fsum(p for _, p, _ in sample.candidate)
    return 0.5 * (control_residual + candidate_residual)


def _control_margin(sample: PositionSample) -> float:
    """Control top1-top2 logit margin for near-tie stratification."""
    logits = sorted((logit for _, _, logit in sample.control), reverse=True)
    return logits[0] - logits[1]


def _top1(sample: PositionSample, side: str) -> int:
    triples = sample.control if side == "control" else sample.candidate
    return max(triples, key=lambda triple: (triple[2], -triple[0]))[0]


def evaluate_distribution_admission(
    evidence: DistributionAdmissionEvidence,
) -> AdmissionVerdict:
    """Evaluate one candidate against every distributional bound.

    The verdict fails closed: any invalid input, missing metric, or missed
    bound rejects the candidate and keeps deterministic serving on the
    exact-parity path. A passing verdict makes the candidate eligible for
    ADMITTED_SERVING_MODE only.
    """
    try:
        evidence.validate()
    except KernelAdmissionPolicyError as error:
        return AdmissionVerdict(
            admitted=False,
            eligible_modes=(),
            rejection_reasons=(f"invalid evidence: {error}",),
        )

    position_pcc: list[float] = []
    position_tv: list[float] = []
    position_residual: list[float] = []
    near_tie_matches = 0
    near_tie_total = 0
    stable_matches = 0
    stable_total = 0

    for prompt in evidence.prompts:
        for sample in prompt.positions:
            try:
                position_pcc.append(_union_pcc(sample))
            except KernelAdmissionPolicyError as error:
                return AdmissionVerdict(
                    admitted=False,
                    eligible_modes=(),
                    rejection_reasons=(f"invalid position probe: {error}",),
                )
            position_tv.append(_topk_tv(sample))
            position_residual.append(_residual_mass(sample))
            control_top1 = _top1(sample, "control")
            candidate_top1 = _top1(sample, "candidate")
            matches = control_top1 == candidate_top1
            if _control_margin(sample) < NEAR_TIE_MARGIN_LOGITS:
                near_tie_total += 1
                near_tie_matches += int(matches)
            else:
                stable_total += 1
                stable_matches += int(matches)

    mean_pcc = math.fsum(position_pcc) / len(position_pcc)
    min_pcc = min(position_pcc)
    mean_tv = math.fsum(position_tv) / len(position_tv)
    max_tv = max(position_tv)
    mean_residual = math.fsum(position_residual) / len(position_residual)

    reasons: list[str] = []
    if mean_pcc < MINIMUM_MEAN_LOGIT_PCC:
        reasons.append(
            f"mean union logit PCC {mean_pcc:.6f} < {MINIMUM_MEAN_LOGIT_PCC}"
        )
    if min_pcc < MINIMUM_MIN_LOGIT_PCC:
        reasons.append(
            f"min union logit PCC {min_pcc:.6f} < {MINIMUM_MIN_LOGIT_PCC}"
        )
    if mean_tv > MAXIMUM_MEAN_TOPK_TV:
        reasons.append(f"mean top-K TV {mean_tv:.6f} > {MAXIMUM_MEAN_TOPK_TV}")
    if max_tv > MAXIMUM_MAX_TOPK_TV:
        reasons.append(f"max top-K TV {max_tv:.6f} > {MAXIMUM_MAX_TOPK_TV}")
    if mean_residual > MAXIMUM_MEAN_TOPK_RESIDUAL_MASS:
        reasons.append(
            f"mean top-K residual mass {mean_residual:.6f} > "
            f"{MAXIMUM_MEAN_TOPK_RESIDUAL_MASS}"
        )
    if stable_total < MINIMUM_STABLE_POSITION_COUNT:
        reasons.append(
            f"stable positions {stable_total} < {MINIMUM_STABLE_POSITION_COUNT}"
        )

    stable_agreement = stable_matches / stable_total if stable_total else None
    near_tie_agreement = (
        near_tie_matches / near_tie_total if near_tie_total else None
    )
    if stable_agreement is not None and stable_agreement < MINIMUM_STABLE_AGREEMENT:
        reasons.append(
            f"stable-position agreement {stable_agreement:.6f} < "
            f"{MINIMUM_STABLE_AGREEMENT}"
        )

    return AdmissionVerdict(
        admitted=not reasons,
        eligible_modes=(ADMITTED_SERVING_MODE,) if not reasons else (),
        rejection_reasons=tuple(reasons),
        mean_logit_pcc=mean_pcc,
        min_logit_pcc=min_pcc,
        mean_topk_tv=mean_tv,
        max_topk_tv=max_tv,
        mean_residual_mass=mean_residual,
        near_tie_positions=near_tie_total,
        stable_positions=stable_total,
        stable_agreement=stable_agreement,
        near_tie_agreement=near_tie_agreement,
        detail=tuple(
            f"pcc={value:.6f}" for value in (mean_pcc, min_pcc, mean_tv, max_tv)
        ),
    )
