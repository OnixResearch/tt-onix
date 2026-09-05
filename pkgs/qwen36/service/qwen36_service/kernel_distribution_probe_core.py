"""Pure teacher-forced probe builder for the sampled-mode kernel admission gate.

r[impl tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

The device screen (Phase 3) runs paired control and candidate roots and reads
back full decode logits per teacher-forced position. This core is the pure
half: it turns those raw logit readbacks into top-K triples with full-softmax
probabilities, assembles the admission evidence, evaluates it with the
distributional admission core, and emits a deterministic receipt. Softmax over
the full readback keeps the truncated top-K residual mass measurable, so the
residual bound stays meaningful.

The core stays device-free and owns no I/O. Digests follow the house
pattern: the core emits the canonical corpus payload string, the shell hashes
it with b3sum, and the core validates the returned BLAKE3 hex before it can
enter a receipt, so a receipt cannot be replayed against a different corpus.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .kernel_distribution_admission_core import (
    ADMITTED_SERVING_MODE,
    MINIMUM_POSITION_COUNT,
    MINIMUM_PROMPT_COUNT,
    REQUIRED_PROBE_MODE,
    REQUIRED_ROOT_ORDER,
    TOP_K_CANDIDATES,
    AdmissionVerdict,
    DistributionAdmissionEvidence,
    PositionSample,
    PromptProbe,
    evaluate_distribution_admission,
)

RECEIPT_SCHEMA = "onix.qwen36.kernel-distribution-probe.v1"

# The readback must cover more than the truncated top-K set, otherwise the
# residual-mass bound is unmeasurable and the probe is invalid.
MINIMUM_READBACK_VOCAB = TOP_K_CANDIDATES + 1

BLAKE3_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProbeBuildError(ValueError):
    """The probe readback violates its closed input policy."""


@dataclass(frozen=True)
class PositionReadback:
    """Full-vocab logit readback for one position of one root."""

    tokens: tuple[int, ...]
    logits: tuple[float, ...]

    def validate(self) -> None:
        if len(self.tokens) != len(self.logits):
            raise ProbeBuildError("token and logit counts differ")
        if len(self.tokens) < MINIMUM_READBACK_VOCAB:
            raise ProbeBuildError(
                f"readback vocab {len(self.tokens)} is smaller than the "
                f"minimum {MINIMUM_READBACK_VOCAB}"
            )
        seen: set[int] = set()
        for token, logit in zip(self.tokens, self.logits, strict=True):
            if isinstance(token, bool) or not isinstance(token, int):
                raise ProbeBuildError("token ids must be integers")
            if token in seen:
                raise ProbeBuildError(f"duplicate token id {token}")
            seen.add(token)
            if not math.isfinite(logit):
                raise ProbeBuildError("logits must be finite")


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    peak = max(values)
    weights = tuple(math.exp(value - peak) for value in values)
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def _top_k_triples(readback: PositionReadback) -> tuple[tuple[int, float, float], ...]:
    probabilities = _softmax(readback.logits)
    pairs = sorted(
        zip(readback.tokens, probabilities, readback.logits, strict=True),
        key=lambda triple: (-triple[2], triple[0]),
    )
    return tuple(
        (token, probability, logit)
        for token, probability, logit in pairs[:TOP_K_CANDIDATES]
    )


def build_position_sample(
    control: PositionReadback, candidate: PositionReadback
) -> PositionSample:
    """Build one admission position from paired full-vocab readbacks."""
    control.validate()
    candidate.validate()
    if len(control.tokens) != len(candidate.tokens):
        raise ProbeBuildError(
            "paired readbacks must share one tokenizer vocabulary length"
        )
    return PositionSample(
        control=_top_k_triples(control),
        candidate=_top_k_triples(candidate),
    )


def build_prompt_probe(
    positions: tuple[tuple[PositionReadback, PositionReadback], ...],
) -> PromptProbe:
    """Build one prompt probe from its paired position readbacks."""
    if not positions:
        raise ProbeBuildError("a prompt probe needs at least one position")
    return PromptProbe(
        positions=tuple(
            build_position_sample(control, candidate)
            for control, candidate in positions
        )
    )


def canonical_corpus_payload(
    corpus: tuple[tuple[tuple[PositionReadback, PositionReadback], ...], ...],
) -> str:
    """Deterministic serialization of the raw probe corpus for b3sum."""
    return json.dumps(
        [
            [
                {
                    "candidate_logits": list(candidate.logits),
                    "candidate_tokens": list(candidate.tokens),
                    "control_logits": list(control.logits),
                    "control_tokens": list(control.tokens),
                }
                for control, candidate in positions
            ]
            for positions in corpus
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def require_corpus_blake3(value: str) -> str:
    """Validate the shell-computed b3sum digest before it enters a receipt."""
    if not isinstance(value, str) or BLAKE3_HEX_PATTERN.fullmatch(value) is None:
        raise ProbeBuildError(
            "corpus_blake3 must be 64 lowercase hex characters from b3sum"
        )
    return value


def evaluate_probe(
    *,
    probe_mode: str,
    root_order: tuple[str, ...],
    corpus: tuple[tuple[tuple[PositionReadback, PositionReadback], ...], ...],
    corpus_blake3: str,
) -> dict:
    """Evaluate paired readbacks end to end and emit the deterministic receipt.

    Any invalid input raises ProbeBuildError instead of producing a receipt,
    so a malformed readback or digest can never yield an admitting receipt.
    """
    require_corpus_blake3(corpus_blake3)
    if probe_mode != REQUIRED_PROBE_MODE:
        raise ProbeBuildError(
            f"probe mode must be {REQUIRED_PROBE_MODE!r}, got {probe_mode!r}"
        )
    if tuple(root_order) != REQUIRED_ROOT_ORDER:
        raise ProbeBuildError(
            f"root order must be {REQUIRED_ROOT_ORDER}, got {tuple(root_order)}"
        )
    if len(corpus) < MINIMUM_PROMPT_COUNT:
        raise ProbeBuildError(
            f"probe corpus needs at least {MINIMUM_PROMPT_COUNT} prompts, "
            f"got {len(corpus)}"
        )
    for positions in corpus:
        if len(positions) < MINIMUM_POSITION_COUNT:
            raise ProbeBuildError(
                f"each prompt needs at least {MINIMUM_POSITION_COUNT} "
                f"positions, got {len(positions)}"
            )

    evidence = DistributionAdmissionEvidence(
        probe_mode=probe_mode,
        root_order=tuple(root_order),
        prompts=tuple(build_prompt_probe(positions) for positions in corpus),
    )
    verdict = evaluate_distribution_admission(evidence)
    return _receipt(verdict, corpus, probe_mode, root_order, corpus_blake3)


def _receipt(
    verdict: AdmissionVerdict,
    corpus: tuple[tuple[tuple[PositionReadback, PositionReadback], ...], ...],
    probe_mode: str,
    root_order: tuple[str, ...],
    corpus_blake3: str,
) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "probe_mode": probe_mode,
        "root_order": list(root_order),
        "prompt_count": len(corpus),
        "position_count": min(len(positions) for positions in corpus),
        "top_k": TOP_K_CANDIDATES,
        "eligible_modes": list(verdict.eligible_modes),
        "admitted": verdict.admitted,
        "admitted_serving_mode": ADMITTED_SERVING_MODE,
        "rejection_reasons": list(verdict.rejection_reasons),
        "corpus_blake3": corpus_blake3,
        "metrics": {
            "mean_logit_pcc": verdict.mean_logit_pcc,
            "min_logit_pcc": verdict.min_logit_pcc,
            "mean_topk_tv": verdict.mean_topk_tv,
            "max_topk_tv": verdict.max_topk_tv,
            "mean_residual_mass": verdict.mean_residual_mass,
            "near_tie_positions": verdict.near_tie_positions,
            "stable_positions": verdict.stable_positions,
            "stable_agreement": verdict.stable_agreement,
            "near_tie_agreement": verdict.near_tie_agreement,
        },
    }
