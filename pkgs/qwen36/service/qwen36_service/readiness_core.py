"""Pure startup warm-up and readiness policy for the Qwen service."""

# r[impl tenstorrent.qwen38_p150x2.startup_readiness]

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

MINIMUM_WARMUP_GENERATION_TOKENS = 2


class StartupReadinessError(ValueError):
    """The startup warm-up plan or transition is invalid."""


class StartupPhase(Enum):
    PLANNED = "planned"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class StartupWarmupPlan:
    widths: tuple[int, ...]
    prompt: str
    generation_tokens: int


@dataclass(frozen=True)
class StartupReadiness:
    plan: StartupWarmupPlan
    phase: StartupPhase
    completed_widths: tuple[int, ...]
    width_elapsed_seconds: tuple[float, ...]
    total_elapsed_seconds: float | None

    @property
    def ready(self) -> bool:
        return self.phase is StartupPhase.READY


def plan_startup_warmup(
    *,
    widths: tuple[int, ...],
    prompt: str,
    generation_tokens: int,
    maximum_width: int,
) -> StartupWarmupPlan:
    if not widths:
        raise StartupReadinessError("startup warm-up widths must not be empty")
    if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
        raise StartupReadinessError("startup warm-up widths must be integers")
    if tuple(sorted(set(widths))) != widths:
        raise StartupReadinessError(
            "startup warm-up widths must be unique and increasing"
        )
    if widths[0] < 1 or widths[-1] > maximum_width:
        raise StartupReadinessError("startup warm-up width is outside capacity")
    if not isinstance(prompt, str) or not prompt.strip():
        raise StartupReadinessError("startup warm-up prompt must be non-empty")
    if (
        isinstance(generation_tokens, bool)
        or not isinstance(generation_tokens, int)
        or generation_tokens < MINIMUM_WARMUP_GENERATION_TOKENS
    ):
        raise StartupReadinessError("startup warm-up must generate at least two tokens")
    return StartupWarmupPlan(
        widths=widths,
        prompt=prompt,
        generation_tokens=generation_tokens,
    )


def initial_readiness(plan: StartupWarmupPlan) -> StartupReadiness:
    return StartupReadiness(
        plan=plan,
        phase=StartupPhase.PLANNED,
        completed_widths=(),
        width_elapsed_seconds=(),
        total_elapsed_seconds=None,
    )


def begin_warmup(state: StartupReadiness) -> StartupReadiness:
    if state.phase is not StartupPhase.PLANNED:
        raise StartupReadinessError("startup warm-up can begin only from planned")
    return StartupReadiness(
        plan=state.plan,
        phase=StartupPhase.WARMING,
        completed_widths=(),
        width_elapsed_seconds=(),
        total_elapsed_seconds=None,
    )


def record_warm_width(
    state: StartupReadiness,
    *,
    width: int,
    elapsed_seconds: float,
) -> StartupReadiness:
    if state.phase is not StartupPhase.WARMING:
        raise StartupReadinessError("warm width requires the warming phase")
    expected_index = len(state.completed_widths)
    if expected_index >= len(state.plan.widths):
        raise StartupReadinessError("all planned warm-up widths are already complete")
    if width != state.plan.widths[expected_index]:
        raise StartupReadinessError("startup warm-up width completed out of order")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, int | float)
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds <= 0
    ):
        raise StartupReadinessError("startup warm-up duration must be positive")
    return StartupReadiness(
        plan=state.plan,
        phase=state.phase,
        completed_widths=(*state.completed_widths, width),
        width_elapsed_seconds=(
            *state.width_elapsed_seconds,
            float(elapsed_seconds),
        ),
        total_elapsed_seconds=None,
    )


def complete_warmup(
    state: StartupReadiness,
    *,
    total_elapsed_seconds: float,
) -> StartupReadiness:
    if state.phase is not StartupPhase.WARMING:
        raise StartupReadinessError("startup can become ready only from warming")
    if state.completed_widths != state.plan.widths:
        raise StartupReadinessError("startup warm-up widths are incomplete")
    minimum_elapsed_seconds = sum(state.width_elapsed_seconds)
    if (
        isinstance(total_elapsed_seconds, bool)
        or not isinstance(total_elapsed_seconds, int | float)
        or not math.isfinite(float(total_elapsed_seconds))
        or total_elapsed_seconds < minimum_elapsed_seconds
    ):
        raise StartupReadinessError("startup total duration is invalid")
    return StartupReadiness(
        plan=state.plan,
        phase=StartupPhase.READY,
        completed_widths=state.completed_widths,
        width_elapsed_seconds=state.width_elapsed_seconds,
        total_elapsed_seconds=float(total_elapsed_seconds),
    )


def fail_warmup(state: StartupReadiness) -> StartupReadiness:
    if state.phase is StartupPhase.READY:
        raise StartupReadinessError("ready startup cannot transition to failed")
    return StartupReadiness(
        plan=state.plan,
        phase=StartupPhase.FAILED,
        completed_widths=state.completed_widths,
        width_elapsed_seconds=state.width_elapsed_seconds,
        total_elapsed_seconds=None,
    )
