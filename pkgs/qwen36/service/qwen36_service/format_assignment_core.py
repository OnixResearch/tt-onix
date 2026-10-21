"""Exact mixed-precision assignment under an average effective-bit budget.

This is the assigning half of an AutoQuantize-style search, with the TT
block-float formats supplied by the caller. It does not assume NVIDIA formats
or a CUDA dependency: each candidate states its effective bits (scales
included) and its measured sensitivity.

Input: groups of operators that share one format decision. Coupled operators
are already aggregated by the caller, as ModelOpt does for QKV and all routed
experts of a layer. The constraint is

    sum_i parameters_i * bits(i, f_i) / sum_i parameters_i <= budget,

and the objective is the sum of the groups' sensitivity scores. A Pareto
frontier over the weighted bit numerator solves the multiple-choice knapsack
exactly; dominated (cost, sensitivity) states are pruned at every group.

The sensitivity values are an input, not something this module measures. The
producer runs a reference pass with the same TT block-float format: quantize one
group at a time, score the output KL against the BF16 reference on
teacher-forced rows, and aggregate operators that must share a format. The
resulting assignment is a recommendation; the existing component PCC, GDN
state-carry and teacher-forced decode-logit gates remain the acceptance
authority. This tool does not depend on CUDA or NVIDIA quantized formats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

MAXIMUM_PARETO_STATES = 200_000


class FormatAssignmentError(ValueError):
    """The assignment input is not a closed, solvable budget problem."""


@dataclass(frozen=True)
class FormatOption:
    name: str
    effective_bits: float
    sensitivity: float


@dataclass(frozen=True)
class FormatGroup:
    name: str
    parameters: int
    options: tuple[FormatOption, ...]


@dataclass(frozen=True)
class FormatAssignment:
    formats: tuple[str, ...]
    effective_bits: float
    sensitivity: float


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatAssignmentError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise FormatAssignmentError(f"{label} must be finite")
    return value


def solve_assignment(
    groups: tuple[FormatGroup, ...],
    *,
    budget_effective_bits: float,
) -> FormatAssignment:
    """Choose one format per group with minimum summed sensitivity."""
    if not isinstance(groups, tuple) or not groups:
        raise FormatAssignmentError("groups must be a non-empty tuple")
    budget = _finite(budget_effective_bits, "budget_effective_bits")
    if budget <= 0:
        raise FormatAssignmentError("budget_effective_bits must be positive")
    names: set[str] = set()
    for group in groups:
        if not isinstance(group, FormatGroup):
            raise FormatAssignmentError("groups must hold FormatGroup values")
        if not group.name or group.name in names:
            raise FormatAssignmentError("group names must be non-empty and unique")
        names.add(group.name)
        if (
            isinstance(group.parameters, bool)
            or not isinstance(group.parameters, int)
            or group.parameters <= 0
        ):
            raise FormatAssignmentError("group parameters must be positive integers")
        if not group.options:
            raise FormatAssignmentError("a group needs at least one format")
        formats: set[str] = set()
        for option in group.options:
            if not isinstance(option, FormatOption):
                raise FormatAssignmentError("options must hold FormatOption values")
            if not option.name or option.name in formats:
                raise FormatAssignmentError("format names must be non-empty and unique")
            formats.add(option.name)
            bits = _finite(option.effective_bits, f"{group.name} {option.name} bits")
            sensitivity = _finite(
                option.sensitivity, f"{group.name} {option.name} sensitivity"
            )
            if bits <= 0:
                raise FormatAssignmentError("effective bits must be positive")
            if sensitivity < 0:
                raise FormatAssignmentError("sensitivity must not be negative")

    total_parameters = sum(group.parameters for group in groups)
    # Costs are exact weighted-bit numerators; `bits` may be 4.5.
    budget_units = Fraction(budget) * total_parameters
    # (cost numerator, sensitivity, chosen option indices)
    states: list[tuple[Fraction, float, tuple[int, ...]]] = [(Fraction(0), 0.0, ())]
    for group in groups:
        candidates: list[tuple[Fraction, float, tuple[int, ...]]] = []
        for cost, sensitivity, choices in states:
            for index, option in enumerate(group.options):
                next_cost = cost + group.parameters * Fraction(option.effective_bits)
                if next_cost <= budget_units:
                    candidates.append((
                        next_cost,
                        sensitivity + option.sensitivity,
                        choices + (index,),
                    ))
        if not candidates:
            raise FormatAssignmentError(
                f"no assignment fits {budget_effective_bits} effective bits"
            )
        # Exact Pareto pruning: lower cost is better, lower sensitivity is better.
        candidates.sort(key=lambda state: (state[0], state[1], state[2]))
        states = []
        best_sensitivity = math.inf
        for state in candidates:
            if state[1] < best_sensitivity:
                states.append(state)
                best_sensitivity = state[1]
        if len(states) > MAXIMUM_PARETO_STATES:
            raise FormatAssignmentError(
                "the Pareto frontier exceeds "
                f"{MAXIMUM_PARETO_STATES} states; group more operators"
            )

    cost, sensitivity, choices = min(
        states, key=lambda state: (state[1], state[0], state[2])
    )
    return FormatAssignment(
        formats=tuple(
            group.options[index].name for group, index in zip(groups, choices)
        ),
        effective_bits=float(cost / total_parameters),
        sensitivity=sensitivity,
    )
