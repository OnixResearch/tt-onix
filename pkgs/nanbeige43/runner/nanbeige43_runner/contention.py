"""Contention screening for measured bench runs (P100/P101).

Two distinct signatures were observed in one batch-16 session: a run whose first
token and decode rate both collapsed, and a run whose first token alone inflated
while decode stayed within 12 %. Host contention explains the first - a compile or
trace cost cannot slow steady decode fourfold - so a run that slow is excluded from
the admitted set instead of being averaged into a median. The second is only flagged,
because its cause is not established and dropping it would hide a real stall.

Both baselines come from the healthy tail. Contention only ever slows a run, so the
fastest decode and the shortest first token are the honest references; a median
baseline is self-defeating when most runs are contaminated, which is exactly how a
29.2 s first token escaped a median-based rule in P99.

This module is deliberately free of device imports so the device-free check suite can
test the rule against recorded measurements.
"""

from __future__ import annotations

DECODE_CONTENDED_RATIO = 0.6
FIRST_TOKEN_OUTLIER_FACTOR = 3.0


def flag_runs(runs: list[dict]) -> dict:
    """Annotate measured runs in place; return the flag counts.

    Requires at least two runs, since a single run has nothing to be compared with.
    Each run must carry ``first_token_seconds`` and
    ``decode_tokens_per_second``.
    """
    flags = {"decode_contended": 0, "first_token_outlier": 0}
    if len(runs) < 2:
        return flags
    best_decode = max(run["decode_tokens_per_second"] for run in runs)
    min_ttft = min(run["first_token_seconds"] for run in runs)
    for run in runs:
        ratio = run["decode_tokens_per_second"] / best_decode
        run["decode_ratio"] = ratio
        run["decode_contended"] = ratio < DECODE_CONTENDED_RATIO
        run["first_token_outlier"] = (
            run["first_token_seconds"] > FIRST_TOKEN_OUTLIER_FACTOR * min_ttft
        )
        flags["decode_contended"] += int(run["decode_contended"])
        flags["first_token_outlier"] += int(run["first_token_outlier"])
    return flags


def admitted(runs: list[dict]) -> list[dict]:
    """The measured runs a summary may use: the ones not decode-contended.

    Falls back to every run when all of them were contended, so a summary still
    exists and the flags show what happened.
    """
    clean = [run for run in runs if not run.get("decode_contended")]
    return clean or list(runs)
