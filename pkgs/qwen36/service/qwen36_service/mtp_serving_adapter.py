"""Request-scoped orchestration for exact greedy speculative rounds."""

import time
from contextlib import ExitStack

import ttnn

from .mtp_serving_core import (
    ROUND_MODE_SPECULATIVE,
    ROUND_WATCHDOG_SECONDS,
    RoundEconomics,
    advance_round_economics,
    commit_verified_round,
    plan_round,
)
from .mtp_state_snapshot_adapter import MtpStateSnapshot, target_state_tensors
from .mtp_trace_adapter import TracedMtpDraftAdapter
from .mtp_watchdog_adapter import run_device_stage


def require_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimeError("request deadline expired")


def restore_target_prefix(owner, snapshot, current, drafts, commit, position, deadline):
    """Restore every state tensor, then rebuild only the committed input prefix.

    This reference recovery trades latency for exact KV and recurrent state.
    It does not rely on masked stale KV rows or incomplete prefix snapshots.
    """
    snapshot.restore()
    replayed = []
    for offset, token in enumerate((current, *drafts[: commit.accepted])):
        require_deadline(deadline)
        replayed.append(
            owner.decode_adapter.decode_eager(token=token, position=position + offset)
        )
    if tuple(replayed) != commit.committed_token_ids:
        raise RuntimeError(
            "speculative verification differs from sequential prefix replay"
        )


def advance_draft_prefix(trace, *, token_inputs, feedback, position):
    """Consume accepted inputs and the bonus with the head's shifted positions."""
    result = None
    for offset, token in enumerate(token_inputs):
        result = trace.replay_token_inputs(
            token_inputs=(token,),
            initial_feedback=feedback,
            start_position=position + offset,
        )
        feedback = result.feedback
    if result is None:
        raise RuntimeError("speculative draft continuation is empty")
    return result.token_ids[-1], feedback


def run_rounds(
    owner,
    *,
    anchor,
    first_draft,
    feedback,
    page_table,
    prompt_length,
    maximum_generation_tokens,
    request_deadline,
    plain_step_seconds,
):
    """Own snapshots and traces for one request. Never retain them for reuse."""
    width = owner.profile.draft_token_count
    generated = [anchor]
    current = anchor
    position = prompt_length
    economics = RoundEconomics()
    accepted_total = 0
    rejected_rounds = 0
    round_durations = []
    round_stage_seconds = []
    verification_margins = []
    disengaged_reason = None
    capture_seconds = 0.0
    verifier = owner._native_verifier
    trace = None

    with ExitStack() as cleanup:
        target = MtpStateSnapshot(lambda: target_state_tensors(owner.model))
        cleanup.callback(target.close)
        saved_feedback = ttnn.clone(feedback)
        cleanup.callback(ttnn.deallocate, saved_feedback)
        cleanup.callback(verifier.close_persistent)
        require_deadline(request_deadline)
        capture_seconds += run_device_stage(
            lambda: verifier.capture_persistent(
                (anchor, *((0,) * width)),
                start_position=position,
                page_table=page_table,
            ),
            seconds=verifier.trace_capture_timeout_seconds,
            stage="verify-capture",
        )
        # Capture restores GDN state itself, but it also writes paged KV.
        target.restore()
        ttnn.synchronize_device(owner.mesh)
        trace = TracedMtpDraftAdapter(
            owner=owner,
            maximum_trace_capture_seconds=owner.trace_profile.maximum_trace_capture_seconds,
            persistent=True,
        )
        cleanup.callback(trace.close)

        while len(generated) < maximum_generation_tokens:
            require_deadline(request_deadline)
            plan = plan_round(
                tokens_generated=len(generated),
                maximum_generation_tokens=maximum_generation_tokens,
                draft_token_count=width,
            )
            if plan.mode != ROUND_MODE_SPECULATIVE:
                break
            round_started = time.monotonic()
            ttnn.copy(feedback, saved_feedback)
            replay = run_device_stage(
                lambda: trace.run(
                    initial_token=first_draft,
                    initial_feedback=feedback,
                    start_position=position,
                    step_count=width - 1,
                ),
                seconds=ROUND_WATCHDOG_SECONDS,
                stage="draft",
            )
            draft_seconds = time.monotonic() - round_started
            if not (
                replay.trace_binding_valid
                and replay.capture_idempotence_valid
                and replay.mtp_kv_prefix_valid
            ):
                raise RuntimeError(
                    "speculative draft trace failed its exactness checks"
                )
            drafts = (first_draft, *replay.token_ids)
            require_deadline(request_deadline)
            verify_started = time.monotonic()
            verification = run_device_stage(
                lambda: verifier.replay_round(
                    (current, *drafts), start_position=position
                ),
                seconds=ROUND_WATCHDOG_SECONDS,
                stage="verify",
            )
            verify_seconds = time.monotonic() - verify_started
            commit = commit_verified_round(
                draft_token_ids=drafts, continuation_token_ids=verification.token_ids
            )
            verification_margins.append(verification.minimum_top_two_margin)
            rollback_seconds = 0.0
            if commit.rejected:
                rejected_rounds += 1
                rollback_started = time.monotonic()
                run_device_stage(
                    lambda: verifier.rollback_to_prefix(commit.accepted),
                    seconds=ROUND_WATCHDOG_SECONDS,
                    stage="target-rollback",
                )
                rollback_seconds = time.monotonic() - rollback_started
                continuation_inputs = (
                    *drafts[: commit.accepted],
                    commit.committed_token_ids[-1],
                )
                continuation_feedback = saved_feedback
                continuation_position = position
            else:
                # Draft generation did not consume its final output token.
                continuation_inputs = (drafts[-1], commit.committed_token_ids[-1])
                continuation_feedback = trace._trace_feedback
                continuation_position = position + width - 1
            require_deadline(request_deadline)
            head_commit_started = time.monotonic()
            first_draft, feedback = run_device_stage(
                lambda: advance_draft_prefix(
                    trace,
                    token_inputs=continuation_inputs,
                    feedback=continuation_feedback,
                    position=continuation_position,
                ),
                seconds=ROUND_WATCHDOG_SECONDS,
                stage="head-commit",
            )
            head_commit_seconds = time.monotonic() - head_commit_started
            ttnn.synchronize_device(owner.mesh)
            require_deadline(request_deadline)
            duration = time.monotonic() - round_started - replay.trace_capture_seconds
            capture_seconds += replay.trace_capture_seconds
            round_durations.append(duration)
            round_stage_seconds.append({
                "draft": round(draft_seconds, 6),
                "verify": round(verify_seconds, 6),
                "rollback": round(rollback_seconds, 6),
                "head_commit": round(head_commit_seconds, 6),
                "emitted": len(commit.committed_token_ids),
                "accepted": commit.accepted,
            })
            generated.extend(commit.committed_token_ids)
            accepted_total += commit.accepted
            current = commit.committed_token_ids[-1]
            position += len(commit.committed_token_ids)
            economics = advance_round_economics(
                economics,
                round_seconds=duration,
                emitted_tokens=len(commit.committed_token_ids),
                plain_step_seconds=plain_step_seconds,
            )
            if economics.disengaged:
                disengaged_reason = "round-economics"
                break

        # No large decode trace is captured while a speculative trace resides.
        trace.close()
        verifier.close_persistent()
        if len(generated) < maximum_generation_tokens:
            require_deadline(request_deadline)
            eligibility = run_device_stage(
                lambda: owner.decode_adapter.ensure_trace(
                    token=current, position=position
                ),
                seconds=ROUND_WATCHDOG_SECONDS,
                stage="plain-capture",
            )
            if not eligibility.eligible:
                raise RuntimeError(
                    f"speculative plain fallback is unavailable: {eligibility.reason}"
                )
            while len(generated) < maximum_generation_tokens:
                require_deadline(request_deadline)
                current = run_device_stage(
                    lambda: owner.decode_adapter.decode_traced(
                        token=current, position=position
                    ),
                    seconds=ROUND_WATCHDOG_SECONDS,
                    stage="plain-step",
                )
                require_deadline(request_deadline)
                generated.append(current)
                position += 1
    return {
        "token_ids": tuple(generated),
        "rounds": len(round_durations),
        "accepted_total": accepted_total,
        "rejected_rounds": rejected_rounds,
        "disengaged_reason": disengaged_reason,
        "round_seconds": tuple(round_durations),
        "round_stage_seconds": tuple(round_stage_seconds),
        "trace_capture_seconds": capture_seconds,
        "minimum_verification_margin": (
            min(verification_margins) if verification_margins else None
        ),
    }
