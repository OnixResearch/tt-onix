"""Pure target-validation signature for the Qwen MTP prewarm control."""

# r[impl tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MINIMUM_TOKEN_ID = 0
MINIMUM_ACCEPTED_PREFIX_LENGTH = 0
CONTINUATION_TOKEN_COUNT = 1


class MtpTargetPrewarmError(ValueError):
    """The target prewarm result cannot support an exact comparison."""


@dataclass(frozen=True)
class MtpTargetValidationSignature:
    """Exact prompt-local facts that select the target validation shape."""

    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    target_verification_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    committed_width: int
    rollback_applied: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "draft_token_ids": list(self.draft_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "target_verification_token_ids": list(self.target_verification_token_ids),
            "accepted_prefix_length": self.accepted_prefix_length,
            "committed_width": self.committed_width,
            "rollback_applied": self.rollback_applied,
        }


def make_target_validation_signature(
    *,
    draft_token_ids: tuple[int, ...],
    target_token_ids: tuple[int, ...],
    target_verification_token_ids: tuple[int, ...],
    accepted_prefix_length: int,
    rollback_applied: bool,
    target_output_valid: bool,
    rollback_valid: bool,
) -> MtpTargetValidationSignature:
    """Return one validated exact signature or reject incomplete evidence."""
    _validate_token_sequence("draft", draft_token_ids)
    _validate_token_sequence("target", target_token_ids)
    _validate_token_sequence("target verification", target_verification_token_ids)
    if len(target_token_ids) != len(draft_token_ids):
        raise MtpTargetPrewarmError("target prewarm token counts differ")
    expected_verification_count = len(draft_token_ids) + CONTINUATION_TOKEN_COUNT
    if len(target_verification_token_ids) != expected_verification_count:
        raise MtpTargetPrewarmError("target prewarm verification count changed")
    if isinstance(
        accepted_prefix_length, bool
    ) or not MINIMUM_ACCEPTED_PREFIX_LENGTH <= accepted_prefix_length <= len(
        draft_token_ids
    ):
        raise MtpTargetPrewarmError("target prewarm accepted prefix is invalid")
    if not isinstance(rollback_applied, bool):
        raise MtpTargetPrewarmError("target prewarm rollback flag is invalid")
    expected_rollback = accepted_prefix_length < len(draft_token_ids)
    if rollback_applied != expected_rollback:
        raise MtpTargetPrewarmError("target prewarm rollback flag changed")
    if not isinstance(target_output_valid, bool) or not target_output_valid:
        raise MtpTargetPrewarmError("target prewarm output is not exact")
    if not isinstance(rollback_valid, bool) or not rollback_valid:
        raise MtpTargetPrewarmError("target prewarm rollback is not exact")
    committed_width = accepted_prefix_length + CONTINUATION_TOKEN_COUNT
    return MtpTargetValidationSignature(
        draft_token_ids=draft_token_ids,
        target_token_ids=target_token_ids,
        target_verification_token_ids=target_verification_token_ids,
        accepted_prefix_length=accepted_prefix_length,
        committed_width=committed_width,
        rollback_applied=rollback_applied,
    )


def target_validation_signatures_match(
    prewarm: MtpTargetValidationSignature,
    measured: MtpTargetValidationSignature,
) -> bool:
    """Return whether prewarm and measured target work are exactly equal."""
    if not isinstance(prewarm, MtpTargetValidationSignature):
        raise MtpTargetPrewarmError("target prewarm signature type is invalid")
    if not isinstance(measured, MtpTargetValidationSignature):
        raise MtpTargetPrewarmError("target measured signature type is invalid")
    return prewarm == measured


def _validate_token_sequence(label: str, token_ids: tuple[int, ...]) -> None:
    if not isinstance(token_ids, tuple) or not token_ids:
        raise MtpTargetPrewarmError(f"{label} token sequence is invalid")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < MINIMUM_TOKEN_ID
        for token_id in token_ids
    ):
        raise MtpTargetPrewarmError(f"{label} token is invalid")
