"""Pure device-free cores for the RWKV-7 LoRA CPU reference.

These functions are deterministic and do not touch the filesystem, the
network, or a framework. The shell injects token IDs and identities.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

SCHEMA_VERSION = 1
RECORD_FIELDS = ("prompt", "response")
EOS_TOKEN_ID = 0
PAD_TOKEN_ID = 0

# Explicit role-based LoRA allowlist. `all-linear` is intentionally absent.
ATTENTION_PROJECTION_ROLES = (
    "attention.receptance",
    "attention.key",
    "attention.value",
    "attention.output",
)
FEED_FORWARD_PROJECTION_ROLES = ("feed_forward.key", "feed_forward.value")
DEFAULT_LORA_ROLES = ATTENTION_PROJECTION_ROLES + FEED_FORWARD_PROJECTION_ROLES
FORBIDDEN_ROLE_TOKENS = ("all-linear", "all_linear", "*")


def blake3_hex(data: bytes) -> str:
    """BLAKE3 hex digest via hashlib when available, else a stand-in note.

    The shell provides the real blake3 module; this helper exists so pure
    manifest tests can run without importing it at module scope.
    """
    import blake3  # type: ignore

    return blake3.blake3(data).hexdigest()


def admit_record(record: Any) -> tuple[str, str]:
    """Admit one JSONL record with nonempty prompt and response strings."""
    if not isinstance(record, dict):
        raise TypeError("record must be a JSON object")
    unknown = set(record) - set(RECORD_FIELDS)
    if unknown:
        raise ValueError(f"record has unknown fields: {sorted(unknown)}")
    missing = [field for field in RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    prompt = record["prompt"]
    response = record["response"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("record prompt must be a nonempty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("record response must be a nonempty string")
    return prompt, response


def plan_token_labels(
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    sequence_length: int,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict[str, list[int]]:
    """Produce deterministic tokens, active labels, and masks for one record.

    Prompt tokens and padding are masked from the supervised loss. Response
    tokens (shifted by one for next-token prediction) contribute labels.
    """
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if not prompt_token_ids:
        raise ValueError("prompt token ids must be nonempty")
    if not response_token_ids:
        raise ValueError("response token ids must be nonempty")

    full = list(prompt_token_ids) + list(response_token_ids) + [eos_token_id]
    if len(full) > sequence_length:
        raise ValueError(
            f"sample length {len(full)} exceeds admitted sequence length {sequence_length}"
        )

    tokens = full + [PAD_TOKEN_ID] * (sequence_length - len(full))
    # Labels predict the next token; positions predicting prompt or padding are masked.
    labels: list[int] = [-100] * sequence_length
    attention: list[int] = [0] * sequence_length
    for position in range(len(full) - 1):
        attention[position] = 1
        # Only response targets (and the EOS that ends the response) are supervised.
        if position >= len(prompt_token_ids) - 1:
            labels[position] = full[position + 1]

    active = [label for label in labels if label != -100]
    if not active:
        raise ValueError("record produced no active labels")

    return {"tokens": tokens, "labels": labels, "attention": attention}


def select_lora_targets(requested_roles: Sequence[str]) -> tuple[str, ...]:
    """Admit an explicit LoRA role set; reject unsafe or malformed plans."""
    roles = tuple(requested_roles)
    if not roles:
        raise ValueError("LoRA plan requires at least one explicit role")
    lowered = tuple(role.strip() for role in roles)
    for token in FORBIDDEN_ROLE_TOKENS:
        if token in lowered:
            raise ValueError(f"LoRA plan forbids selector '{token}'")
    allowed = set(DEFAULT_LORA_ROLES)
    for role in lowered:
        if role not in allowed:
            raise ValueError(f"unknown LoRA role '{role}'")
    if len(set(lowered)) != len(lowered):
        raise ValueError("LoRA plan contains duplicate roles")
    return tuple(sorted(lowered))


def parameter_freeze_plan(
    parameter_roles: Sequence[str], lora_roles: Sequence[str]
) -> dict[str, list[str]]:
    """Split base parameters into frozen and trainable adapter groups.

    Every base tensor stays frozen. Only the selected projection roles gain
    trainable LoRA A/B tensors.
    """
    selected = set(select_lora_targets(lora_roles))
    trainable = sorted(f"{role}.lora_A" for role in selected) + sorted(
        f"{role}.lora_B" for role in selected
    )
    frozen = sorted(parameter_roles)
    return {"frozen": frozen, "trainable": trainable}


def authority_identity(fields: dict[str, Any]) -> str:
    """Deterministic BLAKE3 authority identity over pinned model facts."""
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return blake3_hex(canonical)


def checkpoint_manifest(
    adapter_tensor_blake3: dict[str, str],
    bound: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Build the adapter checkpoint manifest and its BLAKE3 identity."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "safetensors-adapter-only",
        "step": step,
        "adapter_tensor_blake3": dict(sorted(adapter_tensor_blake3.items())),
        "bound": bound,
        "non_claims": [
            "No Tenstorrent execution is established.",
            "No useful convergence is established.",
            "No dataset quality is established.",
            "No production fine-tuning quality is established.",
        ],
    }
    manifest["manifest_blake3"] = authority_identity(manifest)
    return manifest


def validate_resume_manifest(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Fail closed when resume inputs drift from the saved manifest."""
    for key in ("bound", "adapter_tensor_blake3", "schema_version", "format"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"resume rejected: '{key}' identity differs from saved manifest"
            )
