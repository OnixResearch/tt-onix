"""Bounded FP32 CPU continuation for the pinned RWKV-7 model.

The primary path follows the FLA v0.3.0 recurrent-state orientation. The audit
path follows the transposed state equations from the official RWKV-LM RNN
reference. Both paths use the same pinned BF16 checkpoint converted to FP32.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import blake3
import torch
from safetensors import safe_open
from torch.nn import functional

SCHEMA_VERSION: Final = 1
MODEL_ID: Final = "RWKV/RWKV7-Goose-World2.8-0.1B-HF"
MODEL_REVISION: Final = "d81965cb4e1a9f96696b4f70b84212b8f2e43216"
MODEL_PROFILE_BLAKE3: Final = (
    "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88"
)
CHECKPOINT_BLAKE3: Final = (
    "905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5"
)
REFERENCE_METHOD: Final = "pinned_fp32_full_sequence_greedy_argmax"
BINDING_DOMAIN: Final = b"rwkv7-window16-cpu-continuation-v1\x00"
RECURRENT_MATRIX_HASH_DOMAIN: Final = b"rwkv7-window16-fp32-recurrent-matrix-v1\x00"
FLA_REFERENCE_VERSION: Final = "0.3.0"
OFFICIAL_REFERENCE_REVISION: Final = "952102498e9ed367ea0a59ee64106916d474d30f"

LAYER_COUNT: Final = 12
HIDDEN_SIZE: Final = 768
HEAD_COUNT: Final = 12
HEAD_SIZE: Final = 64
INTERMEDIATE_SIZE: Final = 3072
VOCABULARY_SIZE: Final = 65536
DECAY_LOG_SCALE: Final = -0.6065306597126334
LAYER_NORM_EPSILON: Final = 1e-5
GROUP_NORM_EPSILON: Final = HEAD_SIZE * LAYER_NORM_EPSILON
NORMALIZATION_EPSILON: Final = 1e-12
GENERATED_TOKEN_COUNT: Final = 16
REQUEST_COUNT: Final = 4
CPU_THREAD_COUNT: Final = 4
AUDIT_STATE_TOLERANCE: Final = 5e-4
AUDIT_LOGIT_TOLERANCE: Final = 5e-3
SELF_TEST_SEED: Final = 17
SELF_TEST_HEAD_COUNT: Final = 2
SELF_TEST_HEAD_SIZE: Final = 3
SELF_TEST_TOLERANCE: Final = 1e-6
SELF_TEST_LOG_DECAY_SCALE: Final = 0.1
RECURRENT_STATE_RANK: Final = 3
EUCLIDEAN_NORM: Final = 2.0
TOP_LOGIT_COUNT: Final = 2
LAST_DIMENSION: Final = -1
SECOND_TO_LAST_DIMENSION: Final = -2
JSON_INDENT: Final = 2
HASH_BLOCK_BYTES: Final = 1024 * 1024

KNOWN_PROMPT_TOKEN_IDS: Final = (0, 24281, 59, 36786, 65530, 5585, 41693, 59)
KNOWN_GENERATED_PREFIX: Final = (36786, 34, 308, 459)

Mode = Literal["derive", "validate"]


@dataclass
class LayerState:
    attention_previous: torch.Tensor
    ffn_previous: torch.Tensor
    matrix: torch.Tensor


@dataclass
class ModelRun:
    generated_token_ids: list[int]
    state: list[LayerState]
    final_logits: torch.Tensor
    token_margins: list[float]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def blake3_file(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return functional.linear(value, weight, bias)


def layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return functional.layer_norm(
        value,
        (HIDDEN_SIZE,),
        weight=weight,
        bias=bias,
        eps=LAYER_NORM_EPSILON,
    )


def mixed(
    value: torch.Tensor,
    previous: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    return torch.lerp(value, previous, factor)


def lora(
    weights: dict[str, torch.Tensor],
    prefix: str,
    value: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
    has_bias: bool = True,
) -> torch.Tensor:
    hidden = activation(linear(value, weights[f"{prefix}.lora.0.weight"]))
    bias = weights[f"{prefix}.lora.2.bias"] if has_bias else None
    return linear(hidden, weights[f"{prefix}.lora.2.weight"], bias)


def recurrence_fla(
    state: torch.Tensor,
    log_decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalized_key: torch.Tensor,
    attention_gate: torch.Tensor,
    receptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA v0.3.0 state orientation: [head, key, value]."""
    require(
        state.ndim == RECURRENT_STATE_RANK,
        "FLA recurrent state has an invalid rank",
    )
    head_count, key_size, value_size = state.shape
    require(head_count > 0 and key_size > 0 and value_size > 0, "FLA state is empty")
    require(key.numel() == head_count * key_size, "FLA key shape differs")
    require(value.numel() == head_count * value_size, "FLA value shape differs")
    require(receptance.numel() == head_count * key_size, "FLA receptance shape differs")
    require(normalized_key.numel() == key.numel(), "FLA normalized key shape differs")
    require(attention_gate.numel() == key.numel(), "FLA attention gate shape differs")
    decay = torch.exp(log_decay).view(head_count, key_size, 1)
    key_rows = key.view(head_count, key_size)
    value_rows = value.view(head_count, value_size)
    normalized_rows = normalized_key.view(head_count, key_size)
    gate_rows = attention_gate.view(head_count, key_size)
    receptance_rows = receptance.view(head_count, key_size)
    rank_left = normalized_rows * gate_rows
    rank_right = -normalized_rows
    contraction = torch.sum(rank_right.unsqueeze(-1) * state, dim=1)
    next_state = (
        decay * state
        + rank_left.unsqueeze(-1) * contraction.unsqueeze(1)
        + key_rows.unsqueeze(-1) * value_rows.unsqueeze(1)
    )
    output = torch.sum(next_state * receptance_rows.unsqueeze(-1), dim=1)
    return next_state, output


def recurrence_official(
    state: torch.Tensor,
    log_decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalized_key: torch.Tensor,
    attention_gate: torch.Tensor,
    receptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official RWKV-LM state orientation: [head, value, key]."""
    require(
        state.ndim == RECURRENT_STATE_RANK,
        "official recurrent state has an invalid rank",
    )
    head_count, value_size, key_size = state.shape
    require(
        head_count > 0 and key_size > 0 and value_size > 0, "official state is empty"
    )
    require(key.numel() == head_count * key_size, "official key shape differs")
    require(value.numel() == head_count * value_size, "official value shape differs")
    require(
        receptance.numel() == head_count * key_size, "official receptance shape differs"
    )
    require(
        normalized_key.numel() == key.numel(), "official normalized key shape differs"
    )
    require(
        attention_gate.numel() == key.numel(), "official attention gate shape differs"
    )
    decay = torch.exp(log_decay).view(head_count, 1, key_size)
    key_rows = key.view(head_count, key_size)
    value_rows = value.view(head_count, value_size)
    normalized_rows = normalized_key.view(head_count, key_size)
    gate_rows = attention_gate.view(head_count, key_size)
    receptance_rows = receptance.view(head_count, key_size)
    rank_matrix = torch.matmul(
        (-normalized_rows).unsqueeze(-1),
        (normalized_rows * gate_rows).unsqueeze(1),
    )
    outer = torch.matmul(value_rows.unsqueeze(-1), key_rows.unsqueeze(1))
    next_state = decay * state + torch.matmul(state, rank_matrix) + outer
    output = torch.matmul(next_state, receptance_rows.unsqueeze(-1)).squeeze(-1)
    return next_state, output


def model_tensor_names() -> set[str]:
    names = {
        "model.embeddings.weight",
        "model.norm.weight",
        "model.norm.bias",
        "lm_head.weight",
    }
    common_suffixes = (
        "attn_norm.weight",
        "attn_norm.bias",
        "attn.x_r",
        "attn.x_w",
        "attn.x_k",
        "attn.x_v",
        "attn.x_a",
        "attn.x_g",
        "attn.r_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.o_proj.weight",
        "attn.w_lora.lora.0.weight",
        "attn.w_lora.lora.2.weight",
        "attn.w_lora.lora.2.bias",
        "attn.a_lora.lora.0.weight",
        "attn.a_lora.lora.2.weight",
        "attn.a_lora.lora.2.bias",
        "attn.g_lora.lora.0.weight",
        "attn.g_lora.lora.2.weight",
        "attn.k_k",
        "attn.k_a",
        "attn.r_k",
        "attn.g_norm.weight",
        "attn.g_norm.bias",
        "ffn_norm.weight",
        "ffn_norm.bias",
        "ffn.x_k",
        "ffn.key.weight",
        "ffn.value.weight",
    )
    later_layer_suffixes = (
        "attn.v_lora.lora.0.weight",
        "attn.v_lora.lora.2.weight",
        "attn.v_lora.lora.2.bias",
    )
    for layer_index in range(LAYER_COUNT):
        prefix = f"model.layers.{layer_index}."
        names.update(prefix + suffix for suffix in common_suffixes)
        if layer_index == 0:
            names.add(prefix + "pre_norm.weight")
            names.add(prefix + "pre_norm.bias")
        else:
            names.update(prefix + suffix for suffix in later_layer_suffixes)
    return names


def load_fp32_weights(model_path: Path) -> dict[str, torch.Tensor]:
    required_names = model_tensor_names()
    weights: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        available_names = set(checkpoint.keys())
        missing = sorted(required_names - available_names)
        require(not missing, f"checkpoint is missing required tensors: {missing[:1]}")
        for name in sorted(required_names):
            tensor = checkpoint.get_tensor(name)
            require(
                tensor.dtype == torch.bfloat16, f"checkpoint tensor is not BF16: {name}"
            )
            weights[name] = tensor.to(dtype=torch.float32)
    return weights


def zero_state(orientation: Literal["fla", "official"]) -> list[LayerState]:
    require(orientation in ("fla", "official"), "unknown recurrent-state orientation")
    return [
        LayerState(
            attention_previous=torch.zeros(HIDDEN_SIZE, dtype=torch.float32),
            ffn_previous=torch.zeros(HIDDEN_SIZE, dtype=torch.float32),
            matrix=torch.zeros((HEAD_COUNT, HEAD_SIZE, HEAD_SIZE), dtype=torch.float32),
        )
        for _layer_index in range(LAYER_COUNT)
    ]


def run_step(
    weights: dict[str, torch.Tensor],
    state: list[LayerState],
    input_token: int,
    orientation: Literal["fla", "official"],
) -> torch.Tensor:
    require(0 <= input_token < VOCABULARY_SIZE, "input token is outside the vocabulary")
    require(len(state) == LAYER_COUNT, "model state has the wrong layer count")
    hidden = weights["model.embeddings.weight"][input_token]
    value_anchor: torch.Tensor | None = None

    for layer_index in range(LAYER_COUNT):
        prefix = f"model.layers.{layer_index}."
        layer_state = state[layer_index]
        residual = hidden
        if layer_index == 0:
            residual = layer_norm(
                hidden,
                weights[prefix + "pre_norm.weight"],
                weights[prefix + "pre_norm.bias"],
            )
        attention_input = layer_norm(
            residual,
            weights[prefix + "attn_norm.weight"],
            weights[prefix + "attn_norm.bias"],
        )

        x_r = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_r"],
        )
        x_w = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_w"],
        )
        x_k = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_k"],
        )
        x_v = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_v"],
        )
        x_a = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_a"],
        )
        x_g = mixed(
            attention_input,
            layer_state.attention_previous,
            weights[prefix + "attn.x_g"],
        )
        layer_state.attention_previous = attention_input

        receptance = linear(x_r, weights[prefix + "attn.r_proj.weight"])
        raw_key = linear(x_k, weights[prefix + "attn.k_proj.weight"])
        projected_value = linear(x_v, weights[prefix + "attn.v_proj.weight"])
        if layer_index == 0:
            require(value_anchor is None, "layer zero received a stale value anchor")
            value = projected_value
            value_anchor = projected_value
        else:
            require(value_anchor is not None, "later layer is missing the value anchor")
            value_gate = torch.sigmoid(
                lora(weights, prefix + "attn.v_lora", x_v, lambda item: item)
            )
            value = torch.lerp(projected_value, value_anchor, value_gate)

        log_decay = DECAY_LOG_SCALE * torch.sigmoid(
            lora(weights, prefix + "attn.w_lora", x_w, torch.tanh)
        )
        attention_gate = torch.sigmoid(
            lora(weights, prefix + "attn.a_lora", x_a, lambda item: item)
        )
        output_gate = lora(
            weights,
            prefix + "attn.g_lora",
            x_g,
            torch.sigmoid,
            has_bias=False,
        )
        normalized_key = functional.normalize(
            (raw_key * weights[prefix + "attn.k_k"]).view(HEAD_COUNT, HEAD_SIZE),
            dim=LAST_DIMENSION,
            p=EUCLIDEAN_NORM,
            eps=NORMALIZATION_EPSILON,
        ).reshape(HIDDEN_SIZE)
        key = raw_key * (1 + (attention_gate - 1) * weights[prefix + "attn.k_a"])

        if orientation == "fla":
            next_matrix, recurrent_output = recurrence_fla(
                layer_state.matrix,
                log_decay,
                key,
                value,
                normalized_key,
                attention_gate,
                receptance,
            )
        else:
            next_matrix, recurrent_output = recurrence_official(
                layer_state.matrix,
                log_decay,
                key,
                value,
                normalized_key,
                attention_gate,
                receptance,
            )
        layer_state.matrix = next_matrix

        normalized_output = functional.group_norm(
            recurrent_output.reshape(1, HIDDEN_SIZE),
            num_groups=HEAD_COUNT,
            weight=weights[prefix + "attn.g_norm.weight"],
            bias=weights[prefix + "attn.g_norm.bias"],
            eps=GROUP_NORM_EPSILON,
        ).reshape(HEAD_COUNT, HEAD_SIZE)
        correction = (
            receptance.view(HEAD_COUNT, HEAD_SIZE)
            * key.view(HEAD_COUNT, HEAD_SIZE)
            * weights[prefix + "attn.r_k"]
        ).sum(dim=-1, keepdim=True) * value.view(HEAD_COUNT, HEAD_SIZE)
        attention_output = linear(
            (
                (normalized_output + correction)
                * output_gate.view(HEAD_COUNT, HEAD_SIZE)
            ).reshape(HIDDEN_SIZE),
            weights[prefix + "attn.o_proj.weight"],
        )
        after_attention = residual + attention_output

        ffn_input = layer_norm(
            after_attention,
            weights[prefix + "ffn_norm.weight"],
            weights[prefix + "ffn_norm.bias"],
        )
        ffn_mixed = mixed(
            ffn_input,
            layer_state.ffn_previous,
            weights[prefix + "ffn.x_k"],
        )
        layer_state.ffn_previous = ffn_input
        ffn_hidden = torch.relu(linear(ffn_mixed, weights[prefix + "ffn.key.weight"]))
        ffn_output = linear(ffn_hidden.square(), weights[prefix + "ffn.value.weight"])
        hidden = after_attention + ffn_output

    final_hidden = layer_norm(
        hidden,
        weights["model.norm.weight"],
        weights["model.norm.bias"],
    )
    return linear(final_hidden, weights["lm_head.weight"])


def run_generation(
    weights: dict[str, torch.Tensor],
    prompt_token_ids: list[int],
    orientation: Literal["fla", "official"],
) -> ModelRun:
    require(prompt_token_ids, "prompt tokens must not be empty")
    state = zero_state(orientation)
    logits = torch.empty(VOCABULARY_SIZE, dtype=torch.float32)
    for input_token in prompt_token_ids:
        logits = run_step(weights, state, input_token, orientation)

    generated: list[int] = []
    margins: list[float] = []
    for generated_index in range(GENERATED_TOKEN_COUNT):
        top_values, top_indices = torch.topk(logits, k=TOP_LOGIT_COUNT)
        next_token = int(top_indices[0].item())
        generated.append(next_token)
        margins.append(float((top_values[0] - top_values[1]).item()))
        if generated_index + 1 < GENERATED_TOKEN_COUNT:
            logits = run_step(weights, state, next_token, orientation)
    return ModelRun(generated, state, logits, margins)


def normalized_state_matrix(
    matrix: torch.Tensor,
    orientation: Literal["fla", "official"],
) -> torch.Tensor:
    return (
        matrix.transpose(LAST_DIMENSION, SECOND_TO_LAST_DIMENSION)
        if orientation == "fla"
        else matrix
    )


def state_blake3(
    state: list[LayerState],
    orientation: Literal["fla", "official"],
) -> str:
    digest = blake3.blake3(b"rwkv7-window16-fp32-state-v1\x00")
    for layer_index, layer in enumerate(state):
        for label, tensor in (
            ("attention_previous", layer.attention_previous),
            ("ffn_previous", layer.ffn_previous),
            ("matrix", normalized_state_matrix(layer.matrix, orientation)),
        ):
            digest.update(f"{layer_index}:{label}\x00".encode("ascii"))
            digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def recurrent_matrix_blake3(
    state: list[LayerState],
    orientation: Literal["fla", "official"],
) -> str:
    digest = blake3.blake3(RECURRENT_MATRIX_HASH_DOMAIN)
    for layer_index, layer in enumerate(state):
        digest.update(f"{layer_index}:matrix\x00".encode("ascii"))
        digest.update(
            normalized_state_matrix(layer.matrix, orientation)
            .detach()
            .contiguous()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def maximum_state_difference(primary: ModelRun, audit: ModelRun) -> float:
    require(len(primary.state) == len(audit.state), "audit state layer counts differ")
    maximum = 0.0
    for primary_layer, audit_layer in zip(primary.state, audit.state):
        maximum = max(
            maximum,
            float(
                (primary_layer.attention_previous - audit_layer.attention_previous)
                .abs()
                .max()
                .item()
            ),
            float(
                (primary_layer.ffn_previous - audit_layer.ffn_previous)
                .abs()
                .max()
                .item()
            ),
            float(
                (
                    normalized_state_matrix(primary_layer.matrix, "fla")
                    - normalized_state_matrix(audit_layer.matrix, "official")
                )
                .abs()
                .max()
                .item()
            ),
        )
    return maximum


def validate_workload(document: object) -> tuple[list[int], list[int]]:
    require(isinstance(document, dict), "workload must be a JSON object")
    expected_root_fields = {
        "schema_version",
        "cpu_reference_profile_blake3",
        "cpu_reference_method",
        "requests",
    }
    require(set(document) == expected_root_fields, "workload root fields differ")
    require(document["schema_version"] == SCHEMA_VERSION, "workload schema differs")
    require(
        document["cpu_reference_profile_blake3"] == MODEL_PROFILE_BLAKE3,
        "workload CPU profile identity differs",
    )
    require(
        document["cpu_reference_method"] == REFERENCE_METHOD,
        "workload CPU reference method differs",
    )
    requests = document["requests"]
    require(isinstance(requests, list), "workload requests must be an array")
    require(len(requests) == REQUEST_COUNT, "workload request count differs")
    request_fields = {
        "request_id",
        "prompt_token_ids",
        "max_new_tokens",
        "expected_generated_token_ids",
    }
    first_prompt: list[int] | None = None
    first_expected: list[int] | None = None
    request_ids: set[str] = set()
    for request in requests:
        require(isinstance(request, dict), "workload request must be an object")
        require(set(request) == request_fields, "workload request fields differ")
        request_id = request["request_id"]
        require(isinstance(request_id, str) and request_id, "request ID is invalid")
        require(request_id not in request_ids, "request ID is duplicated")
        request_ids.add(request_id)
        prompt = request["prompt_token_ids"]
        expected = request["expected_generated_token_ids"]
        require(
            isinstance(prompt, list) and all(isinstance(item, int) for item in prompt),
            "prompt token list is invalid",
        )
        require(
            isinstance(expected, list)
            and all(isinstance(item, int) for item in expected),
            "expected token list is invalid",
        )
        require(
            request["max_new_tokens"] == GENERATED_TOKEN_COUNT,
            "generated-token bound differs",
        )
        require(
            len(expected) == GENERATED_TOKEN_COUNT,
            "expected token count differs",
        )
        if first_prompt is None:
            first_prompt = prompt
            first_expected = expected
        else:
            require(prompt == first_prompt, "workload prompts differ")
            require(expected == first_expected, "workload expected tokens differ")
    require(first_prompt is not None, "workload has no prompt")
    require(first_expected is not None, "workload has no expected continuation")
    require(
        tuple(first_prompt) == KNOWN_PROMPT_TOKEN_IDS,
        "workload prompt differs from the pinned Hello prompt",
    )
    return first_prompt, first_expected


def build_receipt(
    checkpoint_hash: str,
    primary: ModelRun,
    audit: ModelRun,
    state_difference: float,
    logit_difference: float,
    mode: Mode,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "outcome": "validated_cpu_continuation"
        if mode == "validate"
        else "derived_cpu_continuation",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "profile_blake3": MODEL_PROFILE_BLAKE3,
            "checkpoint_blake3": checkpoint_hash,
            "checkpoint_precision": "bf16",
            "arithmetic_precision": "fp32_from_bf16",
        },
        "method": REFERENCE_METHOD,
        "generated_token_ids": primary.generated_token_ids,
        "known_prefix_matched": primary.generated_token_ids[
            : len(KNOWN_GENERATED_PREFIX)
        ]
        == list(KNOWN_GENERATED_PREFIX),
        "token_margins": primary.token_margins,
        "audit": {
            "primary": f"fla-{FLA_REFERENCE_VERSION}-key_value_state",
            "secondary": f"rwkv-lm-{OFFICIAL_REFERENCE_REVISION}-value_key_state",
            "generated_tokens_equal": primary.generated_token_ids
            == audit.generated_token_ids,
            "maximum_state_abs_difference": state_difference,
            "maximum_final_logit_abs_difference": logit_difference,
            "primary_state_blake3": state_blake3(primary.state, "fla"),
            "secondary_state_blake3": state_blake3(audit.state, "official"),
            "primary_fp32_recurrent_matrix_blake3": recurrent_matrix_blake3(
                primary.state, "fla"
            ),
            "secondary_fp32_recurrent_matrix_blake3": recurrent_matrix_blake3(
                audit.state, "official"
            ),
        },
        "bounds": {
            "prompt_token_count": len(KNOWN_PROMPT_TOKEN_IDS),
            "generated_token_count": GENERATED_TOKEN_COUNT,
            "layer_count": LAYER_COUNT,
            "cpu_thread_count": CPU_THREAD_COUNT,
        },
        "non_claims": [
            "No Tenstorrent execution is established.",
            "No simulator or physical performance is established.",
            "No production window-16 admission is established.",
        ],
    }
    receipt["binding_blake3"] = blake3.blake3(
        BINDING_DOMAIN + canonical_json(receipt)
    ).hexdigest()
    return receipt


def validate_paths(model_path: Path, workload_path: Path, output_path: Path) -> None:
    require(model_path.is_absolute(), "model path must be absolute")
    require(model_path.is_file(), "model path must be a regular file")
    require(workload_path.is_absolute(), "workload path must be absolute")
    require(workload_path.is_file(), "workload path must be a regular file")
    require(output_path.is_absolute(), "output path must be absolute")
    require(not output_path.exists(), "output path already exists")
    require(output_path.parent.is_dir(), "output parent must be a directory")


def self_test() -> dict[str, object]:
    generator = torch.Generator().manual_seed(SELF_TEST_SEED)
    shape = (SELF_TEST_HEAD_COUNT, SELF_TEST_HEAD_SIZE, SELF_TEST_HEAD_SIZE)
    state_fla = torch.randn(shape, generator=generator)
    state_official = state_fla.transpose(
        LAST_DIMENSION, SECOND_TO_LAST_DIMENSION
    ).contiguous()
    vector_shape = (SELF_TEST_HEAD_COUNT * SELF_TEST_HEAD_SIZE,)
    log_decay = (
        torch.randn(vector_shape, generator=generator) * SELF_TEST_LOG_DECAY_SCALE
    )
    key = torch.randn(vector_shape, generator=generator)
    value = torch.randn(vector_shape, generator=generator)
    normalized_key = functional.normalize(
        torch.randn((SELF_TEST_HEAD_COUNT, SELF_TEST_HEAD_SIZE), generator=generator),
        dim=-1,
    ).reshape(vector_shape)
    attention_gate = torch.sigmoid(torch.randn(vector_shape, generator=generator))
    receptance = torch.randn(vector_shape, generator=generator)

    next_fla, output_fla = recurrence_fla(
        state_fla,
        log_decay,
        key,
        value,
        normalized_key,
        attention_gate,
        receptance,
    )
    next_official, output_official = recurrence_official(
        state_official,
        log_decay,
        key,
        value,
        normalized_key,
        attention_gate,
        receptance,
    )
    maximum = max(
        float(
            (
                next_fla.transpose(LAST_DIMENSION, SECOND_TO_LAST_DIMENSION)
                - next_official
            )
            .abs()
            .max()
            .item()
        ),
        float((output_fla - output_official).abs().max().item()),
    )
    require(maximum <= SELF_TEST_TOLERANCE, "recurrence orientations disagree")
    negative_rejected = False
    try:
        recurrence_official(
            torch.zeros(1),
            log_decay,
            key,
            value,
            normalized_key,
            attention_gate,
            receptance,
        )
    except ValueError:
        negative_rejected = True
    require(negative_rejected, "invalid recurrent state was accepted")
    return {
        "outcome": "self_test_pass",
        "positive_orientation_equivalence": True,
        "negative_shape_rejected": True,
        "maximum_abs_difference": maximum,
    }


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("derive", "validate"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed.self_test:
        require(parsed.mode is None, "self-test cannot select a run mode")
        json.dump(self_test(), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    require(parsed.mode is not None, "run mode is required")
    require(parsed.model is not None, "model path is required")
    require(parsed.workload is not None, "workload path is required")
    require(parsed.output is not None, "output path is required")
    model_path: Path = parsed.model
    workload_path: Path = parsed.workload
    output_path: Path = parsed.output
    mode: Mode = parsed.mode
    validate_paths(model_path, workload_path, output_path)

    checkpoint_hash = blake3_file(model_path)
    require(checkpoint_hash == CHECKPOINT_BLAKE3, "checkpoint BLAKE3 differs")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    prompt, expected = validate_workload(workload)

    torch.set_grad_enabled(False)
    torch.set_num_threads(CPU_THREAD_COUNT)
    torch.set_num_interop_threads(1)
    weights = load_fp32_weights(model_path)
    primary = run_generation(weights, prompt, "fla")
    audit = run_generation(weights, prompt, "official")
    require(
        primary.generated_token_ids[: len(KNOWN_GENERATED_PREFIX)]
        == list(KNOWN_GENERATED_PREFIX),
        "derived continuation differs from the accepted four-token prefix",
    )
    require(
        primary.generated_token_ids == audit.generated_token_ids,
        "FLA and official recurrences produce different tokens",
    )
    state_difference = maximum_state_difference(primary, audit)
    logit_difference = float(
        (primary.final_logits - audit.final_logits).abs().max().item()
    )
    require(math.isfinite(state_difference), "state audit difference is nonfinite")
    require(math.isfinite(logit_difference), "logit audit difference is nonfinite")
    require(
        state_difference <= AUDIT_STATE_TOLERANCE,
        f"state audit difference exceeds {AUDIT_STATE_TOLERANCE}: {state_difference}",
    )
    require(
        logit_difference <= AUDIT_LOGIT_TOLERANCE,
        f"logit audit difference exceeds {AUDIT_LOGIT_TOLERANCE}: {logit_difference}",
    )
    if mode == "validate":
        require(
            expected == primary.generated_token_ids,
            "workload continuation differs from the pinned CPU result",
        )

    receipt = build_receipt(
        checkpoint_hash,
        primary,
        audit,
        state_difference,
        logit_difference,
        mode,
    )
    output_path.write_text(
        json.dumps(receipt, indent=JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"rwkv7_cpu_continuation: {error}", file=sys.stderr)
        raise SystemExit(os.EX_DATAERR)
