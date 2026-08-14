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
DEVICE_RECURRENT_PROBE_OUTCOME: Final = (
    "temporary_fp32_recurrent_matrix_boundaries"
)
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
FIRST_MODEL_STEP_INDEX: Final = 0
FIRST_LAYER_INDEX: Final = 0
RECURRENT_VECTOR_ELEMENT_COUNT: Final = HEAD_COUNT * HEAD_SIZE
RECURRENT_MATRIX_ELEMENT_COUNT: Final = HEAD_COUNT * HEAD_SIZE * HEAD_SIZE
RECURRENT_INPUT_OPERATIONS: Final = ("key_rows", "value_rows")
RECURRENT_OPERATION_ORDER: Final = (
    "initial_state",
    "state_dot_a",
    "rank_update",
    "decay",
    "outer_update",
    "first_add",
    "next_state",
)
DEVICE_RECURRENT_PROBE_FIELD_COUNT: Final = 19
DEVICE_RECURRENT_BOUNDARY_FIELD_COUNT: Final = 7

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


def tensor_blake3(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(torch.float32).contiguous().numpy()
    return blake3.blake3(values.tobytes()).hexdigest()


def initial_recurrent_boundaries(
    weights: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prefix = f"model.layers.{FIRST_LAYER_INDEX}."
    hidden = weights["model.embeddings.weight"][KNOWN_PROMPT_TOKEN_IDS[0]]
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
    previous = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
    x_w = mixed(attention_input, previous, weights[prefix + "attn.x_w"])
    x_k = mixed(attention_input, previous, weights[prefix + "attn.x_k"])
    x_v = mixed(attention_input, previous, weights[prefix + "attn.x_v"])
    x_a = mixed(attention_input, previous, weights[prefix + "attn.x_a"])

    raw_key = linear(x_k, weights[prefix + "attn.k_proj.weight"])
    value = linear(x_v, weights[prefix + "attn.v_proj.weight"])
    log_decay = DECAY_LOG_SCALE * torch.sigmoid(
        lora(weights, prefix + "attn.w_lora", x_w, torch.tanh)
    )
    attention_gate = torch.sigmoid(
        lora(weights, prefix + "attn.a_lora", x_a, lambda item: item)
    )
    normalized_key = functional.normalize(
        (raw_key * weights[prefix + "attn.k_k"]).view(HEAD_COUNT, HEAD_SIZE),
        dim=LAST_DIMENSION,
        p=EUCLIDEAN_NORM,
        eps=NORMALIZATION_EPSILON,
    )
    key = (
        raw_key
        * (1 + (attention_gate - 1) * weights[prefix + "attn.k_a"])
    ).view(HEAD_COUNT, HEAD_SIZE)
    value_rows = value.view(HEAD_COUNT, HEAD_SIZE)
    gate_rows = attention_gate.view(HEAD_COUNT, HEAD_SIZE)
    a_rows = -normalized_key
    b_rows = normalized_key * gate_rows
    w_rows = torch.exp(log_decay).view(HEAD_COUNT, HEAD_SIZE)
    initial_state = torch.zeros(
        (HEAD_COUNT, HEAD_SIZE, HEAD_SIZE), dtype=torch.float32
    )
    state_dot_a = torch.matmul(initial_state, a_rows.unsqueeze(-1))
    rank_update = torch.matmul(state_dot_a, b_rows.unsqueeze(1))
    decay = initial_state * w_rows.unsqueeze(1)
    outer_update = torch.matmul(value_rows.unsqueeze(-1), key.unsqueeze(1))
    first_add = decay + rank_update
    next_state = first_add + outer_update
    recurrence_inputs = {"key_rows": key, "value_rows": value_rows}
    operation_boundaries = {
        "initial_state": initial_state,
        "state_dot_a": state_dot_a,
        "rank_update": rank_update,
        "decay": decay,
        "outer_update": outer_update,
        "first_add": first_add,
        "next_state": next_state,
    }
    return recurrence_inputs, operation_boundaries


def parse_device_boundary(
    raw: object,
    expected_operation: str,
    expected_element_count: int,
) -> torch.Tensor:
    require(isinstance(raw, dict), "device recurrent boundary must be an object")
    require(
        len(raw) == DEVICE_RECURRENT_BOUNDARY_FIELD_COUNT,
        "device recurrent boundary field count differs",
    )
    require(
        raw.get("model_step_index") == FIRST_MODEL_STEP_INDEX,
        "device recurrent boundary model step differs",
    )
    require(
        raw.get("layer_index") == FIRST_LAYER_INDEX,
        "device recurrent boundary layer differs",
    )
    require(
        raw.get("operation") == expected_operation,
        "device recurrent boundary operation differs",
    )
    require(raw.get("dtype") == "fp32", "device recurrent boundary is not FP32")
    require(
        raw.get("element_count") == expected_element_count,
        "device recurrent boundary element count differs",
    )
    raw_values = raw.get("values")
    require(
        isinstance(raw_values, list) and len(raw_values) == expected_element_count,
        "device recurrent boundary values differ from its element count",
    )
    require(
        all(isinstance(value, (int, float)) for value in raw_values),
        "device recurrent boundary contains a nonnumeric value",
    )
    values = torch.tensor(raw_values, dtype=torch.float32)
    require(
        bool(torch.isfinite(values).all().item()),
        "device recurrent boundary contains a nonfinite value",
    )
    require(
        raw.get("blake3") == tensor_blake3(values),
        "device recurrent boundary BLAKE3 differs from its values",
    )
    return values


def indexed_device_boundaries(
    raw_boundaries: object,
    operations: tuple[str, ...],
    matrix_operations: set[str],
) -> dict[str, torch.Tensor]:
    require(isinstance(raw_boundaries, list), "device boundaries must be an array")
    require(
        len(raw_boundaries) == len(operations),
        "device recurrent boundary count differs",
    )
    output: dict[str, torch.Tensor] = {}
    for raw, operation in zip(raw_boundaries, operations):
        expected_count = (
            RECURRENT_MATRIX_ELEMENT_COUNT
            if operation in matrix_operations
            else RECURRENT_VECTOR_ELEMENT_COUNT
        )
        output[operation] = parse_device_boundary(raw, operation, expected_count)
    return output


def compare_fp32_values(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, object]:
    expected_values = expected.detach().to(torch.float32).contiguous().reshape(-1)
    actual_values = actual.detach().to(torch.float32).contiguous().reshape(-1)
    require(expected_values.numel() > 0, "FP32 comparison is empty")
    require(
        expected_values.numel() == actual_values.numel(),
        "FP32 comparison element counts differ",
    )
    require(
        bool(torch.isfinite(expected_values).all().item())
        and bool(torch.isfinite(actual_values).all().item()),
        "FP32 comparison contains a nonfinite value",
    )
    different = expected_values.view(torch.int32) != actual_values.view(torch.int32)
    different_indices = torch.nonzero(different, as_tuple=False).reshape(-1)
    first_index = (
        int(different_indices[0].item()) if different_indices.numel() > 0 else None
    )
    maximum_error = float((expected_values - actual_values).abs().max().item())
    require(math.isfinite(maximum_error), "FP32 comparison error is nonfinite")
    return {
        "element_count": expected_values.numel(),
        "cpu_blake3": tensor_blake3(expected_values),
        "device_blake3": tensor_blake3(actual_values),
        "exact": first_index is None,
        "first_differing_element": first_index,
        "cpu_first_differing_value": (
            float(expected_values[first_index].item())
            if first_index is not None
            else None
        ),
        "device_first_differing_value": (
            float(actual_values[first_index].item())
            if first_index is not None
            else None
        ),
        "maximum_absolute_error": maximum_error,
    }


def build_recurrent_diagnosis(
    weights: dict[str, torch.Tensor],
    device_document: object,
    device_document_blake3: str,
) -> dict[str, object]:
    require(isinstance(device_document, dict), "device probe must be an object")
    require(
        len(device_document) == DEVICE_RECURRENT_PROBE_FIELD_COUNT,
        "device probe field count differs",
    )
    require(
        device_document.get("schema_version") == SCHEMA_VERSION,
        "device probe schema differs",
    )
    require(
        device_document.get("outcome") == DEVICE_RECURRENT_PROBE_OUTCOME,
        "device probe outcome differs",
    )
    require(
        device_document.get("physical_hardware") is True
        and device_document.get("simulated") is False,
        "device probe is not physical evidence",
    )
    require(
        device_document.get("physical_device_id") == 0,
        "device probe used the wrong physical device",
    )
    require(
        device_document.get("dispatch") == "normal"
        and device_document.get("wkv_execution_policy") == "generic-reference",
        "device probe used the wrong execution harness",
    )
    require(
        device_document.get("model_step_count") == 1
        and device_document.get("layer_count") == 1,
        "device probe exceeded its step or layer bound",
    )
    require(
        device_document.get("model_step_index") == FIRST_MODEL_STEP_INDEX
        and device_document.get("layer_index") == FIRST_LAYER_INDEX
        and device_document.get("input_token_id") == KNOWN_PROMPT_TOKEN_IDS[0],
        "device probe boundary identity differs",
    )
    model_activation_precision = device_document.get("model_activation_precision")
    require(
        model_activation_precision in ("bf16", "fp32"),
        "device probe model activation precision differs",
    )
    require(
        device_document.get("temporary_raw_tensor_values") is True
        and device_document.get("performance_claim") is False
        and device_document.get("production_physically_admitted") is False,
        "device probe retention or claim boundary differs",
    )

    cpu_inputs, cpu_operations = initial_recurrent_boundaries(weights)
    device_inputs = indexed_device_boundaries(
        device_document.get("recurrence_inputs"),
        RECURRENT_INPUT_OPERATIONS,
        set(),
    )
    matrix_operations = set(RECURRENT_OPERATION_ORDER) - {"state_dot_a"}
    device_operations = indexed_device_boundaries(
        device_document.get("boundaries"),
        RECURRENT_OPERATION_ORDER,
        matrix_operations,
    )

    input_comparisons: dict[str, object] = {}
    for operation in RECURRENT_INPUT_OPERATIONS:
        input_comparisons[operation] = compare_fp32_values(
            cpu_inputs[operation], device_inputs[operation]
        )

    operation_comparisons: list[dict[str, object]] = []
    first_divergence: dict[str, object] | None = None
    prior_matching_operations: list[str] = []
    for operation in RECURRENT_OPERATION_ORDER:
        comparison = compare_fp32_values(
            cpu_operations[operation], device_operations[operation]
        )
        summary = {
            "model_step_index": FIRST_MODEL_STEP_INDEX,
            "layer_index": FIRST_LAYER_INDEX,
            "operation": operation,
            "dtype": "fp32",
            **comparison,
        }
        operation_comparisons.append(summary)
        if first_divergence is None and not comparison["exact"]:
            first_divergence = {
                **summary,
                "prior_matching_operations": prior_matching_operations.copy(),
            }
        elif first_divergence is None:
            prior_matching_operations.append(operation)
    device_input_outer = torch.matmul(
        device_inputs["value_rows"].view(HEAD_COUNT, HEAD_SIZE).unsqueeze(-1),
        device_inputs["key_rows"].view(HEAD_COUNT, HEAD_SIZE).unsqueeze(1),
    )
    outer_control = compare_fp32_values(
        device_input_outer, device_operations["outer_update"]
    )
    input_materialization_differs = any(
        not comparison["exact"] for comparison in input_comparisons.values()
    )
    mechanism_proven = (
        model_activation_precision == "bf16"
        and first_divergence is not None
        and first_divergence["operation"] == "outer_update"
        and input_materialization_differs
        and outer_control["exact"]
    )
    correction_boundary_passed = (
        model_activation_precision == "fp32" and first_divergence is None
    )
    if mechanism_proven:
        outcome = "bf16_recurrence_input_materialization_proven"
    elif correction_boundary_passed:
        outcome = "fp32_model_activation_correction_passed_initial_boundary"
    elif model_activation_precision == "fp32":
        outcome = "fp32_model_activation_correction_failed_initial_boundary"
    else:
        outcome = "first_recurrent_divergence_localized_mechanism_unresolved"
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "profile_blake3": MODEL_PROFILE_BLAKE3,
            "checkpoint_blake3": CHECKPOINT_BLAKE3,
            "checkpoint_precision": "bf16",
            "cpu_arithmetic_precision": "fp32_from_bf16",
            "device_activation_precision": model_activation_precision,
            "device_recurrent_matrix_precision": "fp32",
        },
        "execution_harness": {
            "physical_device_id": 0,
            "simulated": False,
            "dispatch": "normal",
            "wkv_execution_policy": "generic-reference",
            "model_matmul_contract": (
                "fp32_model_activations"
                if model_activation_precision == "fp32"
                else "hifi4_fp32_dest_acc_bf16_output"
            ),
            "model_step_count": 1,
            "layer_count": 1,
            "input_token_id": KNOWN_PROMPT_TOKEN_IDS[0],
        },
        "first_divergence": first_divergence,
        "recurrence_input_comparisons": input_comparisons,
        "operation_comparisons": operation_comparisons,
        "device_input_outer_control": outer_control,
        "mechanism": {
            "name": "bf16_recurrence_input_materialization",
            "proven": mechanism_proven,
            "device_outer_matches_exact_device_inputs": outer_control["exact"],
            "cpu_and_device_recurrence_inputs_differ": input_materialization_differs,
        },
        "correction": {
            "name": "fp32_model_activations",
            "probed": model_activation_precision == "fp32",
            "initial_boundary_exact": correction_boundary_passed,
            "full_sixteen_token_probe_started": False,
            "full_recurrent_matrix_parity": False,
        },
        "source_device_probe_blake3": device_document_blake3,
        "retention": {
            "aggregate_only": True,
            "raw_tensor_values_retained": False,
            "device_probe_must_be_removed_after_comparison": True,
            "cache_retained": False,
            "compiler_output_retained": False,
            "profiler_output_retained": False,
            "operational_log_retained": False,
        },
        "decision": {
            "candidate_v3_created": False,
            "admission_gates_started": False,
            "production_windows": [2, 4, 8],
        },
        "non_claims": [
            "No exact final recurrent-matrix parity is established.",
            "No production window-16 admission is established.",
            "No performance result is established.",
        ],
    }


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
    exact_comparison = compare_fp32_values(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])
    )
    divergent_comparison = compare_fp32_values(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.5])
    )
    require(exact_comparison["exact"] is True, "equal FP32 values differ")
    require(
        divergent_comparison["exact"] is False
        and divergent_comparison["first_differing_element"] == 1
        and divergent_comparison["maximum_absolute_error"] == 0.5,
        "different FP32 values were not localized",
    )
    return {
        "outcome": "self_test_pass",
        "positive_orientation_equivalence": True,
        "negative_shape_rejected": True,
        "positive_boundary_comparison": True,
        "negative_boundary_comparison": True,
        "maximum_abs_difference": maximum,
    }


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("derive", "validate"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--compare-device-recurrent-probe", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed.self_test:
        require(parsed.mode is None, "self-test cannot select a run mode")
        json.dump(self_test(), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    require(parsed.model is not None, "model path is required")
    require(parsed.output is not None, "output path is required")
    model_path: Path = parsed.model
    output_path: Path = parsed.output
    require(model_path.is_absolute(), "model path must be absolute")
    require(model_path.is_file(), "model path must be a regular file")
    require(output_path.is_absolute(), "output path must be absolute")
    require(not output_path.exists(), "output path already exists")
    require(output_path.parent.is_dir(), "output parent must be a directory")
    checkpoint_hash = blake3_file(model_path)
    require(checkpoint_hash == CHECKPOINT_BLAKE3, "checkpoint BLAKE3 differs")

    torch.set_grad_enabled(False)
    torch.set_num_threads(CPU_THREAD_COUNT)
    torch.set_num_interop_threads(1)
    weights = load_fp32_weights(model_path)
    if parsed.compare_device_recurrent_probe is not None:
        require(parsed.mode is None, "recurrent comparison cannot select a run mode")
        require(parsed.workload is None, "recurrent comparison cannot select a workload")
        device_path: Path = parsed.compare_device_recurrent_probe
        require(device_path.is_absolute(), "device probe path must be absolute")
        require(device_path.is_file(), "device probe path must be a regular file")
        device_document_blake3 = blake3_file(device_path)
        device_document = json.loads(device_path.read_text(encoding="utf-8"))
        receipt = build_recurrent_diagnosis(
            weights, device_document, device_document_blake3
        )
        output_path.write_text(
            json.dumps(receipt, indent=JSON_INDENT, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output_path)
        return 0

    require(parsed.mode is not None, "run mode is required")
    require(parsed.workload is not None, "workload path is required")
    workload_path: Path = parsed.workload
    mode: Mode = parsed.mode
    validate_paths(model_path, workload_path, output_path)
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    prompt, expected = validate_workload(workload)

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
