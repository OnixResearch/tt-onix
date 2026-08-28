"""Pure request and response policy for the Qwen P150x2 service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

MINIMUM_GENERATION_TOKEN_COUNT = 1
DEFAULT_GENERATION_TOKEN_COUNT = 64
DEFAULT_REQUEST_CHOICE_COUNT = 1
GREEDY_TEMPERATURE = 0.0


class Endpoint(Enum):
    COMPLETIONS = "/v1/completions"
    CHAT_COMPLETIONS = "/v1/chat/completions"


class RequestValidationError(ValueError):
    """The request does not satisfy the admitted service contract."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    endpoint: Endpoint
    model: str
    max_tokens: int
    prompt: str | None
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    seed: int | None = None


def parse_endpoint(path: str) -> Endpoint:
    for endpoint in Endpoint:
        if path == endpoint.value:
            return endpoint
    raise RequestValidationError(f"unsupported endpoint: {path}")


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")
    return payload


def _parse_max_tokens(payload: dict[str, Any], maximum: int) -> int:
    value = payload.get("max_tokens", DEFAULT_GENERATION_TOKEN_COUNT)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestValidationError("max_tokens must be an integer")
    if not MINIMUM_GENERATION_TOKEN_COUNT <= value <= maximum:
        raise RequestValidationError(
            f"max_tokens must be between {MINIMUM_GENERATION_TOKEN_COUNT} and {maximum}"
        )
    return value


def _validate_common_fields(
    payload: dict[str, Any], admitted_model: str, sampled_serving_enabled: bool
) -> None:
    requested_model = payload.get("model", admitted_model)
    if requested_model != admitted_model:
        raise RequestValidationError(f"model must be {admitted_model!r}")

    if payload.get("stream", False) is not False:
        raise RequestValidationError("streaming responses are not supported")

    choice_count = payload.get("n", DEFAULT_REQUEST_CHOICE_COUNT)
    if choice_count != DEFAULT_REQUEST_CHOICE_COUNT:
        raise RequestValidationError("n must be 1")

    temperature = payload.get("temperature", GREEDY_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise RequestValidationError("temperature must be a number")
    if float(temperature) != GREEDY_TEMPERATURE and not sampled_serving_enabled:
        raise RequestValidationError("only greedy temperature 0 is supported")


def _parse_prompt(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestValidationError("prompt must be a non-empty string")
    return prompt


def _parse_messages(payload: dict[str, Any]) -> tuple[ChatMessage, ...]:
    source = payload.get("messages")
    if not isinstance(source, list) or not source:
        raise RequestValidationError("messages must be a non-empty array")

    admitted_roles = {"system", "user", "assistant"}
    messages = []
    user_message_count = 0
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise RequestValidationError(f"messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in admitted_roles:
            raise RequestValidationError(f"messages[{index}].role is not supported")
        if not isinstance(content, str) or not content.strip():
            raise RequestValidationError(
                f"messages[{index}].content must be non-empty text"
            )
        if role == "user":
            user_message_count += 1
        messages.append(ChatMessage(role=role, content=content))

    if user_message_count == 0:
        raise RequestValidationError("messages must contain a user message")
    return tuple(messages)


def _parse_temperature(payload: dict[str, Any]) -> float | None:
    """Parse the sampling temperature for serving-mode routing.

    Missing temperature is deterministic. Valid temperatures are finite and
    non-negative; classification into a serving route happens in the pure
    router core, which fails closed on invalid values.
    """
    value = payload.get("temperature")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError("temperature must be a number or null")
    temperature = float(value)
    if not math.isfinite(temperature):
        raise RequestValidationError("temperature must be finite")
    if temperature < 0:
        raise RequestValidationError("temperature must not be negative")
    return temperature


def _parse_seed(payload: dict[str, Any]) -> int | None:
    """Parse an optional reproducible sampling seed."""
    value = payload.get("seed")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RequestValidationError("seed must be a non-negative integer or null")
    return value


def parse_completion_request(
    payload: Any,
    endpoint: Endpoint,
    admitted_model: str,
    maximum_generation_tokens: int,
    sampled_serving_enabled: bool = False,
) -> CompletionRequest:
    body = _require_object(payload)
    _validate_common_fields(body, admitted_model, sampled_serving_enabled)
    max_tokens = _parse_max_tokens(body, maximum_generation_tokens)

    if endpoint is Endpoint.COMPLETIONS:
        prompt = _parse_prompt(body)
        messages: tuple[ChatMessage, ...] = ()
    else:
        prompt = None
        messages = _parse_messages(body)

    return CompletionRequest(
        endpoint=endpoint,
        model=admitted_model,
        max_tokens=max_tokens,
        prompt=prompt,
        messages=messages,
        temperature=_parse_temperature(body),
        seed=_parse_seed(body),
    )


def build_completion_response(
    request: CompletionRequest,
    request_id: str,
    created: int,
    generated_text: str,
    prompt_token_count: int,
    generated_token_count: int,
) -> dict[str, Any]:
    usage = {
        "completion_tokens": generated_token_count,
        "prompt_tokens": prompt_token_count,
        "total_tokens": prompt_token_count + generated_token_count,
    }
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        choices = [
            {
                "finish_reason": "length",
                "index": 0,
                "message": {"content": generated_text, "role": "assistant"},
            }
        ]
        object_name = "chat.completion"
    else:
        choices = [{"finish_reason": "length", "index": 0, "text": generated_text}]
        object_name = "text_completion"

    return {
        "choices": choices,
        "created": created,
        "id": request_id,
        "model": request.model,
        "object": object_name,
        "usage": usage,
    }
