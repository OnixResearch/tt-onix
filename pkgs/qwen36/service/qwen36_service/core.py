"""Pure request and response policy for the Qwen P150x2 service."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .chat_output_core import (
    ChatOutputPlan,
    chat_message,
    chat_output_plan,
    parse_chat_output,
)

MINIMUM_GENERATION_TOKEN_COUNT = 1
DEFAULT_REQUEST_CHOICE_COUNT = 1
GREEDY_TEMPERATURE = 0.0
TOOL_CHOICE_AUTO = "auto"
TOOL_CHOICE_NONE = "none"
TOOL_CHOICE_REQUIRED = "required"
# omp's reasoning-effort vocabulary onto the only efforts the chat template admits.
REASONING_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


class Endpoint(Enum):
    COMPLETIONS = "/v1/completions"
    CHAT_COMPLETIONS = "/v1/chat/completions"


class RequestValidationError(ValueError):
    """The request does not satisfy the admitted service contract."""


@dataclass(frozen=True)
class ToolCall:
    """A tool call in the chat history, with its arguments as the template's mapping."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class CompletionRequest:
    endpoint: Endpoint
    model: str
    # Without an explicit limit this is the service cap until the prompt length is known;
    # `admit_prompt` then lowers it to the room left in the sequence.
    max_tokens: int
    prompt: str | None
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    seed: int | None = None
    stream: bool = False
    include_usage: bool = False
    max_tokens_explicit: bool = True
    # The tools offered to the chat template after `tool_choice`; none disables call parsing.
    tools: tuple[dict[str, Any], ...] = ()
    parallel_tool_calls: bool = True
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    reasoning_effort: str | None = None
    # Fixed once the chat prompt is rendered; reads the generated text.
    output_plan: ChatOutputPlan | None = None


def parse_endpoint(path: str) -> Endpoint:
    for endpoint in Endpoint:
        if path == endpoint.value:
            return endpoint
    raise RequestValidationError(f"unsupported endpoint: {path}")


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")
    return payload


def _parse_max_tokens(payload: dict[str, Any], maximum: int) -> tuple[int, bool]:
    """The generation limit and whether the client set it; `max_completion_tokens` is an alias."""
    values = {
        name: payload[name]
        for name in ("max_tokens", "max_completion_tokens")
        if payload.get(name) is not None
    }
    if len(values) == 2 and values["max_tokens"] != values["max_completion_tokens"]:
        raise RequestValidationError("max_tokens and max_completion_tokens differ")
    if not values:
        return maximum, False
    name, value = next(iter(values.items()))
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestValidationError(f"{name} must be an integer")
    if not MINIMUM_GENERATION_TOKEN_COUNT <= value <= maximum:
        raise RequestValidationError(
            f"{name} must be between {MINIMUM_GENERATION_TOKEN_COUNT} and {maximum}"
        )
    return value, True


def _validate_common_fields(
    payload: dict[str, Any], admitted_model: str, sampled_serving_enabled: bool
) -> None:
    requested_model = payload.get("model", admitted_model)
    if requested_model != admitted_model:
        raise RequestValidationError(f"model must be {admitted_model!r}")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestValidationError("stream must be a boolean")
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        if not stream:
            raise RequestValidationError("stream_options requires stream")
        if not isinstance(stream_options, dict):
            raise RequestValidationError("stream_options must be an object")
        if not isinstance(stream_options.get("include_usage", False), bool):
            raise RequestValidationError("stream_options.include_usage must be a boolean")

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
    """Chat history, held to what the chat template renders (TPL 88-160)."""
    source = payload.get("messages")
    if not isinstance(source, list) or not source:
        raise RequestValidationError("messages must be a non-empty array")

    messages = []
    has_user_query = False
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise RequestValidationError(f"messages[{index}] must be an object")
        role = item.get("role")
        if role == "system":
            if index != 0:
                raise RequestValidationError(f"messages[{index}]: a system message must come first")
            message = ChatMessage(role=role, content=_required_text(item.get("content"), index))
        elif role == "user":
            content = _required_text(item.get("content"), index)
            stripped = content.strip()
            has_user_query = has_user_query or not (
                stripped.startswith(TOOL_RESPONSE_OPEN) and stripped.endswith(TOOL_RESPONSE_CLOSE)
            )
            message = ChatMessage(role=role, content=content)
        elif role == "assistant":
            message = _parse_assistant_message(item, index)
        elif role == "tool":
            if not messages or messages[-1].role not in ("assistant", "tool"):
                raise RequestValidationError(
                    f"messages[{index}]: a tool message must follow an assistant or tool message"
                )
            message = ChatMessage(role=role, content=_optional_text(item.get("content"), index))
        else:
            raise RequestValidationError(f"messages[{index}].role is not supported")
        messages.append(message)

    if not has_user_query:
        raise RequestValidationError("messages must contain a user message")
    return tuple(messages)


def _parse_assistant_message(item: dict[str, Any], index: int) -> ChatMessage:
    reasoning = item.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise RequestValidationError(f"messages[{index}].reasoning_content must be a string")
    source = item.get("tool_calls")
    if source is None:
        source = []
    if not isinstance(source, list):
        raise RequestValidationError(f"messages[{index}].tool_calls must be an array")
    return ChatMessage(
        role="assistant",
        content=_optional_text(item.get("content"), index),
        reasoning_content=reasoning,
        tool_calls=tuple(
            _parse_history_tool_call(call, f"messages[{index}].tool_calls[{call_index}]")
            for call_index, call in enumerate(source)
        ),
    )


def _parse_history_tool_call(call: Any, location: str) -> ToolCall:
    if not isinstance(call, dict):
        raise RequestValidationError(f"{location} must be an object")
    if call.get("type", "function") != "function":
        raise RequestValidationError(f"{location}.type must be function")
    function = call.get("function")
    if not isinstance(function, dict):
        raise RequestValidationError(f"{location}.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestValidationError(f"{location}.function.name must be a non-empty string")
    arguments = function.get("arguments")
    # OpenAI clients send arguments as a JSON string; the template iterates a mapping.
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        arguments = {}
    elif isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RequestValidationError(
                f"{location}.function.arguments must be a JSON object"
            ) from error
    if not isinstance(arguments, dict):
        raise RequestValidationError(f"{location}.function.arguments must be a JSON object")
    return ToolCall(name=name, arguments=arguments)


def _message_text(content: Any, index: int) -> str | None:
    """A message's text: a string, or text parts joined as the chat template joins them."""
    if isinstance(content, list):
        texts = []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "text":
                raise RequestValidationError(
                    f"messages[{index}].content[{part_index}] must be a text part"
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise RequestValidationError(
                    f"messages[{index}].content must be non-empty text"
                )
            texts.append(text)
        content = "".join(texts)
    return content if isinstance(content, str) else None


def _required_text(content: Any, index: int) -> str:
    text = _message_text(content, index)
    if text is None or not text.strip():
        raise RequestValidationError(f"messages[{index}].content must be non-empty text")
    return text


def _optional_text(content: Any, index: int) -> str:
    """Assistant and tool content may be empty or null."""
    if content is None:
        return ""
    text = _message_text(content, index)
    if text is None:
        raise RequestValidationError(f"messages[{index}].content must be text")
    return text


def _parse_tools(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    source = payload.get("tools")
    if source is None:
        return ()
    if not isinstance(source, list):
        raise RequestValidationError("tools must be an array")
    for index, tool in enumerate(source):
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise RequestValidationError(f"tools[{index}] must be a function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RequestValidationError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RequestValidationError(f"tools[{index}].function.name must be a non-empty string")
        if function.get("parameters") is not None and not isinstance(function["parameters"], dict):
            raise RequestValidationError(f"tools[{index}].function.parameters must be an object")
        if function.get("description") is not None and not isinstance(function["description"], str):
            raise RequestValidationError(f"tools[{index}].function.description must be a string")
    return tuple(source)


def _select_tools(
    payload: dict[str, Any], tools: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """The tools offered to the template: `none` hides them, a named choice narrows them.

    `required` and a named choice are best effort; nothing constrains decoding."""
    choice = payload.get("tool_choice")
    if choice is None or choice == TOOL_CHOICE_AUTO:
        return tools
    if choice == TOOL_CHOICE_NONE:
        return ()
    if choice == TOOL_CHOICE_REQUIRED:
        if not tools:
            raise RequestValidationError("tool_choice required needs tools")
        return tools
    if isinstance(choice, dict) and choice.get("type", "function") == "function":
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            selected = tuple(tool for tool in tools if tool["function"]["name"] == name)
            if not selected:
                raise RequestValidationError(f"tool_choice names {name!r}, which is not in tools")
            return selected
    raise RequestValidationError(
        "tool_choice must be auto, none, required or {type: function, function: {name}}"
    )


def _thinking_field(
    payload: dict[str, Any], template_arguments: dict[str, Any], name: str
) -> Any:
    """A thinking control from the top level, else from chat_template_kwargs."""
    value = payload.get(name)
    return template_arguments.get(name) if value is None else value


def _parse_thinking(payload: dict[str, Any]) -> tuple[bool | None, bool | None, str | None]:
    """enable_thinking, preserve_thinking and the template's reasoning effort.

    Only these chat_template_kwargs reach the template; other keys are ignored."""
    template_arguments = payload.get("chat_template_kwargs")
    if template_arguments is None:
        template_arguments = {}
    if not isinstance(template_arguments, dict):
        raise RequestValidationError("chat_template_kwargs must be an object")
    flags = []
    for name in ("enable_thinking", "preserve_thinking"):
        value = _thinking_field(payload, template_arguments, name)
        if value is not None and not isinstance(value, bool):
            raise RequestValidationError(f"{name} must be a boolean")
        flags.append(value)
    effort = _thinking_field(payload, template_arguments, "reasoning_effort")
    if effort is not None:
        if effort not in REASONING_EFFORTS:
            raise RequestValidationError(
                "reasoning_effort must be one of " + ", ".join(REASONING_EFFORTS)
            )
        effort = REASONING_EFFORTS[effort]
    return flags[0], flags[1], effort


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
    max_tokens, max_tokens_explicit = _parse_max_tokens(body, maximum_generation_tokens)

    stream = body.get("stream", False)
    request = CompletionRequest(
        endpoint=endpoint,
        model=admitted_model,
        max_tokens=max_tokens,
        prompt=None,
        messages=(),
        temperature=_parse_temperature(body),
        seed=_parse_seed(body),
        stream=stream,
        include_usage=bool(stream and (body.get("stream_options") or {}).get("include_usage", False)),
        max_tokens_explicit=max_tokens_explicit,
    )
    if endpoint is Endpoint.COMPLETIONS:
        return replace(request, prompt=_parse_prompt(body))

    parallel_tool_calls = body.get("parallel_tool_calls", True)
    if not isinstance(parallel_tool_calls, bool):
        raise RequestValidationError("parallel_tool_calls must be a boolean")
    enable_thinking, preserve_thinking, reasoning_effort = _parse_thinking(body)
    return replace(
        request,
        messages=_parse_messages(body),
        tools=_select_tools(body, _parse_tools(body)),
        parallel_tool_calls=parallel_tool_calls,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        reasoning_effort=reasoning_effort,
    )


def template_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    """Chat history in the shape the chat template reads."""
    messages = []
    for message in request.messages:
        rendered: dict[str, Any] = {"content": message.content, "role": message.role}
        if message.reasoning_content is not None:
            rendered["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            rendered["tool_calls"] = [
                {"function": {"arguments": call.arguments, "name": call.name}, "type": "function"}
                for call in message.tool_calls
            ]
        messages.append(rendered)
    return messages


def template_arguments(request: CompletionRequest) -> dict[str, Any]:
    """The thinking controls the client set; unset ones keep the template's defaults."""
    arguments = {
        "enable_thinking": request.enable_thinking,
        "preserve_thinking": request.preserve_thinking,
        "reasoning_effort": request.reasoning_effort,
    }
    return {name: value for name, value in arguments.items() if value is not None}


def admit_prompt(
    request: CompletionRequest, prompt: str, prompt_token_count: int, maximum_sequence_length: int
) -> CompletionRequest:
    """Fit generation to the sequence and fix how the chat output will be read.

    Without an explicit limit a request may generate up to the service cap or the room the
    prompt leaves, whichever is smaller."""
    max_tokens = request.max_tokens
    if not request.max_tokens_explicit:
        max_tokens = max(
            MINIMUM_GENERATION_TOKEN_COUNT,
            min(max_tokens, maximum_sequence_length - prompt_token_count),
        )
    total_token_count = prompt_token_count + max_tokens
    if total_token_count > maximum_sequence_length:
        # OpenAI's wording, which clients such as omp recognize as a context overflow and
        # answer by compacting the conversation.
        raise RequestValidationError(
            f"This model's maximum context length is {maximum_sequence_length} tokens. "
            f"The prompt ({prompt_token_count} tokens) and generation ({max_tokens} tokens) "
            f"require {total_token_count}. Please reduce the length of the messages or max_tokens."
        )
    output_plan = (
        chat_output_plan(prompt, request.tools, request.parallel_tool_calls)
        if request.endpoint is Endpoint.CHAT_COMPLETIONS
        else None
    )
    return replace(request, max_tokens=max_tokens, output_plan=output_plan)


FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"


def load_stop_token_ids(model_path: str | Path, tokenizer: Any) -> frozenset[int]:
    """End-of-sequence ids: generation_config's eos_token_id (id or list) and the tokenizer's."""
    ids: set[int] = set()
    config_path = Path(model_path) / "generation_config.json"
    if config_path.is_file():
        value = json.loads(config_path.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(value, int) and not isinstance(value, bool):
            ids.add(value)
        elif isinstance(value, list):
            ids.update(item for item in value if isinstance(item, int) and not isinstance(item, bool))
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and not isinstance(eos, bool):
        ids.add(eos)
    return frozenset(ids)


def finish_generation(tokens: list[int], stop_token_ids: frozenset[int]) -> tuple[list[int], str]:
    """Tokens up to (excluding) the first stop token, and the OpenAI finish reason."""
    for index, token in enumerate(tokens):
        if token in stop_token_ids:
            return list(tokens[:index]), FINISH_STOP
    return list(tokens), FINISH_LENGTH


def chat_finish_reason(generation_finish: str, completed_call_count: int, open_call: bool) -> str:
    """`tool_calls` only for complete calls ended by a stop token.

    A run out of budget stays `length` even after tool deltas went out, so a client never
    executes a truncated call."""
    if generation_finish == FINISH_LENGTH:
        return FINISH_LENGTH
    if completed_call_count and not open_call:
        return FINISH_TOOL_CALLS
    return FINISH_STOP


def build_completion_response(
    request: CompletionRequest,
    request_id: str,
    created: int,
    generated_text: str,
    prompt_token_count: int,
    generated_token_count: int,
    finish_reason: str = FINISH_LENGTH,
) -> dict[str, Any]:
    """The complete response; `finish_reason` says how generation ended (stop or length)."""
    usage = {
        "completion_tokens": generated_token_count,
        "prompt_tokens": prompt_token_count,
        "total_tokens": prompt_token_count + generated_token_count,
    }
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        if request.output_plan is None:
            raise ValueError("a chat response needs the output plan of its rendered prompt")
        output = parse_chat_output(generated_text, request.output_plan)
        choices = [
            {
                "finish_reason": chat_finish_reason(
                    finish_reason, output.completed_call_count, output.open_call
                ),
                "index": 0,
                "message": chat_message(output),
            }
        ]
        object_name = "chat.completion"
    else:
        choices = [{"finish_reason": finish_reason, "index": 0, "text": generated_text}]
        object_name = "text_completion"

    return {
        "choices": choices,
        "created": created,
        "id": request_id,
        "model": request.model,
        "object": object_name,
        "usage": usage,
    }
