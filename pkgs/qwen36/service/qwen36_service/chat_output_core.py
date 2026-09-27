"""Pure parser that splits Qwen3.8 chat output into reasoning, content and tool calls.

The chat template (chat_template.jinja) opens the reasoning section in the prompt
(`<think>\\n`) and teaches this call format:

    REASONING\\n</think>\\n\\nCONTENT<tool_call>\\n<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>

One incremental parser serves the streaming and the complete responses: text arrives as
deltas, each delta is scanned once, and a tail that could still become a tag is held back.
Feeding a text in one piece or character by character yields the same result.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

THINK_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"
# The template writes a value as `\n{value}\n</parameter>`; one newline on each side is framing.
VALUE_CLOSE = "\n" + PARAMETER_CLOSE
# A prompt ending here leaves the model inside its reasoning; one ending in `</think>\n\n` does not.
REASONING_PROMPT_SUFFIX = "<think>\n"

EVENT_REASONING = "reasoning"
EVENT_CONTENT = "content"
EVENT_TOOL_CALL_START = "tool_call_start"
EVENT_TOOL_CALL_ARGUMENTS = "tool_call_arguments"

JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
STRING_ONLY = frozenset({"string"})

_REASONING = "reasoning"
_CONTENT = "content"
_SEEK_FUNCTION = "seek-function"
_IN_FUNCTION = "in-function"
_STRING_VALUE = "string-value"
_BUFFERED_VALUE = "buffered-value"
_BETWEEN_CALLS = "between-calls"
_FUNCTION_MARKERS = (PARAMETER_OPEN, FUNCTION_CLOSE, TOOL_CALL_CLOSE)
_BETWEEN_MARKERS = (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)

# A parameter type map: function name -> parameter name -> admitted JSON types (None: unknown).
ArgumentTypes = Mapping[str, Mapping[str, frozenset[str] | None]]


def new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ChatOutputPlan:
    """How to read one chat generation, fixed when its prompt is rendered."""

    starts_in_reasoning: bool
    parse_tool_calls: bool
    parallel_tool_calls: bool = True
    argument_types: ArgumentTypes = field(default_factory=dict)


@dataclass(frozen=True)
class OutputEvent:
    kind: str
    text: str = ""
    index: int = -1
    call_id: str = ""
    name: str = ""


@dataclass(frozen=True)
class ParsedToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatOutput:
    reasoning: str
    content: str
    tool_calls: tuple[ParsedToolCall, ...]
    completed_call_count: int
    open_call: bool


def _value_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def schema_types(schema: Any) -> frozenset[str] | None:
    """The JSON types a parameter schema admits, or None when they cannot be told."""
    if not isinstance(schema, dict):
        return None
    declared = schema.get("type")
    if isinstance(declared, str):
        return frozenset({declared}) if declared in JSON_TYPES else None
    if isinstance(declared, list):
        if declared and all(isinstance(item, str) and item in JSON_TYPES for item in declared):
            return frozenset(declared)
        return None
    for key in ("anyOf", "oneOf"):
        members = schema.get(key)
        if isinstance(members, list) and members:
            admitted: set[str] = set()
            for member in members:
                member_types = schema_types(member)
                if member_types is None:
                    return None
                admitted |= member_types
            return frozenset(admitted)
    values = schema.get("enum")
    if isinstance(values, list) and values:
        return frozenset(_value_json_type(value) for value in values)
    if "const" in schema:
        return frozenset({_value_json_type(schema["const"])})
    return None


def argument_types(tools: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, frozenset[str] | None]]:
    """Parameter types per function from OpenAI `tools` entries."""
    types: dict[str, dict[str, frozenset[str] | None]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        types[function["name"]] = (
            {name: schema_types(schema) for name, schema in properties.items()}
            if isinstance(properties, Mapping)
            else {}
        )
    return types


def chat_output_plan(
    prompt: str, tools: Iterable[Mapping[str, Any]], parallel_tool_calls: bool
) -> ChatOutputPlan:
    """Seed the parser from the rendered prompt and the tools offered to the template."""
    offered = tuple(tools)
    return ChatOutputPlan(
        starts_in_reasoning=prompt.endswith(REASONING_PROMPT_SUFFIX),
        parse_tool_calls=bool(offered),
        parallel_tool_calls=parallel_tool_calls,
        argument_types=argument_types(offered),
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def coerce_argument(raw: str, admitted: frozenset[str] | None) -> Any:
    """A parameter value from its rendered text: raw for strings, JSON for the rest.

    The template writes string values raw and every other value as JSON. A value whose JSON
    reading is not an admitted non-string type stays the raw text.
    """
    if admitted == STRING_ONLY:
        return raw
    try:
        value = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
    except ValueError:
        return raw
    if admitted is None:
        return value
    value_type = _value_json_type(value)
    if value_type == "string":
        return raw
    if value_type in admitted or (value_type == "integer" and "number" in admitted):
        return value
    return raw


def _json_string_fragment(text: str) -> str:
    """JSON escaping works per character, so escaped fragments concatenate to the escaped whole."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _partial_suffix(text: str, markers: Iterable[str]) -> int:
    """Length of the longest tail of `text` that is a proper prefix of one of `markers`."""
    longest = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), longest, -1):
            if marker.startswith(text[-length:]):
                longest = length
                break
    return longest


class _TrimmedText:
    """A text channel without leading newlines or trailing whitespace, however it is chunked."""

    def __init__(self) -> None:
        self.started = False
        self._pending = ""

    def feed(self, text: str) -> str:
        if not self.started:
            text = text.lstrip("\n")
            if not text:
                return ""
            self.started = True
        body = text.rstrip()
        if not body:
            self._pending += text
            return ""
        emitted = self._pending + body
        self._pending = text[len(body) :]
        return emitted


class ChatOutputParser:
    """Turn decoded chat text deltas into reasoning, content and tool-call events."""

    def __init__(
        self, plan: ChatOutputPlan, new_call_id: Callable[[], str] = new_tool_call_id
    ) -> None:
        self._plan = plan
        self._new_call_id = new_call_id
        self._state = _REASONING if plan.starts_in_reasoning else _CONTENT
        self._buffer = ""
        # Characters of the buffer already searched for a closing tag in the buffered states.
        self._scanned = 0
        self._final = False
        self._reasoning = _TrimmedText()
        self._content = _TrimmedText()
        self._events: list[OutputEvent] = []
        self._call_index = -1
        self._call_emitted = False
        self._call_parameter_count = 0
        self._parameter_prefix = ""
        self._parameter_types: frozenset[str] | None = None
        self._function_types: Mapping[str, frozenset[str] | None] = {}
        self.completed_call_count = 0
        self.open_call = False

    def push(self, text: str) -> list[OutputEvent]:
        if self._final:
            raise RuntimeError("the parser has finished")
        self._buffer += text
        return self._drain()

    def finish(self) -> list[OutputEvent]:
        """Flush held-back text; close the JSON of a call the output left open."""
        if self._final:
            return []
        self._final = True
        while self._step():
            pass
        state = self._state
        if state == _SEEK_FUNCTION:
            # `<tool_call>` without a function header is not a call; keep the text.
            self._emit_content(TOOL_CALL_OPEN + self._buffer)
        elif state == _STRING_VALUE:
            self._arguments(_json_string_fragment(self._buffer) + '"')
            self._close_call(completed=False)
        elif state == _BUFFERED_VALUE:
            self._arguments(self._parameter_prefix + self._coerced(self._buffer))
            self._close_call(completed=False)
        elif state == _IN_FUNCTION:
            self._close_call(completed=False)
        self._buffer = ""
        events, self._events = self._events, []
        return events

    def _drain(self) -> list[OutputEvent]:
        while self._step():
            pass
        events, self._events = self._events, []
        return events

    def _consume(self, length: int) -> None:
        self._buffer = self._buffer[length:]
        self._scanned = 0

    def _hold(self, markers: Iterable[str]) -> int:
        return 0 if self._final else _partial_suffix(self._buffer, markers)

    def _find(self, marker: str) -> int:
        """Find `marker` in the buffer, skipping text an earlier call already searched."""
        position = self._buffer.find(marker, max(0, self._scanned - len(marker) + 1))
        if position < 0:
            self._scanned = len(self._buffer)
        return position

    def _step(self) -> bool:
        """Advance over the buffer; False once more text is needed."""
        state = self._state
        if state == _REASONING:
            return self._step_reasoning()
        if state == _CONTENT:
            return self._step_content()
        if state == _SEEK_FUNCTION:
            return self._step_seek_function()
        if state == _IN_FUNCTION:
            return self._step_in_function()
        if state == _STRING_VALUE:
            return self._step_string_value()
        if state == _BUFFERED_VALUE:
            return self._step_buffered_value()
        return self._step_between_calls()

    def _step_reasoning(self) -> bool:
        position = self._buffer.find(THINK_CLOSE)
        if position < 0:
            end = len(self._buffer) - self._hold((THINK_CLOSE,))
            self._emit(EVENT_REASONING, self._reasoning.feed(self._buffer[:end]))
            self._consume(end)
            return False
        self._emit(EVENT_REASONING, self._reasoning.feed(self._buffer[:position]))
        self._consume(position + len(THINK_CLOSE))
        self._state = _CONTENT
        return True

    def _step_content(self) -> bool:
        if not self._plan.parse_tool_calls:
            self._emit_content(self._buffer)
            self._consume(len(self._buffer))
            return False
        position = self._buffer.find(TOOL_CALL_OPEN)
        if position < 0:
            end = len(self._buffer) - self._hold((TOOL_CALL_OPEN,))
            self._emit_content(self._buffer[:end])
            self._consume(end)
            return False
        # Whitespace before the call stays pending in the content channel: it separates this
        # content from any content that follows the calls.
        self._emit_content(self._buffer[:position])
        self._consume(position + len(TOOL_CALL_OPEN))
        self._state = _SEEK_FUNCTION
        return True

    def _step_seek_function(self) -> bool:
        header = self._find(FUNCTION_OPEN)
        close = self._buffer.find(TOOL_CALL_CLOSE)
        if close >= 0 and (header < 0 or close < header):
            # A call block without a function header is not a call; keep it as content.
            end = close + len(TOOL_CALL_CLOSE)
            self._emit_content(TOOL_CALL_OPEN + self._buffer[:end])
            self._consume(end)
            self._state = _CONTENT
            return True
        if header < 0:
            return False
        name_end = self._buffer.find(">", header + len(FUNCTION_OPEN))
        if name_end < 0:
            return False
        name = self._buffer[header + len(FUNCTION_OPEN) : name_end].strip()
        self._consume(name_end + 1)
        self._open_call(name)
        return True

    def _step_in_function(self) -> bool:
        stripped = self._buffer.lstrip()
        if len(stripped) != len(self._buffer):
            self._consume(len(self._buffer) - len(stripped))
        buffer = self._buffer
        if not buffer:
            return False
        if buffer.startswith(PARAMETER_OPEN):
            name_end = buffer.find(">", len(PARAMETER_OPEN))
            # The newline after `>` is framing; wait for the character that tells.
            if name_end < 0 or (name_end + 1 == len(buffer) and not self._final):
                return False
            name = buffer[len(PARAMETER_OPEN) : name_end].strip()
            value_start = name_end + 1
            if buffer.startswith("\n", value_start):
                value_start += 1
            self._consume(value_start)
            self._open_parameter(name)
            return True
        if buffer.startswith(FUNCTION_CLOSE):
            self._consume(len(FUNCTION_CLOSE))
            self._close_call(completed=True)
            return True
        if buffer.startswith(TOOL_CALL_CLOSE):
            self._consume(len(TOOL_CALL_CLOSE))
            self._close_call(completed=True)
            return True
        if not self._final and any(marker.startswith(buffer) for marker in _FUNCTION_MARKERS):
            return False
        # Stray text inside a call: skip to the next tag.
        positions = [
            position
            for position in (buffer.find(marker) for marker in _FUNCTION_MARKERS)
            if position > 0
        ]
        if positions:
            self._consume(min(positions))
            return True
        self._consume(len(buffer) - self._hold(_FUNCTION_MARKERS))
        return False

    def _step_string_value(self) -> bool:
        close = self._buffer.find(PARAMETER_CLOSE)
        if close < 0:
            end = len(self._buffer) - self._hold((VALUE_CLOSE, PARAMETER_CLOSE))
            if end > 0:
                self._arguments(_json_string_fragment(self._buffer[:end]))
                self._consume(end)
            return False
        value = self._buffer[:close]
        if value.endswith("\n"):
            value = value[:-1]
        self._arguments(_json_string_fragment(value) + '"')
        self._consume(close + len(PARAMETER_CLOSE))
        self._state = _IN_FUNCTION
        return True

    def _step_buffered_value(self) -> bool:
        close = self._find(PARAMETER_CLOSE)
        if close < 0:
            return False
        value = self._buffer[:close]
        if value.endswith("\n"):
            value = value[:-1]
        self._arguments(self._parameter_prefix + self._coerced(value))
        self._consume(close + len(PARAMETER_CLOSE))
        self._state = _IN_FUNCTION
        return True

    def _step_between_calls(self) -> bool:
        stripped = self._buffer.lstrip()
        if len(stripped) != len(self._buffer):
            self._consume(len(self._buffer) - len(stripped))
        buffer = self._buffer
        if not buffer:
            return False
        if buffer.startswith(TOOL_CALL_OPEN):
            self._consume(len(TOOL_CALL_OPEN))
            self._state = _SEEK_FUNCTION
            return True
        if buffer.startswith(TOOL_CALL_CLOSE):
            self._consume(len(TOOL_CALL_CLOSE))
            return True
        if not self._final and any(marker.startswith(buffer) for marker in _BETWEEN_MARKERS):
            return False
        # Text after a call is content, not dropped.
        self._state = _CONTENT
        return True

    def _emit(self, kind: str, text: str) -> None:
        if not text:
            return
        last = self._events[-1] if self._events else None
        if last is not None and last.kind == kind and kind in (EVENT_REASONING, EVENT_CONTENT):
            self._events[-1] = OutputEvent(kind=kind, text=last.text + text)
        else:
            self._events.append(OutputEvent(kind=kind, text=text))

    def _emit_content(self, text: str) -> None:
        self._emit(EVENT_CONTENT, self._content.feed(text))

    def _open_call(self, name: str) -> None:
        # Without parallel calls only the first call is reported; later ones are parsed and skipped.
        self._call_emitted = self._plan.parallel_tool_calls or self._call_index < 0
        self._call_parameter_count = 0
        self._function_types = self._plan.argument_types.get(name, {})
        self._state = _IN_FUNCTION
        if not self._call_emitted:
            return
        self._call_index += 1
        self.open_call = True
        self._events.append(
            OutputEvent(
                kind=EVENT_TOOL_CALL_START,
                index=self._call_index,
                call_id=self._new_call_id(),
                name=name,
            )
        )

    def _open_parameter(self, name: str) -> None:
        opener = "{" if self._call_parameter_count == 0 else ", "
        self._call_parameter_count += 1
        self._parameter_prefix = opener + json.dumps(name, ensure_ascii=False) + ": "
        self._parameter_types = self._function_types.get(name)
        if self._parameter_types == STRING_ONLY:
            self._arguments(self._parameter_prefix + '"')
            self._state = _STRING_VALUE
        else:
            self._state = _BUFFERED_VALUE

    def _coerced(self, value: str) -> str:
        return json.dumps(coerce_argument(value, self._parameter_types), ensure_ascii=False)

    def _arguments(self, fragment: str) -> None:
        if not self._call_emitted or not fragment:
            return
        last = self._events[-1] if self._events else None
        if (
            last is not None
            and last.kind == EVENT_TOOL_CALL_ARGUMENTS
            and last.index == self._call_index
        ):
            self._events[-1] = OutputEvent(
                kind=EVENT_TOOL_CALL_ARGUMENTS, text=last.text + fragment, index=self._call_index
            )
        else:
            self._events.append(
                OutputEvent(kind=EVENT_TOOL_CALL_ARGUMENTS, text=fragment, index=self._call_index)
            )

    def _close_call(self, *, completed: bool) -> None:
        self._arguments("}" if self._call_parameter_count else "{}")
        if self._call_emitted:
            self.open_call = not completed
            if completed:
                self.completed_call_count += 1
        self._state = _BETWEEN_CALLS


def collect_output(
    events: Iterable[OutputEvent], completed_call_count: int, open_call: bool
) -> ChatOutput:
    """Join parser events into one complete message."""
    reasoning: list[str] = []
    content: list[str] = []
    calls: dict[int, list[str]] = {}
    for event in events:
        if event.kind == EVENT_REASONING:
            reasoning.append(event.text)
        elif event.kind == EVENT_CONTENT:
            content.append(event.text)
        elif event.kind == EVENT_TOOL_CALL_START:
            calls[event.index] = [event.call_id, event.name, ""]
        else:
            calls[event.index][2] += event.text
    return ChatOutput(
        reasoning="".join(reasoning),
        content="".join(content),
        tool_calls=tuple(
            ParsedToolCall(call_id=call_id, name=name, arguments=arguments)
            for call_id, name, arguments in (calls[index] for index in sorted(calls))
        ),
        completed_call_count=completed_call_count,
        open_call=open_call,
    )


def parse_chat_output(
    text: str, plan: ChatOutputPlan, new_call_id: Callable[[], str] = new_tool_call_id
) -> ChatOutput:
    """Parse one complete generation with the streaming parser."""
    parser = ChatOutputParser(plan, new_call_id)
    events = parser.push(text) + parser.finish()
    return collect_output(events, parser.completed_call_count, parser.open_call)


def chat_message(output: ChatOutput) -> dict[str, Any]:
    """The OpenAI assistant message: null content beside tool calls, reasoning when present."""
    message: dict[str, Any] = {
        "content": output.content if output.content or not output.tool_calls else None,
        "role": "assistant",
    }
    if output.reasoning:
        message["reasoning_content"] = output.reasoning
    if output.tool_calls:
        message["tool_calls"] = [
            {
                "function": {"arguments": call.arguments, "name": call.name},
                "id": call.call_id,
                "type": "function",
            }
            for call in output.tool_calls
        ]
    return message


def event_deltas(events: Iterable[OutputEvent]) -> list[dict[str, Any]]:
    """OpenAI chat-chunk deltas for parser events, in order."""
    deltas: list[dict[str, Any]] = []
    for event in events:
        if event.kind == EVENT_REASONING:
            deltas.append({"reasoning_content": event.text})
        elif event.kind == EVENT_CONTENT:
            deltas.append({"content": event.text})
        elif event.kind == EVENT_TOOL_CALL_START:
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "function": {"arguments": "", "name": event.name},
                            "id": event.call_id,
                            "index": event.index,
                            "type": "function",
                        }
                    ]
                }
            )
        else:
            deltas.append(
                {"tool_calls": [{"function": {"arguments": event.text}, "index": event.index}]}
            )
    return deltas
