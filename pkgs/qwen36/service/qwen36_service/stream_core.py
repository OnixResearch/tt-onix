"""Pure OpenAI-compatible streaming (server-sent events) policy for the Qwen service."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .core import CompletionRequest, Endpoint

SSE_DONE = b"data: [DONE]\n\n"
# A decode that ends in U+FFFD stopped inside a multi-byte character; wait for more tokens.
REPLACEMENT_CHARACTER = "\ufffd"


def sse_event(payload: dict[str, Any]) -> bytes:
    """One `data:` event."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return b"data: " + body.encode("utf-8") + b"\n\n"


def _object_name(request: CompletionRequest) -> str:
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        return "chat.completion.chunk"
    return "text_completion"


def stream_chunk(
    request: CompletionRequest,
    request_id: str,
    created: int,
    *,
    text: str | None = None,
    role: bool = False,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """A chunk with a text delta, the assistant role, or the finish reason."""
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        delta: dict[str, Any] = {}
        if role:
            delta["role"] = "assistant"
        if text is not None:
            delta["content"] = text
        choice: dict[str, Any] = {"delta": delta, "finish_reason": finish_reason, "index": 0}
    else:
        choice = {"finish_reason": finish_reason, "index": 0, "text": text or ""}
    return {
        "choices": [choice],
        "created": created,
        "id": request_id,
        "model": request.model,
        "object": _object_name(request),
    }


def usage_chunk(
    request: CompletionRequest, request_id: str, created: int, usage: dict[str, Any]
) -> dict[str, Any]:
    """The trailing `stream_options.include_usage` chunk: no choices, only usage."""
    return {
        "choices": [],
        "created": created,
        "id": request_id,
        "model": request.model,
        "object": _object_name(request),
        "usage": usage,
    }


def response_text(request: CompletionRequest, response: dict[str, Any]) -> str:
    choice = response["choices"][0]
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        return choice["message"]["content"]
    return choice["text"]


class IncrementalDetokenizer:
    """Turn generated tokens into text deltas without splitting a character.

    Each push decodes the whole sequence so far, as the complete response does, and emits
    only the new suffix. A stop token ends the stream, as `finish_generation` ends the text.
    """

    def __init__(
        self, decode: Callable[[list[int]], str], stop_token_ids: frozenset[int]
    ) -> None:
        self._decode = decode
        self._stop_token_ids = stop_token_ids
        self._tokens: list[int] = []
        self._sent = ""
        self.stopped = False
        self.diverged = False

    def push(self, token: int) -> str:
        if self.stopped:
            return ""
        if token in self._stop_token_ids:
            self.stopped = True
            return ""
        self._tokens.append(token)
        text = self._decode(self._tokens)
        if text.endswith(REPLACEMENT_CHARACTER) or not text.startswith(self._sent):
            return ""
        delta = text[len(self._sent) :]
        self._sent = text
        return delta

    def finish(self, final_text: str) -> str:
        """The rest of the complete response text that the stream has not sent."""
        if not final_text.startswith(self._sent):
            # Sent text cannot be retracted; record the mismatch instead.
            self.diverged = True
            return ""
        delta = final_text[len(self._sent) :]
        self._sent = final_text
        return delta
