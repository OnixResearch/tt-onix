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
    delta: dict[str, Any] | None = None,
    text: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """A chunk with a chat delta or completion text, or the finish reason."""
    if request.endpoint is Endpoint.CHAT_COMPLETIONS:
        choice: dict[str, Any] = {
            "delta": delta or {},
            "finish_reason": finish_reason,
            "index": 0,
        }
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


def generated_token_ids(response: dict[str, Any]) -> list[int]:
    """The response's generated tokens, stop token excluded: the raw text the stream reconciles to."""
    return list(response["x_onix_decode_evidence"]["generated_token_ids"])


class IncrementalDetokenizer:
    """Turn generated tokens into text deltas without splitting a character.

    Each push decodes only the tokens since the last emitted text plus the ones before them
    that fix its context (vLLM's prefix and read offsets), so the cost per token stays bounded
    however long the generation grows. Text ending in U+FFFD stopped inside a multi-byte
    character and waits for the next token. A stop token ends the stream, as
    `finish_generation` ends the complete response.
    """

    def __init__(
        self, decode: Callable[[list[int]], str], stop_token_ids: frozenset[int]
    ) -> None:
        self._decode = decode
        self._stop_token_ids = stop_token_ids
        self._tokens: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0
        self.stopped = False
        self.diverged = False

    def push(self, token: int) -> str:
        if self.stopped:
            return ""
        if token in self._stop_token_ids:
            self.stopped = True
            return ""
        self._tokens.append(token)
        return self._advance(final=False)

    def _advance(self, *, final: bool) -> str:
        prefix_text = self._decode(
            self._tokens[self._prefix_offset : self._read_offset]
        )
        text = self._decode(self._tokens[self._prefix_offset :])
        if len(text) <= len(prefix_text) or (
            not final and text.endswith(REPLACEMENT_CHARACTER)
        ):
            return ""
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._tokens)
        return text[len(prefix_text) :]

    def finish(self, final_token_ids: list[int]) -> str:
        """The text of the complete response's tokens that the stream has not sent yet."""
        sent = len(self._tokens)
        if list(final_token_ids[:sent]) != self._tokens:
            # Sent text cannot be retracted; record the mismatch instead.
            self.diverged = True
            return ""
        self._tokens.extend(final_token_ids[sent:])
        return self._advance(final=True)
