# SPDX-License-Identifier: LGPL-2.1-only
"""Fixed prompts and the teacher-forced corpus shared by the reference and device runs."""

from __future__ import annotations

import bz2
from pathlib import Path

import models.tt_transformers.tests as tt_transformers_tests  # namespace package: no __file__

PROMPTS = (
    {"name": "plain-capital", "kind": "plain", "text": "The capital of France is"},
    {
        "name": "chat-sky",
        "kind": "chat",
        "messages": [
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}
        ],
        "enable_thinking": False,
    },
    {"name": "plain-fibonacci", "kind": "plain", "text": "def fibonacci(n):\n"},
)
CORPUS_NAME = "tale-of-two-cities.txt.bz2"


def chat_ids(tokenizer, messages, enable_thinking: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        (encoded,) = encoded
    return [int(token) for token in encoded]


def encode(tokenizer, prompt: dict) -> list[int]:
    if prompt["kind"] == "plain":
        return [
            int(token)
            for token in tokenizer.encode(prompt["text"], add_special_tokens=True)
        ]
    if prompt["kind"] == "chat":
        return chat_ids(tokenizer, prompt["messages"], prompt["enable_thinking"])
    raise ValueError(f"unknown prompt kind {prompt['kind']!r}")


def corpus_path() -> Path:
    for directory in tt_transformers_tests.__path__:
        candidate = Path(directory) / CORPUS_NAME
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"{CORPUS_NAME} is not installed under models/tt_transformers/tests"
    )


def corpus_tokens(tokenizer, length: int) -> list[int]:
    text = bz2.open(corpus_path(), "rt", encoding="utf-8").read()
    tokens = tokenizer.encode(text, add_special_tokens=True)[:length]
    if len(tokens) != length:
        raise SystemExit(f"corpus encodes to {len(tokens)} tokens, need {length}")
    return [int(token) for token in tokens]
