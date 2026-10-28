"""Checkpoint location and prompt rendering shared by the device runner and the oracle."""

from __future__ import annotations

import os

# The bare tokenizers library does not match the non-special " thinking" added token.
THINK_TOKEN_ID = 124901


def default_checkpoint_dir() -> str:
    return os.environ.get("LFM3_CHECKPOINT_DIR", os.path.expanduser("~/.cache/lfm3"))


def render_prompt(prompt: str, repeat: int) -> str:
    """The checkpoint's chat template for one user turn, up to the assistant header."""
    return (
        "<|startoftext|><|im_start|>user\n"
        + "\n".join([prompt] * repeat)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
