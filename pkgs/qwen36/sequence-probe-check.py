"""Device-free checks for the isolated diagnostic sequence adapter."""

from types import SimpleNamespace

import torch
from qwen36_service.decode_component_adapter import (
    DecodeComponentAdapterError,
    DecodeComponentProbeAdapter,
)


def check():
    generation_tokens = 3
    prompt_count = 5
    initial_token = 7
    events = []
    adapter = object.__new__(DecodeComponentProbeAdapter)
    layer = SimpleNamespace(
        is_full_attention=False,
        attention=SimpleNamespace(_decode_fusion_mode="norm-state"),
    )
    adapter.model = SimpleNamespace(layers=[layer])
    adapter.profile = SimpleNamespace(
        maximum_sequence_length=prompt_count + generation_tokens
    )

    def prefill(prompt, *, generation_tokens):
        events.append(("prefill", generation_tokens))
        return prompt_count, initial_token

    def replay(*, token, position):
        events.append(("replay", position))
        return token + 1

    adapter._prefill = prefill
    adapter.decode_adapter = SimpleNamespace(
        ensure_trace=lambda **kwargs: SimpleNamespace(eligible=True),
        decode_traced=replay,
    )
    result = adapter.run_sequence(object(), generation_tokens=generation_tokens)
    assert result["fusion_mode"] == "norm-state"
    assert result["generated_token_ids"] == list(
        range(initial_token, initial_token + generation_tokens)
    )
    assert events == [("prefill", generation_tokens)] + [
        ("replay", prompt_count + offset) for offset in range(generation_tokens - 1)
    ]
    for invalid in (0, -1, True, str(generation_tokens)):
        events.clear()
        try:
            adapter.run_sequence(object(), generation_tokens=invalid)
        except DecodeComponentAdapterError:
            assert not events
        else:
            raise AssertionError("invalid sequence length was accepted")
    adapter.decode_adapter.ensure_trace = lambda **kwargs: SimpleNamespace(
        eligible=False, reason="fixture denial"
    )
    events.clear()
    try:
        adapter.run_sequence(object(), generation_tokens=generation_tokens)
    except DecodeComponentAdapterError:
        assert events == [("prefill", generation_tokens)]
    else:
        raise AssertionError("denied trace was accepted")
    for layers in (
        [],
        [
            layer,
            SimpleNamespace(
                is_full_attention=False,
                attention=SimpleNamespace(_decode_fusion_mode="off"),
            ),
        ],
    ):
        adapter.model.layers = layers
        events.clear()
        try:
            adapter.run_sequence(object(), generation_tokens=generation_tokens)
        except DecodeComponentAdapterError:
            assert not events
        else:
            raise AssertionError("missing or mixed GDN modes were accepted")
    try:
        DecodeComponentProbeAdapter._prefill(
            adapter,
            torch.zeros((1, prompt_count), dtype=torch.int32),
            generation_tokens=adapter.profile.maximum_sequence_length,
        )
    except DecodeComponentAdapterError as error:
        assert "exceeds the sequence bound" in str(error)
    else:
        raise AssertionError("oversized sequence was accepted")
    print("PASS diagnostic sequence")


if __name__ == "__main__":
    check()
