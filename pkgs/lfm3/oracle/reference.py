"""CPU/PyTorch fp32 oracle for LFM3-1.7B: hidden states, prefill logits, greedy decode.

Run this outside the packaged runtime: the checkpoint's remote code declares
``transformers_version = 5.12.1`` and imports
``transformers.integrations.use_kernel_func_from_hub_with_fallback``, which the
nixpkgs transformers does not export. Use a virtualenv with transformers >=
5.12.1 and torch, then point ``lfm3-check --reference`` at the npz this writes.

    python reference.py --checkpoint-dir ~/.cache/lfm3 --tokens 32
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The bare tokenizers library does not match the non-special " thinking" added token.
THINK_TOKEN_ID = 124901


def default_checkpoint_dir() -> str:
    return os.environ.get("LFM3_CHECKPOINT_DIR", os.path.expanduser("~/.cache/lfm3"))


def render_prompt(prompt: str, repeat: int) -> str:
    return (
        "<|startoftext|><|im_start|>user\n"
        + "\n".join([prompt] * repeat)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def main() -> None:
    ckpt = default_checkpoint_dir()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=ckpt)
    ap.add_argument("--prompt", default="What is C. elegans?")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ckpt, "reference.npz"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    # The chat template's generation prompt ends with the single added token " thinking";
    # append it by id so this oracle and the device runner share the exact prompt.
    text = render_prompt(args.prompt, args.repeat)
    ids = torch.tensor([tok(text).input_ids + [THINK_TOKEN_ID]])
    print("prompt_text:", json.dumps(text))
    print("input_ids:", ids.tolist())

    layer_outputs = {}
    handles = [
        layer.register_forward_hook(
            lambda module, inputs, output, i=i: layer_outputs.__setitem__(
                i, (output[0] if isinstance(output, tuple) else output).detach()
            )
        )
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=True)
    for handle in handles:
        handle.remove()
    logits = out.logits.float()
    embed = model.model.embed_tokens(ids).detach()
    final = model.model.embedding_norm(
        layer_outputs[len(model.model.layers) - 1]
    ).detach()
    hidden = (
        [embed.float()]
        + [layer_outputs[i].float() for i in range(len(model.model.layers))]
        + [final.float()]
    )

    cur = int(logits[0, -1].argmax())
    generated = [cur]
    cache = out.past_key_values
    step_logits = []
    for _ in range(args.tokens - 1):
        with torch.no_grad():
            step = model(torch.tensor([[cur]]), past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        lg = step.logits.float()[0, -1]
        step_logits.append(lg.numpy())
        cur = int(lg.argmax())
        generated.append(cur)
    print("generated:", generated)
    print("text:", tok.decode(generated))

    np.savez(
        args.out,
        input_ids=ids.numpy(),
        logits=logits[0, -1].numpy(),
        prefill_last_logits=logits[0, -1].numpy(),
        hidden=np.stack([h[0].numpy() for h in hidden]),
        generated=np.array(generated, dtype=np.int32),
        step_logits=np.stack(step_logits)
        if step_logits
        else np.zeros((0, logits.shape[-1]), np.float32),
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
