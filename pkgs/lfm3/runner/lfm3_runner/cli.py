"""LFM3-1.7B on one Blackhole P150: prefill + greedy decode with timings.

The checkpoint directory holds the Hugging Face files (config.json,
model.safetensors, tokenizer.json). ``--dtype float32`` is the accurate mode
(the residual stream is float32 and attention uses an explicit float32
softmax); ``bfloat16`` is the fast mode.
"""

# r[impl tenstorrent.lfm3.p150.gpu.generate]
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import ttnn
from tokenizers import Tokenizer

from .model import Lfm3Config, Lfm3Model
from .prompt import THINK_TOKEN_ID, default_checkpoint_dir, render_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=default_checkpoint_dir())
    ap.add_argument("--prompt", default="What is C. elegans?")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the user text N times (long-context smoke)",
    )
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    ap.add_argument("--weight-dtype", default="", choices=["", "bfloat16", "float32"])
    ap.add_argument(
        "--decode-attention",
        default="sdpa",
        choices=["sdpa", "fp32"],
        help="fp32 keeps the KV cache wide but is ~4x slower per token",
    )
    ap.add_argument(
        "--save", default="", help="write the generated token ids to this .npy file"
    )
    args = ap.parse_args()

    ckpt = args.checkpoint_dir
    tok = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))
    text = render_prompt(args.prompt, args.repeat)
    ids = tok.encode(text).ids + [THINK_TOKEN_ID]
    print("prompt_tokens", len(ids), json.dumps(text))

    cfg = Lfm3Config.from_json(os.path.join(ckpt, "config.json"))
    dtype = ttnn.float32 if args.dtype == "float32" else ttnn.bfloat16
    wd = ttnn.float32 if args.weight_dtype == "float32" else ttnn.bfloat16
    weight_dtype = dtype if not args.weight_dtype else wd
    dev = ttnn.open_device(device_id=0, l1_small_size=1 << 15)
    try:
        t0 = time.perf_counter()
        model = Lfm3Model(
            dev,
            cfg,
            os.path.join(ckpt, "model.safetensors"),
            max_seq=args.max_seq,
            dtype=dtype,
            weight_dtype=weight_dtype,
            decode_attention=args.decode_attention,
        )
        print(f"load_s {time.perf_counter() - t0:.2f}")

        t0 = time.perf_counter()
        logits = model.prefill(ids)
        t1 = time.perf_counter()
        cur = int(logits.argmax())
        generated = [cur]
        for _ in range(args.tokens - 1):
            cur = int(model.decode(cur).argmax())
            generated.append(cur)
        decode_s = time.perf_counter() - t1
        print(f"prefill_s {t1 - t0:.3f} ({len(ids) / (t1 - t0):.1f} tok/s)")
        if args.tokens > 1:
            print(f"decode_s {decode_s:.3f} ({(args.tokens - 1) / decode_s:.1f} tok/s)")
        print("generated", generated)
        print("text", json.dumps(tok.decode(generated)))
        if args.save:
            np.save(args.save, np.array(generated, dtype=np.int32))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
