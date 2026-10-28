"""Device numerical gate: prefill/decode against the fp32 HF oracle dump.

The oracle is produced by ``lfm3-reference`` on a CPU host. This gate compares
every captured hidden state (embedding, 24 layers, final norm), the prefill
logits, and the greedy decode tokens/logits.
"""

# r[verify tenstorrent.lfm3.accuracy.oracle.short]
# r[verify tenstorrent.lfm3.accuracy.oracle.long]
from __future__ import annotations

import argparse
import os

import numpy as np
import ttnn

from .model import Lfm3Config, Lfm3Model
from .prompt import default_checkpoint_dir


def stats(name: str, got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    diff = np.abs(got - want)
    denom = np.maximum(np.abs(want), 1e-6)
    print(
        f"{name:24s} max_abs={diff.max():.6g} mean_abs={diff.mean():.6g} "
        f"max_rel={np.max(diff / denom):.6g} pcc={np.corrcoef(got.ravel(), want.ravel())[0, 1]:.8f}"
    )
    return float(diff.max())


def main() -> None:
    ckpt = default_checkpoint_dir()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=ckpt)
    ap.add_argument("--reference", default=os.path.join(ckpt, "reference.npz"))
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    ap.add_argument(
        "--decode-attention",
        default="sdpa",
        choices=["sdpa", "fp32"],
        help="fp32 keeps the KV cache wide but is ~4x slower per token",
    )
    args = ap.parse_args()

    ref = np.load(args.reference)
    ids = ref["input_ids"][0].tolist()
    cfg = Lfm3Config.from_json(os.path.join(args.checkpoint_dir, "config.json"))
    dtype = ttnn.float32 if args.dtype == "float32" else ttnn.bfloat16
    dev = ttnn.open_device(device_id=0, l1_small_size=1 << 15)
    try:
        model = Lfm3Model(
            dev,
            cfg,
            os.path.join(args.checkpoint_dir, "model.safetensors"),
            max_seq=args.max_seq,
            dtype=dtype,
            weight_dtype=dtype,
            decode_attention=args.decode_attention,
        )
        captured = model.prefill(ids, capture_layers=True)
        worst = 0.0
        for i, (got, want) in enumerate(zip(captured, ref["hidden"])):
            name = (
                "embed"
                if i == 0
                else ("final_norm" if i == len(captured) - 1 else f"layer{i - 1}")
            )
            worst = max(worst, stats(name, got.float().numpy(), want))
        logits = model.prefill(ids)
        last = stats("prefill_logits", logits.numpy(), ref["prefill_last_logits"])
        print(
            "prefill argmax",
            int(logits.argmax()),
            "want",
            int(ref["prefill_last_logits"].argmax()),
        )

        if not args.skip_decode:
            generated = ref["generated"].tolist()
            got_tokens, got_logits = [], []
            for t in generated[:-1]:
                lg = model.decode(t)
                got_logits.append(lg.numpy())
                got_tokens.append(int(lg.argmax()))
            print("decode tokens", got_tokens, "want", generated[1:])
            for j, lg in enumerate(got_logits):
                stats(f"decode_logits[{j}]", lg, ref["step_logits"][j])
        print("SUMMARY worst_hidden_max_abs", worst, "prefill_logits_max_abs", last)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
