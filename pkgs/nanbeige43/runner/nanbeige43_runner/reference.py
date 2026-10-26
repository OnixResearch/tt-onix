# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.reference_agreement]
"""CPU Hugging Face reference for the device comparison.

For every fixed prompt it records the greedy continuation and the logits at the
positions the device produces: the prefill row and one row per teacher-forced
decode step. It can also record the teacher-forced corpus: the reference
tokens, the top-5 at every position, and the logits over the compared window.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .authority import load_authority, sha256_file, verify_checkpoint
from .prompts import PROMPTS, corpus_tokens, encode

SCHEMA = "nanbeige43-reference/v1"


def last_logits(model, ids: list[int], first: int) -> torch.Tensor:
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.logits[0, first:].float()


def greedy(model, ids: list[int], steps: int) -> list[int]:
    with torch.inference_mode():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            max_new_tokens=steps,
            min_new_tokens=steps,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    return [int(token) for token in out[0, len(ids) :]]


def build(
    checkpoint: Path, output: Path, steps: int, corpus_length: int, dtype: torch.dtype
) -> dict:
    authority = load_authority()
    identity = verify_checkpoint(checkpoint, authority)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
    model.eval()
    load_seconds = time.time() - started

    prompts = []
    for prompt in PROMPTS:
        ids = encode(tokenizer, prompt)
        continuation = greedy(model, ids, steps)
        logits = last_logits(model, ids + continuation, len(ids) - 1)
        assert logits.shape[0] == steps + 1
        # Cached greedy decode and one uncached forward can round differently; record, do not hide.
        mismatches = int((logits[:-1].argmax(-1) != torch.tensor(continuation)).sum())
        prompts.append({
            **prompt,
            "prompt_ids": ids,
            "continuation_ids": continuation,
            "continuation_text": tokenizer.decode(continuation),
            "greedy_forward_argmax_mismatches": mismatches,
            "logits": logits.clone(),
        })
        print(
            f"[reference] {prompt['name']}: {len(ids)} prompt tokens -> {tokenizer.decode(continuation)!r}",
            flush=True,
        )

    teacher_forced = None
    if corpus_length:
        tokens = corpus_tokens(tokenizer, corpus_length)
        split = corpus_length // 2
        with torch.inference_mode():
            logits = (
                model(input_ids=torch.tensor([tokens]), use_cache=False)
                .logits[0]
                .float()
            )
        top5 = torch.topk(logits, k=5, dim=-1).indices[:-1]
        teacher_forced = {
            "reference_tokens": torch.tensor([tokens]),
            "top5_tokens": top5,
            "split": split,
            # Rows split-1 .. corpus_length-2: the prefill row and every compared decode row.
            "window_logits": logits[split - 1 : corpus_length - 1]
            .to(torch.bfloat16)
            .clone(),
        }
        hits = (
            (top5[split - 1 :, 0] == torch.tensor(tokens[split:])).float().mean().item()
        )
        print(
            f"[reference] corpus: {corpus_length} tokens, reference self top-1 on window {hits:.4f}",
            flush=True,
        )

    artifact = {
        "schema": SCHEMA,
        "checkpoint": identity,
        "dtype": str(dtype).removeprefix("torch."),
        "continuation_steps": steps,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "torch_threads": torch.get_num_threads(),
        "load_seconds": load_seconds,
        "total_seconds": time.time() - started,
        "prompts": prompts,
        "teacher_forced": teacher_forced,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    torch.save(artifact, partial)
    partial.rename(output)
    summary = {
        "schema": SCHEMA,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "checkpoint": identity,
        "dtype": artifact["dtype"],
        "versions": artifact["versions"],
        "load_seconds": round(load_seconds, 1),
        "total_seconds": round(artifact["total_seconds"], 1),
        "prompts": [
            {
                "name": p["name"],
                "prompt_tokens": len(p["prompt_ids"]),
                "continuation_ids": p["continuation_ids"],
                "continuation_text": p["continuation_text"],
                "greedy_forward_argmax_mismatches": p[
                    "greedy_forward_argmax_mismatches"
                ],
            }
            for p in prompts
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation-steps", type=int, default=32)
    parser.add_argument(
        "--corpus-length",
        type=int,
        default=1024,
        help="0 disables the teacher-forced corpus",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    summary = build(
        args.checkpoint,
        args.output,
        args.continuation_steps,
        args.corpus_length,
        getattr(torch, args.dtype),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
