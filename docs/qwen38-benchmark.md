# Qwen serving benchmark (Tenstorrent compatibility pointer)

> [!IMPORTANT]
> The portable `qwen38-benchmark` runner, frozen workloads (`corpus.json`, `agentic_trace.json`), and shared command/metric/agentic documentation have migrated to [`OnixResearch/inference-lab`](https://github.com/OnixResearch/inference-lab/blob/main/docs/qwen38-benchmark.md) under [`inference-lab#7`](https://github.com/OnixResearch/inference-lab/issues/7).
>
> `tt-onix` retains `pkgs/qwen38-benchmark/default.nix` as a pinned compatibility package (`nix run .#qwen38-benchmark`, `nix build .#checks.x86_64-linux.qwen38-benchmark`) referencing `OnixResearch/inference-lab@fd90c12231f6664c64e72c47050d90fd14f968e6`.
> **Explicit rollback pin:** `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.

See [`OnixResearch/inference-lab/docs/qwen38-benchmark.md`](https://github.com/OnixResearch/inference-lab/blob/fd90c12231f6664c64e72c47050d90fd14f968e6/docs/qwen38-benchmark.md) for the full portable command reference, accounting rules, `compare` tamper/parity validation, and `agentic` SLO sweep documentation.

## Compatibility command and self-test

```sh
nix run .#qwen38-benchmark -- --self-test
nix build .#checks.x86_64-linux.qwen38-benchmark
```

The existing CLI flags, default Qwen3.8-27B B=8 eight-prompt corpus (`corpus.json`), default eight-call frozen agentic trace (`agentic_trace.json`), and `onix.qwen38-benchmark.v1` schema are preserved identically by the pinned compatibility package.

## Tenstorrent P150x2 setup and comparison rules

- **Model alias:** When `--model` is omitted, the runner uses the first ID from `/v1/models`. The onix P150x2 service accepts only its admitted alias; a wrong `--model` fails with the server's validation error.
- **Exclusive card ownership:** The onix service and a vLLM service normally cannot own the same two P150 cards simultaneously. Run targets sequentially in ABBA order, reset the cards between owners (`tt-smi -r` after stopping the previous owner via `ttwkv7-owner-control`), and record the exact package, runtime, mesh, cache mode, and server flags in `--note`.
- **Cache configuration:** The reviewed vLLM TT path requires prefix caching disabled, while the onix service may serve with its hybrid prefix cache. Match cache configurations and declare `--cache-mode` honestly when comparing endpoints.
- **vLLM status:** The official Tenstorrent plugin is pinned at `bef89e429e202caa38d3e3c8a24da4e8cc02405a` and targets vLLM `0.26.0` built with `VLLM_TARGET_DEVICE=empty`. This repository does not yet package that stack; see [the dependency preflight](vllm-tt-dependency-preflight.md), [the Qwen vs vLLM comparison](qwen-vllm-comparison.md), and [the upgrade evaluation](tenstorrent-upgrade-evaluation.md).
- **Admission receipts remain separate:** `qwen38-benchmark` is a client-side benchmark, not an admission receipt and not a substitute for the B=8, prefix-cache, or single-stream hardware admission receipts in `pkgs/qwen36/evidence/`.

## Follow-on ownership split

Generic client-side extensions (event timing, stream-stall/deadline reducers, reusable workloads, and cross-backend comparisons) live in `OnixResearch/inference-lab`. Tenstorrent-specific endpoint qualification, device-side phase attribution, kernel/runtime integration, and physical P150x2 receipts remain tracked in `OnixResearch/tt-onix` (including `#17`, `#19`, `#22`, `#27`, `#32`).
