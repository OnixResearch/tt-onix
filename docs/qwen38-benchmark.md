# Qwen serving benchmark

`qwen38-benchmark` measures the client-visible behavior of any OpenAI-compatible
Qwen endpoint on a frozen corpus. It exists so the onix service and a vLLM
serving stack can be compared with the same prompts, the same greedy settings,
and the same accounting.

It is a benchmark, not an admission receipt. It never starts a server, touches a
device, reads server-private evidence, or claims that a result is a production
throughput measurement.

## Build and self-test

```sh
nix run .#qwen38-benchmark -- --self-test
nix build .#checks.x86_64-linux.qwen38-benchmark
```

The self-test starts a loopback fake OpenAI server and exercises streaming,
batch release, summaries, and parity. It needs no Tenstorrent device.

## Frozen workload

The packaged corpus is `pkgs/qwen38-benchmark/corpus.json`: the eight prompts
from the admitted Qwen3.8-27B P150x2 B=8 batch profile. The default run uses
64 generated tokens, greedy decoding, streaming, one warmup, three measured
runs, and a batch width of 8.

Changing the corpus, prompt order, `--max-tokens`, concurrency, stream mode, or
run counts changes the workload digest. `compare` refuses results whose digests
differ.

## Measure one endpoint

```sh
nix run .#qwen38-benchmark -- run \
  --label onix \
  --base-url http://127.0.0.1:8000 \
  --output onix.json

nix run .#qwen38-benchmark -- run \
  --label vllm \
  --base-url http://127.0.0.1:8001 \
  --output vllm.json
```

`--model` is optional. When omitted, the harness takes the first ID from
`/v1/models`. The onix service accepts only its admitted alias; a wrong
`--model` fails with the server's validation error.

Put credentials in `--api-key` or `QWEN38_BENCH_API_KEY`, never in the base URL.
A base URL with userinfo, a query, or a fragment is rejected. A secret embedded
in the URL path is still recorded, so do not put one there. Selected environment
values that are absolute host paths are recorded as `<absolute-path>`; Nix store
paths are retained. The endpoint's served model name and health payload are
recorded as returned, so keep secrets and host paths out of those fields.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--concurrency 4,8` | `8` | Measure one or more batch widths greater than one |
| `--runs 5` | `3` | More measured roots |
| `--warmup-runs 2` | `1` | Prime compile, trace, and cache state |
| `--no-stream` | off | Fallback for an endpoint without SSE; TTFT and decode rate become unavailable |
| `--api-key` | `QWEN38_BENCH_API_KEY` | Bearer token; never written to the result |
| `--header NAME: VALUE` | unset | Additional request header, repeatable |
| `--request-timeout 900` | `600` | Per-request timeout |

## Compare

```sh
nix run .#qwen38-benchmark -- compare onix.json vllm.json \
  --output comparison.json
```

The first result is the baseline unless `--baseline-label` names another. The
comparison emits candidate/baseline ratios for:

- single-stream delivered, decode, and end-to-end throughput;
- single-stream TTFT and p95 wall;
- batch delivered and aggregate decode throughput;
- batch per-user decode throughput, TTFT, and p95 wall;
- exact greedy-text parity per mode and prompt.

A parity mismatch exits non-zero. That is deliberate: if the two endpoints do
not produce the same greedy text, their throughput numbers are not a controlled
comparison. `compare` validates each result first: the stored workload digest
must match its workload, every mode and run must be present, every record must
name the corpus prompt it used, every single-stream run must cover the whole
corpus, and the stored summary must match a summary recomputed from the runs. A
malformed or edited result is rejected instead of producing a ratio.

To cancel linear host drift, run the targets in ABBA order:

```sh
# A, B, B, A across two invocations each
nix run .#qwen38-benchmark -- run --label onix --base-url http://127.0.0.1:8000 --output onix-a.json
nix run .#qwen38-benchmark -- run --label vllm --base-url http://127.0.0.1:8001 --output vllm-a.json
nix run .#qwen38-benchmark -- run --label vllm --base-url http://127.0.0.1:8001 --output vllm-b.json
nix run .#qwen38-benchmark -- run --label onix --base-url http://127.0.0.1:8000 --output onix-b.json
nix run .#qwen38-benchmark -- compare onix-a.json vllm-a.json --output compare-a.json
nix run .#qwen38-benchmark -- compare onix-b.json vllm-b.json --output compare-b.json
```

The onix service and a vLLM service normally cannot own the same two cards at
the same time. Run them sequentially, reset the cards between owners, and record
which package and runtime each endpoint reported. The result includes the
endpoint's `/v1/models` and health response, selected TT-Metal/vLLM environment
variables, host platform, and the repository revision when available.

## Accounting

- **TTFT** is request send to the first streamed content chunk with non-empty
  text. Empty role or usage chunks do not count.
- **Decode rate** is `(completion_tokens - 1) / (last_content_time - TTFT)`.
  The token count must come from `usage.completion_tokens`. A stream or
  non-stream response without that field is rejected; the harness does not
  substitute a content-chunk count for a token count.
- **Delivered throughput** is completion tokens over request or batch wall.
  Batch wall starts at the first request released by the barrier and ends when
  the slowest request finishes.
- **Aggregate decode throughput** sums per-request decode tokens over the span
  from the earliest TTFT to the latest content chunk. It excludes prefill but
  not queueing.
- **Per-user decode throughput** is the median of each request's decode rate.
  For a batch it is the median per-request rate; aggregate throughput is
  reported separately.

Batch mode releases every request from one barrier after its connection is
established. That removes client setup from the window, but it cannot prove the
server actually formed one batch or kept all slots active. Treat an aggregate
number as client-delivered throughput for the measured release pattern, not as a
server-side occupancy measurement.

A non-stream response with empty text or zero completion tokens is rejected.
Without that check two broken endpoints could "agree" on empty text and produce
a meaningless parity pass.

## vLLM status

The official Tenstorrent plugin is pinned at
`bef89e429e202caa38d3e3c8a24da4e8cc02405a` and targets vLLM 0.26.0 built with
`VLLM_TARGET_DEVICE=empty`. This repository does not yet package that stack:
the dependency profile conflicts with the deployed Qwen environment, and no
vLLM server run has been recorded here. See
[the dependency preflight](vllm-tt-dependency-preflight.md) and
[the upgrade evaluation](tenstorrent-upgrade-evaluation.md).

Once a vLLM endpoint is running, this benchmark needs no repository change:
point `--base-url` at it. Keep the compared services on the same checkpoint,
tokenizer, mesh, and greedy settings, and record the exact vLLM command in
`--note`.

Match the cache configuration too. The reviewed vLLM TT path requires prefix
caching disabled, while the onix service may serve with its hybrid prefix cache.
A target with a warm cache can look faster on a repeated corpus, so record the
cache setting for both and compare like with like.

## Non-claims

- Not an admission receipt and not a substitute for the B=8, prefix-cache, or
  single-stream admission receipts in `pkgs/qwen36/evidence/`.
- Not a profiler, thermal, or device-health measurement.
- Not a sampled-generation or accuracy benchmark. Greedy text parity is a
  validity gate for throughput comparison, not a model-quality claim.
- A `--no-stream` result has no TTFT or decode-rate claim.
- The comparison is only as controlled as the two endpoint configurations. A
  different checkpoint, tokenizer, precision recipe, mesh, or serving flag is a
  different workload even when the prompt digest matches.
