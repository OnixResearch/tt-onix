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

## Agentic mode

`run` answers "how fast is one stream?" and `agentic` answers "how many
concurrent agents meet a response-time target on a fixed multi-call trace?"
It is still a client-side harness: it never reads server cache counters or
starts a server.

```sh
nix run .#qwen38-benchmark -- agentic \
  --label onix \
  --base-url http://127.0.0.1:8000 \
  --agents 1,2,4,8 \
  --ttft-p95 0.5 \
  --itl-p95 0.05 \
  --task-p95 30 \
  --cache-mode disabled \
  --output onix-agentic.json
```

Add a power boundary only when it is measured:

```sh
nix run .#qwen38-benchmark -- agentic ... \
  --power-boundary wall --watts 900 \
  --power-method "wall meter, 1 s samples, five-minute mean"
```

### Frozen trace

The packaged trace is `pkgs/qwen38-benchmark/agentic_trace.json`: one system
prompt, eight coding-agent calls, a 300-token generation bound per call, and
seven one-second tool waits. Each call's message list extends the previous
call's list with a frozen assistant turn, a frozen tool result, and the next
user turn. Assistant turns are frozen rather than fed back from live output so
every target receives the same workload text; the live reply is still measured
and hashed per call.

The history grows, so call *n* reuses the entire prompt of call *n-1* as a
client-known prefix. `--tool-wait-scale` multiplies every wait uniformly; it is
recorded in the workload digest and must match across compared targets.
`--trace` points at a different frozen file.

### SLOs and the sweep

`--ttft-p95`, `--itl-p95`, and `--task-p95` are required and are in seconds.
They are defined before the run and recorded in the result. For each agent
width, every agent runs the whole trace closed-loop from one barrier; a run
repeats the sweep `--runs` times and discards `--warmup-runs` sweeps. The
summary reports, per width, p50/p95 call TTFT, p50/p95 mean ITL, p50/p95/max
task time, tasks/s, output tokens/s, and whether each SLO passed. It reports
the largest width meeting all three SLOs, plus the width with the highest
task throughput as the saturation point. The `curve` array is the knee curve.

Per-call ITL is the mean inter-token interval,
`(last_content_time - TTFT) / (completion_tokens - 1)`, which is the reciprocal
of the per-call decode rate. Task time covers the calls plus the replayed tool
waits for one agent. Every run's summary is retained, so a run whose ITL
deviates from the session spread can be discarded instead of silently averaged
into the curve.

Agentic results use their own schema; `compare` accepts only `run` results.
Two agentic results are directly comparable when `workload_sha256` and the three
SLOs match, and their `curve` arrays can then be read side by side.

### Token accounting

OpenAI-compatible usage reports prompt and completion tokens; it does not
guarantee a server-side prefix-cache report. Every call therefore records:

- `prompt_tokens` and `completion_tokens`, from usage;
- `reused_prefix_characters`, the exact client-known prefix overlap with the
  previous call in the same trace;
- `reused_prefix_fraction`, that overlap over the serialized prompt length;
- `reused_prefix_tokens_estimate` and `new_input_tokens_estimate`, a
  client-side apportionment of the reported `prompt_tokens` by that fraction;
- `server_reported_cached_tokens` only when the endpoint returns
  `usage.prompt_tokens_details.cached_tokens`, with its source named;
- `cache_confirmation: "client-prefix-overlap-only"`.

The estimate is an accounting label, not evidence that the server kept those
tokens in a prefix cache. A result never claims a server cache hit unless the
endpoint itself reported cached tokens.

### Held conditions

The result records the endpoint label, base URL, resolved model and probe; the
trace content, digest, per-call message digests, generation bounds, and scaled
tool-wait schedule; the SLOs; the agent widths, runs, warmup, settle time,
greedy temperature, and streaming flag; the declared cache mode
(`cold`, `warm`, `disabled`, or `unknown`); and the power boundary, watts, and
method. `--power-boundary` and `--watts` must be supplied together; otherwise
`agents_at_slo_per_watt` is `null`. The existing environment and source
revision fields are recorded unchanged.

Cache mode is operator-declared; the client cannot observe or reset a server's
cache. Server-side flags such as `--page-blocks`, `--max-seq-len`, and
`--batched-prefill` are not client-visible, so record the exact server command
in `--note`. `--power-boundary` and `--watts` must be supplied together;
otherwise `agents_at_slo_per_watt` is `null`.

The self-test starts a loopback fake OpenAI server and runs this mode on the
packaged trace with `--tool-wait-scale 0`, so it needs no device. It also checks
that the SLO maximum, the per-watt derivation, reused-prefix accounting, and
tamper rejection are consistent; a tampered call, curve point, trace, or
missing run is rejected.

## Non-claims

- Not an admission receipt and not a substitute for the B=8, prefix-cache, or
  single-stream admission receipts in `pkgs/qwen36/evidence/`.
- Not a profiler, thermal, or device-health measurement.
- Not a sampled-generation or accuracy benchmark. Greedy text parity is a
  validity gate for throughput comparison, not a model-quality claim.
- A `--no-stream` result has no TTFT or decode-rate claim.
- Agentic mode is a client-side closed-loop measurement of a fixed trace. It
  does not prove one server batch, slot occupancy, or a production arrival
  process, and its tool waits are replayed sleeps rather than real tool calls.
- `reused_prefix_tokens_estimate` is a client-side apportionment, not a server
  prefix-cache hit count. Only `server_reported_cached_tokens` restates an
  endpoint's own usage report.
- `agents_at_slo_per_watt` is only reported inside the stated power boundary
  using the supplied watts. It is not a facility-efficiency or cost claim.
- The comparison is only as controlled as the two endpoint configurations. A
  different checkpoint, tokenizer, precision recipe, mesh, or serving flag is a
  different workload even when the prompt digest matches.
