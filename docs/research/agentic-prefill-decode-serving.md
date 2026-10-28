# Tenstorrent: serving prefill and decode for agentic AI

Notes on Ben Hall, [Serving Prefill and Decode for Agentic AI](https://tenstorrent.com/developers/blog/serving-prefill-and-decode-for-agentic-ai),
Tenstorrent developer blog, 2026-10-27. Read on 2026-10-28. Every number below is
quoted from the post; nothing here was measured in this repository.

Follow-ups filed from this note:

- [#11 Prefix (KV) caching for multi-turn/agentic serving in the Nanbeige serving stack](https://github.com/OnixResearch/tt-onix/issues/11)
- [#12 Agentic serving benchmark: concurrent agents at a TTFT/ITL target](https://github.com/OnixResearch/tt-onix/issues/12)

## Summary

**Agents turn one task into many model calls.** An agent loops: generate, wait for a
tool, read the result, call the model again with the instructions, earlier messages and
prior outputs. The server sees a series of related requests with steadily growing
context, where a long input is often mostly material the model has already processed.

**Faster generation helps less than it looks.** Dependent calls wait for generation to
finish, so delays add up along each chain. Modeled example: eight sequential calls, 300
generated tokens each, 0.5 s before generation begins on each call, seven 1 s tool waits.
At 50 tok/s the task takes 59 s; at 200 tok/s it takes 23 s - 4x faster generation, about
2.6x faster task, because the other waits stay fixed (queueing excluded). Throughput still
has to be economical, so the post's target metric is **the number of concurrent agents a
deployment can support at the required response time**.

**Count cached input, new input and output, not ISL/OSL.** ISL and OSL do not say how
much input has already been processed. The post tracks:

1. cached input (reuses existing KV state),
2. new input (needs fresh prefill),
3. generated output (needs decode).

Prefix caching skips computation for an exact reusable prefix when its state is
available to the worker. Example: 100,000 input tokens and 1,000 output is ISL/OSL 100:1;
if 99,000 input tokens come from cache, new input/output is 1:1. Neither ratio is a
hardware allocation rule. An illustrative billing example (assumed prices of $0.20 per
million cached, $2 per million new input, $8 per million output; total $0.0298) puts
cached tokens at 66.4 % of the token bill - customer spend, not infrastructure cost.
Reuse is not free: KV state occupies memory, is still read by attention, costs bandwidth
to move between workers, and must be rebuilt if evicted.

**Disaggregated pools need a moving balance.** Prefill processes many positions in
parallel; decode advances each sequence one step at a time and, at small batch, is often
bound by reading weights and KV rather than arithmetic. Disaggregated serving (the post
cites DistServe) gives each phase its own pool and transfers KV between them, which adds
KV transfer and routing work that must not lose prefix reuse, and in a fixed fleet every
worker in one pool is taken from the other. Fresh documents raise prefill demand, reused
prefixes lower it, longer reasoning raises decode; model, context length and response
target also move it. "No single P/D ratio fits all of these conditions." Hardware that can
serve either phase keeps the split adjustable after purchase, though reassignment is not
instant (requests drain, weights or layouts may change), and whether it saves anything
depends on measured utilization and transition costs.

**Galaxy hardware (aggregate specifications, not application throughput).** A 32-chip
Blackhole Galaxy has 6.2 GB on-chip SRAM, 1 TB GDDR6, a 23 PFLOPS Block FP8 peak, and
ships as a 6U air-cooled chassis with no direct liquid-cooling connection. SRAM holds
active data near compute; GDDR6 holds weights and KV that exceed on-chip capacity. In
Tenstorrent's inference architecture prefill distributes work across servers while decode
pipelines work using both SRAM and DRAM.

**One projection, two regimes (modeled).** DeepSeek-V3's attention output projection is
16,384 x 7,168 (about 117 M weights, about 117 MB at one byte per weight). At an idealized
512 GB/s, one full weight read is about 229 us. Arithmetic assumes 719 T ops/s (23 PFLOPS
/ 32 chips) and a multiply-add counted as two operations:

| | Prefill | One decode step |
| --- | --- | --- |
| Positions | 32,768 prompt tokens | 32 users x 1 token |
| Arithmetic | 7.7 T ops | 7.5 B ops |
| Arithmetic time at assumed peak | 10,700 us | 10.5 us |
| One full weight read | 229 us | 229 us |

The long prefill is arithmetic-bound; the small decode batch is memory-bound. Neither
peak compute nor peak bandwidth alone predicts serving performance.

**Data placement.** Kernels split into reader (DRAM to SRAM), compute and writer code,
overlapped in time; when several cores need the same data the on-chip network distributes
one copy instead of each core fetching from DRAM. Prefill reuses inputs and keeps
intermediates on chip; decode spreads weights across memory banks and coordinates reads.
Weights, KV and temporary buffers share the finite GDDR6. Attention changes layout by
phase: prefill divides query positions across cores; decode, with one new query per
sequence, divides the existing KV context across cores and combines partial results (KV
read in chunks).

**Testing.** Benchmark the whole deployment, not a kernel or phase: run coding-agent
traffic under concurrent load and report how many agents a system serves at a given
response time, holding model, workload, cache conditions and response targets constant,
stating the power boundary, and keeping customer token prices separate from serving cost.

## Measured versus modeled

| Claim in the post | Status |
| --- | --- |
| FlashAttention on one Wormhole n150 ran 9-44x faster than a basic attention implementation on the same card (BF16 and BFP8, sequence 512-16,384, batches totaling 16,384 tokens); the baseline wrote intermediates to DRAM, the optimized kernel kept them in SRAM and skipped unneeded causal work | **Measured** (published kernel report; Wormhole, not Blackhole; not a service result or cross-vendor comparison) |
| 8-call task, 59 s vs 23 s, ~2.6x | Modeled |
| 100:1 vs 1:1 ratio, 66.4 % billing share | Illustrative, assumed prices |
| DeepSeek-V3 projection table (7.7 T / 7.5 B ops, 10,700 / 10.5 / 229 us) | Modeled, idealized bandwidth and peak |
| Galaxy 6.2 GB SRAM, 1 TB GDDR6, 23 PFLOPS BFP8 | Vendor aggregate specification |
| P/D flexibility saves cost | Not claimed; "depends on measured utilization and transition costs" |

No Blackhole serving throughput, latency, agents-at-SLO, or power figure is reported.

## Relevance to tt-onix

**Decode is memory-bound here too, and measured.** The post's decode row matches what the
Nanbeige4.3-33B P150x2 campaign measured: matmuls hold 90.6 % of the traced decode kernel
time at a bucket rate of about 290 GB/s, close to what the machine sustains at the model's
per-op weight sizes, with no supported knob demonstrated to close the gap to the 512 GB/s
pin rate (README, Nanbeige section, P63). Per-user decode is a fitted constant of 14.21
tok/s across batch 1-32 (commit 7d4e8d362c), i.e. batch is free capacity up to 32 because
the step is weight-streaming bound. The post's idealized 512 GB/s is the same pin rate
this repo quotes as the unreached ceiling for P150 decode.

**Batch beyond 32 is a decode-layout limit, not arithmetic.** Batch 64 is rejected
upstream (`supported_batches = {1, 2, 4, 8, 16, 32}`), and more streaming cores need
multi-row DRAM sharding inside tt-metal: the packaged `tt_transformers` asserts
`dram_grid_size.y == 1` (README, P96). The post's "spread weights across memory banks and
coordinate reads" for decode is exactly the axis that is closed for this model on the
packaged stack.

**Prefill is the compute regime; batching it pays.** Batched prefill is opt-in
(`--batched-prefill`) and measures a paired 2.81-3.51x first-token gain at batch 32
(README), consistent with the post's point that many token positions amortize each weight
read. Batched prefill's greedy ids differ from the sequential path's at token 2 (batch-variant
reduction order upstream), and the three serving receipts at batch 8, 16 and 32 emit
identical 64-token outputs (commit c913ee6115).

**Prefix reuse is the missing piece for agentic traffic on Nanbeige.** The post's central
accounting (cached / new / output) assumes prefix caching. Nanbeige has paged KV
(`pkgs/nanbeige43/runner/nanbeige43_runner/run.py`) but no prefix reuse and no serving
endpoint. Constraints already measured: `--page-block` must stay at 32 (block 128 is 7.4x
worse decode, commit e9a35c08cd); `--max-seq-len 32768` costs 2.65x decode at batch 1
(commit cbed41ce59), so cached long prefixes are not free to keep resident, matching the
post's "reuse is not free"; and batched prefill caps `--page-blocks` at 4096 at batch 32.
Upstream also falls back to sequential prefill when a prefix is cached (comment in
`run.py`). Qwen3.8 already has an opt-in hybrid prefix cache
(`.cairn/changes/qwen38-hybrid-prefix-cache/`, `pkgs/qwen36/prefix-cache-*.ncl`), which
must snapshot recurrent state; Nanbeige, a plain Llama, needs only KV. Tracked in
[#11](https://github.com/OnixResearch/tt-onix/issues/11).

**Our numbers are not the post's target metric.** Every Nanbeige figure is a host-wall
smoke measurement, not a serving benchmark (README). `pkgs/qwen38-benchmark`
([docs](../qwen38-benchmark.md)) measures TTFT, decode rate and delivered throughput on a
frozen corpus and already warns that a warm prefix cache must be matched between targets;
`scripts/qwen38-subagent-benchmark.py` runs OMP task-subagent roots against the Qwen3.8
service. Neither reports concurrent agents at a TTFT/ITL target, cache state, or a power
boundary. Tracked in [#12](https://github.com/OnixResearch/tt-onix/issues/12).

**Kernel structure.** The post's reader/compute/writer split is the same decomposition
the TTWKV7 decode contract binds at source level for the RWKV-7 decodeL path
([TTWKV7 decode composition](../ttwkv7-decode-composition.md), commit ba6e95c55d); that
check is device-free and makes no hardware claim.

**Small models.** LFM3-1.7B on one P150 ([bring-up](../lfm3-1.7b-p150.md), commit
402c2829f2) is single-card, greedy, with no batch scheduler or service wrapper, so none of
the serving points apply yet; its gated short-convolution state would make prefix
reuse hybrid like Qwen3.8's rather than KV-only.

**Not applicable here.** Disaggregated P/D across pools needs more than the one P150x2
host this repo targets; Galaxy figures are not comparable to P150 numbers.
