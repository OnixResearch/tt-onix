## Context

`S` = `${tt-metal.src}` = `/nix/store/5cafvl5cvjzck5hjl36ixj48mpx6j28f-source`
(TT-Metal 0.77.0; `nix eval .#tt-metal.src.outPath` matches, and
`.#tt-metal.outPath` is the existing `7hplkynh…-tt-metal-0.77.0`, so no runtime
rebuild is needed). Device-free findings, read from that source:

- **Loading.** `ModelArgs` requires `HF_MODEL`. It sets
  `CACHE_PATH = $TT_CACHE_PATH/<device_name>` and derives `model_name` from the
  path basename (`S/models/tt_transformers/tt/model_config.py:586-605`).
  `load_state_dict` always loads the full HF model through
  `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto")`. It drops
  layers `>= n_layers` only afterwards (`model_config.py:3190-3237`). Every run
  therefore needs about 66 GB of host RAM, including a one-layer test, unless
  it is pointed at a reduced checkpoint.
- **No allowlist blocks an unknown model.** Every per-model table falls back.
  `MAX_PREFILL_CHUNK_SIZES_DIV1024` falls back to 4 (4096-token chunks,
  `model_config.py:2475-2484`). `default_padded_cores` falls back to 0
  (`:2812-2819`). The tokenizer loads directly from `HF_MODEL` before any
  fallback mapping (`:3640-3648`). The stop list is `[eos_token_id]`
  (`:3706-3707`), which is 166101 `<|im_end|>`.
- **Mesh.** Two Blackhole devices map to `device_name` `P300` (`:4727-4734`),
  and the demo maps `MESH_DEVICE=P300` to `(1, 2)`
  (`S/models/tt_transformers/demo/simple_text_demo.py:842-856`). The cluster
  type for two ethernet-connected P150 cards is `P150_X2`
  (`S/tt_metal/api/tt-metalium/cluster.hpp:36`). It is not in the ring list, so
  `ccl_topology()` returns `Linear` (`model_config.py:2575-2596`). The existing
  Qwen3.8 P150x2 path relies on the same `P300` name
  (`pkgs/qwen36/p150x2-tp.patch:76-85`).
- **Shapes under TP2.** 28 query heads and 4 KV heads per device (GQA 7, the
  same ratio as Qwen2.5-7B on N300). The QKV width is 9216, or 4608 per device
  (144 tiles). The padded vocabulary is `nearest_multiple(166912, 64) = 166912`,
  so no padding is added (`model_config.py:99-103, 2756`). That is 83456
  columns per device. On Blackhole, with two devices, the LM head splits at
  `128256 // 8 = 16032` columns (`model_config.py:2256-2278`;
  `S/models/tt_transformers/tt/lm_head.py:38-56`). This gives 5×16032 + 3296
  per device. The decode MLP DRAM-sharded grid needs a core count that divides
  both K = 224 and N = 344 tiles. `find_grid_k_n` therefore selects 8 cores
  (`model_config.py:795-805, 3372-3401`). On hardware that grid made the FF3
  decode program's static CBs fill L1 to 1,220,928 B, and it clashed at decoder
  layer 2 (see Decision: Pad the MLP to 32 decode cores). The runner therefore
  sets `PAD_MLP_CORES=16`, which pads to 22528 (`:2821-2833`).
- **RoPE.** transformers 5.5.4 (the Qwen package environment) parses
  `config.json` to `rope_parameters={'rope_theta': 120000000, 'rope_type': 'default'}`
  (run 2026-10-25). `get_rope_theta` reads it, and `get_rope_scaling` returns
  `None` for `rope_type` `default` (`S/models/tt_transformers/tt/common.py:165-199`).
  `rope_scaling` is therefore `None` and `use_scaled_rope` is false
  (`model_config.py:2865-2871, 2908-2910`). No Llama-3 scaling is applied. QKV
  weights are permuted to Meta RoPE order generically through
  `convert_hf_to_meta` (`S/models/tt_transformers/tt/load_checkpoints.py:193-199, 451-466`).
- **Precision presets.** An unknown model name gets `ModelOptimizations.accuracy`
  by default. This is BF16 WQKV, WO, and KV cache, BFP8 MLP, and HiFi4 for
  attention (`model_config.py:130-196, 666-667`). `performance` sets FF1/FF3 to
  BFP4 with LoFi and keeps everything else at BFP8 (`:199-237, 288-304`). The LM
  head uses the model dtype (BFP8 via `create_tt_model(dtype=ttnn.bfloat8_b)`).
  The embedding is BF16, sharded on hidden (`S/models/tt_transformers/tt/model.py:55-63`,
  `embedding.py:26-28`).
- **Sampling.** On-device sampling requires `vocab_size // num_devices <= 65536`
  (`model.py:155-157`). Here 83456 exceeds it, so the generator returns logits
  to the host (`generator.py:783-787`). This is the same host-sampling
  constraint the Qwen P150x2 path carries (`pkgs/qwen36/p150x2-tp.patch:59-69`).
  It suits validation, because full logits reach the host.
- **Tokenizer.** The revision ships `tokenizer.json`, `tokenizer.model`, and a
  `LlamaTokenizer` config. Under transformers 5.5.4, `AutoTokenizer` and
  `use_fast=False` resolve to the same class and produce identical IDs on the
  probes. BOS `<|im_start|>` is 166100, EOS `<|im_end|>` is 166101, pad `<unk>`
  is 0, and the length is 166107. `apply_chat_template(tokenize=True)` emits
  one leading 166100 (no duplicated BOS), and `enable_thinking=False` emits
  an empty `<think>` block. Model rows 166107–166911 are unused vocabulary.
- **Simulator.** The packaged TTsim 1.10.6 provides only `x1` runtimes.
  `blackhole-x2` is blocked (`docs/ttsim-runtime.md`). The 0.77 GDN numerical
  gate stops at an unmodeled matmul flag (`docs/tt-metal-0.77-port.md:82-85`);
  whether the Llama op set avoids that flag is unproven.

## Decisions

### Decision: Reuse `tt_transformers` from the runtime-matched source, unpatched

**Choice:** Install `${tt-metal.src}/models` (88 MB, the same tree the runtime
was built from). Use it unpatched until an observed failure needs a patch.
`pkgs/qwen36` instead ships the PR-53543 `models/` tree, which is not
runtime-matched for `tt_transformers`.

**Rationale:** Every shape divides, every model-name table falls back, and RoPE
needs no special case. The match between the runtime's ops and the Python model
code is the main correctness risk, and the matched tree removes it. New kernels
are not justified.

### Decision: Own the runner; do not drive the pytest demo in production

**Choice:** Add a repo-authored LGPL-2.1-only `nanbeige43_runner` module. It
opens the mesh like `pkgs/qwen36/service/qwen36_service/server.py:1508-1538`:
`get_pcie_device_ids() == [0, 1]`, `FABRIC_1D`, `MeshShape(1, 2)`, and a
two-device check. It then calls `create_tt_model` (`common.py:892-944`) and
`Generator`, and owns prompt encoding (chat template with explicit
`enable_thinking`), host greedy sampling, and teacher forcing. A
`--smoke-1x1` mode is the AGENTS.md hardware-neutral mode: it opens a `1x1`
mesh, accepts only a reduced-layer checkpoint, and skips the PCIe-ID gate. It
has no simulator-specific branch.

**Rationale:** The demo's `TokenAccuracy` reads a cwd-relative `.refpt`. It also
requires centralized `model_targets.yaml` thresholds that do not exist for this
model (`simple_text_demo.py:35-90`). Its fixtures depend on the repository-root
pytest `conftest.py`. The Qwen package already drives the model API directly
for the same reasons.

### Decision: Precision. Use `accuracy` first and measure `performance` second

| Preset (per device, TP2, 8k-token KV pool) | Layers | LM head | Embed | KV | Total |
|---|---|---|---|---|---|
| BF16 everywhere (reference only) | 30.73 | 1.20 | 1.20 | 0.87 | 33.99 GB: under 0.3 GB of headroom |
| `accuracy`: BF16 attn/KV, BFP8 MLP | 19.19 | 0.64 | 1.20 | 0.87 | 21.89 GB |
| all BFP8 (default settings) | 16.32 | 0.64 | 1.20 | 0.46 | 18.62 GB |
| `performance`: BFP4 FF1/FF3, BFP8 rest | 12.22 | 0.64 | 1.20 | 0.46 | 14.52 GB |

BFP8 is 1.0625 B/element and BFP4 0.5625 B/element. KV per token per device
is 106,496 B in BF16 (52 layers × 2 × 4 heads × 128 × 2). A P150 addresses
8 DRAM banks × 4,278,190,080 B = 34.23 GB
(`S/tt_metal/soc_descriptors/blackhole_140_arch.yaml:112-113`). The feasibility
report's "33.4 GB against 32 GB" mixed decimal and binary units. BF16
everywhere still leaves under 0.3 GB of the device for activations, programs,
and trace, so it is not viable. Activations, trace, and program buffers come on
top of these totals. Bring-up uses `accuracy` with batch 1, `max_seq_len`
8192, and paged blocks of 32 × 256. `performance` is admitted only if it meets
the same teacher-forced gate.

### Decision: One shared Metalium overlay fragment

**Choice:** `pkgs/tt-metal/metalium-overlay.nix` returns the overlay shell
fragment that `pkgs/qwen36` used inline. That fragment is the writable
symlink tree of the runtime, the package-matched `api/` and `ttnn/cpp`, the
restored patched JIT directories, and the Tracy tools. Both `pkgs/qwen36` and
`pkgs/nanbeige43` interpolate it.

**Rationale:** Approved on the condition that the qwen38 derivations stay
byte-identical. On 2026-10-25 they did, before and after:
`.#qwen38` and `.#checks.x86_64-linux.qwen38` evaluated to
`af68qaa6rimz65g49yhgkvv031znrsp4-qwen38-0.1.0.drv`, and `.#qwen38-gdn-grid`
to `ckqjqw3rq7n14hdh9lgzfh4kbw876i37-qwen38-0.1.0.drv`. The fragment keeps the
original comments verbatim for that reason.

### Decision: Pad the MLP to 32 decode cores

**Choice:** The runner sets the upstream `PAD_MLP_CORES=16` on every mesh and
keys the weight cache path by it (`mlp-pad-16`), because cache file names do
not include shape. The padding is zero, so the math is exact. Preset, dtypes,
and gates are unchanged.

**Rationale:** The first full-model run threw a static-CB/L1 clash in the FF3
decode matmul at decoder layer 2. With the unpadded 8-core grid, that program
fills L1 to 1,220,928 of 1,461,376 B, and decode-time fragmentation placed the
FF1 output at 1,218,304 B. Upstream already pads the Qwen2.5 32B/72B MLPs the
same way. The evidence file records the trace.

### Decision: Decode optimizations admitted by predeclared gates

**Choice:** Traced decode is the default (`--trace`; `--no-trace` selects
eager). The preset stays `accuracy`. `--preset performance` remains selectable
but failed its gate (O2). `bench` and `compare` take `--batch` 1, 4, 16, or
32, with identical inputs per row. Batch 1 stays the default because it gives the
highest per-user rate. Batch 16 gives the highest aggregate rate. Batch 8 is
refused. `compare --batch N` now retains fixed-prompt evidence for every row
and summarizes each step using worst-row PCC and all-row top-1 equality.
Teacher-forced corpus gates also use the worst row. Each row's KV budget is
its share of the page pool, `page_block × page_blocks / N`. Historical O3
receipts cover all corpus rows but only row 0 of fixed prompts.

**Rationale:** Traced decode was bit-identical to eager and 1.31× faster
(7.42 → 9.69 tok/s). The `performance` preset lost 6.6 teacher-forced top-1
points. Decode step time barely depends on batch size up to 16 (9.69 → 9.16
tok/s per user), so batching multiplies aggregate throughput (146.5 tok/s at
16). Batch 4 is bit-identical to batch 1. Batch 16 and batch 32 change the SDPA
reduction order and diverge from batch 1 at id 101, but both pass the L3b
teacher-forced gate on every row; batch 32 is byte-identical to batch 16.
Batch 8 corrupts rows 5–7 at every page-pool size tried. Its root cause is
open, so the runner fails closed.

O5 screened all-BFP8 on TP2 after it passed the O2-class accuracy gate.
Its deterministic traced bench reached only 5.714879 tok/s median against
the predeclared > 9.69 tok/s threshold, so it was not promoted. The raw
runner receipt's `passed=true` covers deterministic ids, not speed admission;
the separate `O5-admission.json` records rejection. `accuracy` stays default,
and batch 4/16 claims remain accuracy-only. Named BFP4 mixes remain explicit
screening selections, not admitted defaults. The O5 run is not evidence that
lower precision costs speed: `accuracy` already runs a BFP8 MLP, so the mix
changes only WQKV/WO and KV (-14.2 % streamed bytes), it also drops attention
from HIFI4 to HIFI2, and the same run's prefill was above `accuracy` at 1024/4096 tokens
(2719/2753 vs 2548/2530 tok/s) and below at 128 (726 vs 790) while decode
samples ranged 8.456-7.682 tok/s. The decode degradation is
unattributed.

P2 (2026-10-25) added batch 32 after declaring the gates first: the traced
O0-protocol bench produced 313.551 tok/s aggregate (min 261.548) with 32/32
rows identical and ids exactly equal to batch 16, and `compare --batch 32`
passed the O3 fallback gate on every row (worst top-1 0.912109375, top-5
0.99609375, prefill PCC >= 0.99692). Batch 1 remains the default; the only
runner source delta is the accepted-size tuple, and no precision, device, or
kernel behavior changed. Receipts are `P2-*` in the change evidence.

Post-campaign ceiling (2026-10-25, measured P0 replacement plus source-read
arithmetic): the named traced decode moves 20.125 GB per device per step at
262.6 GB/s whole-step (289.8 GB/s matmul-only), 51.3 %/56.6 % of the p150
512 GB/s pin rate, with 8 sender cores streaming and about 1.6 FLOP/B. The
supported prefetcher is the only topology lever, and it is closed: at its
reachable rings (8/16/24/64/80) the smallest per-core footprint is
1,305,600 B against an 850,000 B budget; the arithmetic minimum is ring 115
unpadded or 118 padded, i.e. 120 as a multiple of 8, beyond both the legal
receiver counts and the 110 available workers this run reported. Within the
supported precision-preserving knob set, no decode lever remains. A
prefetcher entry or a hand-built DRAM-sharded program config is unsupported
and would need its own declared experiment and re-acceptance; this design
adopts neither.

The Phase 6 TP1/DP ladder stopped at the accuracy screens: all-BFP8 passes
accuracy but exceeds physical TP1 capacity (36.92 GB > 34.23 GB), while D,
M27, and E failed accuracy retention. G/H were heuristically pruned and remain
untested; their failure and the impossibility of every TP1 scheme are not
proven. No full-model single-card or two-instance path ran. The unexercised
submesh/model-loop code and wrappers are removed rather than shipped; the
reduced-layer `smoke-1x1` mode is unchanged.

### Decision: Slice active rows before host greedy conversion (P1)

**Choice:** `Session.decode_greedy` keeps the blocking readback, uses the
ordinary supported `ttnn.to_torch` on the proven all-gathered replica, slices
`[0, 0, :batch, :vocab]` before FP32 conversion and greedy argmax, and returns
tokens only. `generate` and `bench` use it. `compare` and teacher forcing keep
`Session.decode_rows`, which still returns full host logits. There is no
zero-copy, device, kernel, layout, or precision change.

**Rationale:** O4 measured 22.522 ms of a 106.8 ms traced step in
`process_decode_output_host`, which converted the full gathered vocabulary.
P1 was approved against that measurement alone; P0's four-way device
attribution remains incomplete. The replacement profiler run recorded in the
evidence file names every traced operation, but the dominant matmul bucket
still mixes attention, MLP, and LM-head work. The gates were declared before measurement:
real-logit old/new equality, O1 ids, exact teacher-forced equality, stable
O0-protocol bench ids, requalified batch 4/16 rows, and wall decode improvement.
On the P1 package `/nix/store/zc0fsdycib8kazc0wp0rk4n24wxjh8bq-nanbeige43-0.1.0`,
batch 1 and 4 differential runs matched old/new tokens, active logits, and
every gathered replica; the O0-protocol bench stayed id-identical to O1 on all
seven runs and reached 11.178 tok/s median on that package (O1 9.694; the
shipped kn85 identity re-measured 9.966, P3 receipt); compare batch 1
reproduced all 512 teacher-forced predictions and every prompt row exactly;
batch 4 rows stayed O1-identical; and batch 16 rows stayed identical to the
admitted O3 receipt. The new path is unreachable from `compare`, so the
existing batch 4/16 compare receipts were not rerun. No gate failed, so the
accepted baseline is replaced rather than retained. P1 did not establish a batch-1 wall-clock gain (shipped 9.966 vs O1
9.694, within run-to-run noise); it is retained for strictly less host work
with byte-identical tokens, not for measured speed. Batch 32's aggregate
313.6 vs 160.8 tok/s is the only established throughput improvement. Receipts and non-claims are
in `evidence/bring-up-2026-10-25.md` and `evidence/receipts/P1-*`.

### Decision: Reduced-layer checkpoint for simulator and fast physical stages

**Choice:** A device-free tool writes `<checkpoint>-L<n>`, holding layers
`0..n-1`, the full embedding, the final norm, and the LM head, with
`num_hidden_layers=n` (about 6–7 GB for n = 1–2). It is written outside the git
tree under `/home/brittonr/models`, and the tool records its source hashes.

**Rationale:** `load_state_dict` otherwise loads 66 GB per invocation, and
`reference_decoder` would load it again. The reduced checkpoint keeps
production shapes for every tensor.

## Validation ladder

1. **Device-free.** Checkpoint authority. The config-derived constraints above
   (parameter count, padding, grids, RoPE, host sampling, memory budget for the
   chosen preset and context). The tokenizer and chat template probes. The
   upstream device-free tests (`tests/test_model_config_utils.py`,
   `tests/test_load_checkpoints.py`).
2. **CPU reference** (transformers 5.5.4, BF16, 66 GB of host RAM; 118 GB were
   available on 2026-10-25). Teacher-forced top-5 at every position for 1024
   tokens of `tale-of-two-cities` in the `generate_reference_hf.py` format, plus
   last-position logits and 32-token greedy continuations for fixed plain and
   chat prompts. The same artifacts are produced for the reduced checkpoint.
3. **TTsim `1x1`, simulated.** The one-layer reduced checkpoint, the
   `smoke-1x1` mesh, slow dispatch, `TT_METAL_DISABLE_SFPLOADMACRO=1`, a
   prefill of each fixed prompt, and two teacher-forced decode steps, under a
   named 60-minute timeout. Gate: device open, firmware, weight load, and for
   every compared row a logits PCC ≥ 0.99 with top-1 equality against the
   float32 reduced-checkpoint CPU reference. An unmodeled op is recorded as the
   blocker, and the ladder moves to one bounded hardware attempt. TP2 shapes
   and CCL are not exercised here because `x2` is unavailable.
4. **Hardware `1x2`, bounded.** (a) The two-layer reduced checkpoint with TP2
   and CCL, against its float32 CPU reference, under the L2 row gate.
   (b) The full model: cold weight-cache build, then prefill logits PCC ≥ 0.94
   for each fixed prompt and for the 512-token corpus prefix. 0.94 is the
   upstream full-model accuracy-mode floor (`S/models/tt_transformers/tests/test_model.py:136`),
   fixed before any device result. Then teacher-forced decode against the
   reference: top-1 ≥ 87 % and top-5 ≥ 97 % over 512 positions, in line with
   upstream floors for comparable models (`S/models/model_targets.yaml`).
   (c) Greedy chat generation with `enable_thinking=False`, with a receipt that
   binds runtime, package, checkpoint, preset, and mesh.

## Risks / Trade-offs

- MLP padding to 22528 adds 2.3 % MLP weight bytes (22.19 GB per device in the
  accuracy preset) and changes the MLP accumulation partition relative to the
  unpadded layout. The zero padding itself is exact.
- A host RAM peak of about 66–75 GB per launch. The CPU reference and a device
  run cannot run at the same time.
- The weight cache under `$TT_CACHE_PATH/P300` holds about 42 GB for `accuracy`
  and more for each extra preset. It must live on the root filesystem (about
  125 GiB free). `/tmp` (5.8 GB) and `/home/brittonr/git` (5 GB) cannot hold it.
- Batched prefill variance on Blackhole meshes is disabled upstream only for
  named models (`model_config.py:644-648`). Bring-up runs batch 1. Batched
  decode (O3) prefills each row separately. Batch 8 decode corrupts rows 5–7
  for a reason not yet established, so the runner refuses it.
- Tokenizer parity with the model card's transformers-4.51 slow tokenizer is
  not proven. The reference and device runs share one tokenizer, so numerical
  gates do not depend on it.
