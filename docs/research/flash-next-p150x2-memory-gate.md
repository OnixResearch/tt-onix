# Flash Next on P150x2: Strata checkpoint memory screen

**Decision (2026-09-29): neither Q2_0 nor IQ2_XS is admitted to a Flash Next TT runtime or the conditional one-layer numerical gate.** GGUF byte spans fit an incomplete nominal two-card floor, but a subsequent TT-Metal 0.77 format screen proves that even optimistic *all-tensor BFLOAT4_B* residency exceeds the two cards' combined 64-GB nominal capacity. Direct GGUF-preserving kernels or weight streaming would be a different, unimplemented execution path. This screen did not download complete weight shards, open a device, change the running Qwen3.8-27B service, or establish throughput. The machine-readable [receipt](../evidence/flash-next-strata-q2-iq2-memory-gate.json) records each unknown as `null` and both `memory_admitted` values as `false`.

## Inputs and reproducibility

- [Quantized release and model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09), pinned at `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The [Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0) and [IQ2_XS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS) first shards have respectively 37,623,740,192 and 39,225,954,592 bytes. Both use the 28,800,138,432-byte n-gram second shard. These are exact repository file sizes, not 32-GiB-class device capacity estimates.
- The [Strata GGUF reader](https://github.com/Niko1221/Strata/blob/3ce2523c2823687de5372be3af58534f56cbf286/tools/gguf_reader.py) supplies GGML type/block sizes; [QSA implementation](https://github.com/Niko1221/Strata/blob/3ce2523c2823687de5372be3af58534f56cbf286/include/strata/kernels/qsa.hpp) keeps pooled indexer keys in FP32. Neither GGUF Q2_0 nor IQ2_XS is a Tenstorrent BLOCKFP2 encoding. Strata's CUDA/CPU kernels and its SSD-mapped n-gram lookup are **ideas, not TT operators**.
- Public [Blackhole product specifications](https://github.com/tenstorrent/tenstorrent.github.io/blob/c37781bf680fc269cfd31097e2e3e47cc54aa414/core/aibs/blackhole/index.md) list 32 GB GDDR6 per p150a/b; the screen uses **32,000,000,000 nominal bytes per card**, not 32 GiB or an assertion about allocatable TT-NN DRAM. [Package-matched TT-NN memory configurations](https://github.com/tenstorrent/tt-metal/blob/v0.74.0-rc6/docs/source/ttnn/ttnn/tensor.rst) distinguish DRAM, L1, and sharded tensors. The **1,073,741,824-byte per-card trace floor** comes only from the existing *different* [27B decode profile](../../pkgs/qwen36/decode-profile.ncl); it is not a Flash Next trace measurement.
- The [0.77 candidate port](../tt-metal-0.77-port.md) records a blocked GDN numerical gate; its basic simulator smoke is not this model's recurrent-state evidence. This is separate from the bounded GDN hardware precision candidate.
- The pinned [TT-Metal 0.77 tensor types](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/tt_metal/api/tt-metalium/tensor/tensor_types.hpp#L26-L40) expose BFLOAT16, BFLOAT8_B, and BFLOAT4_B, but no BFLOAT2_B tensor type. Their [format mapping](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/tt_metal/impl/tensor/tensor_types.cpp#L91-L125), [tile byte sizes](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/tt_metal/api/tt-metalium/tt_backend_api_types.hpp#L121-L139), and [physical shape padding](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/tt_metal/impl/tensor/spec/layout/tensor_layout.cpp#L329-L365) govern the second, format-specific calculation below.
- The pinned [llama.cpp Metalium fork](https://github.com/marty1885/llama.cpp/blob/04602ca95c1a6adb6c1533e024ad9c79ee6c428d/ggml/src/ggml-metalium/ggml-metalium.cpp#L282-L366) only maps Wormhole in its GGML-to-TT type table; it rejects IQ2_XS, IQ4_NL, and TQ2_0 there. It is **not** an existing Blackhole Flash Next GGUF loader.

Download only the exact GGUF directories into a disposable directory, then run the Python 3 standard-library screen. Each first-shard payload begins at byte 11,024,672 and the n-gram payload begins at byte 192: the ranges below stop immediately before those bytes. The tool rejects wrong/truncated/oversized prefixes and verifies the parsed-directory SHA-256 in addition to tensor storage offsets, layouts, layer count, n-gram shard size, and model geometry. Its LFS SHA-256 fields are **repository metadata, not a full-file integrity check**: the full shards were not retrieved.

```sh
headers=$(mktemp -d)
repo=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09
curl -fSL --range 0-11024671 --max-filesize 11024672 \
  "$repo/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" -o "$headers/q2.header"
curl -fSL --range 0-11024671 --max-filesize 11024672 \
  "$repo/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" -o "$headers/iq2.header"
curl -fSL --range 0-191 --max-filesize 192 \
  "$repo/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" -o "$headers/ngram.header"
python3 scripts/flash_next_memory_gate.py --headers "$headers" \
  --context 65536 --disk-path /home/brittonr/models
```

The exact HTTP requests above returned `206` and the requested 11,024,672 / 11,024,672 / 192 bytes; the final receipt used those downloaded directory-only prefixes. Earlier exploratory 32-MiB / 1-MiB prefixes included bounded tensor payload bytes. Those excess prefixes were discarded before the receipt; **no complete tensor-weight shards were retrieved**. Free host memory and SSD space are observations at execution time, not constant features of these checkpoints.

## Placement and bounds

Scenario: one text request, 65,536 tokens, no vision projector or speculative/MTP draft, BF16 K/V, two physical P150 cards in an **hypothetical layer pipeline**. Do not substitute the existing 27B tensor-parallel allocation or its KV pool. Each first-shard GGUF has 48 layers, 144 fused routed-expert tensors, and five non-layer tensors. An exact padded **GGUF-on-disk** tensor-span sum puts the input embedding on chip 0 and the output head on chip 1. Of the 47 nonempty layer-boundary splits, minimizing the larger per-card GGUF byte floor chooses layers 0–23 / 24–47 for both variants. Each card then holds six QSA and eighteen GDN layers. These GGUF storage bytes are *not* TTNN DRAM allocations.

For each card the floor is first-shard tensor spans + BF16 K/V + FP32 pooled indexer + FP32 GDN state + the *provisional* 1-GiB trace reservation. At 65,536 tokens, each card's QSA cache is 805,306,368 bytes; indexer is `6 * (ceil(65536/4) + 3) * 128 * 4 = 50,340,864` bytes (the final three-token tail is conservatively reserved); GDN state is `18 * 3,268,608 = 58,834,944` bytes. The indexer is FP32, **not** BF16. These are arithmetic reservations; no TT tensor layout, allocator or maximum prompt workspace has been measured.

| Variant | Chip | Weight spans (bytes) | Known floor (bytes) | 32-GB nominal remainder (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Q2_0 | 0 | 18,682,937,600 | 20,671,161,600 | 11,328,838,400 |
| Q2_0 | 1 | 18,929,777,920 | 20,918,001,920 | 11,081,998,080 |
| IQ2_XS | 0 | 19,281,027,840 | 21,269,251,840 | 10,730,748,160 |
| IQ2_XS | 1 | 19,933,902,080 | 21,922,126,080 | 10,077,873,920 |

The **host CPU fallback planning floor** is all GGUF routed experts plus a provisional 6-GB non-expert policy allowance: 39,973,862,400 bytes (Q2_0) or 41,454,976,000 bytes (IQ2_XS). It is not proof of a working CPU offload. `/proc/meminfo` reported 110,533,877,760 `MemAvailable` bytes for the recorded receipt; host pinning and n-gram page-cache peaks remain unknown. The two required shards occupy 66,423,878,624 or 68,026,093,024 SSD bytes, respectively, against 1,194,044,121,088 available bytes in `/home/brittonr/models` at receipt time. The 28.8-GB n-gram table must remain an SSD/host lookup rather than being silently included in card weight spans; the proposed paging mechanism itself has not been ported to TT.

## TTNN tiled-format capacity screen

For an intentionally favorable comparison, store **every first-shard tensor** once as a standard 32×32 TTNN TILE tensor in one uniform format, including tensors that may actually require higher precision. The pinned 0.77 implementation pads the final two dimensions (a 1-D vector occupies at least one full tile row, padded to 32 rows), and each physical tile occupies 2,048 / 1,088 / 576 bytes in BFLOAT16 / BFLOAT8_B / BFLOAT4_B. Every 3-D expert tensor in both pinned directories has tile-aligned dimensions. The receipt computes the physical tile count separately for all 1,223 tensors and places them at the *GGUF-selected* layer split; these are a **hypothetical size projection**, not executed conversions, operating buffers, or numerical-equivalence evidence. Both variants have identical tensor shapes and therefore identical projected TTNN footprints.

The pinned type histograms still differ: Q2_0 has 202 GGUF type-42/TQ2_0 tensors; IQ2_XS has 22 type-16/IQ2_XS tensors and 53 type-42/TQ2_0 tensors. The directory name is not a uniform device dtype; the n-gram shard is a separate type-20/IQ4_NL lookup.

| Uniform TTNN format | Bytes per tile | First-shard weights on both cards (bytes) | Chip 0 / chip 1 weights at layers 0–23 / 24–47 (bytes) |
| --- | ---: | ---: | ---: |
| BFLOAT16 | 2,048 | 251,641,724,928 | 125,848,059,904 / 125,793,665,024 |
| BFLOAT8_B | 1,088 | 133,684,666,368 | 66,856,781,824 / 66,827,884,544 |
| BFLOAT4_B | 576 | 70,774,235,136 | 35,394,766,848 / 35,379,468,288 |

Even the BFLOAT4_B **weights alone** exceed the two nominal cards' combined 64,000,000,000 bytes by 6,774,235,136 bytes; no different split of *fully resident* BFLOAT4_B tensors can fix that total. Including the existing 1,988,224,000-byte per-card KV/indexer/GDN/trace reservation raises the projected chip floors to 37,382,990,848 and 37,367,692,288 bytes. Actual usable DRAM and peak scratch only tighten this rejection. GGUF Q2_0 (including GGUF type 42/TQ2_0 tensors) and IQ2_XS are not TTNN BFLOAT4_B: conversion would also change the model's quantized values and needs a separate numerical comparison. Metalium defines a low-level Bfp2 tile, but TTNN 0.77 exposes no BFLOAT2_B tensor type or Flash Next operator; **do not count Bfp2 as an admitted workaround**.

**Admission remains closed.** The original GGUF-only per-card/host/SSD floors fit, but standard fully resident TTNN tiled weight formats do not fit even under the BFLOAT4_B assumption. A direct low-bit GGUF-preserving kernel with proven state and routed-expert math, or a measured expert-streaming/host route, must first replace that unsupported resident layout. The TT implementation also lacks Flash Next QSA/MoE/GDN placement and the SSD n-gram lookup. `allocatable_dram_bytes`, `prompt_scratch_peak_bytes`, `measured_trace_peak_bytes`, `weight_materialization_overhead_bytes`, `pinnable_bytes`, and `ngram_page_cache_bytes` remain unknown. After establishing a viable actual weight path, measure its peak allocations and only then attempt a bounded one-layer routed-expert plus recurrent-state numerical comparison with the checkpoint's full layer weights and a CPU reference. The conditional comparison **was not run**, because neither checkpoint has passed the gate and no complete checkpoint shard was retrieved. No simulator or silicon pass is implied.

The quant repository YAML labels its files Apache-2.0, but its card explicitly says the quantized weights inherit the [base model's Qwen Community 1.0 license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Resolve that discrepancy before redistributing weights; no model weights were committed here.
