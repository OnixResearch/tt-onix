# Packaged MLP reader transactions and residency — 2026-10-25

## Scope and result

P1.1 is a **completed negative characterization** of the configured TP2 decode
baseline at B1 and B8: the selected packaged reader requests **each weight tile
once per projection invocation**, with no duplicate weight-page requests and no
out-of-range weight reads. Gate/up use **39 active weight-reader/compute cores
inside an 11×4 capacity**, not 44 active readers; down uses **32 inside 11×3**,
not 33. The three projections issue **130,560 weight read requests and
97,484,800 B per device per layer invocation**. B8 shares that stream across
rows of one M tile; it does not multiply the stream by eight.

This is source-bound host enumeration, including execution of the actual
installed Python configuration builder and a compiled extraction of the exact
selected reader's page-address loop. It is **not** execution of a TT device
kernel, a simulator run, a DRAM/NoC counter, a profile, or a timing measurement.
Absolute allocation bases, live cache metadata, runtime routing, and physical
DRAM-controller commands remain outside this evidence.

P1.3 closes as **rejected**, not as a legal/admitted implementation: bank
matching cannot remove a baseline duplicate that does not exist, and cannot
reduce the mandatory unchanged-dtype matrix payload. An eight-bank repartition
can change request lengths and placement; no candidate request-count or
performance advantage was executed or established. No successor or P2 prototype
is introduced.

## Authorities and identities

Root `AGENTS.md`, `docs/tt-metal-0.77-port.md`, the existing P1 layout/design
records, and the packaged bandwidth report were read before this disposition.
Earlier evidence's 0.74/example-reader restrictions are historical bounds, not
a restriction on this new exact-0.77 audit.

Abbreviations below identify immutable paths, not the latest-reference checkout:

- **S** = `/nix/store/5cafvl5cvjzck5hjl36ixj48mpx6j28f-source`.
  `pkgs/tt-metal/default.nix:49-58` pins `v0.77.0`, including submodules.
  The matched Nix 2.36 `nix hash path S` returned
  `sha256-5NY5orbs7w9BTOUsylg8eAXIxJK+smthhkjd6HFePGA=`, exactly the pin.
  No fetch/build of a replacement runtime was needed.
- **R** = `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- **Q** = `/nix/store/wxnsxbhlx59vx4cfwi0881szxjfb32lm-qwen38-0.1.0`.
  Its `bin/qwen38-serve:2-5` binds Q's model Python files to R's Python
  runtime and Q's installed Metalium root. `pkgs/qwen36/source.nix:14-19`
  pins model revision `d102e7f4512060e09e580e40cf26a870bfa88c55`, tree hash
  `sha256-jxoNGFOPUHbF2eOvkduBThCOJ7JjUtEAx0pLyusVt9Y=`. The inspected
  files are Q's **patched installed files**, not the unpatched revision.
- **M** = `Q/lib/python3.13/site-packages/models/demos/blackhole/qwen36/tt`.
- **F** = `S/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`.
- **K** = `S/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`.
  `sha256sum` of K and its installed copy under
  `R/libexec/tt-metalium/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/`
  both returned `86ce5ffac79eb48e660532b7182c487b1b359bc1c6e0f5dd629666879970c101`.
- **A** = `S/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp`.

The release authority is `pkgs/qwen36/authority.ncl:10-11,40-42,91-97`:
Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The [pinned config](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
was read and its exact raw SHA-256 verified by the throwaway harness as
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
(the authority's base64 SHA-256). Its `text_config` supplies hidden 5120,
intermediate 17408, and 64 layers. The installed base
`models/tt_transformers/tt/model_config.py:2735,2765-2769,2811-2836`
binds those dimensions; Qwen3.8 has no default `PAD_MLP_CORES` override.
`M/tp_common.py:326-342` transposes `[out,in]` and mesh-shards the named axis.
Thus the fresh/default TP2 shapes are verified, not inferred from a 9B comment:

| Projection | Mesh split after transpose | Per-device `[K,N]` | `Kt,Nt` | Weight dtype |
| --- | --- | --- | --- | --- |
| Gate w1 | columns, `dim=-1` | `[5120,8704]` | `160,272` | bfloat4_b |
| Up w3 | columns, `dim=-1` | `[5120,8704]` | `160,272` | bfloat4_b |
| Down w2 | rows, `dim=0` | `[8704,5120]` | `272,160` | bfloat8_b |

`M/mlp.py:17-21,121-147` supplies dtypes and DRAM-interleaved placement.
The rejected `QWEN36_MLP_DOWN_BF4=1` override, `PAD_MLP_CORES` overrides,
rounded-register opt-in, alternative DRAM-sharded path, and nonmatching cached
tensor metadata are **not** this baseline. No opt-in/default/admission changed.

## Exact selected path and arguments

`M/model_config.py:228-254` sets `mlp_1d_decode=True`, builds configurations
with `M=1`, requests capacities 44/44/33, and passes the device worker width.
For the stated P150 configuration that width is 11. This is an explicit source
configuration input, not a new live-device grid observation.
`M/mlp.py:285-304,334-368` interleaves activation input in L1, calls the two
configured linears, multiplies in L1, then calls separately configured down;
its collective remains separate (`:371-380`). Decode compute configuration
is LoFi, FP32 destination accumulation, `packer_l1_acc=True` (`:205-207`).

`M/model.py:1077-1089` reshapes TP embeddings to `[1,1,B,dim_frac]` before
layer calls. B1/B8 occupy one padded 32-row tile. `fuse_batch=True` in the
builder and `F:5473-5477` produce `in0_B=in1_B=1` and `Mt=1`;
`device/utilities/matmul_utilities.hpp:94-102` defines fused M by padded
volume/tile height. Hence user batch B is **not** the reader's outer `batch`.

The installed `create_matmul_1d_decode_progcfg` was actually called, with gate's
SILU parameter and up/down's `None`, using the package interpreter
`/nix/store/qwm7vyvj1jbr0q65jdmshfqk6k9l4isc-python3-3.13.13-env/bin/python`.
It returned for **both** B cases:

| Parameter | Gate/up | Down |
| --- | --- | --- |
| Configured grid / capacity | `(11,4)` / 44 | `(11,3)` / 33 |
| `in0_block_w` | 8 | 8 |
| `out_subblock_h,w` | `1,1` | `1,1` |
| `out_block_h,w` | `1,7` | `1,5` |
| `per_core_M,N` | `1,7` | `1,5` |
| `fuse_batch,mcast_in0` | true,true | true,true |
| `gather_in0,stream_in1,untilize_out` | false,false,false | false,false,false |
| `allowed_worker_cores` before normalization | None | None |
| Active weight readers | `ceil(272/7)=39` | `ceil(160/5)=32` |

`A:2093-2122` sets broadcast-B true because weights have batch size one.
`A:2127-2151` selects `MatmulMultiCoreReuseMcast1DProgramFactory`:
there is no gather-in0 or global CB. **The current normal path is the
descriptor builder**, not the illustrative/example reader or just the older
legacy helper: `F:5404-5477,5495-5568` calls
`create_program_mcast_in0_descriptor`. Its weight-reader binding is
`F:3686-3700`: K on RISCV_0, preferred DRAM-read NoC, active cores only;
`SKIP_MCAST=1` (`F:3550-3568`), named CBs in1=1/bias=3/out=4/sparsity=7,
`num_active=0`. No in1-sharded, output-sharded, global-CB, sparse, bias, fused
all-gather, or fused reduce-scatter branch applies. Gate's SILU affects compute,
not the weight-read stream. Activation multicast is separate from weight reads.

### Complete weight reader/writer positional contract

`F:3445-3504` provides the positional compile arguments, with indices understood
by `K:50-107,138-166`. The instantiated vectors (indices 0 through 38) are:

```text
gate/up:
[1,272,2176,7,7,8,56,20,1,1,0,0,0,0,43520,1,1,0,0,
 1,272,1,272,7,272,1,1,1,272,0,0,0,0, 2,576, 0,0, 0,2048]
down:
[1,160,1280,5,5,8,40,34,1,1,0,0,0,0,43520,1,1,0,0,
 1,160,1,160,5,160,1,1,1,160,0,0,0,0, 2,1088, 0,0, 0,2048]
```

In order: in1 strides w/h/next-K-block/next-width-block (0–3), in1 block
w/h/tile-count (4–6), inner/width/height block counts (7–9), unused multicast
semaphores and counts (10–13, zero under SKIP_MCAST), KtNt/batch/bcast_B
(14–16), sparse batch/page size (17–18, zero), output strides (19–24),
output subblock w/h/count (25–27), MtNt (28), absent-bias stride (29),
fused-op booleans and compact-output (30–32, zero). Accessors append
`[IsDram=2,page_bytes]`, null sparse `[0,0]`, and interleaved BF16 L1 output
`[0,2048]`. The accessor encoding follows
`tt_metal/impl/buffers/tensor_accessor_args.cpp:146-157,189-205` and
`hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp:17-20`.
These are derived instantiated arguments; no runtime allocation was fabricated.

For active core ordinal `i`, logical `(x,y)=(i%11,floor(i/11))`,
`c=7` for gate/up or `5` for down, and `a=c*i`, the **entire runtime vector**
(indices 0–20) from `F:4070-4130` is:

```text
[weight_base,a, 0,0,0,0, 0, output_base,a, last_w,
 1,1,0, c,last_w,1,0,c-last_w, 0,0,1]
```

`last_w=7` for gate/up i=0..37, **6** for i=38; down has `last_w=5`
for all i=0..31. The two bases are symbolic device allocation addresses,
not observed addresses. Weight starts are 0,7,…,266 and 0,5,…,155.
`F:3246-3249,3974-3983` determines work and edge padding; row-major
active-core assignment does not schedule the spare capacity as weight readers.
There are no bias runtime extensions or fused-op runtime extensions.

## Exact page, bank, and request stream

For core i, ordered loops are outer batch=1, dense batchB limit=1,
height-block=1, width-block=1, then K-block `j=0..Kt/8-1`,
`h=0..7`, `w=0..c-1`. With the guard `w < last_w`, K's **actual selected**
interleaved block `:438-466` requests:

```text
p = c*i + (8*j+h)*Nt + w
source bank b = p % 8
within-bank page q = floor(p / 8)
source byte interval = [weight_base + bank_offset[b] + q*s,
                        weight_base + bank_offset[b] + (q+1)*s)
length s = 576 (gate/up) or 1088 (down)
```

The order among different concurrently running cores is **not** specified by
this enumeration; each core's order and the aggregate multiset are specified.
K advances next-K-block by `8*Nt` and does not reset within that single
height/width block (`:445-463`). No padded edge read is emitted. It still
publishes a nominal full CB block (`:513-515`); unused compute/output lanes
are not an extra DRAM transaction.

Mapping authority is not just a Python `DRAM_CORES` constant:
`api/tensor/tensor_accessor.h:380-411` selects InterleavedAddrGen for these
weights; `internal/dataflow/dataflow_api_addrgen.h:18-48,284-310` supplies
bank modulo, within-bank quotient, bank offsets, and NoC endpoint lookup.
The packaged `soc_descriptors/blackhole_140_arch.yaml:11-76` declares eight
DRAM views, zero view address offsets, with endpoint mapping; allocator
`impl/allocator/l1_banking_allocator.cpp:223-251` obtains views/offsets from
the SoC descriptor. The formula retains the offset and runtime bank table
rather than pretending a source YAML is an observation of the running boards.

### Per-core and per-bank ranges (complete parametric description)

Every range is inclusive in tile/page coordinates. These formulas enumerate
**every** core/bank without hiding noncontiguous page ownership in a bounding
range:

- Gate/up i=0..37 owns columns `[7i,7i+6]`; i=38 owns `[266,271]`,
  over all k=0..159. Its page set is `{272k+n}`. A column n belongs to bank
  `b=n%8` and bank pages `{34k+floor(n/8): k=0..159}`. Each of the seven
  owned banks gets exactly 160 requests per normal core; the omitted bank is
  `(7i+7)%8`. The final core owns banks 2..7, 160 each; banks 0/1 get zero.
  Normal core count=1120, final=960. Normal/final issued bytes are
  645,120 / 552,960.
- Down i=0..31 owns columns `[5i,5i+4]`, over k=0..271;
  page set `{160k+n}`. Bank `b=n%8` receives bank pages
  `{20k+floor(n/8): k=0..271}`. Exactly the five banks
  `{(5i+w)%8: w=0..4}` get 272 requests each; the others get zero.
  Every core issues 1360 requests / 1,479,680 B.
- Across all active cores **each projection** covers global tile IDs
  `[0,43519]` once. **Every bank 0..7** covers all its within-bank pages
  `[0,5439]` once, with no holes/duplicates. Its total byte interval relative
  to its tensor base is `[0,3133440)` for gate/up or `[0,5918720)` for down.
  These are separate tensor allocations; equal relative page IDs between
  gate/up/down are not duplicate reads of the same physical weight tensor.

### One API read is one emitted request here

Define a transaction narrowly as **one software-emitted NoC read request**,
not a response flit, API call in general, or a DRAM-controller command.
`api/dataflow/noc.h:149-177` resolves the accessor source/destination and calls
`noc_async_read` for this tile's size.
`api/dataflow/dataflow_api.h:551-573` uses the any-length path by default.
Blackhole `internal/tt-1xx/blackhole/noc_nonblocking_api.h:859-875` emits
full maximum chunks then one remainder; `noc/noc_parameters.h:286-290`
sets that maximum to 16,384 B. A 576- or 1088-B read therefore emits exactly
**one** request. There is no software coalescing across loop iterations in
this selected path. Hardware packet transport/controller behavior is unmeasured.

| B1 **and** B8, per device/layer | Requests | Duplicate excess | Padded reads issued | Payload B | Exponent-inclusive bytes requested |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate | 43,520 | 0 | 0 | 22,282,240 | 25,067,520 |
| Up | 43,520 | 0 | 0 | 22,282,240 | 25,067,520 |
| Down | 43,520 | 0 | 0 | 44,564,480 | 47,349,760 |
| Total | **130,560** | **0** | **0** | **89,128,960** | **97,484,800** |

Gate/up each skip **160** nominal padded tile slots, rather than reading them.
All K/N dimensions are already tile-aligned. The 64-layer multiplication is
8,355,840 requests / **6,239,027,200 B per device per decode step**, excluding
all other weights, activations and collectives. TP2 doubles these totals across
the two devices, not on each device. Payload omits exponent bytes; packed sizes
come from `tt_metal/api/tt-metalium/constants.hpp:19-21`. These are software
request bytes, **not measured GDDR traffic**.

## Persistent GDDR versus bounded L1 residency

`M/mlp.py:178-207` owns `self.weights` for the layer object; subsequent forwards
reuse those DRAM allocations. `M/tp_common.py:313-342`'s tensorbin cache is a
**host/on-disk weight-load cache**, not decode-time L1 weight retention. Cache
hits return stored tensor metadata as-is, so this audit assumes fresh or
matching-layout cache tensors; no running model's cache file was inspected.

The selected reader has no cross-invocation weight-cache lookup or reuse guard.
Each invocation starts its address loop again. The selected compute kernel
`bmm_large_block_zm_fused_bias_activation.cpp` (binding `F:3740-3743`) consumes
and pops in1 blocks (`:476-477`); reader push follows each block. Program
compilation/cache reuse does not make those reads conditional or retain the
whole matrix in its input CB.

`F:3182-3217,3820-3833` and buffering depth 2 at
`device/utilities/matmul_utilities.hpp:21-23` bound the **allocated weight CB**:

| Projection invocation | Per-core in1 CB | Active-core total | Full normal reader's weight slice |
| --- | ---: | ---: | ---: |
| Gate or up | `2*8*7=112` tiles / **64,512 B** | **2,515,968 B** over 39 cores | 645,120 B (10× CB) |
| Down | `2*8*5=80` tiles / **87,040 B** | **2,785,280 B** over 32 cores | 1,479,680 B (17× CB) |

The tail gate/up core still allocates the nominal CB size. Gate and up execute
as separate programs; adding their allocated CB sizes does not establish that
both simultaneously remain resident. L1 activations/output/intermediates,
firmware and allocator headroom are separate tenants. The descriptor gives
1,572,864 B worker L1 (`blackhole_140_arch.yaml:109-110`); an isolated slice
capacity comparison is not a proof of legal whole-model persistent caching.
The baseline simply does not configure that mechanism. Across 64 layers the
MLP weight requirement is 6.239 GB/device, much greater than even the
unharvested 140-worker raw L1 sum of 220,200,960 B. A claim that **all** layer
weights persist in worker L1 is therefore unsupported both by selected code
and capacity. This does not assert that one layer could never fit under some
other, unimplemented allocation.

## Executed extraction and falsifiable checks

Two throwaway host files and an executable were used, then removed; no
permanent test or runtime tool was added. The Python harness imported the real
installed builder, fetched only the pinned small config (hash checked), and
instantiated all six cases. It translated the surrounding source-bound loop
counts and enumerated every `(core,page,bank,bank-page,length)` request.
Independently, a C++ harness compiled **K lines 445–463 verbatim** with a
recording `noc.async_read` endpoint in place of hardware, and source-derived
per-core starts, edge widths and K-block wrapper loops. This runs the selected
address/edge loop itself; it does not model firmware, synchronization, compute,
real NoC execution or any cache.

Extraction uses `"\n".join(K.read_text().splitlines()[444:463])`, without a
trailing newline; the fragment SHA-256 is
`b52a9753ba936d642c966463767f2346b84c60f26126cc34b9dde49022fc3293`.
The recording API is equivalent to:

```cpp
struct Page { uint32_t page_id; };
struct Offset { uint32_t offset_bytes; };
struct Noc {
    void async_read(int, int, uint32_t bytes, Page p, Offset) {
        std::cout << p.page_id << ":" << bytes << "\n";
    }
};
```

The actual extracted source fills `Page`/`Offset` with its existing designated
initializers. The wrapper supplies `in1_block_h=8`, block width and strides from
the installed configuration/source vector, `num_blocks_w_dim=1,bw=0`, starts
`c*i`, and loops `block<Kt/8` for every active i. C++ was built with
`g++ -std=c++20 -O2`; the package interpreter ran the harness. For **each** B1
and B8 case, the entire emitted ordered `page_id:bytes\n` sequence equalled
the independent Python enumeration, not just its length:

```text
COMPILED_FRAGMENT_MATCH 1 gate 43520 32b81195b89866f586261a7f69d6b3eec3d253421ed7d4733f0c4d9517e9a58e
COMPILED_FRAGMENT_MATCH 1 up   43520 32b81195b89866f586261a7f69d6b3eec3d253421ed7d4733f0c4d9517e9a58e
COMPILED_FRAGMENT_MATCH 1 down 43520 a827cf31f3d686f91d7e0504750544166f8b608530f76c8c4b2f09d9e905980d
COMPILED_FRAGMENT_MATCH 8 gate 43520 32b81195b89866f586261a7f69d6b3eec3d253421ed7d4733f0c4d9517e9a58e
COMPILED_FRAGMENT_MATCH 8 up   43520 32b81195b89866f586261a7f69d6b3eec3d253421ed7d4733f0c4d9517e9a58e
COMPILED_FRAGMENT_MATCH 8 down 43520 a827cf31f3d686f91d7e0504750544166f8b608530f76c8c4b2f09d9e905980d
TOTAL 1 pages 130560 bytes 97484800 payload 89128960.0 64layers 6239027200
TOTAL 8 pages 130560 bytes 97484800 payload 89128960.0 64layers 6239027200
PASS: six selected-config loop enumerations; complete page coverage, multiplicity one, exact bank sums, no OOB; not device execution
```

Harness assertions were: global set exactly `range(Kt*Nt)`; multiplicity exactly
one per page; per-core sets equal the assigned column rectangle; eight bank
counts each 5440; unique `(bank,relative-byte-address,length)` tuples;
maximum global page 43519; compiled fragment stream equals host enumeration
pair-for-pair. Therefore this is stronger than merely computing `Kt*Nt` or
using an unrelated analytic estimator. No `tenstorrent-eval` traffic model was
needed for selected-reader proof.

## P1.3 rejection and legality boundary

1. **Traffic hurdle fails.** The now-bound baseline has no duplicate weight
   reads to eliminate. An unchanged-dtype eight-way column split has 34 tile
   columns/bank for gate/up or 20 for down: each bank still reads 5440 tiles.
   It therefore removes **0 mandatory weight bytes and 0 duplicate reads**.
   This is a lower-bound/repartition comparison, not a claim to have executed
   a nonexistent candidate. Coalescing could change the number/length of
   software requests, but cannot count as fewer bytes/rereads here.
2. **A tensor shard grid is not a physical worker binding.** The existing
   helper uses logical DRAM coordinates x=0..7,y=0 (`M/tp_common.py:17-19,97-109`),
   i.e. `(x,y)=(8,1)`. Calling that “1×8” in row×column prose must not be
   implemented as an x=1,y=8 DRAM grid. DRAM coordinates name banks, not the
   compute rectangle. Merely changing per_core_N/grid is not a source-proved
   bank-to-worker ownership configuration.
3. **The specialized program is not a drop-in legal version of the stated
   unchanged surface.** `A:1263-1312` requires WIDTH_SHARDED in0 and sharded
   output, ROW_MAJOR orientation, `Mt=per_core_M=1`, tile height at least 16,
   and K/shard divisibility by in0_block_w. Current decode in0/output are L1
   interleaved. The DRAM-sharded config exposes no compute-grid field
   (`device/config/matmul_program_config_types.hpp:75-80`); its factory
   `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:91-119,144-179`
   obtains physical reader assignment from
   `get_optimal_dram_bank_to_reader_assignment`, derives compute N separately,
   and may pad compute subblocks. Setting the Python helper's `per_core_N`
   is not proof of the requested exact eight-worker grid binding. No claim of
   legal candidate tensor buffers, allocation, compilation or numerics is made.
4. **Ceiling unchanged.** The previous source model's 97,484,800 packed bytes
   is now corroborated for this baseline's issued requests. Pairing unchanged
   bytes with the historical 335.3 µs forward solely as arithmetic gives
   290.74 GB/s; claiming an unchanged-byte 2× speedup implies 581.5 GB/s,
   above the packaged 512 GB/s bound. Payload-only 2× similarly gives
   531.6 GB/s. Those interpretations remain inadmissible, not measurements.

These failures are sufficient to complete P1.3 **as rejection** without building
a legality module or proving every possible alternative legal. The specialized
path's potential routing/congestion differences remain outside this mechanism;
`tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md:24-31,49-56`
explains that distinction and reports other architectures, not a new Blackhole
result. Dtype/equations/compute fidelity/accumulation/collective order and every
other projection remain untouched. P2 remains rejected/blocked; no prototype,
hardware probe, simulator, default switch or performance promotion is authorized.

## Residual assumptions and coverage bounds

- Fresh or matching-cache, full 32×32 TILE weights and the configured default
  BF4/BF4/BF8 dtype path; no malformed/stale live tensor metadata inspected.
- TP2, hidden/intermediate dimensions from the pinned release, no padding/env
  override, width 11, no relocated allowed-worker/subdevice override. Such
  changes require re-instantiation, not extrapolation of these core counts.
- The exact selected source stream is proven under those source inputs; physical
  bank endpoint tables and allocation bases remain symbolic. This is not proof
  that a currently running service uses Q or has those tensor addresses.
- Repeated invocations software-request the weights again. No source-only claim
  about realized GDDR cache hits/misses, effective bandwidth, NoC contention,
  controller burst counts, silicon behavior or runtime cache residence follows.
- No candidate stream was executed. Request coalescing, alternate legality,
  alternative persistent-cache policies, numerical retention and performance
  of a different mechanism are not silently claimed or invented as P2 work.
