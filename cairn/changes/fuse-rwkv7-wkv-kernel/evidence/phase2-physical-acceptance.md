# Fused WKV physical acceptance evidence

Date: 2026-08-10

## Verdict

The Blackhole fused WKV candidate passed physical tensor correctness, exact generated-token parity, and stable B16 A/B gain.

Strict P150x2 latency admission did not pass. Production dispatch stays closed, and `generic_wkv_step` stays authoritative.

## Physical maximum-batch probe

The retained physical probe used one P150 card and batch 16. It ran two recurrent steps.

- First and second output maximum absolute error: `0.0`.
- First and second state maximum absolute error: `0.0`.
- State dtype: FP32.
- Vector and output dtype: BF16.
- `physical_hardware`: `true`.
- `production_physically_admitted`: `false`.

Receipt: `phase2-fused-wkv-physical-probe-receipt.json`.

## Full-model B16 A/B result

The final run used the typed B16 profile and 32-request workload. It ran three order-balanced pairs for one card and two cards.

Every generated token matched the pinned FP32 CPU reference. The timed A/B samples compared 384 request streams. Each stream produced four tokens.

| Topology | Generic tokens/s | Fused tokens/s | Aggregate gain |
| --- | ---: | ---: | ---: |
| One card | 358.0837631395328 | 437.98668576683207 | 1.223x |
| Two cards | 337.3229076103619 | 722.4472805047268 | 2.142x |

Every paired result improved:

- one-card speedups: `1.2508837408212021`, `1.2300160498287833`, `1.187905183992303`;
- two-card speedups: `2.219560350366525`, `4.732432255516413`, `1.4038548119232248`.

The A/B receipt reports `stable_gain: true` and `status: stable_physical_gain`.

Receipts:

- `phase2-fused-wkv-physical-ab-receipt.json`;
- `phase2-fused-wkv-physical-throughput-receipt.json`;
- `phase2-generic-wkv-physical-throughput-receipt.json`.

## Program, launch, and device-cycle facts

Profiler samples used the same runtime and B16 workload. The raw profiler CSV files were streamed into bounded summaries and removed.

The summaries have no unmatched start or end events.

| Fact | Generic | Fused | Change |
| --- | ---: | ---: | ---: |
| Complete zone events | 1,173,823 | 1,155,480 | -18,343 (-1.5627%) |
| TRISC kernel starts | 326,976 | 317,805 | -9,171 (-2.8048%) |
| Summed zone cycles | 19,070,035,058 | 6,870,320,814 | -63.97% |

Summed zone cycles include nested zones and many cores. Do not read this value as wall time.

The generic and fused program signatures differ. The retained single-card signatures are:

- generic: `50454000942601883e95f951fa2ec7172754d2b427a7be7bad9c1d3069fbff3b`;
- fused: `db0593434141cf62a38262e7b2b35e71e4c69c159841a2f3afb141cc463377b7`.

The dual-card card-0 signatures are also retained. The profiler attestation binds physical device IDs 0 and 1.

## Memory facts

The B16 worker recorded these checked card-local values:

- recurrent state: `37,748,736` bytes;
- token-shift state: `589,824` bytes;
- temporary tensors: `83,886,080` bytes;
- safety reserve: `268,435,456` bytes;
- total required: `390,660,096` bytes;
- device budget: `3,221,225,472` bytes.

Receipt: `phase2-fused-wkv-physical-memory-facts.json`.

## Machine-fact binding

`tt-smi -s --snapshot_no_tty` captured devices 0 and 1 before and after the run.

- before BLAKE3: `4920c7ab31d91bdfb64adfc20f0bf2d5cd33deca60055f5bee4659c3b36563cb`;
- after BLAKE3: `074d3390032c15affca875d953e29c3df0f239c8878f2e6844e6bc918210e8fa`.

The complete temporary run root was not retained. The portable receipts in this directory are the retained evidence.

## H3 stall investigation

The first physical A/B launcher let the two card workers enter warmup and timed decode independently. The corrected launcher now:

- waits for both workers after initialization;
- releases one shared warmup-start barrier;
- waits for both workers before timed decode;
- releases one shared timed-start barrier;
- records monotonic worker start and end times;
- rejects a wave when start skew exceeds `0.01` seconds;
- disables TT-Metal Inspector in benchmark workers.

The corrected dual-card samples had start skews from `0.000145062` to `0.000586911` seconds.

| Sample | Generic wave seconds | Fused wave seconds |
| --- | ---: | ---: |
| 0 | 2.167732937 | 0.976649694 |
| 1 | 4.211254129 | 0.88987098 |
| 2 | 1.133233418 | 0.807229785 |

The slow card changed between runs and policies. This pattern did not isolate the stall to the fused WKV kernel.

A second diagnostic ran eight complete, synchronized fused waves. Each worker used one warmup token and one generated token across 32 requests.

The stall capture used a profiler capacity of `40000`. It rejected dropped markers, retained BLAKE3 summaries, and deleted all raw profiler CSV files.

Sample 2 was the fastest wave at `0.719589264` seconds. Sample 1 was the slowest wave at `2.534321332` seconds.

The slow wave was `3.521899865365418` times longer. Its start skew was only `0.00026138` seconds.

The card-0 and card-1 model-submission ratios were `3.4537093166911976` and `3.5325222263615386`. Their blocking-readback ratios were `1.2684559986817652` and `0.977476986844158`.

All 12 layer-submission ratios increased on both cards. They ranged from `3.175938040353203` through `3.926601936320196`.

Device work did not increase. Total profiler-zone cycles fell to `0.9024900293902967` and `0.8992396738414313` of the normal sample.

The measured stall boundary is the host-visible return from each layer submission. It is not blocking readback or increased device-kernel execution.

The receipt does not identify a smaller call inside each layer. A kernel change is not justified by this evidence.

Real-time-profiler synchronization logs did not overlap the timed normal or slow windows. Therefore, an attribution-removing runtime variant was not built.

Two host-side alternatives were also rejected:

- parity and contiguous CPU-affinity partitions made dual fused execution slower;
- Ethernet dispatch failed because `idle_erisc.elf` exceeded the code region on Ethernet firmware `1.4.2`.

Bounded evidence:

- `phase2-fused-wkv-synchronized-timing.json`;
- `phase2-fused-wkv-stall-capture.json`.

## Strict admission blocker

The synchronized fused candidate reached a two-card speedup of `1.6494731551938797`. This passes the required minimum of `1.5`.

Its p99 latency regression was `0.24129836831923815`. The profile permits at most `0.10`.

The retained admission result is `fused_candidate_scaling_rejected`. The runtime constant `kFusedWkvPhysicallyAdmitted` remains `false`.

This result does not admit production dispatch. It does not remove the observed intermittent device stall risk.
