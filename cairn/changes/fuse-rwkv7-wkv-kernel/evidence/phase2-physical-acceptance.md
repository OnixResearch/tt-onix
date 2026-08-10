# Fused WKV physical acceptance evidence

Date: 2026-08-10

## Verdict

The Blackhole fused WKV candidate passed physical tensor correctness, exact generated-token parity, and stable B16 A/B gain.

Strict P150x2 scaling admission did not pass. Production dispatch stays closed, and `generic_wkv_step` stays authoritative.

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
| One card | 307.94323614990446 | 375.6423841464654 | 1.219842945222559x |
| Two cards | 445.98736406997796 | 560.6308989845617 | 1.2570555673783517x |

Every paired result improved:

- one-card speedups: `1.1739560208517679`, `1.2825253935028924`, `1.2044758611722342`;
- two-card speedups: `1.033296863270333`, `1.7587348393203222`, `1.3301446381676707`.

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
| Summed zone cycles | 18,960,886,000 | 6,782,786,846 | -64.2275% |

Summed zone cycles include nested zones and many cores. Do not read this value as wall time.

The generic and fused program signatures differ. The retained single-card signatures are:

- generic: `e8d5f28dd8430da5ada418d5e407787dbaab7930a343470d19be4e48ddd1575c`;
- fused: `cefd6f52d0b6c7466ab1d20b3dc925c7cc160bbfe5de07ad4fe7864f48a46164`.

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

- before BLAKE3: `6b41bdf65dd6653fe7e97f647fac66f4f5971fba9d6d0f39b7c1627140b28579`;
- after BLAKE3: `9cc4abd70f8d5fd4d78ec8a923309c02809b24586338a9bd24c14af8a1ba7251`.

The complete temporary run root was not retained. The portable receipts in this directory are the retained evidence.

## Strict admission blocker

The fused candidate reached a two-card speedup of `1.4924591117651091`. The profile requires at least `1.5`.

Its p99 latency regression was `1.084927367376163`. The profile permits at most `0.10`.

The retained admission result is `fused_candidate_scaling_rejected`. The runtime constant `kFusedWkvPhysicallyAdmitted` remains `false`.

This result does not admit production dispatch. It does not remove the observed intermittent device stall risk.
