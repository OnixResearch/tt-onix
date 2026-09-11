# Fused WKV physical acceptance evidence

Date: 2026-08-11

## Verdict

The Blackhole fused WKV candidate passed physical tensor correctness, exact generated-token parity, stable B16 A/B gain, and strict P150x2 admission.

Production fused dispatch requires CPUs 2 and 6 with FIFO priority 1. All other production configurations fail closed to `generic_wkv_step`.

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

A bounded fetch-queue capture then compared eight complete waves. An isolated diagnostic TT-Metal runtime collected the counters.

The production TT-Metal package remained unchanged. The fastest sample was sample 0. The slowest sample was sample 2.

The wall slowdown was `2.3682617537923503`. Card 0 added `0.000509701` seconds of fetch-queue wait across `0.888030787` seconds of extra wall time.

Card 1 added `0.00022285000000000002` seconds across `0.7931732840000001` seconds. These waits covered only `0.05739677131261441%` and `0.02809600430263609%` of the excess.

Both normal and slow workers made `23404` fetch-queue reservations and `15` wait checks. Each check completed after one poll.

This result falsifies fast-dispatch fetch-queue backpressure as the slow-wave cause.

A second bounded capture measured process and thread CPU time. Its fastest sample was sample 2 at `0.648936506` seconds.

Its slowest sample was sample 5 at `1.069046948` seconds. The wall slowdown was `1.6473829690820323`.

Model-submission process CPU time increased by `1.6465754067048122` and `1.6401340798887116` on cards 0 and 1.

Model-submission thread CPU time increased by `1.6248402533371622` and `1.6232483422583652`. Model-submission wall time increased by `1.6491990824393288` and `1.6488247412273533`.

The slow wave therefore consumed more CPU time on the primary host thread. It was not an off-CPU dispatch wait.

The capture does not distinguish more executed host instructions from lower CPU throughput, cache contention, or memory contention.

A third bounded capture attached 18 root eBPF probes to eight synchronized waves. It gated all probes after the single warmup call.

Sample 6 was the fastest wave at `0.857235608` seconds. Sample 7 was the slowest at `2.253611945` seconds.

The eBPF wave slowdown was `2.628929461128964`. Every worker made the same number of traced calls:

- `32` `advance_tokens_fast` calls;
- `384` fused `ttnn::generic_op` calls;
- `23368` calls to each measured TT-Metal dispatch subphase.

Card 0 `enqueue_mesh_workload` time increased by `2.724100369754593`. Its full `advance_tokens_fast` time increased by `2.7343316712068986`.

Card 1 `enqueue_mesh_workload` time increased by `2.435828729141766`. Its full `advance_tokens_fast` time increased by `2.4691254018304383`.

Reserve, command-sequence, subgrid, and fused generic-op times all increased by similar ratios. Enqueue remained about `18%` of total time in both waves.

Additional off-CPU time explained only `10.399732151173252%` and `6.043197508176343%` of the two cards' excess time.

Migration and switch counts increased. Probe overhead prevents a causal scheduler claim, and the capture cannot support admission.

The direct uprobe boundaries identify broad host-duration scaling without sampled-stack symbols. No smaller blocking TT-NN or TT-Metal subphase dominates the stall.

A fourth eight-sample capture used pinned, nonmultiplexed Linux hardware counters. Sample 6 was fastest, and sample 1 was slowest.

The wall slowdown was `2.38536978179575`. Instruction counts changed by only `1.0030606612412043` and `1.0094673558406875`.

Task-clock counts increased by `2.2284258192332365` and `2.208739315233314`. IPC fell to `0.48960730185785767` and `0.49662296641146986`.

Instruction rate fell to `0.45012073212575704` and `0.4570332718209687`. Average core frequency fell to `0.9193505293277583` and `0.9202821913843998`.

Cache misses increased by `2.155221719594046` and `2.120324652123489`. Cache references increased by only `1.2207529772552914` and `1.204381348800584`.

Context-switch ratios were `11.166666666666666` and `5.411504424778761`. CPU-migration ratios were `14.155172413793103` and `15.788461538461538`.

Device-zone cycles fell to `0.8789044762706084` and `0.8835978448227058`. Fetch-queue waits covered less than `0.034%` of excess time.

The slow wave executes the same host instructions at much lower throughput. Frequency, cache, and scheduler effects combine in the measured loss.

This result justified a primary-thread-only affinity test. It did not justify another fused-kernel or fetch-queue change.

The first affinity test pinned card 0 to CPU 2 and card 1 to CPU 6. Both CPUs use the 96 MiB L3 CCD.

All measured CPU migrations fell to zero. The fastest wave was `0.598603319` seconds, and the slowest was `0.694035274` seconds.

The maximum wave ratio fell to `1.1594243666397046`. Median wall time fell from `0.7576579765` to `0.6107231615` seconds.

Population standard deviation fell from `0.2604570536328125` to `0.036949182658166574` seconds. The ratio was `0.1418628604708738`.

This test pinned only each primary submission thread. Process-wide TT-Metal worker threads kept their normal CPU freedom.

A split-CCD control moved card 1 from CPU 6 to CPU 10. Its maximum wave ratio increased to `2.339356316019673`.

Its median wall time was `0.707549031` seconds, which was `1.1585429792153052x` the same-CCD median.

Its wall-time standard deviation was `0.28169732485257454` seconds. This was `7.623912210959646x` the same-CCD result.

The split placement exposed substantial off-CPU time on the 32 MiB L3 CCD. The same-CCD CPU 2 and 6 layout remained the candidate.

A full B16 A/B rerun then used the same-CCD placement. Exact generated-token parity passed.

Fused one-card throughput was `463.7091348870274` tokens/s. Fused two-card throughput was `799.8082690394812` tokens/s.

The fused scaling ratio increased to `1.7248059373131326`. However, the p99 regression increased to `0.381053554753073`.

One two-card fused-versus-generic pair was only `0.9091285937922943`. Therefore, primary-thread affinity alone failed admission.

A helper-thread exclusion control then reserved each primary CPU and its SMT sibling. The control produced a rejected `4.785382972`-second wave.

The primary thread consumed only `1.566409136` seconds of task clock. It had 1,736 context switches and zero migrations.

The corresponding profiler summary had two unmatched starts and two unmatched ends. The control was rejected, and its implementation was removed.

A bounded root control then tested FIFO priority 1 for only the pinned primary threads. Normal helper threads retained default scheduling.

All eight waves passed complete profiler validation. The fastest wave was `0.595375554` seconds, and the slowest was `0.646890378` seconds.

The maximum wave ratio fell to `1.0865249230572205`. The selected card instruction ratios were `0.9978448175303972` and `0.9979924993718348`.

Cache misses increased by only `1.0244258802180406` and `1.0824105095176326`. CPU migrations remained zero.

This result supports bounded real-time scheduling for the primary submission threads. It does not admit production without a full B16 A/B pass.

The runtime restores normal scheduling immediately after timed decode. FIFO use requires root or `CAP_SYS_NICE`.

Real-time-profiler synchronization logs did not overlap the timed normal or slow windows. Therefore, an attribution-removing runtime variant was not built.

Two host-side alternatives were also rejected:

- parity and contiguous CPU-affinity partitions made dual fused execution slower;
- Ethernet dispatch failed because `idle_erisc.elf` exceeded the code region on Ethernet firmware `1.4.2`.

Bounded evidence:

- `phase2-fused-wkv-synchronized-timing.json`;
- `phase2-fused-wkv-stall-capture.json`;
- `phase2-fused-wkv-fetch-queue-stall-capture.json`;
- `phase2-fused-wkv-host-cpu-stall-capture.json`;
- `phase2-fused-wkv-ebpf-host-subphase-capture.json`;
- `phase2-fused-wkv-hardware-counter-stall-capture.json`;
- `phase2-fused-wkv-primary-thread-affinity-stall-capture.json`;
- `phase2-fused-wkv-split-ccd-affinity-stall-capture.json`;
- `phase2-fused-wkv-primary-thread-affinity-throughput-receipt.json`;
- `phase2-fused-wkv-primary-thread-affinity-throughput-admission.json`;
- `phase2-fused-wkv-primary-thread-affinity-fused-wkv-ab-receipt.json`;
- `phase2-fused-wkv-primary-thread-affinity-generic-throughput-receipt.json`;
- `phase2-fused-wkv-primary-thread-affinity-generic-throughput-admission.json`;
- `phase2-fused-wkv-primary-thread-fifo-stall-capture.json`;
- `phase2-fused-wkv-primary-thread-fifo-throughput-receipt.json`;
- `phase2-fused-wkv-primary-thread-fifo-throughput-admission.json`;
- `phase2-fused-wkv-primary-thread-fifo-fused-wkv-ab-receipt.json`;
- `phase2-fused-wkv-primary-thread-fifo-generic-throughput-receipt.json`;
- `phase2-fused-wkv-primary-thread-fifo-generic-throughput-admission.json`;
- `phase2-fused-wkv-primary-thread-fifo-profiler-attestation.json`;
- `phase2-fused-wkv-conditioned-production-dispatch-smoke.json`.

## Strict admission resolution

The final fused candidate reached a two-card speedup of `1.9004959052068073`. This passes the required minimum of `1.5`.

Its p99 latency regression was `0.06701184998181789`. This passes the maximum of `0.10`.

Fused one-card and two-card throughput were `502.27251306237105` and `954.5668543729688` tokens/s.

Every fused-versus-generic pair improved. Exact generated-token parity passed, and no profiler markers were dropped.

The retained admission status is `fused_candidate_scaling_pass`. Its receipt identity is `7e716eb826b8e889e0def2c8f22d922de868fd9580d1fb5f39fe4462b1a47a58`.

The runtime constant `kFusedWkvPhysicallyAdmitted` is `true`. Production still requires the measured per-card CPU and FIFO scheduling contract.

Workers without that exact contract use `generic_wkv_step`. Explicit `generic-reference` workers preserve the authoritative A/B fallback.

A bounded production-policy smoke confirmed `production_physically_admitted: true` on both physical cards. Exact token parity and profiler validation passed.

The smoke used a shortened diagnostic profile, so its p99 result is not admission evidence. The full B16 A/B receipt remains authoritative.
