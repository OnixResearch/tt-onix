## Why

The current RWKV-7 P150x2 runtime executes the complete model on both cards, but
it uses full-lockstep replication. Both cards process the same token and retain the
same state. This mode proves physical execution. It provides no aggregate
throughput gain and no single-request latency gain.

The runtime also mixes performance work with evidence work. It reads recurrent
state, flushes profiler data, and runs control comparisons inside the generation
path. Its RWKV recurrence uses several generic TT-NN operations for each head and
layer. No retained physical profiler receipt identifies the measured bottleneck.

The first optimization target is weak-scaling inference throughput. One card runs
one independent replica workload. Two cards run the same workload per replica.
The named target is at least `1.50` times the single-card aggregate token rate,
with no more than a `0.10` p99 inter-token latency regression per stream. These
values belong to the typed measurement profile, not runtime source constants.

## What Changes

- Add a fast inference mode that is separate from the existing evidence mode.
- Replace full-lockstep generation with one independent model replica per P150A.
- Add deterministic continuous batching for independent sequence states.
- Keep recurrent state on the device across decode steps and use FP32 recurrence.
- Remove state readback, profiler flush, and control replay from the timed interval.
- Capture and replay the static decode graph when package-matched TT-Metal supports
  the required dynamic token and state bindings.
- Use profiler evidence to select any custom fusion. Start with the smallest change
  that removes the measured bottleneck.
- Add a simulator smoke gate before a new physical run.
- Add a physical one-card versus two-card throughput receipt with per-card
  attribution, latency distributions, launch counts, and utilization facts.
- Add positive and negative tests for state isolation, metric admission, profiler
  completeness, and false speedup claims.

## Impact

- **Runtime**: `pkgs/rwkv7-p150x2` gains a production-oriented throughput mode.
- **Configuration**: a typed Nickel profile owns batching and performance bounds.
- **Scheduling**: requests map to independent card-local replicas and state slots.
- **Precision**: recurrent state changes from BF16 storage to FP32 recurrence.
- **Receipts**: a machine-bound receipt compares one-card and two-card runs.
- **Tests**: device-free fixtures cover scheduling, state isolation, and admission.
- **Documentation**: the package states the exact throughput claim and all remaining
  non-claims.

## Out of Scope

- A single-stream latency claim.
- Full tensor parallelism or layer pipeline parallelism.
- Vocabulary-parallel output projection.
- Context-parallel prompt processing.
- Weight quantization or checkpoint pin changes.
- LoRA training throughput.
- A speedup claim from simulator or device-free evidence.
