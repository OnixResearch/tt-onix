# Host prefill scheduling search frame

## Goal

Explain a material part of the candidate's measured `1.024 s` wall-prefill gap
using host enqueue, command-queue wait, or non-program synchronization spans.
Select no optimization unless matched evidence attributes a material cause.

## Completion evidence

- One immutable package exposes package-matched `tracy-capture` and
  `tracy-csvexport` through the qwen36 overlay.
- Device-free discovery and positive/negative comparator self-tests pass.
- One bounded physical pilot runs in exact order
  `eager,candidate,candidate,eager` on PCIe IDs `0` and `1`.
- A deterministic receipt ranks host-side differences or records the exact
  remaining blocker.
- Eager production is restored and healthy.

## False completions

- Device-program records, simulator output, static source review, or one
  isolated operation do not prove a host scheduling cause.
- A capture that lacks the profiled prefill boundary, `EnqueueProgram` zones,
  matched identities, or duplicate roots is invalid.
- Profiled responses and captures are ineligible for admission.
- Agreement between profiling tools is not an optimization result.

## Risks

- Tracy protocol mismatch between the runtime client and capture tool.
- Full-process Python profiling changes timing and creates unbounded traces.
- Captures include startup or decode zones outside the prefill boundary.
- Overlapping host zones can cause false gap totals.
- Teardown faults can occur after complete responses and must not erase valid
  evidence; production restoration remains mandatory.

## Budgets

- Source authority: package-matched tt-metal `0.74.0-rc6` sources and local
  package implementation only.
- Retrieval: one targeted documentation/source pass, then discriminating reads.
- Physical: four roots, one warm-up plus one measured request per root, one
  generated token, exact `1x2` mesh, no simulator claim.
- Execution: one bounded pilot; stop on missing capture boundaries or invalid
  identities.
- Agents: none (no user consent for subagents).

## Approach registry

| Family | Mechanism | Evidence | State | Exact blocker / next check |
|---|---|---|---|---|
| Tracy host zones | Build capture/export tools from the exact vendored fork; enable partial Python profiling only around `_profiled_prefill_device`; use existing C++ `EnqueueProgram` and finish zones | Packaged runtime contains Tracy client `0.13.3`; tools build and report `0.13.3 / package-matched`; device-free discovery passes | active | Run the balanced physical pilot and compare gap totals |
| TTNN operation hooks | Register pre/post TTNN Python operation hooks and timestamp each call | Package-matched tests skip hooks on Blackhole | falsified | Blackhole support would need upstream runtime work |
| Service-only timers | Add timers around the model call and final synchronize | Existing `prefill_device_seconds` already measures that wall span | blocked | It cannot split enqueue gaps from queue waits |
| External probes | Attach eBPF/perf uprobes to command-queue functions | Could observe native functions without rebuilding | blocked | More authority and symbol plumbing than package-matched Tracy; use only if Tracy lacks required zones |

The passes are serial and correlated because no subagent consent was granted.

## Allowed outcomes

- **Validated**: matched host spans explain a material part of the wall gap.
- **Blocked**: captures are valid but the required span is absent.
- **Exhausted**: the single physical budget completes without material host
  attribution.
- **User decision required**: only if a wider outage or runtime rebuild becomes
  necessary.
