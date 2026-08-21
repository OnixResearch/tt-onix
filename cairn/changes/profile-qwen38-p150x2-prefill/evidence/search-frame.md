# Prefill profiling portfolio-search frame

Date: 2026-08-21

## Exact goal

Attribute the physical P150x2 device-program difference that makes traced-candidate prefill slower than eager prefill. Select one source-backed optimization candidate, or return the strongest checked blocker.

## Completion evidence

- A package-matched profiler API check.
- A deterministic pure summary over explicit device records.
- Positive and negative tests for core and adapter boundaries.
- Balanced `eager,candidate,candidate,eager` physical profile roots.
- Duplicate-root signature coverage and identity agreement.
- A deterministic attribution receipt that names exact kernel sources.

## False completions

- Reusing aggregate device-prefill timing as operation attribution.
- Host timers around asynchronous TTNN calls.
- Per-operation synchronization that changes the device schedule.
- A profiled TTFT result used for admission.
- Static source frequency, simulator timing, or another mesh.
- One root, one prompt, or an unmatched package identity.
- Summed program duration presented as wall-clock TTFT.

## Audit risks

- Callback delivery can lag device synchronization.
- Runtime IDs repeat for cached programs.
- Callback order can differ between chips.
- Profiler resources can perturb the workload.
- A kernel-source signature can contain more than one semantic operation.
- Missing records can create a false attribution.

## Budgets

- Authority and source reads: at most `14` targeted reads.
- Implementation rounds: at most `2`.
- Physical profiling rounds: at most `1`.
- Later unprofiled admission rounds: `0` in this change.
- Workload per root: `4` prompts, `1` warm-up, `3` measured runs, and `64` generated tokens.
- Production outages: at most `1` bounded outage.
- No subagents are authorized. The lenses are serial and correlated.

Allowed terminal states are `validated`, `blocked`, `exhausted`, and `user-decision-required`.

## Approach registry

| Family | Mechanism | Claim | Evidence | State | Next check |
|---|---|---|---|---|---|
| realtime-programs | TTNN real-time callback records device timestamps and kernel sources | Exact program signatures can explain the prefill difference | Package-matched docs and bindings expose callback, active check, timestamps, frequency, chip, and kernel sources | active | Verify the packaged Nix binding and physical P150x2 activation |
| tracy-ops-csv | Run the service under Tracy and parse signposted operation CSV | Full TTNN operation rows can attribute the gap | Package-matched docs support `python -m tracy` and signposts | blocked | Use only if real-time signatures cannot discriminate; capture tooling adds a larger lifecycle boundary |
| host-method-timers | Time model methods on the Python host | Host durations identify slow operations | TT-Metal docs state that asynchronous calls mix queueing and wait time | falsified | None |
| per-op-sync | Synchronize after each candidate operation | Each synchronized interval yields device time | The method would serialize asynchronous work and alter the measured schedule | falsified | None |
| layer-bisection | Synchronize selected model layer boundaries | Layer groups can localize the gap | It needs invasive upstream model edits and still changes scheduling | blocked | Reconsider only if program signatures stay ambiguous |

## Current strongest route

Use the package-matched real-time callback as a diagnostic adapter. Drain its receiver only after the unchanged device-prefill end timestamp. Mark the response ineligible for admission because the drain delays readback. Aggregate exact signatures by chip in a pure core, and compare only roots collected with the same profiling mode.

## Adversarial audit

The selected route does not prove wall-clock attribution. Program durations can overlap with host work, and signature sums can differ from TTFT. The comparator must report this non-claim. It must reject incomplete duplicate coverage and must not select a candidate from one root.
