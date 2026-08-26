## Why

The deployed P150x2 service has physical batching and readiness evidence, but it has no typed batch-admission receipt. Its batch evidence also measures TTFT through trace preparation, although the existing decode admission contract records trace preparation as a separate phase. These gaps prevent a truthful formal decision.

## What Changes

- Add a closed Nickel profile for matched B=1 versus B=4 served-throughput admission.
- Add a pure receipt evaluator for identity, parity, state isolation, throughput, TTFT, tail latency, and reset evidence.
- Correct batch TTFT evidence so trace preparation stays separate from first-token work.
- Keep normal production responses admission-ineligible. Enable eligible evidence only through an explicit bounded operator mode.
- Run physical roots in exact `single,batch,batch,single` order on devices 0 and 1.
- Admit only if every predeclared gate passes. Preserve an exact rejected receipt otherwise.
- Clarify that the older continuous-batching prohibition does not override the later bounded batching requirement.

## Impact

- **Files**: `pkgs/qwen36`, `scripts/qwen36-batch-receipt.rs`, the Qwen runtime spec, and physical evidence.
- **Testing**: Package checks, positive and negative receipt tests, Cairn gates, deterministic profile export, matched physical roots, and production restoration.
