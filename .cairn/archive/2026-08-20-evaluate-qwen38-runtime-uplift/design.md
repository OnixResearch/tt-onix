## Context

Qwen3.8 runs with TT-Metal `0.74.0-rc6` and an overlaid model source at `d102e7f4512060e09e580e40cf26a870bfa88c55`.

The package retains compatibility paths for missing or unsafe capabilities.
These paths include sequential GDN and TP2 prefill without the blocked fused reduction.

The original evaluation did not select an immutable candidate runtime or package-matched candidate model source.

## Goal and Success Contract

The goal is an explicit unselected disposition for this runtime-uplift evaluation.

The disposition is complete when:

- the current closure and retained fallbacks are recorded;
- no candidate identity, capability, or performance result is invented;
- runtime inputs and `flake.lock` remain unchanged;
- future evaluation requires a new validated change with one immutable candidate;
- Cairn validation and lifecycle gates pass.

## Decisions

### Decision: Keep the current runtime

**Choice:** Keep TT-Metal `0.74.0-rc6` as the Qwen3.8 production runtime.

**Rationale:** No complete isolated candidate closure exists for comparison.

### Decision: Retain compatibility fallbacks

**Choice:** Keep every current fallback selected.

**Rationale:** No candidate capability result proves a safe replacement.

### Decision: Make no lockfile change

**Choice:** Do not change runtime inputs or regenerate `flake.lock`.

**Rationale:** A lock update without a selected candidate has no bounded authority.

### Decision: Require a new change for a future candidate

**Choice:** A future evaluation must name immutable runtime, model, patch, ABI, firmware, driver, and topology identities.

**Rationale:** A release number or operation name cannot establish compatibility.

## Architecture

No product code, port, adapter, service, Nix candidate, or configuration is added.

The accepted specification owns the runtime-admission boundary.
The evidence file owns the current closure facts and unselected verdict.

## Validation Strategy

Cairn validates the proposal, design, completed tasks, and specification delta.

The positive scenario keeps the current runtime when no complete candidate exists.
The negative scenario prevents a later release from becoming admitted by version or availability alone.

## Risks and Mitigations

- **Source drift:** Require immutable candidate identities in a new change.
- **Unsafe fallback removal:** Keep all current fallbacks.
- **Lockfile damage:** Make no input or lockfile change.
- **False speed claim:** Record no candidate performance result.

## Non-Goals

- Proving that newer runtimes are incompatible.
- Comparing public release notes.
- Building an unbounded runtime candidate.
- Changing production service behavior.
