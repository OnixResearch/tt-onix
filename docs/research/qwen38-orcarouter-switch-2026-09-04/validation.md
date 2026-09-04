# Checkpoint-switch plan validation

Date: 2026-09-04.

## Scope and baseline

This work extends `research/qwen38-p150x2-20260904` from commit `4292d68`. The baseline Cairn validation reported nine active changes, no findings, and `valid=true`.

The change adds a checkpoint migration plan. It does not implement or deploy that migration.

## Artifact workflow

OpenSpec 1.7.0 validated the initial proposal, delta specification, design, and tasks with strict validation. `openspec-input-validation.json` records that result.

The reviewed migration plan copied only the new package into `.cairn/changes/qwen38-orcarouter-checkpoint-switch/`. `migration.json` records `blocked=false` and `mutated=true`.

The original OpenSpec input remains under `.pi/orca-openspec-input/`. It is not a second tracked active queue. The native package uses Cairn dependency metadata. A later source review tightened its metadata and cache rules.

## Final checks

| Check | Result |
| --- | --- |
| Cairn repository validation | `valid=true`, ten active changes, no findings |
| New proposal gate | PASS |
| New design gate | PASS |
| New tasks gate | PASS |
| Dependency update | Valid, one new edge to the evidence-contract package |
| New implementation tasks | Ten open, none complete |
| New requirements and scenarios | Seven requirements, fourteen scenarios |
| Whitespace check | PASS |
| BLAKE3 file identities | PASS |
| Runtime changes | None |

The saved `gate-*.json`, `validate.json`, `dependency.json`, and `change-graph.json` records establish these structural results. A successful process exit alone does not establish a passing gate.

The Cairn package is `/nix/store/bwjb9hzl3xrndwxbg48ck624pdyakf3p-cairn-0.1.0`. The explicit policy path is `/home/brittonr/git/OnixResearch/cairn/cairn-policy/generated/cairn-policy.json`. The semantic policy hash is recorded in each gate receipt.

## Evidence boundaries

The public HF API snapshots bind the observed candidate repository revisions. They are not weight-file or tensor manifests. Anonymous access to the pinned BF16 configuration failed, so authenticated compatibility proof remains open.

The local source review distinguishes three mechanisms: native checkpoint loading, direct GGUF substitution, and a separate Metalium backend. These were serial passes from the same assistant, not independent reviews.

The earlier optimization report and its `BLAKE3SUMS` belong to snapshot `4292d68`. This follow-up changes the README. The earlier digest set must therefore be verified against that earlier snapshot, not this updated checkout.

This directory's `BLAKE3SUMS` binds the current migration artifacts, README, report, public metadata, and validation receipts. From the repository root:

```sh
b3sum --check docs/research/qwen38-orcarouter-switch-2026-09-04/BLAKE3SUMS
```

Digest agreement proves file identity, not model correctness or authorization.

## Non-claims

No weights, tokenizer files, or configuration files were downloaded from the restricted repositories. No model code, simulator, physical device test, runtime test suite, service restart, host activation, or recovery command ran.

No runtime pin, live model, cache, accepted specification, or device authority changed. No implementation task was newly marked complete. The branch remains local and unpushed.
