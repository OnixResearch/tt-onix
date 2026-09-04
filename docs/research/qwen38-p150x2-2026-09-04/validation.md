# Research validation

Date: 2026-09-04.

## Scope

This record covers research documents and native Cairn packages. It does not establish implementation, model correctness, performance, deployment, or hardware health.

The worktree started from `origin/main` at `51e07b8079b1b91e7c64fbf15c692d0df98fcc35`. The primary checkout remained unchanged. No service, simulator, or physical device command ran.

## Tool identity

- Cairn package: `/nix/store/bwjb9hzl3xrndwxbg48ck624pdyakf3p-cairn-0.1.0`.
- Policy file BLAKE3: `ad92f8f61ae3625f92acf4103215c36e0d124dcfe9da99e1965149caf93d085b`.
- Semantic policy hash in gate receipts: `8280151b7a53822eed460149ecb600bf11418d7463c935e0742009b334f4e7dd`.

The system Cairn binary did not match the current policy schema. The recorded package accepted the explicit policy path. A default invocation also failed because it could not locate the policy from this repository.

## Before document changes

Cairn rejected the existing MTP package. Its specification lacked recognized requirement blocks, and its numbered checkboxes did not count as substantive tasks.

Ruff reported one source error:

```text
undefined-name: Undefined name `started`
   --> pkgs/qwen36/service/qwen36_service/mtp_drafter_adapter.py:832:51
```

The runtime source remains unchanged. The same Ruff finding remains after the document changes. This is retained negative evidence, not a passing runtime check.

## Final results

| Check | Result |
| --- | --- |
| Repository Cairn validation | `valid=true`, no findings |
| Active change packages | Nine: eight new, one revised |
| Proposal gates | Nine PASS |
| Design gates | Nine PASS |
| Tasks gates | Nine PASS |
| Dependency graph | No graph issues, eleven dependency edges |
| Task records | Sixty total, three historical completed, fifty-seven open |
| Runtime source diff | Empty |
| Whitespace diff check | PASS |
| Ruff F821 source check | One existing error retained |
| Physical or simulator performance | Not run, no new claim |

The first draft omitted a concurrency marker on each initial task. Cairn rejected those task gates. Explicit serial markers corrected the documents, and the complete gate set then passed.

Saved raw receipts are in `gates/`, `validate.json`, and `change-graph.json`. The retained Ruff output is `ruff-mtp-baseline.txt`.

`BLAKE3SUMS` binds the research report, README, change artifacts, and saved validation evidence. Run `b3sum --check docs/research/qwen38-p150x2-2026-09-04/BLAKE3SUMS` from the repository root. The digest check establishes file identity, not runtime correctness.

## Reproduction

Use the recorded Cairn package and the policy with the recorded BLAKE3 identity.

```sh
CAIRN=/nix/store/bwjb9hzl3xrndwxbg48ck624pdyakf3p-cairn-0.1.0/bin/cairn
POLICY=/home/brittonr/git/OnixResearch/cairn/cairn-policy/generated/cairn-policy.json
"$CAIRN" validate --root . --policy "$POLICY"
"$CAIRN" gate proposal qwen38-optimization-evidence-contract --root . --policy "$POLICY"
"$CAIRN" gate design qwen38-optimization-evidence-contract --root . --policy "$POLICY"
"$CAIRN" gate tasks qwen38-optimization-evidence-contract --root . --policy "$POLICY"
ruff check --select F821 pkgs/qwen36/service/qwen36_service/mtp_drafter_adapter.py
```

Repeat the stage gates for each change in the research report. Read the JSON verdicts. Process exit alone does not establish that a gate passed.

## Non-claims

No accepted specification was synchronized or archived. No implementation task was newly marked complete. No package build, full test suite, firmware change, service restart, board reset, or push occurred.

Cairn validates document structure and task links. It does not prove the proposed speedups or the sufficiency of future tests.
