# Search frame

## Goal

Explain and enforce the authority boundary between the unused eight-entry mapper set and the pinned Qwen3.8 64-layer hybrid configuration.

## Completion evidence

A valid result must:

- identify every consumer of the static mapper symbol;
- trace runtime layer dispatch from checkpoint input to layer construction;
- bind the claimed 48/16 split to the pinned configuration identity;
- reject missing, malformed, unknown, or wrong-length layer types;
- preserve the valid legacy fallback only for its exact layer count;
- correct historical prose only where named receipt fields prove a label reversal; and
- pass device-free package, authority, Cairn, and repository-quality gates.

## False completion

The following outputs are not completion:

- counting a source constant without proving that runtime code reads it;
- inferring checkpoint structure from layer cache sizes or weight names;
- changing the model pin to match current upstream code;
- treating prior physical execution as proof for malformed configurations;
- changing measured receipt values or selecting an optimization from rejected evidence.

## Audit risks

- The adopted source and latest upstream source can differ.
- The 32-layer fallback can be valid for a legacy model but invalid for Qwen3.8.
- Grep output order can reverse candidate and eager labels.
- A source cleanup can accidentally alter tensor remapping or runtime dispatch.

## Budget

Use the adopted source, pinned checkpoint configuration, accepted authority record, three host-attribution receipts, and affected documentation. Use at most two search rounds and twelve focused commands before implementation. No subagents are authorized, so the lenses are serial and correlated.

## Approach registry

| Family | Mechanism | Claim | Artifact | State | Next check |
| --- | --- | --- | --- | --- | --- |
| Symbol reachability | Search every adopted-source reference to `FULL_ATTENTION_LAYERS` | The eight-entry set is dead metadata | One declaration and no consumers | Validated | Remove it and rerun package checks. |
| Runtime dataflow | Trace `text_config.layer_types` through `attention_type_list` and `is_full_attention_layer` | Checkpoint data owns layer dispatch | `model_config.py` and `layer.py` call chain | Validated | Add direct resolver tests. |
| Checkpoint authority | Read the pinned configuration and its recorded identity | Qwen3.8 has 64 entries with a 48/16 split | Pinned `config.json`, BLAKE3 `a19b6657...` | Validated | Check every expected full-attention index. |
| Legacy fallback | Separate absent metadata from malformed metadata | Only the exact 32-layer legacy model can use the fallback | Direct accepted and rejected resolver tests | Validated | None. |
| Narrative audit | Read receipt objects by named JSON key | The repeated-sample archive reversed candidate and eager | `candidate.valid = true`, `eager.valid = false` | Validated | None. |
| Adversarial configuration | Supply unknown, missing, and wrong-length values | Invalid authority fails before model layer construction | Pure negative tests and the device-free package gate | Validated | None. |

## Allowed outcomes

- **Validated:** the source dataflow is enforced and all focused gates pass.
- **Blocked:** an adopted-source consumer or checkpoint mismatch contradicts the proposed boundary.
- **Exhausted:** the bounded source and receipt set cannot establish dispatch authority.
- **User decision required:** a valid legacy configuration needs behavior outside the exact fallback contract.

## Terminal result

Validated. The adopted mapper set had no consumer. The package now validates checkpoint-owned dispatch, and the corrected archive matches named receipt fields.
