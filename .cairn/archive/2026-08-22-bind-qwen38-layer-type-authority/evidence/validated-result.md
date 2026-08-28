# Validated result

## Layer-type authority

The adopted source is commit `d102e7f4512060e09e580e40cf26a870bfa88c55`. Its `weight_mapping.py` contained one `FULL_ATTENTION_LAYERS` declaration and no consumer. The remapper did not use the set for tensor filtering, renaming, splitting, or dispatch.

Runtime dispatch follows this path:

1. `Qwen36ModelArgs` reads `text_config.layer_types`.
2. `resolve_attention_type_list` validates the complete list and declared layer count.
3. `Qwen36ModelArgs.attention_type_list` stores the immutable result.
4. Each `Qwen36TransformerBlock` calls `is_full_attention_layer(layer_num)`.
5. The block constructs either full attention or GDN from that result.

The pinned configuration has BLAKE3 `a19b66576a666e181293906ba585c13edcb2caa946d342a6a42b9d6a873a119d`. The packaged resolver returned these facts from that file:

- total layers: `64`
- linear-attention layers: `48`
- full-attention layers: `16`
- full-attention indices: `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]`

The new authority JSON has BLAKE3 `c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd`. It records the interval and exact indices. The package build `/nix/store/35xq3yn6h45cvvahhr6yw818dphi14bv-qwen36-0.1.0` passed the device-free resolver tests.

## Historical narrative audit

The host-stability receipt has named top-level `candidate` and `eager` objects. The candidate comparability object is valid. Its wall, enqueue, and gap ratios are `1.1673619615353883`, `1.0199312943580248`, and `1.1955534484861683`.

The eager comparability object is invalid. Its wall, enqueue, and gap ratios are `1.3928346091490837`, `1.2300527123313059`, and `1.4276727516605083`.

The archived physical narrative had these labels reversed. This change corrects the labels and the drift explanation. It does not change the receipt, values, verdict, or BLAKE3 identity. The signpost and earlier host-timeline narratives already match their named receipt objects.

## Portfolio result

All six approach families reached a validated state. The serial passes are correlated because no subagents were authorized. The adversarial tests rejected missing non-legacy metadata, wrong-length lists, unknown types, invalid containers, and invalid layer counts.

## Validation

These gates passed:

- `.#checks.x86_64-linux.qwen36`
- `.#checks.x86_64-linux.qwen36-authority-negative`
- `.#checks.x86_64-linux.qwen36-nixos-service`
- focused Python, Nix, policy, Cairn, and artifact quality gates
- Ruff formatting, Ruff lint, Python compilation, and Nix formatting
- Cairn validation and proposal, design, and tasks gates

Global Tracey coverage still reports the existing repository-wide policy-scope gaps. The new Python implementation is outside the current Tracey evidence roots, which include only `crates` and `tools`. Focused Cairn and package gates passed.

## Non-claims

- This source validation adds no new silicon, parity, performance, or admission evidence.
- Historical physical receipts remain bound to their original package and authority identities.
- Production remains on the deployed eager package.
