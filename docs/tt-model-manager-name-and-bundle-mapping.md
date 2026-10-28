# tt-model naming and tt-model-manager bundle mapping

## Decision

This repository keeps the `tt-model` package and CLI name. The upstream
[`tenstorrent/tt-model-manager`](https://github.com/tenstorrent/tt-model-manager)
tool is a different product and will be packaged, if at all, as
`tt-model-manager` with a `tt-model-manager` binary. It will not replace or
alias the local `tt-model`.

Rationale:

- the local `tt-model` is the runtime-profile registry, record, switch, and run
  tool used by `nixos/examples/qwen38-tt-model-switch.nix` and the
  `checks/tt-model-*.nix` gates;
- upstream `tt-model-manager` is experimental, not on PyPI, and its CLI is also
  named `tt-model`; adopting that name would silently change the meaning of a
  command in existing host configuration;
- a compatibility alias would have to decide which product wins for the same
  command, which is exactly the ambiguity this decision avoids.

Upstream is therefore documented as wrapped under another name. No code rename
is part of this change. A future package must install
`bin/tt-model-manager`, not `bin/tt-model`.

## Source boundary

Reviewed `tenstorrent/tt-model-manager` at commit
`0ca9e4001e81e14504a12a4f398ad1858b92d137` (2026-10-26). The v6 thin manifest
is `tt_kernel_manifest.json`; its typed model is
`src/tt_kernel/manifest.py` (`Manifest`, `WeightsRef`, `Mesh`, `Entrypoint`,
`Resources`, `Capabilities`, `Deps`). The producer works through
`tt-model package-thin` and the bundle's `install.sh` / `run.sh`.

The local model target contract is `model-targets/contract.ncl`
(`ModelProfile`). It describes model identity, architecture shape, recurrent
state, resource floors, device topology, consumer tolerances, and artifact
hashes for this repository's harnesses. It is not a serving-bundle manifest.

## Field mapping

| Local `ModelProfile` | Upstream v6 field | Status |
| --- | --- | --- |
| `target_name` | `Manifest.name` | Equivalent bundle identity; local name may be a target label rather than a Hub repo name. |
| `model.id` | `weights.repo` (JSON alias of `WeightsRef.repo_id`) | Direct pointer, but local `id` is not required to be an HF repo. |
| `model.revision` | `weights.revision` | Direct. Both require a pinned revision for a reproducible bundle. |
| `model.license` | none | The v6 manifest has no license field; keep the local value and the model card. |
| `model.architecture_family` | `entrypoint.arch_name` | Partial. Upstream expects the HF architecture registration name, not a family label. |
| `model.checkpoint_layout`, `model.checkpoint_shard_count` | none | Derived from the HF repo; not part of the v6 manifest. Keep local. |
| `model.precision` | none first-class | Expressed through `deps.requirements` / `deps.constraints` and `resources.tt_additional_config`; no exact field equivalent. |
| `architecture.*` layer/head/expert counts | none | The v6 manifest does not describe internal architecture. Keep local; do not drop these facts when producing a bundle. |
| `state.*` kind, per-layer elements, carry semantics | none | Serving-stack semantics, not bundle metadata. Keep local. |
| `resources.checkpoint_bytes`, `layer_count`, `hidden_size`, `vocabulary_size` | none | Not represented. `resources.max_model_len` is the only nearby capacity field. |
| `resources.maximum_prompt_tokens` | `resources.max_model_len` | Partial: local is a prompt bound; upstream is the model context bound. |
| `adapters.architecture` | `entrypoint.arch_name` | Partial, same distinction as `architecture_family`. |
| `adapters.evidence` | `deps.verify` | Different purpose: local evidence names a repository proof; `verify` is an install-time Python assertion. |
| `topology.id` | `mesh.topology` | Partial/naming difference. |
| `topology.mesh_shape` | `mesh.devices` + `mesh.topology` | Lossy: v6 records a device count and an optional topology string, not an arbitrary shape. |
| `topology.physical_device_ids` | none | Deliberately host-local; v6 must not record physical PCIe IDs. |
| `topology.replication` | none | Not represented. |
| `consumer.dimensions` | none | Harness dimensions, not bundle metadata. |
| `consumer.precisions` | `deps.requirements` / `deps.constraints` | Partial: dependency pins, not named tolerance semantics. |
| `consumer.tolerances` | none | Harness acceptance bounds; keep local. |
| `consumer.adapter_format` | `entrypoint.cls` (indirect) | The adapter class is named, but the local format string and the class are not the same thing. |
| `artifacts[].role/name/url/nix_hash/blake3` | `deps.wheels`, `deps.custom_ops`, `deps.requirements` | Partial. v6 lists bundle-relative wheel paths; it does not carry the local role/url/Nix-hash tuple. |

## Gaps that must stay visible

The v6 manifest cannot express, and a converter must not silently drop:

- architecture layer/head/expert counts and the linear/full attention split;
- recurrent state kind, per-layer element count, and carry semantics;
- physical device IDs or replication policy;
- consumer dimensions, precision labels, and tolerance bounds;
- artifact role/url/Nix-hash provenance beyond bundle-relative wheel paths.

A future `model-target-report` converter should emit the mapped v6 subset and a
machine-readable `unmapped` list for the rest. Until then, this document is the
mapping authority and no local target is claimed to be a v6 bundle.

## What is not done

- Upstream `tt-model-manager` is not packaged. It remains experimental and has
  no release.
- No local CLI rename, alias, or NixOS option rename is made.
- No v6 bundle is produced or validated.
