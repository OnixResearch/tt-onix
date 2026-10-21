# Repository quality gate

Run the complete device-free gate from the repository root:

```sh
nix run .#quality
```

The command evaluates every supported system. Then it builds `checks.x86_64-linux.repository-quality` with Nix-owned tools.

The launcher evaluates the Git-tracked worktree, not a raw directory copy:
ignored `.pi/`, `generated/`, and downloaded model files stay out of the Nix
store even when present locally. Add new source or fixture files to the Git
index before running the gate; untracked files are not tested. The gate still
does not review tracked files for publication safety.

## Baseline from 2026-08-07

The pre-change flake evaluation and three focused Nix builds passed.

Clippy rejected two long layer-harness argument lists. It then found one manual slice-size calculation after those errors were fixed.

ShellCheck found 11 scripts without interpreters. It also found one unused variable and two unsafe optional-argument expansions.

Ruff found 39 format or lint errors. The tracked repository used 79.7 MB, including 70.1 MB of private runtime logs.

Cairn validation passed with nine active changes. The Cairn command used the pinned policy from the Cairn flake input.

## Gate contents

The aggregate gate runs these device-free groups:

1. Rust formatting, Clippy, and tests for every repository Cargo package; the
   in-tree `metalium-rs` workspace runs its own full test tier in the
   `metalium-rs` package check.
2. ShellCheck, interpreter checks, and Bash syntax checks.
3. Ruff formatting and lint checks for maintained Python files.
4. Nickel policy export and generated-data freshness checks.
5. Artifact policy, model-target freshness, focused package checks, and Cairn validation.

`quality/required-checks.ncl` owns the required package inventory. The aggregate includes `qwen36`, `qwen36-authority-negative`, `qwen36-nixos-service`, and the device-free `metalium-rs-rwkv7-tokenizer` gate. Its Nix selector rejects missing checks, unregistered checks, duplicates, and non-derivation entries. Negative controls remove each required check from the available set.

Cairn output passes through `repository-quality cairn-verdict RECEIPT`. The verifier requires a true top-level verdict, valid root resolution, and empty issue lists. It rejects duplicate keys, missing fields, wrong types, unsupported fields, malformed JSON, and excessive input. The JSON adapter enforces named byte, item, depth, and field limits. The core owns the acceptance decision. The CLI owns bounded file reads and the exit status.

The gate runs in a Nix sandbox. The sandbox must not expose `/dev/tenstorrent`.

Hardware checks remain separate. The quality gate does not stop device owners, change firmware, or run a hardware workload.

## Parser inventory

The limits originate in `quality/policy.ncl`. The checked JSON projection is `quality/generated/repository-quality.json`.

| Family | Main boundary | Trust boundary | Current limit |
| --- | --- | --- | --- |
| JSON | `pkgs/model-target-report`, runtime and evidence receipts | Caller or saved evidence | 1 MiB |
| Receipt | `pkgs/rwkv-ttwkv7-persistent-receipt-validator` | Hardware wrapper output | 2 MiB |
| YAML | `pkgs/rwkv7-p150x2-evidence` | TT-Metal profiler output | 8 MiB |
| CSV | `pkgs/rwkv7-p150x2-evidence` | TT-Metal profiler output | 8 MiB, 100000 rows, 256 fields |
| Binary frame | `pkgs/rwkv-layer-harness` | Persistent process transport | 16 MiB |
| Vocabulary | RWKV tokenizer parsers | Pinned external model artifact | 16 MiB, 100000 items |

The central observation tests verify policy for supplied counts and error flags. Actual byte-level tests run in the parser-owner crates. They cover all six families with accepted counterparts and rejected seeds.

The [seed index](../quality/fuzz-seeds/README.md) lists the parser entry points and fixture paths.

After a fuzz run finds an error, add a minimized seed to its parser's fixture directory.

Do not add runtime entropy to the default gate. The normal gate replays only checked seeds and deterministic properties.

## Artifact workflow

The typed artifact classes are source, generated source, portable evidence, private log, large fixture, and external large object.

`quality/policy.ncl` owns the explicit `evidence_roots` registry. It currently registers all files under `docs/research`. These files require BLAKE3 manifest entries and private-marker checks, regardless of filename, extension, size class, or nested build-directory names. Similar names such as `docs/research-more` do not match this root. The legacy `/evidence/` classification remains.

The pure registry validator rejects missing, empty, noncanonical, overlapping, reserved, and unowned roots. Registered evidence cannot disable BLAKE3 or permit private content. The filesystem adapter rejects symbolic links in evidence scopes without following their targets. Private-marker checks inspect bytes, so invalid UTF-8 does not hide a marker.

This registry defines repository evidence ownership. It does not identify every receipt by content or prove mutable-filesystem confinement. Moving a receipt outside registered scopes invalidates its existing manifest. Deliberate policy changes still require review.

Run this command after an approved generated or evidence artifact changes:

```sh
nix run .#repository-quality -- manifest . > quality/generated/artifact-manifest.json
```

Then run `nix run .#quality`. The gate rejects stale hashes, private content, unapproved large files, and tracked private logs.

Keep runtime inspector and watcher logs in `.pi/` or another secure local path. The repository ignores the top-level `generated/` root.

The gate uses BLAKE3 identities for generated source, portable evidence, large fixtures, and external large objects.

## Non-claims

A passing quality gate does not prove hardware function, model correctness, performance, convergence, thermal stability, or simulator completion.
