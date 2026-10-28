# tt-sw-manifest golden drift

The packaged stack is compared with
[`tt-sw-manifest` v4.0.0 `golden.json`](https://github.com/tenstorrent/tt-sw-manifest/releases/tag/v4.0.0)
by `nix run .#golden-drift` (report) and `nix run .#golden-drift -- --strict`
(exit 1 on drift). The non-strict report is also the `golden-drift-report`
flake check; it never fails the quality gate.

## Current report (2026-10-28)

| Component | golden v4.0.0 | packaged | status |
| --- | --- | --- | --- |
| kmd | 2.11.0 | 2.11.0 | match |
| smi | 6.7.0 | 6.7.0 | match |
| flash | 4.1.0 | 4.1.0 | match |
| sfpi | 7.71.0 | 7.71.0 | match |
| tt-metal | v0.78.0 | 0.78.0 | match |
| firmware | 19.15.0 | not pinned | documented-only |

## Diagnostics and driver stack

The diagnostics and driver stack now follows golden v4.0.0 as one coordinated
set:

- `tt-kmd` 2.11.0.
- `tt-smi` 6.7.0 with `pyluwen~=0.10.0` and `tt-umd==0.9.12`.
- `tt-flash` 4.1.0 with `pyluwen~=0.10.0` and `tt_tools_common~=1.5`.
- `pyluwen` 0.10.0, `tt-umd` 0.9.12, `tt-tools-common` 1.6.0.

`tt-smi` no longer needs a `pythonRelaxDeps` entry: its exact `tt-umd==0.9.12`
pin and its `pyluwen~=0.10.0` / `tt_tools_common~=1.6.0` ranges are satisfied by
the packages above. `tt-flash` drops its `pyluwen` relaxation for the same
reason but keeps `pyyaml` and `tabulate`: it pins `pyyaml==6.0.2` and
`tabulate==0.9.0`, while nixpkgs carries the newer compatible 6.0.3 and 0.10.0.

The NixOS training profile was updated with the bump:
`admittedTrainingDriverVersion` in `nixos/module.nix` and the matching TT-KMD
assertion in `flake-modules/onix-packages.nix` now name 2.11.0. `tt-burnin`
keeps its own `pyluwen` relaxation because 0.4.4 still declares `~=0.9.0`.

## Runtime migration

TT-Metal 0.78.0 and SFPI 7.71.0 are now packaged. The production `gdn-*` patch
set applies clean at fuzz 0; `tt-metal`, `qwen38`, and `nanbeige43` build and
their install checks pass. The diagnostic-only `fetch-queue-diagnostics.patch`
was regenerated against 0.78 for the required call-site and sources.cmake
context.

All five compared components now match golden v4.0.0. Strict mode exits 0.
Firmware 19.15.0 is the expected bundle for `tt-flash` users; it is not pinned
by this repository.
