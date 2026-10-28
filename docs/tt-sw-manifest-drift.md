# tt-sw-manifest golden drift

The packaged stack is compared with
[`tt-sw-manifest` v4.0.0 `golden.json`](https://github.com/tenstorrent/tt-sw-manifest/releases/tag/v4.0.0)
by `nix run .#golden-drift` (report) and `nix run .#golden-drift -- --strict`
(exit 1 on drift). The non-strict report is also the `golden-drift-report`
flake check; it never fails the quality gate.

## Current report (2026-10-28)

| Component | golden v4.0.0 | packaged | status |
| --- | --- | --- | --- |
| kmd | 2.11.0 | 2.10.0 | drift |
| smi | 6.7.0 | 6.2.1 | drift |
| flash | 4.1.0 | 3.10.0 | drift |
| sfpi | 7.71.0 | 7.70.0 | drift |
| tt-metal | v0.78.0 | 0.77.0 | drift |
| firmware | 19.15.0 | not pinned | documented-only |

## Why the bumps are not in this change

The three Python diagnostics and the runtime are coupled:

- `tt-smi` 6.7.0 requires `pyluwen~=0.10.0` and `tt-umd==0.9.12`; this tree
  carries pyluwen 0.9.0 and tt-umd 0.9.9. A version-only override would change
  the Python/TT-NN ABI closure used by the admitted Qwen receipts.
- `tt-flash` 4.1.0 also requires `pyluwen~=0.10.0` and `tt_tools_common~=1.5`;
  the current packages are pyluwen 0.9.0 and tt-tools-common 1.6.0.
- SFPI 7.71.0 follows tt-metal 0.78.0. Moving tt-metal from 0.77.0 requires
  re-running the `gdn-*` patch stack and Qwen/Nanbeige checks; the 0.77 GDN
  simulator gate is still blocked.

The safest next step is one coordinated branch: bump pyluwen, tt-umd,
tt-tools-common, tt-smi, tt-flash and tt-kmd together; prove the Python/TT-NN
ABI and the diagnostics; then take tt-metal 0.78.0 + SFPI 7.71.0 with the full
patch-stack and model gates. Until then the drift report is intentionally red in
strict mode and does not block normal builds.

Firmware 19.15.0 is the expected bundle for `tt-flash` users; it is not pinned
by this repository.
