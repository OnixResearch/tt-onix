# Physical result

## Scope

The staged control used physical PCIe devices `0` and `1`, mesh `(1, 2)`, CPUs `8-31`, and package `/nix/store/gqpbdg75m69wx9gk7g9h3wjgr83438gj-qwen36-0.1.0`.

The package used runtime `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0` and checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Each case had a named `1200`-second bound and a `120`-second kill grace.

## Prompt 1 re-prefill

The route completed MTP trace capture, two draft replay steps, target sequential decode, MTP re-prefill, four controlled replay steps, device synchronization, and recovery-state readback.

The last completed boundary was `recovery-complete` at `199.16399158997228` elapsed seconds. The next boundary, `native-verifier-start`, was recorded at `199.17003613198176` seconds.

The process remained inside native verifier setup while TT-Metal compiled target kernels. It reached the `1200`-second bound with status `124`. No receipt was produced.

Ledger:

- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-stage-reprefill-progress.json`
- BLAKE3 `de1f81db3e9c9b279aacc8f28fcedb9fd6f00f71efbe7ada0e1bab2a79514d35`

## Prompt 2 snapshot

The route completed MTP trace capture, two draft replay steps, target sequential decode, rejected-row restoration, one controlled replay step, device synchronization, and recovery-state readback.

The last completed boundary was `recovery-complete` at `89.87513524800306` elapsed seconds. The next boundary, `target-reference-state-start`, was recorded at `89.87818524602335` seconds.

The process remained inside target reference-state construction while TT-Metal compiled target kernels. It reached the `1200`-second bound with status `124`. No receipt was produced.

Ledger:

- `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-stage-snapshot-progress.json`
- BLAKE3 `64de84aa0cbe1a32ec1ed88b43a992a69500116b3b031cac92e3132a37e8c556`

## Restoration incident

The first timeout restored eager service normally.

After the second timeout, the service restart loop failed with:

`RuntimeError: Read 0xffffffff over PCIe ID 0: the board should be reset.`

The operator isolated the service and ran the package-matched reset path:

`tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1`

The reset completed. The operator restored the service. It is active, and `/healthz` returns `{"model": "Qwen3.8-27B", "status": "ok"}`.

## Decision

The staged result localizes both bounded noncompletions after MTP recovery. It does not prove that every recovery root is stable or correct.

The current blocker is cold or incomplete target validation work after recovery. The two observed boundaries are native verification and target reference-state construction.

No complete receipt exists. Persistent recovery tracing, MTP, and traced target decode remain rejected. Production remains eager.
