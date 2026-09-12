# Deployment gap, target artifact, and readiness gate

Recorded 2026-09-11 for change `deploy-qwen38-measured-runtime`, tasks A1-A3, I1, I3.

## A1: what production runs today

| Item | Value |
| --- | --- |
| system generation | `nixos-system-britton-desktop-26.11.20260908.e9b9cbe` (2026-09-08) |
| service unit | `qwen38-p150x2.service`, active, `/healthz` ok |
| package | `/nix/store/jsivpglnl9b8k35wq1rcs60rkhwgxiq3-qwen36-0.1.0` |
| runtime identity | `/nix/store/djchdwzksw0k5psjfwylqpfyixdzhq69-tt-metal-0.74.0` |
| execution identity | `/nix/store/yarrawqwbfyhgmk66170g1av7k4lhzl4-qwen36-execution-identity-0.1.0` |
| decode mode | `candidate` |

No held receipt names that runtime. The three receipts this repository carries
name three other identities: B=4 admitted on 0.74 (`1l9nlvvv`), B=8 rejected on
0.74 (`mg0ha5ns`), B=8 admitted on the 0.77 chunked variant (`6ccynlvv`,
execution identity `z31p1j9w`). The deployment is therefore three runtimes
behind its own evidence.

## A2: what a rebuild would deploy

The module's `qwen38.package` default is `pkgs.qwen38`. Evaluated and built from
this repository at HEAD:

| Item | Value |
| --- | --- |
| package | `/nix/store/4jrx0g3rafcypgn3ndssw5s3dnblwhm7-qwen38-0.1.0` |
| runtime identity | `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0` (shared) |
| execution identity | `/nix/store/3wnk77930rdz2nipxbzln47plnbkp2v6-qwen36-execution-identity-0.1.0` |
| serving wrapper flags | `--decode-mode candidate`, no batch admission |
| batched admission binary runtime | `dr9bj8qw…` (shared) |

Note the identity pairing, which matters for reading the receipts: the *shared*
package carries identity `3wnk7793…`, and the *chunked variant*
(`inmzbg5r…`) carries `z31p1j9w…`, the identity the admitted B=8 receipt
records. The B=8 admission is therefore specifically a chunked-variant
admission; the module default does not deploy it.

## A3: what the switch does and does not carry

- **Carries:** the 0.74 -> 0.77 runtime move for the single-stream service, with
  the same `candidate` decode mode production already runs.
- **Does not carry:** the admitted B=8 batch capability. The module default
  selects the shared runtime, where the width-8 fused decode cannot launch at
  all, and the service does not pass `--batch-admission`. Even on the chunked
  variant, the admitted receipt's own non-claim excludes arbitrary continuous
  batching, so enabling a batched serving route would exceed the evidence.
- **Does not move:** any admission binding, any profile, or any receipt.

## I1: readiness gate on the target artifact

`/tmp/gdn-deploy/readiness.sh` isolated the production service, started
`/nix/store/4jrx0g3r…-qwen38-0.1.0/bin/qwen38-p150x2-serve` with the module's
argument list, waited for readiness, served one greedy request from the frozen
corpus, and restored the service.

```
ready=true
{"model": "Qwen3.8-27B", "ready": true, "startup_warmup": {"elapsed_seconds": 120.618…, "required": true, "widths": [1, 4]}, "status": "ok"}
text: ' Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe '
tokens: [11751, 13, 198, 760, 6511, 314, 9564, 369]
decode_path: batched-traced width: 1
runtime_identity: /nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0
package_identity: /nix/store/3wnk77930rdz2nipxbzln47plnbkp2v6-qwen36-execution-identity-0.1.0
physical_devices: [0, 1] simulated: False
```

The token prefix `11751, 13, 198, 760, …` is the admitted reference prefix
recorded by the earlier B=1 and B=8 evidence, so the target artifact starts,
serves, and reproduces the known output on both physical devices.

## I2: the host edit, and its blocker

The switch requires a host configuration that imports `nixos/module.nix`. That
file is **not in this repository**: the flake defines no `nixosConfigurations`,
there is no `/etc/nixos`, no registry entry names it, and a bounded search of
`~/git` and `~` for `hardware.tenstorrent.qwen38`, `qwen36` service settings,
and this repository's module name found nothing. The deployed generation's
closure does not carry the configuration source.

The edit the host needs is two lines in its module instantiation:

```nix
hardware.tenstorrent.qwen38.package = pkgs.qwen38;   # already the default
# and, if the chunked variant is ever the target:
# hardware.tenstorrent.qwen38.package = pkgs.qwen38-gdn-grid;
```

No module change is required for the shared 0.77 target; selecting the variant
is the only case that needs a non-default package. Tasks I2 and V1-V4 stay open
until that configuration is located.

## I3: artifact manifest repair

`quality/generated/artifact-manifest.json` became invalid JSON at the merge
`e0035c3`, which spliced two manifest revisions and dropped one entry
terminator. The repair inserts the missing `},` and `{` at the single splice
point (file line 1719); no entry text is added, removed, or reordered.

```
before: repository-quality: artifact manifest is invalid JSON: expected `,` or `}` at line 1719 column 7
after:  repository-quality: artifact manifest entry count drift: expected 333, found 507
```

The gate is still red, now for the pre-existing reason that commit `680afa7`
recorded and chose to keep visible. Regenerating would produce 507 entries via
`repository-quality manifest ROOT` — that remains its owner's decision, and the
receipt this session added
(`pkgs/qwen36/evidence/2026-09-11/qwen38-27b-batch-admission-b8-receipt.json`)
is included in a regenerated manifest.
