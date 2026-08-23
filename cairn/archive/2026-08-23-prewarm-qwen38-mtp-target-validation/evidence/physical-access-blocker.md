# Physical access blocker

## Prepared control

The validated package is `/nix/store/b3w97pzld9hg2ndim8y9pkjx4rkx613k-qwen36-0.1.0`.

The control uses:

- command `qwen36-p150x2-mtp-recovery-prewarm-stage`;
- prompt index `2`;
- route `snapshot`;
- PCIe devices `0` and `1`;
- mesh `(1, 2)`;
- CPUs `8-31`;
- one loaded process;
- one `1800`-second command bound; and
- one `120`-second termination grace.

Prompt 2 exercises rejected-row restoration, target reference-state construction, and native verification. The same-prompt prewarm runs before persistent trace capture.

## Safety sequence

The operator must isolate `qwen38-p150x2.service` with `/run/current-system/sw/bin/ttwkv7-owner-control`.

If the command reaches its bound or stops without clean device closure, the operator must keep the service isolated. The operator must reset both devices before restoration.

The reset command is:

`tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1`

The operator must restore the eager service and require the expected `/healthz` response.

## Required authorization

A new explicit bounded outage authorization is required before this physical case can run. Device-free work does not authorize the outage.
