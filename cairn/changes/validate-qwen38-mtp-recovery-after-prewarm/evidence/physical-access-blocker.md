# Physical access blocker

## Prepared control

The validated package is `/nix/store/y625zi8mwspwgdcs031nlaizfhf4kmdl-qwen36-0.1.0`.

The control uses:

- command `qwen36-p150x2-mtp-recovery-prewarmed-control`;
- five exact nonpersistent prewarm prompts;
- ten discarded persistent warmup roots;
- sixty measured roots in the accepted balanced schedule;
- PCIe devices `0` and `1`;
- mesh `(1, 2)`;
- CPUs `8-31`;
- one loaded process;
- one `1800`-second command bound; and
- one `120`-second termination grace.

The command writes one atomic progress ledger. It fails if a warmup or measured signature changes. It also fails if the program-cache count changes after measurement starts.

## Safety sequence

The operator must isolate `qwen38-p150x2.service` with `/run/current-system/sw/bin/ttwkv7-owner-control`.

If the command reaches its bound or stops without clean device closure, the operator must keep the service isolated. The operator must reset both devices before restoration.

The reset command is:

`tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1`

The operator must restore the eager service and require the expected `/healthz` response.

## Required authorization

A new explicit authorization for this exact bounded outage is required before physical execution. Device-free validation does not authorize the outage.
