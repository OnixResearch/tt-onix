## Why

A single short prompt took 2.3 to 2.6 s to its first token on P150x2. Almost
all of that time was host dispatch of about 23,000 TT-NN operations. A traced
replay of the same 128-token forward takes 0.20 s on the device. Existing prefill
traces served only prompts exactly one bucket long, because the GDN padding mask
and the conv-window selection were built from the prompt length on the host.
The batch worker also had no route for 11 or 13 concurrent requests. Width 11
failed decode trace capture on the KV-update shard grid, and width 13 has no
one-user-per-core rectangle on the 11x10 grid.

## What Changes

- Pass the prompt length to the GDN seq prefill as device data: a conv-window
  selector and a scan mask. Each is a persistent input that a replay refreshes.
- Capture one B=1 masked prefill trace per bucket (128, 256, 512, and 1,024).
  Replay the smallest bucket that holds each prompt. Inside the trace, reset the
  GDN scratch, compute the logit, and install state and logits into the staged
  decode slot with one custom program. A batch's replays are enqueued back to
  back and read once. Prompts up to 2,047 tokens continue from the 1,024 bucket
  into a tail bucket, with a staged predicate that keeps the carried state.
  Output is identical to eager prefill.
- Capture at startup, after decode at every selectable width and route, the
  eager prefill shapes, and every decode-slot install. If a program first
  compiles after the capture, release the traces and prefill eagerly until
  restart.
- Prefill with full-width page-table rows, so each prefill shape has one set of
  programs.
- Shard the width-11 KV update as one grid row. Return the newest request of a
  13-request batch to the queue.
- Add the NixOS option `hardware.tenstorrent.qwen38.tracedPrefill`, default on.

## Impact

- **Files**: `pkgs/qwen36/masked-prefill-traces.patch` (model, layer, GDN and
  attention TP modules, seq and FIR kernels),
  `pkgs/qwen36/service/qwen36_service/{batch_decode_adapter,batch_service}.py`,
  `pkgs/qwen36/check.py`, `pkgs/qwen36/warmup-tail-check.py`,
  `pkgs/qwen36/default.nix`, `nixos/module.nix`, and
  `flake-modules/onix-packages.nix`.
- **Startup**: capture adds 86 to 220 s after the existing warm-up, depending
  on host load.
- **Production**: the host configuration lives outside this repository.
