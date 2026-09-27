# t5: deployment on britton-desktop (2026-09-27)

Both t4 receipts admit package `/nix/store/a32pfd9h7lkkidnpmni2lk7gs138dx0x-qwen38-0.1.0`,
so it was deployed.

- **Configuration.** onix-core `e8d1a26c` (branch `underclass-mesh-desktop`) pins
  `tenstorrent-nix-qwen` to `30b4d1b`. The configuration is otherwise unchanged:
  `continuousSlotScheduler` and `prefixCache` stay enabled.
- **System.**
  `/nix/store/0ig5wqjkc7dazpsvm89g2ng8q1s7xqlw-nixos-system-britton-desktop-26.11.20260908.e9b9cbe`,
  switched at about 23:50 UTC.
  - The `britton-desktop-accelerator-inventory` and `mesh-llm-sidecars` checks built.
  - dry-activate restarted only `qwen38-p150x2`, `polkit` and `kiln-aspen-canary-lattice`, and
    reloaded `dbus-broker`.
  - The switch reported failures that predate this change: two ZFS activation snippets,
    `celld`, `celld-site`, `collie-serve` and `kiln-aspen-ci-host`.
- **Served unit.** `qwen38-p150x2.service` runs `a32pfd9h` with `--continuous-slot-policy`
  and `--prefix-cache-policy` from that package. `/healthz` reported ready after a 127 s
  warm-up.
- **Production check.** A three-turn w3 conversation on port 8000:
  - Trace captures were 1, 0, 0: the service's first turn after the restart captured,
    later turns kept the trace.
  - The longest pause between tokens was 0.89 s, then 0.053 s and 0.056 s.
  - Hits were 0, 3,072 and 5,120 tokens.
  - Every turn decoded the same tokens as the admitted cold root p7 a1.
- **New log fields.**
  - `request-passed` now carries `prefix_hit_tokens`, `prefix_saved_tokens` and
    `trace_captures`.
  - `continuous-slot-group-started` carries `decode_trace` (`preserve-for-reuse` after the
    first group) and `decode_trace_reason`.
- **Mesh.** A request from aspen3 through port 9337 was served by `continuous-slot` with
  execution identity
  `/nix/store/g5pdvpin2n221v8c95j1mxbw55zk98pj-qwen36-execution-identity-0.1.0` and 0 trace
  captures.
