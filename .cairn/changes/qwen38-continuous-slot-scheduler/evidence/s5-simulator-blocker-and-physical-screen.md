# s5: simulator blocker, then one physical B4 screen (2026-09-27)

Task s5 allows a representative simulator refill, or its exact blocker, before an
approved physical B4 screen. The simulator cannot run the refill path; the blocker is
reproduced below on the current pins. The physical screen then ran once on
britton-desktop under the standing Tenstorrent maintenance authority.

## Simulator: the exact blocker, reproduced on the current pins

The continuous-slot path is TP=2 only: `BatchedTracedDecodeAdapter` refuses a mesh with
one device. On the simulator it would therefore need `libttsim_bh_x2.so`. AGENTS.md
allows the two-chip model only after the single-chip smoke passes for the operations
used. Every Qwen refill and decode step runs the GDN kernels, and the single-chip GDN
smoke still fails.

- Command: `nix build .#checks.x86_64-linux.ttsim-gdn-reference-smoke --no-link -L`,
  run under a 60-minute timeout; it stopped after 65 s and finished at 16:46 UTC on
  2026-09-27.
- Inputs:
  - TTsim 1.10.6: `/nix/store/0qvnw9q02agnjbnm5pwvfpxhmrkqczii-ttsim-1.10.6`, which also
    ships `libttsim_bh_x2.so`.
  - TT-Metal: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`, the runtime
    of the deployed and the screened package.
- Result, from the derivation
  `/nix/store/91xvmi6ganw3f5pfzydlhmfd3k3n5vfm-ttsim-gdn-fp32-reference-smoke.drv`:

  ```text
  ttsim-gdn-fp32-reference-smoke> [7470] ERROR: UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled
  ```

This is the same diagnostic as `docs/evidence/tt-metal-0.77/gdn-blocker.log`. The
release notes of TTsim 1.10.7 through 1.10.10 (the latest, 2026-09-25) do not mention
the flag. The pin stays at 1.10.6, because a newer release is not by itself a reason to
move it. No simulator receipt is claimed.

## Physical B4 screen

- Package: qwen38 `/nix/store/jxhspibs4ml8sbspj16bynhb1yp887vb-qwen38-0.1.0`
  (tt-onix `65b33de`). Execution identity:
  `/nix/store/h73yynndm34hkzr3aimlfnif6r3402iy-qwen36-execution-identity-0.1.0`.
  Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- How it ran:
  1. `qwen38-p150x2.service` was stopped at 16:49:55 UTC.
  2. The candidate ran as the transient unit `qwen38-screen-continuous`, with the
     production unit's environment (`QWEN35_GDN_DECODE_FUSION=step`,
     `QWEN36_SAMPLED_FUSION=1`, `QWEN36_TRACED_PREFILL=1`). The command was:

     ```text
     qwen38-p150x2-serve --host 127.0.0.1 --port 8011 \
       --continuous-slot-policy $pkg/share/qwen36/continuous-slot-policy.json
     ```

  3. The driver, `scripts/qwen38-continuous-slot-screen.py`, checked the listener from
     its own evidence: decode path `continuous-slot`, devices `[0, 1]`, not simulated.
  4. The production unit was restarted at 17:15:08 UTC.
- Readiness took 230 s, with 9,919,391,744 B of device DRAM free after startup. The
  group-coordinator baseline on the same package reported the same DRAM.
- Workload: eight chat requests with thinking disabled, all with `max_tokens` 1,024.
  - Two long requests at t=0 run to the cap: l1 has a 2,226-token prompt, l2 3,097.
  - Six short requests arrive at 4–24 s, with prompts of 24 (s1), 980 (s2),
    9,618 (s3), 24 (m1), 26 (s4) and 2,055 (s5) tokens. The passages come from
    `.pi/references/rwkv-7/rwkv-7-goose-arxiv-2503.14456v2.txt`.
  - s3 exceeds the 8,192-token refill budget. It was admitted as the only refill at
    its boundary, because the first refill at a boundary is always allowed.
  - Each request first ran alone as its reference, then in two staggered passes.
  - The full data is `pkgs/qwen36/evidence/2026-09-27/qwen38-continuous-slot-continuous-root.json.gz`.

### What the screen exercised (both passes)

| Question left open by `refill-coexistence.md` | Observed |
| --- | --- |
| Refilling a slot that held an earlier user | Slot 2 took s1, s2, s3, s4, s5 in turn: generations 11, 12, 13, 15, 16 in the first pass and 19, 20, 21, 23, 24 in the second. |
| More than one refill per group | Each pass ran as one group with 7 refills beside live rows (`continuous-slot-group-started` 16 times in total: 8 solos, 1 per pass, 6 in the extra checks). |
| Idle rows decoding into the scratch block | Slot 3 stayed idle until m1 arrived at 16 s. Slot 2 was idle between the short requests. |
| The fed-back greedy stream across refills | Every window used `decode_slots_greedy`, with one capture per pass and a restage after each refill and retirement. |
| Structured reads beside a resident trace | A read sent 3 s into a long request waited for the group to drain, as designed, and returned the same answers as an idle read. A short request queued behind the read was served after it and matched its reference. |
| Sampled routes | Two seeded requests at temperature 0.6 (route `full`), run beside each other and beside a greedy request, reproduced their solo tokens exactly. The greedy request did too. |
| Program compilation during a refill | None. There were no `continuous-slot-group-draining`, `masked-prefill-traces-released`, `continuous-slot-group-failed` or `engine-quarantined` events. |

**Isolation.** Every request in both passes decoded exactly the tokens it decoded alone
on the same path: 16 of 16 requests, including both 1,024-token long requests. The s5
checks are in `pkgs/qwen36/evidence/2026-09-27/qwen38-continuous-slot-s5-checks.json`,
and every one is true.

**Refill stalls.** A refill pauses the live rows for its prefill. The measured pauses
match the prefill times from the refill evidence:

| Refill | Prefill (s) | Pause seen by the long rows (s) |
| --- | --- | --- |
| `l2` beside `l1` | 1.87 | 1.92 |
| `s3` | 5.42 | 5.48 |
| `s5` | 1.33 | 1.38 |

All prefills used the masked traces.

### Finding: output depends on trace width, and the candidate fixes the width

The two paths do not produce the same greedy tokens. With the width-4 slot trace (1 to
4 active rows), l1, l2 and s3 first differ from the group coordinator's solo run
(width 1) at tokens 46, 32 and 29.

The group coordinator also changes its own output with load. In both passes its
width-6 group gave s3 84 tokens, against 80 alone, first diverging at token 29.

The candidate always decodes at width 4, so a request's tokens did not depend on what
else was running. This is a numerical property of the width-specific programs, not a
scheduler defect. It is recorded because it changes which reference is exact.

## Not claimed

- No simulated evidence: the simulator path is blocked as recorded above.
- One host and one run of the screen. This is correctness and safety coverage; the
  latency comparison belongs to s6 (`s6-queue-inclusive-admission.md`).
- Refills that compile a program were not provoked on hardware. That path, the unsafe
  refill draining its group, is covered device-free only.
