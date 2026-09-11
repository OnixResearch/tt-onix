## Context

Phase 1 of this change inspected the pinned runtime and falsified the original
premise. The fused GDN decode op is already vendored, built, wired, and admitted
for sampled mode and B=4. The evidence is in `evidence/upstream-authority.md`.

The surviving problem is drawn from `.cairn/archive/2026-08-27-fuse-qwen38-gdn-decode-state-update/evidence/parity-root-cause.md`.
Its approach registry closed eight families:

| Family | State |
| --- | --- |
| Arithmetic equivalence | blocked - needs staged kernels preserving each TTNN precision boundary |
| Mixed unpack precision | blocked - direct mode in the mixed kernel corrupted state |
| Decay arithmetic | exhausted - no live repair in this kernel shape |
| Layer subset | falsified - no weaker subset can establish semantic exactness |
| Margin sensitivity | validated - drift crosses late argmax margins |
| TP2 mapping | validated |
| Trace lifecycle | validated |
| Identity contamination | falsified |

An independent effort reached a compatible conclusion: a separate Tenstorrent
branch reports `fused_recurrent_gated_delta_rule` at `0.345` ms per layer
against `0.541` for the composite. That is a different kernel from this
repository's `gated_delta_decode`, and this repository measured neither.

## Decisions

### Decision: Keep this change and re-scope it to the precision repair

**Choice:** Do not delete the change. Convert it from "vendor the op" to "make
the existing fused op precision-equivalent to the composite".

**Rationale:** The change's phase 1 exists to test its premise. The premise
failed and the falsification is recorded. The underlying goal, faster greedy
decode, is still unmet and still has exactly one named route.

### Decision: Reproduce the composite's precision boundaries, do not loosen parity

**Choice:** Stage the recurrence so each boundary the composite performs is
reproduced. Keep exact token parity as the gate. Do not move to a
distributional bound for greedy serving.

**Rationale:** A distributional bound already exists for sampled serving and
already admitted this candidate. Greedy and deterministic serving keep exact
parity by explicit policy. Weakening that would trade a user-visible guarantee
for speed, which the repository has already declined to do.

### Decision: The staging core is pure and device-free

**Choice:** Own the stage order, the boundary list, and the closed validation in
a pure core. Reject an unknown stage, a missing boundary, a duplicated boundary,
and an out-of-order stage list.

**Rationale:** The staging order is the correctness argument. It must be
testable without a device and reviewable as a list.

### Decision: Treat this as a research risk with a named exit

**Choice:** The archived search marked this family blocked, not promising. If
the staged design cannot reproduce parity within its budget, the change must
record an exhausted result and keep the fused route restricted to sampled
serving.

**Rationale:** The archived search already spent three runs and three
discriminating follow-ups on this. A second open-ended search is not warranted.
A bounded attempt with an explicit exit is.

## Risks / Trade-offs

- The composite's TF32 unpack behavior may not be reproducible in a fused
  kernel at all. Exhaustion is a likely and acceptable outcome.
- Staging adds precision-boundary work that can erase the speed gain. The
  full-step ratio bound decides this.
- This is the lowest-confidence change in the set. It must not block the
  mechanism and admission changes that carry the measurable near-term gains.
