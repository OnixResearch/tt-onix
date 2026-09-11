## ADDED Requirements

### Requirement: MTP target verification requires bounded hybrid-state evidence

r[tenstorrent.qwen38_p150x2.mtp_verification_probe] The repository MAY provide an operator-enabled teacher-forced target-verification probe for the pinned P150x2 runtime. The probe MUST remain diagnostic-only. It MUST compare widths 1, 2, and 4 with matched sequential traced decode after the same prompt state. It MUST use at least five complete samples per width. Every sample MUST preserve target-token parity and bounded GDN recurrent and convolution state equivalence. The deterministic evaluator MUST reject missing identities, unstable samples, invalid durations, wrong widths, token mismatch, or state mismatch. A valid target-only result MUST NOT select or admit MTP without a measured drafter, local acceptance, rollback correctness, and matched end-to-end evidence.

#### Scenario: Teacher-forced target verification is complete

- GIVEN matched P150x2 sequential and teacher-forced samples for widths 1, 2, and 4
- WHEN all identity, timing, target-token, and hybrid-state checks pass
- THEN the receipt MAY report target-verification amortization for the pinned runtime
- AND it MUST state that MTP remains unselected and unadmitted.

#### Scenario: Verification evidence is incomplete

- GIVEN a missing width, too few samples, identity drift, invalid duration, target-token mismatch, or GDN state mismatch
- WHEN the deterministic evaluator runs
- THEN it MUST reject target-verification feasibility
- AND production MUST remain eager.
