## ADDED Requirements

### Requirement: Prewarmed persistent recovery control preserves matched repeated evidence

r[tenstorrent.qwen38_p150x2.mtp_recovery_prewarmed_control]

The diagnostic P150x2 command MUST prewarm every configured prompt through nonpersistent target validation before persistent recovery warmups and measured roots. It MUST reuse the accepted persistent recovery profile, prompt set, route schedule, trace-capture count, exact state checks, stability gate, and recovery-ratio gate. It MUST atomically preserve ordered progress before and after each prewarm, warmup, measured root, evaluation, cleanup, and receipt boundary. Program-cache observations MUST be nonnegative and monotonic before measurement. The program-cache count MUST NOT change after measurement starts. Every warmup and measured target-validation signature MUST match its prompt prewarm signature. Missing progress, order drift, route drift, signature drift, program-cache drift, incomplete cleanup, or existing evaluator failure MUST fail closed. The command MUST remain admission-ineligible and MUST NOT change eager production behavior.

#### Scenario: Prewarmed repeated control completes

- **GIVEN** the pinned package, runtime, checkpoint, five prompts, balanced recovery schedule, and physical P150x2 devices
- **WHEN** the command prewarms each prompt, completes persistent warmups, and completes every measured root without program-cache growth
- **THEN** it SHALL write a diagnostic receipt with exact signatures, stable cache observations, complete recovery evidence, and the existing fail-closed decision

#### Scenario: Repeated control evidence is incomplete or changes

- **GIVEN** a missing boundary, order drift, route drift, signature mismatch, measured program-cache change, state mismatch, unstable root, or incomplete cleanup
- **WHEN** the command evaluates or preserves the control
- **THEN** it MUST reject the control without selecting MTP, traced target decode, or production behavior
