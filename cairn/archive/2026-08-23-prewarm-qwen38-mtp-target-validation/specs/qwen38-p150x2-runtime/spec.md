## ADDED Requirements

### Requirement: Same-process target prewarm isolates post-recovery compilation

r[tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]

The diagnostic P150x2 command MUST prewarm the same prompt through the existing nonpersistent target-validation path before persistent MTP recovery. It MUST preserve ordered prewarm and measured boundaries with nonnegative program-cache observations. It MUST reject a prewarm whose target-validation signature differs from the measured signature. The prewarm and its timing MUST remain admission-ineligible. The command MUST NOT change eager production behavior.

#### Scenario: Exact prewarm completes before the persistent route

GIVEN the pinned P150x2 package, runtime, checkpoint, prompt, and recovery route
WHEN the operator runs the target-prewarm diagnostic command
THEN the command preserves prewarm progress, runs persistent recovery in the same process, and compares exact target-validation signatures before receipt creation

#### Scenario: The prewarm evidence is incomplete or mismatched

GIVEN a missing boundary, a negative or regressing program-cache count, or a changed target-validation signature
WHEN the command evaluates the diagnostic result
THEN the command fails closed without selecting MTP, traced target decode, or production behavior
