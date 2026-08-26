## Context

The matched receipt is valid for the standalone Python 3.13 package. Onix evaluates the same source with Python 3.14 and produces another package identity. The deployed smoke confirms the new package works, but one B=1/B=4 pair cannot replace the required matched roots.

## Decisions

### Decision: Withhold deployment admission

**Choice:** Keep the receipt and its passing metrics, but set the production batch profile to unadmitted.

**Rationale:** Performance and correctness evidence do not transfer across package identities under the accepted contract.

### Decision: Require a stable execution identity before another attempt

**Choice:** A later change must either run matched roots on the deployed package or split the execution artifact from mutable admission metadata.

**Rationale:** Rebuilding a monolithic package after binding a receipt changes its store identity. A stable execution artifact removes this promotion cycle.

## Risks / Trade-offs

- Production remains operational in explicit candidate mode.
- The repository makes no formal B=4 production admission claim.
- The passing candidate receipt remains useful for future stable-artifact design.
