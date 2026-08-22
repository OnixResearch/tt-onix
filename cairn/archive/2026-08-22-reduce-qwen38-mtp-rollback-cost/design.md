## Context

The diagnostic drafter writes two MTP decode cache rows while producing three drafts. Its rejection path currently reruns MTP prefill and each accepted draft before it can reconstruct the committed prefix. The latest physical control kept exact outputs but measured MTP cache re-prefill on the zero-accept prompt. Full-cache copies would move sequence-capacity data to recover at most two overwritten rows.

## Decisions

### Decision: Snapshot only the rows that draft decode can overwrite

**Choice:** Clone the two bounded MTP K/V rows before draft decode. Keep prefix feedback in preallocated device buffers. After verification, restore only rejected rows and then execute the committed target token.

**Rationale:** The fixed three-draft route writes only two decode rows. Bounded row snapshots avoid maximum-sequence cache copies and preserve explicit cache authority.

### Decision: Compare complete next-round MTP state

**Choice:** Compare the next draft token, feedback tensor, and committed MTP K/V prefix against cache re-prefill. Use the natural accepted prefix for each fixed prompt.

**Rationale:** Token parity alone cannot prove a coherent MTP continuation state.

### Decision: Select only a diagnostic rollback mechanism

**Choice:** A stable exact gain may select snapshot rollback for later MTP work. It cannot admit MTP, traced target decode, or production behavior.

**Rationale:** Draft and target timing remain unstable, and sustained multi-round evidence does not exist.

## Risks / Trade-offs

- The five prompts naturally cover accepted widths zero and three, not widths one and two.
- Restoring cache rows needs package-matched `paged_update_cache` input layout.
- Device allocations during an active trace are unsafe. Snapshot and feedback buffers must exist before capture.
- The mechanism can improve rollback while end-to-end MTP remains unadmitted.
