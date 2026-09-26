## ADDED Requirements

### Requirement: Structured reads serve the Jev decision contract

r[tenstorrent.qwen38_p150x2.structured_reads]

The batch-worker service MUST answer `POST /v1/systemone` with the Jev body and answer shapes of vLLM PR 57250's example server. It MUST support that server's schema extensions except images. Each answer slot's distribution MUST come from the next-token logits at the row before the slot, in one prefill of the prompt and the answer template. The reads MUST run on the batch worker between generation batches. They MUST NOT change generation output.

#### Scenario: A decision reads its slots from one prefill

r[tenstorrent.qwen38_p150x2.structured_reads.valid]

- **GIVEN** a valid Jev body whose labels are single tokens in the answer template
- **WHEN** the service decides it
- **THEN** each slot's logits MUST equal those of a prefill that ends at the row before the slot
- **AND** a batched read MUST equal the same sequence read alone
- **AND** greedy generation MUST stay token-identical to the service without structured reads.

#### Scenario: An unreadable request is refused

r[tenstorrent.qwen38_p150x2.structured_reads.invalid]

- **GIVEN** a body with images, an unknown question type, a label that is not one template token, a dependency cycle, a thought over the generation bound, or a read longer than the admitted sequence
- **WHEN** the service parses it
- **THEN** it MUST answer 422 before any device work
- **AND** invalid JSON MUST answer 400, and a service without the batch worker MUST answer 503.

### Requirement: Structured-read draws follow the autoregressive model

r[tenstorrent.qwen38_p150x2.structured_read_draws]

The first draw MUST fill each answer slot with a placeholder token. It MUST use noise instead where no single placeholder token exists, or when the request sets `first_fill` to `noise`. By default, the first draw MUST then refine to a fixed point, where each slot holds the previous read's most likely label. Each later draw MUST read once, with each slot filled by a label sampled from the refined first draw. Identical prefills within a decision MUST reach the device once. The auto policy MUST add draws only when a slot after the first passes the entropy threshold.

#### Scenario: Draws that cannot differ cost one read

r[tenstorrent.qwen38_p150x2.structured_read_draws.valid]

- **GIVEN** a one-question decision with any sample count or step count
- **WHEN** it is decided
- **THEN** exactly one device read MUST run.

### Requirement: Structured reads reuse a long shared prefix

r[tenstorrent.qwen38_p150x2.structured_read_prefix]

A traced single-sequence read whose plan starts with a full 1,024-token chunk MUST save the carried GDN state after that chunk. A later read through the same page-table row MAY skip that chunk only when its first 1,024 tokens are identical and none of its read positions lie inside them. It MUST then restore the saved state and replay only the remaining segments. Its logits MUST equal an uncached read of the same tokens. Before any prefill writes a KV block of the saved prefix, the saved prefix MUST be dropped.

#### Scenario: A reused prefix reads the same logits

r[tenstorrent.qwen38_p150x2.structured_read_prefix.valid]

- **GIVEN** a saved prefix, followed by a generation below full width or by batched reads
- **WHEN** a read with the same first 1,024 tokens runs
- **THEN** it MUST replay only the segments after the prefix
- **AND** its logits MUST equal an uncached read bit for bit.

#### Scenario: An overwritten prefix is not reused

r[tenstorrent.qwen38_p150x2.structured_read_prefix.invalid]

- **GIVEN** a saved prefix
- **WHEN** a full-width generation prefill, a read from position 0 through the same row, or a prefill through a device page table runs
- **THEN** the saved prefix MUST be dropped
- **AND** the next read MUST replay from position 0.
