## ADDED Requirements

### Requirement: Masked prefill traces reproduce eager prefill for short prompts

r[tenstorrent.qwen38_p150x2.masked_prefill_traces]

The repository MAY serve P150x2 prompts by replaying B=1 masked prefill traces, one per bucket. A prompt longer than the largest bucket MUST continue from the previous replay's state, as eager chunked prefill does. The prompt length MUST enter the trace as device data: the GDN conv-window selector and the scan mask. The replayed output and installed state MUST equal eager masked-bucket prefill for the same page-table rows. The traces MUST be captured after every program the batch worker can run. After a later first-time compile they MUST be released and never replayed. No device buffer allocated after a capture MAY stay alive across a replay. The batch worker MUST NOT select a decode width that has no one-user-per-core grid rectangle.

#### Scenario: Traced prefill serves the batch exactly

r[tenstorrent.qwen38_p150x2.masked_prefill_traces.valid]

- **GIVEN** a service started with traced prefill
- **WHEN** a batch of 1 to 16 users prefills
- **THEN** each user MUST replay the smallest captured bucket that holds its prompt
- **AND** first-token logits, installed state, and generated text MUST equal eager prefill
- **AND** users no trace plan holds MUST prefill eagerly before any replay of the same batch, and batches that reuse a parked decode trace MUST prefill eagerly.

#### Scenario: A post-capture compile or an unshardable width

r[tenstorrent.qwen38_p150x2.masked_prefill_traces.invalid]

- **GIVEN** a program compiled after the capture, or 13 compatible queued requests on an 11x10 grid
- **WHEN** the next batch prefills or the worker takes a batch
- **THEN** the traces MUST be released before any replay, and prefill MUST run eagerly
- **AND** the worker MUST return the newest request to the queue and take 12.
