## ADDED Requirements

### Requirement: Prefill state staging stays device-resident without parked traces

r[tenstorrent.qwen38_p150x2.prefill_device_staging]

The service's per-user prefill path MAY stage GDN state through device-resident clones instead of the per-user host snapshot round trip. Device staging MUST be gated on the absence of parked bucket and chunk prefill traces and MUST fall back to the host snapshot path when either trace is resident. Slot writes MUST go through the existing write_slot ownership path with fresh clone allocations as sources. First-token evidence MUST fire before slot writes with unchanged TTFT meaning. Decode RoPE inputs MUST use cached per-position tables bitwise identical to the per-step trig they replace. A production throughput or TTFT claim for the staged path MUST use matched physical roots under the existing batch admission discipline on the changed execution identity.

#### Scenario: Staged prefill serves a batch

- GIVEN no parked prefill trace and a released decode trace
- WHEN a multi-user prefill runs
- THEN per-user GDN state MUST be cloned on device and written through write_slot
- AND the matched B=1 versus B=4 control MUST preserve exact token parity.

#### Scenario: A parked trace forces the host path

- GIVEN a resident bucket or chunk prefill trace
- WHEN prefill stages user state
- THEN the host snapshot round trip MUST be used
- AND no clone may alias trace intermediates.

#### Scenario: Admission evidence is required

- GIVEN the changed execution identity
- WHEN a production claim is made
- THEN a fresh matched admission receipt MUST bind the identity
- AND the prior receipt MUST NOT be transferred.
