## ADDED Requirements

### Requirement: The serialized greedy route runs speculative MTP rounds behind a default-off gate

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

The repository MAY provide a gated speculative decode path for the serialized greedy serving route. The gate MUST default off, and off MUST leave the serving loop, startup, and admission behavior unchanged. The path MUST load the bf16 MTP head only when enabled, MUST capture its draft, verification, and decode traces in a hygiene order that never captures while another trace is resident, and MUST replay them without per-round recapture. Each round MUST draft exactly `draft_token_count` tokens, verify them with the target backbone teacher-forced, and commit the longest accepted prefix plus the verification bonus token. For greedy requests the committed stream MUST equal sequential greedy output for every acceptance pattern. The path MUST restore rejected KV and recurrent state before the next round and MUST disengage specululation for the rest of a request when measured round economics fall below the plain step rate for a configured number of consecutive windows. A forced trace recapture MUST disengage speculation for the request and record the reason. The path MUST NOT change sampled, batch, or deterministic-route behavior, and it MUST NOT admit reduced precision or admit itself into production without matched physical evidence and owner review of the operating point.

### Requirement: The round schedule and commit rule are owned by a pure device-free core

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]

The repository MUST own the round schedule and the exact greedy commit rule in a pure, device-free core. The scheduler MUST select a speculative round only when the request has more than `draft_token_count` tokens remaining and MUST select plain steps otherwise. The commit rule MUST accept the longest draft prefix whose tokens equal the verification argmaxes and MUST append the verification argmax at the rejection position, or the final argmax on full acceptance. The evaluator MUST reject negative or non-integer counters, draft widths outside the closed range 2..6, length-mismatched drafts and verification argmaxes, out-of-range token identifiers, and unknown round modes.

### Requirement: A valid device-free probe demonstrates scheduling, commit, and bound behavior

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]

A valid device-free probe MUST demonstrate: speculative scheduling strictly inside the generation tail with plain steps at the boundary; committed streams equal to a reference sequential stream for full acceptance, rejection at the first draft, rejection at an interior position, and rejection at the last draft; and round plans that keep the generated count within the request bound.

### Requirement: The probe rejects malformed round inputs with specific diagnostics

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]

The probe MUST reject mismatched draft and verification lengths, draft widths outside the closed range, negative generated counters, remaining-token counts that underflow the request bound, and non-integer token identifiers, each with a specific diagnostic.
