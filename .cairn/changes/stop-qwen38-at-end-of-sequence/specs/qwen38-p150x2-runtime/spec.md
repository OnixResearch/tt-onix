## ADDED Requirements

### Requirement: Generation ends at the model's end of sequence

r[tenstorrent.qwen38_p150x2.end_of_sequence]

The service MUST read the model's end-of-sequence ids from its generation configuration and tokenizer. Outside admission measurements, generation MUST stop once every user of a batch has produced one of them. Each response MUST contain only the tokens before the user's first end-of-sequence id, and MUST report `finish_reason` `stop` in that case or `length` when `max_tokens` was reached first. Admission measurements MUST keep generating exactly `max_tokens` tokens.

#### Scenario: A chat answer ends before the bound

r[tenstorrent.qwen38_p150x2.end_of_sequence.valid]

- **GIVEN** a chat request whose answer ends before `max_tokens`
- **WHEN** the service generates it, alone or in a batch
- **THEN** decoding MUST stop once every user in the batch has ended
- **AND** the response text MUST equal the fixed-length output up to the end-of-sequence token, with `finish_reason` `stop`.

#### Scenario: No stop id or an admission measurement

r[tenstorrent.qwen38_p150x2.end_of_sequence.invalid]

- **GIVEN** generation that never produces a stop id, a stop set with no source, or an admission measurement
- **WHEN** the response is built
- **THEN** it MUST contain every generated token and report `finish_reason` `length`.
