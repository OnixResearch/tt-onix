## ADDED Requirements

### Requirement: Greedy batches decode with device-fed tokens

r[tenstorrent.qwen38_p150x2.greedy_stream]

The greedy batch trace MUST combine the shards' winners on the device and write the global tokens into the next step's token input. The combine MUST choose the highest maximum and keep the lowest device on equal maxima, as the host combine did. A greedy batch MUST decode with at most one step queued ahead of the host's readback. It MUST stage only positions and RoPE after the first step. Its tokens MUST equal step-by-step decode. Sampled batches MUST keep host sampling per step.

#### Scenario: A greedy batch streams its steps

r[tenstorrent.qwen38_p150x2.greedy_stream.valid]

- **GIVEN** a prepared greedy trace at any width
- **WHEN** the batch decodes through the stream
- **THEN** every step's tokens MUST equal the step-by-step tokens
- **AND** the readback MUST be one transfer from one replica.

#### Scenario: A reader meets the other mode's tail

r[tenstorrent.qwen38_p150x2.greedy_stream.invalid]

- **GIVEN** a sampled candidate tail read as greedy, or a greedy tail read as sampled or streamed
- **WHEN** the adapter reads it
- **THEN** it MUST raise a batch adapter error instead of returning tokens.
