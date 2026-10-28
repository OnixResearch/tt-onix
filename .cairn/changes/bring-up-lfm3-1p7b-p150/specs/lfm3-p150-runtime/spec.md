## ADDED Requirements

### Requirement: The runner executes the checkpoint's tri-hybrid math on one device

r[tenstorrent.lfm3.p150.gpu]

The package MUST provide a device command that loads a local Hugging Face
checkpoint directory and runs prefill and greedy decode of LFM3-1.7B on a single
Blackhole device. It MUST implement the checkpoint's layer operators as
declared by `config.json` and `modeling_lfm3.py`: gated depthwise causal
convolutions with kernel 7 and a six-row decode window, sliding-window attention
with window 256 and full RoPE, global attention with partial RoPE (0.5) over the
high-frequency half, per-head sigmoid output gating, QK RMSNorm, and sandwich
RMSNorm around both residual branches. It MUST NOT add simulator-specific
runtime branches.

#### Scenario: A prompt produces a token stream

r[tenstorrent.lfm3.p150.gpu.generate]

- **GIVEN** the checkpoint directory with `config.json`, `model.safetensors`, and
  `tokenizer.json`
- **WHEN** the device command runs with a chat prompt and a token count
- **THEN** it MUST open one device, prefill the prompt, decode the requested
  tokens, and print the prompt token count, prefill and decode rates, the token
  ids, and the decoded text

#### Scenario: The checkpoint is supplied at runtime

r[tenstorrent.lfm3.p150.gpu.checkpoint]

- **GIVEN** a checkpoint path from `--checkpoint-dir` or `LFM3_CHECKPOINT_DIR`
- **WHEN** the command loads it
- **THEN** it MUST read the configuration from that directory and MUST NOT
  require a network fetch

### Requirement: The float32 mode agrees with the CPU fp32 oracle

r[tenstorrent.lfm3.accuracy.oracle]

With `--dtype float32`, the device command MUST produce the same greedy token
stream as the CPU fp32 oracle for the reference prompt and the long-context
prompt. The device gate MUST report, for every captured hidden state, the
prefill logits, and every decode step, the maximum absolute error, the mean
absolute error, and the Pearson correlation against the oracle dump.

#### Scenario: Short reference prompt

r[tenstorrent.lfm3.accuracy.oracle.short]

- **GIVEN** the 17-token chat prompt and an oracle dump for 32 generated tokens
- **WHEN** the gate replays it on the device in float32 mode
- **THEN** all 32 greedy tokens MUST match
- **AND** the gate MUST print a deviation row for the embedding, all 24 layers,
  the final norm, the prefill logits, and each decode step

#### Scenario: Long-context prompt

r[tenstorrent.lfm3.accuracy.oracle.long]

- **GIVEN** the 430-token prompt that crosses the 256-token sliding window and an
  oracle dump for 32 generated tokens
- **WHEN** the gate replays it on the device in float32 mode
- **THEN** all 32 greedy tokens MUST match
- **AND** the gate MUST report the drift of the last hidden state rather than
  hide it

### Requirement: Decode attention keeps a fast default and a wide option

r[tenstorrent.lfm3.decode.attention]

The device command MUST default to the packaged bf16 SDPA decode kernel over a
bf16 KV cache and MUST offer a float32 decode path that widens the cache and
runs attention over the used rows with a float32 softmax. The documented
evidence MUST record the token agreement and the per-token cost of both paths
for the long-context prompt.

#### Scenario: Default decode path

r[tenstorrent.lfm3.decode.attention.default]

- **GIVEN** the long-context prompt and float32 mode
- **WHEN** the command runs with the default decode attention
- **THEN** the greedy stream MUST match the oracle
- **AND** the documented per-token rate MUST be the fast-kernel rate

#### Scenario: Wide decode path

r[tenstorrent.lfm3.decode.attention.wide]

- **GIVEN** the same prompt
- **WHEN** the command runs with `--decode-attention fp32`
- **THEN** the greedy stream MUST match the oracle
- **AND** the documented per-token rate MUST disclose the wide-path cost

### Requirement: The precision floor is measured and disclosed

r[tenstorrent.lfm3.precision.disclosure]

The package documentation MUST state the measured matmul fidelity of the
packaged runtime, MUST NOT claim per-element equality with a CPU fp32 reference,
and MUST mark the bfloat16 mode as a throughput mode rather than an accuracy
gate.

#### Scenario: Precision probe is recorded

r[tenstorrent.lfm3.precision.disclosure.matmul]

- **GIVEN** the packaged TT-NN runtime
- **WHEN** the documentation reports matmul fidelity
- **THEN** it MUST cite measured maximum and mean absolute error for every
  fidelity and dtype configuration it compares

#### Scenario: Throughput mode is labelled

r[tenstorrent.lfm3.precision.disclosure.bfloat16]

- **GIVEN** the bfloat16 mode
- **WHEN** the documentation reports its agreement
- **THEN** it MUST state the divergence index and MUST NOT present the mode as
  an acceptance gate
