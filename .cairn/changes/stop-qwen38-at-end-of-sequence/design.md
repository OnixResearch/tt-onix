## Context

`BatchGenerationCoordinator._run_group` looped
`while len(generated[0]) < max_tokens`. `build_completion_response` hard-coded
`finish_reason: "length"`. Qwen3.8's `generation_config.json` lists
`eos_token_id: [248046, 248044]` (`<|im_end|>`, `<|endoftext|>`).

## Decisions

### Decision: Stop the batch when every user has stopped

**Choice:** The batch keeps a `stopped` flag per row. It leaves the decode loop
when every row has produced a stop id or the bound is reached. Rows that
stopped early keep decoding with the batch, and their extra tokens are dropped.

**Rationale:** A batch runs one trace per step at a fixed width. Removing a
finished row would need a width change and a trace recapture per step.

### Decision: Truncate before the first stop id

**Choice:** `finish_generation(tokens, stop_ids)` returns the tokens before the
first stop id, together with `"stop"`, or all tokens with `"length"`. Every
generation path shapes its response through it.

**Rationale:** This is the OpenAI contract. The tokens kept are exactly the
tokens the fixed-length loop produced up to that point, so the answer text is
unchanged.

### Decision: Admission modes keep the fixed length

**Choice:** With `--batch-admission` or `--trace-reuse-admission`, the stop set
is empty.

**Rationale:** Those receipts compare decode rates over a known number of steps.
