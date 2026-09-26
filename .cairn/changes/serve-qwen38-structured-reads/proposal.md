## Why

vLLM PR 57250 (merged as `1b3b88e`) adds structured reads to DiffusionGemma. The
answer template is seeded into the canvas with noise at each answer slot. One
denoise step then gives a distribution over each slot's labels. An example server
turns a question schema into those reads behind Jev's `POST /v1/systemone`
contract. The reads run on CUDA only. DiffusionGemma has no TT-NN port here.
The vLLM Tenstorrent plugin targets vLLM 0.26, so the PR cannot run on the P150x2
cards as it stands.

The owner chose to serve the same contract from the validated Qwen3.8-27B P150x2
service. Qwen3.8 is autoregressive, not a diffusion model.

## What Changes

- Add `POST /v1/systemone` to the batch-worker service. It covers the PR's Jev
  body, answer shapes, and schema extensions: `samples`, `auto_max`,
  `auto_threshold`, `steps`, `think`, `ask`, `chunk_rows`, `chunk_prompt`,
  `sequential`, `depends_on`, `ask_if`, and `alone`.
- A structured read is one prefill of the prompt plus the answer template. The
  masked prefill traces' logit tile widens from one to 32 rows. Row 0 stays the
  generation logit. The other rows carry the rows before each answer slot. The
  selector, final norm, LM head, and gather still cost one tile row, and greedy
  generation stays token-identical.
- Add batched read traces: four sequences share one replay of a 128- or
  256-token bucket. The worker merges queued read jobs into these replays.
- Reuse a long prompt's first 1,024-token chunk. Single reads write KV through
  the last page-table row. A read saves the carried GDN state after a full first
  chunk, and a later read with the same first chunk replays only its tail.
- Adapt the read policy to an autoregressive model:
  - reads are memoized per decision;
  - the first draw fills slots with a `?` placeholder, not random tokens, and
    by default refines to its fixed point, the greedy answer chain;
  - later draws read once, under earlier answers sampled from the first draw;
  - the auto policy ignores the first slot, which no draw can change.

## Impact

- **Files**: `pkgs/qwen36/structured-read-rows.patch` (new; `model.py`,
  `gdn/tp.py`); `pkgs/qwen36/default.nix`;
  `pkgs/qwen36/service/qwen36_service/{structured_read_core,batch_decode_adapter,batch_service,server}.py`;
  `pkgs/qwen36/check.py`; and `docs/qwen38-structured-reads.md`.
- **Behavior**:
  - A new route. Existing routes keep their responses.
  - Startup captures two more traces and allocates about 87 MiB per device for
    the saved prefix state.
  - Eager (MTP) mode answers the route with 503.
  - The execution identity changes because the patch set changed.
