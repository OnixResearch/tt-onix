## Context

vLLM PR 57250 (merged as `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`) reads
DiffusionGemma's canvas. The answer template is seeded, noise fills every answer
slot, and one denoise step returns a distribution per slot. The PR's example server
turns a Jev question schema into those reads. DiffusionGemma has no TT-NN port in
this repository, and the Tenstorrent vLLM plugin targets vLLM 0.26. The owner chose
to serve the contract from the validated Qwen3.8-27B P150x2 service, which is
autoregressive.

The service's masked prefill traces (`trace-qwen38-masked-prefill`) already compute
the whole bucket's hidden states. A one-hot `[1, 1, 1, bucket]` selector picks the
last row for the LM head. That logit and the prompt's GDN state are installed into a
decode slot.

## Decisions

### Decision: Read slots from spare rows of the prefill's logit tile

**Choice:** The selector becomes `[1, 1, 32, bucket]`. Row 0 keeps the generation
logit, and rows 1–31 select the rows before each answer slot. The rows are staged
as device data, like the prompt length.

**Rationale:**

- A 32-row selector is one tile row, so the selector matmul, final norm, LM head,
  and gather cost what the single row did.
- A read is one prefill of the prompt plus the template. It needs no decode, and
  no program compiles at request time.
- Greedy generation stays token-identical.

### Decision: Batch four short reads per replay

**Choice:**

- Two more traces run four sequences through one forward, for the 128- and
  256-token buckets.
- GDN, norm, and MLP layers run over every row at once. Full-attention layers run
  per user through page-table rows fixed at capture.
- The GDN path writes no state, so it needs no masks. Right padding cannot reach
  an earlier row of a causal conv or scan.
- The worker merges queued read jobs, and a pure planner groups them shortest
  first. A group reads batched only above a measured minimum per bucket.

**Rationale:**

- The seq GDN kernel maps one (user, head) row per core. At 24 local value heads,
  four users fit the 110-core grid.
- A batched replay costs the same for any fill: 0.42 s at 128 tokens and 0.68 s at
  256.
- Every body compiles before the first capture. Capturing the batched body after
  the single-sequence traces hung the device.

### Decision: A placeholder first draw that refines to its fixed point

**Choice:**

- The first draw fills each slot with the `?` token and refines by default (up to 8
  steps). Each step refills the slots with the previous argmax labels, until they
  stop changing.
- Later draws read once, under earlier answers sampled from the refined first
  draw.
- `first_fill: noise` keeps the PR's random first fill.

**Rationale:**

- An autoregressive model reads a later slot conditioned on the earlier ones.
  Random tokens there are out of distribution: a 40-question schema scored 29/40
  with one noisy read.
- The fixed point is the greedy answer chain, which scored 40/40.
- Later draws that were refined would each return to the same fixed point, so they
  read once to measure how the plausible earlier answers move a later one.

### Decision: Memoize prefills and gate the auto policy past the first slot

**Choice:** A decision computes each distinct prefill once. The auto policy adds
draws only when a slot after the first passes the entropy threshold.

**Rationale:** The first slot's row sees no earlier fill, so no draw can change
it. Together these cut device reads from 87 to 66 on the 42-decision subset, with
unchanged accuracy.

### Decision: Reuse a long prompt's first chunk

**Choice:**

- A single read writes its KV through the last page-table row.
- A read whose plan starts with a full 1,024-token chunk copies the carried GDN
  state after that chunk into persistent buffers.
- A later read with the same first chunk, and no read position inside it, restores
  that state and replays only its tail.
- Every prefill entry point drops the saved prefix before it writes one of the
  prefix's KV blocks.

**Rationale:**

- Refinement steps and later draws of a long state share the whole prompt, and
  the first chunk dominates their cost.
- The buffers and copy programs exist before any trace capture, so no replay can
  overwrite them and no request compiles a program.
- The last row is written only by full-width generation. Batched reads use the
  first rows.
- Prompts under 1,024 tokens get no reuse. Splitting their single replay to
  create a boundary costs a second replay floor of about 0.22 s, more than a
  decision's later reads would save.

### Decision: Reads run on the batch worker

**Choice:** Read jobs queue on the batch worker between generation batches.
Eager (MTP) mode answers the route with 503.

**Rationale:**

- The worker owns the device and the traces.
- A read releases any parked decode trace first, because a replay overwrites
  device buffers allocated after its capture.
