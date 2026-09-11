# Qwen3.8 MTP drafter search frame

## Goal

Determine whether the pinned one-layer MTP drafter can produce a correct and useful speculative loop with the validated P150x2 target verifier.

## Completion evidence

A valid result must:

- load the exact `mtp.*` tensors from the pinned checkpoint without changing target weights;
- implement the checkpoint equation: normalize next-token embedding and target hidden, concatenate, project, run one full-attention decoder layer, and final-normalize;
- share the target embedding and LM head because the checkpoint stores neither dedicated tensor;
- initialize and advance the MTP layer cache with target hidden states and shifted token inputs;
- measure three drafts, greedy accepted-prefix length, target verification, and emitted-token latency across at least five fixed prompts;
- preserve target greedy output across accepted and rejected drafts;
- restore target GDN prefix state and keep target and drafter KV positions coherent after rejection;
- stay within the measured width-four target budget or return an exact blocker;
- remain diagnostic-only and leave eager production unchanged.

## False completion cases

- Treating Qwen MTP as a Gemma-style Q-only assistant that reads target KV.
- Running only the last hidden state without initializing the MTP layer cache.
- Reporting checkpoint or external acceptance as local evidence.
- Comparing drafts with misaligned target logits.
- Ignoring target GDN rollback because paged KV can be overwritten.
- Counting trace capture, model load, or prompt prefill as steady draft latency.
- Selecting MTP from target-only verification or one prompt.

## Audit risks

- The packaged Transformers version ignores `mtp.*`, so framework loading is not an oracle.
- MTP input token IDs are shifted relative to target hidden rows.
- The target hidden contract can be pre-norm or post-final-norm.
- MTP attention has its own KV cache and needs prompt or accepted-token re-prefill.
- Target GDN state cannot use implicit paged-KV overwrite rollback.
- Shared target embedding and LM head have tensor-parallel layouts.
- BF16 MTP weights add memory pressure beside the target model.
- Near-tie logits can change greedy acceptance without changing target correctness.

## Search budget

- Primary facts: checkpoint config, safetensor index and shapes, pinned target source, and physical target receipt.
- Implementation references: vLLM and SGLang Qwen3.5 MTP source. Treat them as semantic references, not package-matched authority.
- Mechanism families: at most five.
- Physical budget: one bounded loader/parity smoke, then one five-prompt acceptance and rollback pilot if the smoke passes.
- Allowed outcomes: validated, blocked, exhausted, or user-decision-required.

## Approach registry

| Family | Mechanism | Claim | State | Exact blocker or next check |
| --- | --- | --- | --- | --- |
| Qwen full-layer MTP | Shared target embedding/head, FC fusion, one full-attention decoder layer with its own cache | Matches checkpoint and vLLM/SGLang semantics | Rejected | Aggregate speedup `0.9687222124830936`; draft spread `2.059840539341421`. |
| Gemma assistant reuse | Q-only layer cross-attends into target KV at one fixed position | Reuse existing TT Gemma drafter flow | Falsified | Qwen checkpoint stores Q, K, V, O and advances draft positions. |
| Last-hidden shortcut | Run the MTP layer only on the final target hidden without cache initialization | Smallest implementation | Falsified | Omits prompt history required by the MTP full-attention layer. |
| Transformers oracle | Load MTP through packaged Hugging Face model code | Supplies CPU parity reference | Blocked | Packaged Transformers explicitly ignores `^mtp.*`. |
| External runtime reuse | Run vLLM or SGLang MTP directly | Avoids a local implementation | Blocked | Their CUDA runtime and cache machinery do not match the pinned TTNN P150x2 execution base. |
