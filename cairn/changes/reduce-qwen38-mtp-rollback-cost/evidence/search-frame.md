## Goal and boundary

The goal is exact MTP continuation after target rejection without rerunning the 128-token MTP prefill. A lower local duration is insufficient. The recovered next draft, feedback, and committed K/V prefix must match re-prefill exactly in stable matched P150x2 roots.

## Source evidence

- Package-matched `TPAttention.reset_state()` allocates internal K/V tensors with shape `[B, 1, max_seq_len, head_dim]`.
- Package-matched decode uses `ttnn.experimental.paged_update_cache` with an explicit position tensor.
- Three MTP drafts require two draft-decode steps and therefore overwrite at most two post-prefill rows.
- The traced route already proves same-input capture writes are exact and exposes stable prefix feedback outputs.
- The natural prompt set accepts lengths `3,0,3,3,3`, so physical evidence covers zero-accept rejection and full acceptance.

## Mechanism portfolio

| Family | Mechanism | Evidence | Disposition |
|---|---|---|---|
| Full rebuild | Rerun 128-token MTP prefill and accepted drafts | Existing exact fallback | reference route |
| Full cache clone | Copy maximum-sequence K/V tensors | Moves unrelated sequence-capacity state | rejected |
| Row snapshot | Clone only two writable K/V rows and prefix feedback | Fixed three-draft shape bounds writes | selected candidate |
| Zero overwrite | Replace rejected rows with assumed zero tensors | Current reset starts at zero, but later rounds need prior values | rejected as too narrow |
| Prompt replay | Recompute only the prompt tail | Still needs target hidden alignment and broader replay | retained alternative |
| Confidence width | Avoid drafts on low-confidence prompts | Needs calibrated held-out multi-round evidence | deferred |

## Simulator disposition

The pinned simulator does not establish host trace allocation safety or silicon timing. Simulator evidence cannot select this rollback mechanism.
