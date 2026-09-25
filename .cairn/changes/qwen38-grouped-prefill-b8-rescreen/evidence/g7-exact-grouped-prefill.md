# g7: grouped prefill installs any width, matches per-user exactly, and serves by default

Recorded 2026-09-25 on P150x2 (devices 0 and 1). The fused GDN decode step was
active for these runs.

## The g4/g5 blocker was a leaked zero source

`_alloc_gdn_scratch_b` and `_alloc_gdn_scratch_b1` call `reset_state` on every GDN
layer, which also replaces `_zero_rec` and `_zero_conv_carry` with scratch-shaped
tensors. `_restore_gdn_batched` restored the state bindings but not those zero
sources. After a width-4 group, the next per-user prefill therefore ran
`reset_state_inplace` and copied a `[4, 24, 128, 128]` zero into the
`[1, 24, 128, 128]` B=1 scratch, the recorded failure. The prior tuple now carries
both zero sources, and `_restore_gdn_batched` frees the scratch-shaped ones and
restores the originals.

With the leak gone, the existing row assembly (`fit_rows`) installs groups
narrower than the 16-row decode buffer and several groups in order. The adapter's
`group-width-below-decode-capacity` and `multi-group-assembly-unsupported`
refusals are removed.

## Grouped numerics now equal per-user numerics

The first silicon run with the refusals removed installed correctly at widths 4,
8 and 16, but it was not exact. Teacher-forced top-1 agreement with per-user
prefill was 0.9816 over 924 tokens. First-token logits differed by up to 0.53,
and the difference was identical for group sizes 1, 2 and 4. That identity showed
the cause was the path, not the batching. The grouped forward differed from the
per-user forward in four places, and `grouped-prefill-exact.patch` makes them the
same:

- GDN output: the grouped path now uses `_silu_mul` and the tuned `_row_proj`
  (or the fused reduce-scatter when enabled) on `[1, B*T, *]` rows. It used to use
  an auto-configured `ttnn.linear`.
- Norms: both paths now take `Qwen36DecoderLayer.norm_configs("prefill")`.
- Logits: the grouped path now uses the per-user `_masked_bucket_logits_tp`
  (one-hot row select, norm, LM head).
- RoPE: the grouped path now clears staged per-request RoPE as the per-user path
  does.

After the change:

| Check | Result |
| --- | --- |
| First-token logits vs per-user, group sizes 1, 2, 4 | max abs diff 0.0 for every user |
| Per-user repeat (determinism) | max abs diff 0.0 |
| Greedy 25 tokens, widths 1/4/8/16, eight distinct prompts of 8 to 61 tokens | 25/25 for every user |
| Width-1 per-user prefill after a grouped run | succeeds (the leak regression) |
| Packaged service, 64 greedy tokens, widths 1/4/8/16 × 3 rounds | text identical to per-user in 84 of 84 responses |

## Service latency

Package `/nix/store/j017mx2i4lkz6ylcz7jlnmbi8hka8xmi-qwen38-0.1.0` was run with
`QWEN35_GROUPED_PREFILL=0` and then `=1`. Each width had three rounds of
concurrent 64-token greedy requests. The host was shared (load average about 40
to 70), so each value below is the median of three rounds. Per-token decode
latency is unchanged.

| Width | TTFT max, per-user | TTFT max, grouped | Aggregate tok/s, per-user | Aggregate tok/s, grouped |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 24.6 s | 6.0 s | 8.4 | 24.6 |
| 8 | 30.1 s | 15.8 s | 13.8 | 25.4 |
| 16 | 53.3 s | 23.5 s | 13.7 | 35.3 |

The adapter harness shows the same effect (prefill seconds, two calls each):
width 4 went from 21.4/22.4 to 8.8/8.4, width 8 from 56.2/53.0 to 17.7/15.6, and
width 16 from 161.4/113.3 to 29.2/28.4.

## Serving decision

The NixOS option `hardware.tenstorrent.qwen38.groupedPrefill` (default `true`)
sets `QWEN35_GROUPED_PREFILL=1`. The output is identical to per-user prefill, so
no numerical gate applies. The per-user route remains in three cases:

- single users;
- any batch with a prompt above 256 tokens;
- batches that plan to reuse a parked decode trace (`decode-trace-parked`), so
  that a first-time program compile cannot meet a parked trace.

Widths that the startup warm-up does not cover (8 and 16) compile their grouped
programs on first use.
