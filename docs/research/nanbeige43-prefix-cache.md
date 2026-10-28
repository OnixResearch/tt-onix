# Nanbeige4.3-33B prefix-cache design

## Status

Implemented as a bounded, device-free-verified slice (issue #11):

- `pkgs/nanbeige43/runner/nanbeige43_runner/prefix_cache_core.py` is the pure
  32-token block index, LRU, refcount and accounting layer. It imports neither
  ttnn nor torch and is checked by `nanbeige43-check` without cards.
- `Session.prefill_rows` forwards `start_pos`, and the runner's `prefix-cache`
  mode runs one cold/warm pair in a single session on the requested batch.
- One batch-1 hardware receipt passes every declared gate:
  `prefix-cache-b1-layers` (see Device evidence). The first probe
  (`prefix-cache-b1`) diverged at token 0 because of the packaged non-traced
  resumed-prefill page table, now patched in
  `patches/0002-prefix-cache-full-page-table.patch`. After the patch, cold and
  warm prefill logits agree at pcc 0.997825 with equal top-1, the cached prefix's
  K/V is bit-identical in all 52 layers, and the greedy ids diverge at token 1
  because the recomputed suffix differs at 51 of 52 layers (numerics, below).

This note describes the process-local reuse boundary the runner now owns. It is
not a serving design: there is no request lifecycle, no admission control, no
cross-process persistence, and no cross-tenant sharing.

## Reuse boundary

A prefix entry is identified by the exact token prefix plus:

- checkpoint file digest and revision;
- package/execution identity and TT-Metal/TT-NN runtime identity;
- precision preset (`bfp8` default or `accuracy`/mix);
- mesh and model configuration;
- execution knobs that change the KV layout (`prompt_mode`, page size, pool
  size, decode trace).

`prefix_cache_core.cache_identity` digests exactly this fixed key set; a missing
or extra key is an error rather than a silently merged identity. Tokens are bound
by a SHA-256 digest chain over 32-token blocks rooted at that identity, so a
different token at any position — or the same tokens under another preset or
mesh — produces different block keys and therefore a miss.

## Pool and block model

- The paged pool uses 32-token blocks (`--page-block 32`); `prefix-cache` mode
  fails closed on any other page size because a 32-token entry would misalign.
- Whole 32-token blocks are the residency unit and are shared by reference. Each
  block tracks the entries holding it and the leases reading it; a leased block
  is never freed, and an entry with an active lease is never evicted.
- Blocks are read-only to requests. `Lease.write_start_block` is the first block
  past the cached prefix, `Lease.copy_tail_block` marks a partial tail that must
  be copied, and `PrefixCache.assert_writable` rejects any KV write into a shared
  prefix block. Only whole blocks are cached, so this slice never produces the
  partial-tail case; the marker and guard exist for callers that ask for one.
- Eviction is LRU over entries with no active lease, under
  `--prefix-cache-blocks` (default: the prefix's own block count). A prefix that
  cannot fit the budget after evicting every idle entry is rejected before the
  index mutates. The index budget is separate from the paged pool: the runner's
  page table and `--page-blocks` stay the authority for KV capacity.
- The cache is process-local. Blocks, entries and counters live in one Python
  object for the life of one run.

## Prefill path

Upstream `models/tt_transformers/tt/generator.py` accepts `start_pos` (cached
prefix lengths) on `prefill_forward_text` and slices
`tokens[user, num_cached:seq_len]` before the prefill. Batched prefill is not
compatible: it zeroes the cached offset. The runner therefore:

1. requires the sequential per-row path (`--batched-prefill` is rejected in this
   mode) and records the effective upstream `disable_batched_prefill` flag;
2. resolves the longest resident block prefix under the exact identity;
3. mirrors upstream's resume-offset flooring (`lcm(page block, traced SDPA
   q_chunk)`: 64 below a 2048-token padded suffix, 256 above) on the host and
   fails closed when the requested prefix would be silently recomputed;
4. prefills the full prompt with `start_pos=[cached]*batch` while keeping the
   session's KV and page table;
5. fails closed rather than falling back silently: a rejected prefix, a wrong
   batch, or a pool that cannot hold prompt plus decode is refused before the
   mesh is opened.

Prefill traces stay disabled (`enable_trace=False` on prefill), as before. The
decode trace is reused between the cold and warm passes, which is why the warm
decode rate is not a cache claim.

### Packaged resume-path defect and patch

The first device probe (`prefix-cache-b1`) diverged at greedy token 0. The cause
is in the packaged v0.78.0 non-traced resumed prefill, not in the index:
`_prefill_forward_text_impl` built the per-user page table with
`_get_prefill_user_page_table(..., prefill_seq_len=padded_suffix)`, which
truncates it to the new chunk's block count. `prefill_forward_single_user_text`
then needs `num_blocks_in_seq(padded_suffix + cached_tokens)` blocks and pads the
missing new pages with zeros, so the new K/V landed in physical block 0 and the
cached prefix was never attended. Only the traced path passed the full prompt
table (`use_full_prompt_len=True`).

`patches/0002-prefix-cache-full-page-table.patch` passes the full prompt's
cached and new pages whenever a prefill is resumed, which is what that
function's own `page_table` contract ("Cached and new pages") already requires.
The batched path is untouched (it has no cached offset), and the trace path is
unchanged. `nanbeige43-check` fails if the installed generator loses the patch.

## Device evidence (batch 1/8/32)

Receipts:
`~/models/nanbeige43-evidence/receipts/prefix-cache-b1-layers.json`,
`.../prefix-cache-b8.json`, and `.../prefix-cache-b32.json`.
BLAKE3: `ae2e2b96eef533fb8471ec215d1aae65b9175fd6da18f64314af6dde2fe742a6`,
`d39e60b2bcad862ee31beb6e97d8fc80033ab609d923f36d114e3e02a4b3c640`,
`1ab45d4c20a97333062b5770508d41a66a21aef910e1b3f06c5f9ebe8497d83c`.

All three use package
`/nix/store/n5va3rk1v871jfy1plx64j0zw44mlck6-nanbeige43-0.1.0`, runtime
`/nix/store/lz9hma19v7hfhc53fldq6qhdkyyyjmrh-tt-metal-0.78.0`, checkpoint
revision `c1093c501cc6460427c0ac634db7223d0450c0ee`, preset `bfp8`, page block
32, 128 cached tokens, 256 prompt tokens, 8 decode tokens (batch 1 used 16).

| Batch | Cold prefill / TTFT | Warm prefill / TTFT | Cold decode | Warm decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.516 / 23.520 s | 12.793 / 12.793 s | 18.762 s | 1.072 s (13.99 tok/s) |
| 8 | 41.876 / 41.893 s | 11.974 / 11.979 s | 18.771 s | 0.511 s (13.69 tok/s) |
| 32 | 18.839 / 18.865 s | 8.460 / 8.466 s | 13.423 s | 0.532 s (13.16 tok/s) |

Accounting is cold 0 cached / 256 new / 8 (or 16) output and warm 128 cached /
128 new / 8 (or 16) output per request; warm batch totals are 128×batch cached
and 128×batch new. Every row was identical in both passes.

Correctness in all three receipts:

- cold/warm prefill logits: pcc 0.997825, max absolute difference 1.25, top-1
  equal, not bit-identical;
- cached prefix pages (positions 0..127): SHA-256 page digests equal in all 52
  layers, so the warm pass never rewrote cached K/V;
- recomputed suffix pages (positions 128..255): layer 0 identical, layers 1..51
  differ (102 page-digest mismatches over the two caches), the expected
  floating-point consequence of recomputing the suffix at a different padded
  shape;
- greedy ids: cold `[12384, 7652, 264, 764, 44196, 13, ...]`, warm
  `[12384, 324, 437, 764, 26213, 13, ...]`, first divergence at token 1 and
  recorded, not gated. Exact greedy parity is not a valid bfp8 gate for a
  differently shaped suffix prefill.

The decode-rate deltas are not a cache speed claim: the cold pass carries the
one-time decode-trace capture and the warm pass reuses that trace. Timings vary
by seconds between sessions.

## Runner mode and accounting

`nanbeige43-p150x2 prefix-cache` arguments:

- `--batch 1|8|32` (exactly the row counts issue #11 names; other sizes fail
  closed with the admitted list);
- `--prefix-tokens` (default 128): must be a multiple of 32 and survive the
  resume alignment;
- `--prefix-prompt-length` (default 256): corpus tokens in the shared prompt;
- `--prefix-decode-tokens` (default 32): greedy tokens per pass;
- `--prefix-cache-blocks`: index block budget;
- `--prefix-kv-check` (default off): after each pass, copy layer 0's K/V for the
  prompt's physical pages to the host and require the cached pages to match.

Cold pass: `Session.reset()`, full prefill, greedy decode N, then insert the
prefix entry. Warm pass: no reset, full prompt prefill with
`start_pos=[prefix]*batch`, greedy decode N, then release the lease. Each request
records:

- `cached_input_tokens`, `new_input_tokens`, `output_tokens` (per request, plus
  batch totals for the warm pass);
- prefix hit/miss, entry generation, resident and evicted blocks;
- prefill seconds, first-token seconds, decode seconds and decode rate per pass.

The receipt gains a `prefix_cache` block with the identity keys and digest, the
requested/effective prefix, the entry view, the accounting block, both pass
records, token parity, the first divergence index, `prefill_logits` (cold/warm
prefill logits pcc, max absolute difference, bit identity and top-1 equality per
row) and, with `--prefix-kv-check`, `kv_parity` (per-row K and V equality and
max absolute difference for the prompt's physical pages). Existing modes and
gates are unchanged.

## Correctness and measurement gates

Device-free (run by `nanbeige43-check`):

- exact-token identity: a changed token, a different preset, or a different
  token sequence misses or hits only the unchanged whole blocks;
- block accounting at 31/95/96 tokens; shared-block refcounts on overlapping
  prefixes; a leased entry cannot be evicted; LRU evicts the untouched entry;
  an over-budget prefix is rejected without mutating the index;
- the partial-tail copy marker and the shared-block write guard;
- cached/new/output accounting rejects unaligned, negative and over-prompt
  counts;
- the CLI preflight rejects batch 2, batched prefill, the decode prompt path, a
  64-token page, an unaligned prefix, a prefix as long as the prompt, a budget
  below the prefix, zero decode tokens and a pair that overflows the per-row
  pool;
- the alignment mirror agrees with the packaged helpers: 128 over 256 settles at
  128 with alignment 64, 32 over 256 floors to 0 and is rejected.

Device (not measured; these are the gates a receipt must satisfy):

- cold/warm prefill logits pcc >= 0.99 (the runner's existing row floor) and
  top-1 equal; the greedy ids and their first divergence are recorded, not
  gated, because the warm suffix is recomputed under a different prefill shape;
- every batch row identical in both passes;
- with `--prefix-kv-check`, the cached prefix's K/V page digests identical in
  every layer; suffix page mismatches are reported per layer as the numeric
  fingerprint of the recompute;
- TTFT and decode quoted together, cold and warm in one session;
- no claim until the exact package/checkpoint/preset/mesh identity is recorded.

## Non-claims and remaining gaps

- Device evidence is the batch 1/8/32 receipts above. All three used upstream's
  sequential per-user `start_pos` path and identical packet shapes; batch 8/32
  were measured on the P150x2 mesh.
- One entry per process and one request at a time: no eviction pressure is
  exercised on a device run (the default budget equals the prefix's blocks and
  only one entry is inserted), and LRU/refcount behavior is covered by unit
  checks only.
- No cross-process or cross-tenant sharing, no persistence, no serving capacity
  or throughput claim.
- Upstream may floor the resume offset further; the runner rejects a mismatch
  instead of reporting the floored count as a hit.
- The decode trace is reused by the warm pass, so decode-rate deltas are not
  attributed to the cache. Timings from the receipt are one session each.
- Exact greedy equality across cold and warm is not claimed at bfp8: the
  measured receipts diverge at token 1 while the cached prefix K/V is exact and
  prefill logits stay at pcc 0.9978. The batch 1/8/32 receipts all show the
  same divergence point and cached-prefix K/V equality.
