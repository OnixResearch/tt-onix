# Design

## Round shape

A round drafts `k` tokens and verifies `k+1` input rows. The seed row is the last committed token. The other rows are the drafts.

1. The MTP head produces the draft chain from its request-owned feedback.
2. The target verifier consumes `[last_committed, d0..d(k-1)]` at positions `[p..p+k]`.
3. The core commits the accepted prefix and one target continuation token.
4. After rejection, the adapter restores the GDN prefix and rebuilds the head continuation.

The verifier returns rows `a0..ak`, where row `a_i` predicts position `p+i+1`. On rejection at `i`, the commit is `drafts[:i] + (a_i,)`. On full acceptance, it is `drafts + (a_k,)`.

If verifier rows agree with sequential evaluation and state recovery is correct, this rule preserves sequential greedy output. Bounded token comparisons establish neither condition for arbitrary contexts.

## Trace residency

The large verifier trace captures first, with no other trace resident. The prefix-restoration candidate then captures one small trace per prefix buffer. The draft trace captures after those traces. Each speculative round only replays existing traces.

Before plain fallback captures its width-1 trace, the adapter releases the draft, prefix, and verifier traces. Prefix traces release before their borrowed state buffers. No request reuses these traces across request boundaries.

The [Adartras bundle](https://huggingface.co/Adartras/qwen3.6-a3b-blackhole) describes small commit traces after verifier capture. The inspected revision is `42a71032659a9d0e3b860b6104c59cfa6ca82149`. It is an implementation lead, not local acceptance. The physical screen must validate this candidate capture order.

## Prefix-restoration traces

Each restoration trace records the existing per-layer recurrent copy, convolution copies, and carry synchronization. It borrows fixed prefix and destination buffers. The verifier replay updates prefix contents before any restoration replay.

The adapter compares Python tensor-object identities before replay. It does not inspect device addresses. The owner must keep the borrowed allocations live. Invalid indices, incomplete capture, or changed identities reject the replay before device effects. Capture failure ends the active capture and releases all created prefix traces. The parent restores its original GDN snapshot after prefix capture, including the failure path.

Capture time includes these additional traces. Round timing includes their replay and synchronization. Faster recovery does not imply faster end-to-end requests because capture also has a cost.

## State and exactness boundary

On full acceptance, draft generation has not consumed `d(k-1)`. Head continuation consumes that last draft and then the bonus `a_k`. On partial acceptance, it consumes accepted inputs and the correction from round-entry feedback.

The current target adapter restores GDN state but leaves speculative KV rows. It relies on position masks and rewrite-before-read for that tail. This remains a gap against the delta specification's requirement to restore rejected KV state. Bounded token parity does not close that requirement. Production admission, spec sync, and archive remain blocked.

The next round resumes at `p + accepted + 1` with seed `a_accepted`.

## Economics guard and defaults

The pure core compares emitted tokens and measured round time with the supplied plain-step rate. Consecutive losing rounds disengage speculation for the request. Re-engagement is forbidden. Neither the two-draft profile nor the prefix-trace candidate changes this threshold.

Production speculation remains off by default. The two-draft profiles are explicit diagnostic alternatives to the unchanged three-draft defaults. Checkpoint, precision, sampled behavior, and batch behavior stay unchanged.

## Evidence boundary

The five-prompt hw6 receipt compared 320 tokens, not 185. It accepted 22 drafts across 15 rounds. Its measured round-only ratio was 0.543, not an end-to-end speedup.

The later row-window pair preserved 320 tokens and the same 18-round acceptance pattern. Mean verifier device time fell from 113.664 to 110.303 ms. The candidate round-only ratio was 0.849, still slower than plain decode.

The two-draft hw9 screen preserved all 320 tokens and reduced verifier device time to about 94.5 ms. Its round-only ratio was 0.770. A 489 ms rollback outlier motivated the prefix-trace candidate. The outlier does not prove allocator causality.

See `docs/qwen38-mtp-serving-progress.md` and the immutable private run directories for measurements. Historical pilot averages and design estimates are not serving-loop acceptance evidence.

## Unadmitted target-KV row recovery

The diagnostic prototype reuses request-entry snapshots for rejected rows. The pure planner requires a contiguous committed boundary and a unique page map. The adapter restores offsets `accepted+1` through `draft_count`, with the seed retained. It validates all bindings and formats before writes. This approach assumes that no other writer changes the uncommitted tail between rounds.

The simulated operator test found a negative-zero bit mismatch against its pre-upload host input. A later stage audit found the sign already absent at mesh upload/readback, before recovery. Matched host-only controls preserved it. Thus, the earlier mismatch does not establish a rollback defect. The exact cause within that transfer/readback boundary remains unknown. The bitwise criterion remains unchanged. A valid restoration test must establish the device-entry baseline and keep a separate upload control. This is simulated evidence, not a silicon defect claim.

The `diagnostic_kv_tail_restore` seam therefore defaults to false. No server or CLI route enables it. The existing serving behavior and the open rejected-KV requirement remain unchanged. The device test, host mutation, and failed receipts preserve the counterexample for further work.
