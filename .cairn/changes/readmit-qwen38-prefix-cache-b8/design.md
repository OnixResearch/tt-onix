## Decisions

### Decision: a new schema, not an edited policy

**Choice:** add schema 5 with one extra field, `scheduler_profile_id`, and leave the
schema-2 policy, its contracts, and its receipt exactly as they are.

**Rationale:** the schema-2 policy and the p7 receipt are historical artifacts with a
published digest. Editing either would make the exported policy's provenance
ambiguous, and a receipt could no longer be re-verified from the stored roots. A new
schema keeps the old evidence verifiable and makes the new binding explicit.

### Decision: the policy names the scheduler profile, and the roots must match it

**Choice:** the policy declares the continuous-slot profile every root must report,
and the core rejects a root that omits it or names another one.

**Rationale:** the prefix cache and the scheduler are separate features with separate
receipts. Binding the scheduler profile in the policy turns "the cache was admitted"
into "the cache was admitted in this scheduler configuration", and the rejection is
per root, so a campaign that ran the wrong profile cannot produce a receipt.

### Decision: the roots differ only in the cache

**Choice:** the cold and cached roots run the same eight-slot scheduler profile, at
the served limits, with the cache policy added to the cached roots only.

**Rationale:** the ABBA protocol measures the cache, so the cache must be the only
difference between the roots. This is the p6 and p7 arrangement, with the newer
scheduler profile.

### Decision: the workload filler is a pinned text, and its digests are recorded

**Choice:** the campaign uses the pinned RWKV-7 paper text as its filler and records
the resulting workload digests.

**Rationale:** the filler behind the p6 and p7 workload digests is not recoverable:
the committed roots store the digest and the counts, not the prompt text, and no
copy of that filler is in the repository or on this host. The gates do not depend on
the digest's value; they compare hits against misses inside one campaign. The new
digests are recorded so this campaign's workload is identifiable.

### Decision: the served package carries both receipts

**Choice:** re-run the B=8 scheduler admission on the same package as the prefix-cache
campaign, then move the served profile to it.

**Rationale:** a receipt binds a package and an execution identity. The served
package must be one that both receipts name, or the deployment rests on a receipt for
a package that is no longer served.

## Re-admission on the promoted package (2026-10-26)

The prefix-cache admission was re-earned on the promoted package
`/nix/store/gzwghi7spl93iv4y6qb680cg8wyfbq6r-qwen38-0.1.0` and identity
`/nix/store/qwnxwcppn16za67i710qm5drnzs57vg9-qwen36-execution-identity-0.1.0` with the
schema-5 policy, four ABBA roots, and both declared workloads. Receipt:
`pkgs/qwen36/evidence/2026-10-26/qwen38-prefix-cache-abba-admission-receipt.json`,
BLAKE3 `11288cecce8aa47a8308d605ade20de9b4e4295f65d0627b12c6a3ebdae4a109`; all eight
gates pass. The 2026-10-21 receipt and the `wlsl6pnx…`/`l4arf461…` package and identity
are superseded as the current admission and retained as history.

## Risks / Trade-offs

- The campaign takes about forty-five minutes of device time: four fresh processes
  per workload, each paying the served startup.
- The DRAM-drop gate is unchanged at 512 MiB. The eight-slot profile's wider trace
  does not change the cache's state slots, so the drop should stay near the measured
  365 MB; if it does not, the gate says so.
- The host's other work drifts during the campaign. The ABBA order is what cancels
  that drift, and the protocol is the one already predeclared for this cache.
