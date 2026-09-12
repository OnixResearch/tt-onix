# I1-I3: rebuilt package, fail-closed gates, untouched B=4

Recorded 2026-09-11 for change `re-admit-qwen38-p150x2-b8-after-batched-prefill`.

## I1: the rebuilt package and its new identities

```
nix build .#qwen38 .#qwen38-gdn-grid --no-link --print-out-paths
/nix/store/rim3imrishs3s4z9wfii1h8g8cn38qpb-qwen38-0.1.0   (shared runtime)
/nix/store/mgi46qcz8mxs3sbmaq4ivyhs71y1jpac-qwen38-0.1.0   (chunked runtime)
```

| Identity | Value |
| --- | --- |
| admission package | `/nix/store/mgi46qcz8mxs3sbmaq4ivyhs71y1jpac-qwen38-0.1.0` |
| runtime identity | `/nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0` (chunked GDN grid variant) |
| package execution identity | `/nix/store/z31p1j9wa4af27vakffyzqp9cyypsjr1-qwen36-execution-identity-0.1.0` |
| admission binary | `$package/bin/qwen38-p150x2-b8-admission` |
| profile passed by the wrapper | `$package/share/qwen36/batch-admission-b8-profile.json` |
| decode mode | `candidate` with `QWEN35_GDN_DECODE_FUSION=full` exported |

The execution identity is new (`z31p1j9w…`) because the B=8 contracts file is
an identity input; the previous measurement identity was `spf5pcxnp…`. Every
response this run produces carries the new identity, which is why the roots
were re-measured instead of reusing the earlier diagnostic roots.

## I2: negative contract tests that reject a weakened gate

### Export layer (Nickel contracts, package build chokepoint)

`pkgs/qwen36/batch-admission-b8-contracts.ncl` now pins the frozen protocol
values that mirror the named constants in the policy file: widths 1 and 8,
64 generated tokens, three measured runs, one warmup run, eight prompts, and
the four admission bounds 6.0, 0.75, 9.0, 1.5. The root order was already
pinned. Each mutation was applied to a copy of the policy and exported:

| Mutation | Result |
| --- | --- |
| aggregate speedup threshold 6.0 -> 5.5 | REJECTED: contract broken by `minimum_aggregate_decode_speedup_ratio` |
| TTFT threshold 9.0 -> 12.0 | REJECTED: contract broken by `maximum_ttft_ratio` |
| corpus eight prompts -> seven | REJECTED: contract broken by `prompts` |
| measured runs 3 -> 1 | REJECTED: contract broken by `measured_runs` |
| root order changed | REJECTED: contract broken by `execution_order` |
| generated tokens 64 -> 32 | REJECTED: contract broken by `generated_tokens` |

The unmutated profile exports with exit 0, so the gate rejects edits rather
than everything. The B=4 contracts file is untouched, so the B=4 profile keeps
its own validated shape.

### Receipt layer (`scripts/qwen36-batch-receipt.rs --self-test`)

The self-test already rejected a token-parity mismatch and ineligible
evidence. It now also rejects, with named diagnostics:

| Injection | Diagnostic required |
| --- | --- |
| one batch root removed | `exactly two single and two batch roots are required` |
| order changed in the arguments | `execution order must be single,batch,batch,single` |
| order changed inside the profile | `profile execution order differs from the command` |
| corpus shortened by one prompt | `measured samples, expected` |

`--self-test` exits 0 over both widths after the additions, so the negative
cases fail only when a gate is actually weakened.

## I3: the B=4 profile, contracts, binding, and receipt are byte-identical

- `git diff --name-only origin/main -- 'pkgs/qwen36/*.ncl'` lists only
  `batch-admission-b8-contracts.ncl`.
- No file under `pkgs/qwen36/evidence` changed.
- B=4 profile export, pre-change versus post-change: byte-identical,
  BLAKE3 `d3ee3895c9718343b2fefaa4f63cbe50cd1bdbee27e79c0ddd104193369e66cd`,
  and identical to the deployed artifact inside the rebuilt package.
- The B=4 binding still declares `physically_admitted = true` with receipt
  `489f2fbcb414ae04e536fc055857864c3255d727e4c12b79dc701a86ded3d028`, unchanged.
