## Why

Two confirmed guards in the pinned runtime cap a GDN kernel at one program
instance per Tensix core. Both are in real source, not in upstream prose.

**Decode.** `pkgs/tt-metal/gdn-decode-fp32.patch` line 324, in the fused
`ttnn.transformer.gated_delta_decode` program factory:

```c
TT_FATAL(num_heads <= compute_grid.x * grid_y, "num_heads exceeds the available compute grid");
```

and line 329 maps head `h` to core `{h / grid_y, h % grid_y}`. One head, one
core, no chunking.

**Prefill.** `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`
line 137:

```c
const uint32_t ncores = grid.x * grid.y;
TT_FATAL(BH <= ncores, "num_heads {} exceeds compute cores {}", BH, ncores);
```

with a second cap at line 143, `BH * cand <= ncores`.

The decode guard binds the admitted fast path to a batch ceiling. At TP=2 the
fused op reports `H=24`, so head count is `B x 24`. A Blackhole die offers about
110 compute cores:

| Batch | Instances | Fits |
| ---: | ---: | --- |
| 1 | 24 | yes |
| 4 | 96 | yes, barely |
| 8 | 192 | **no** |

The fresh B=4 fused admission passed with all four correctness flags true and a
`3.5293` aggregate speedup. That is also the last width the fused path can
reach. B=8 cannot use it at all.

The prefill guard is the recorded root cause of the rejected B=8 batch
admission: eight serialized per-user prefills, TTFT ratio `9.0792` against a
bound of `9.0`, per-user retention `0.7352` against `0.75`.

Upstream PR 53482 already carries the fix as its second hardening item. It
lifts the guard and gives each active core a contiguous chunk of
`ceil(BH / ncores)` instances, with `(bh_start, n_inst)` passed as runtime
arguments. The mapping is unchanged when `BH` is at or below the core count.

Applicability is verified, not assumed. `patch -p1 --dry-run` of the full PR
against the pinned source tree exits `0` with 18 files checked and zero hunk
failures.

## What Changes

- Lift the one-instance-per-core guard in the fused GDN decode program factory
  and assign each active core a contiguous instance chunk.
- Lift the same guard in the GDN chunk-prefill phased program factory, and keep
  its `BH * cand <= ncores` sub-split search bounded.
- Add a pure, device-free instance-assignment core that computes contiguous
  chunks and rejects malformed ranges. It takes over the validation the removed
  fatal used to provide.
- Keep the in-range mapping byte-identical, so every existing prefill and decode
  receipt stays valid.
- Add exact parity evidence for both sites against their current references.

This change improves two mechanisms and admits no batch width, no timing claim,
and no serving route.

## Impact

- **Files**: `pkgs/tt-metal/gdn-decode-fp32.patch` or a follow-up patch beside
  it, `pkgs/qwen36/` check probes and the instance-assignment core, this change
  directory, and the physical evidence tree.
- **Testing**: qwen36 device-free check with positive and negative assignment
  probes, package build, kernel hazard audits, a fresh Blackhole simulator
  smoke, and matched physical parity roots for decode and prefill.
- **Production**: unchanged. No admission binding moves. The change creates a
  patched tt-metal derivation so the shared `tt-metal` 0.77 output stays intact
  for RWKV-7 and the training platform.
