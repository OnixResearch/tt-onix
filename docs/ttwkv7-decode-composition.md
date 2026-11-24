# TTWKV7 decode composition contract

## Scope

`pkgs/ttwkv7/decode-composition-check.nix` is a device-free, source-bound check
for the existing decodeL **reader → compute → writer** composition. It binds:

- the ordered decode ABI in `ttwkv7-decode-abi.h`;
- the host construction introduced by `share-decode-runtime-abi.patch`
  (`make_decode_runtime_arguments` + `SetRuntimeArgs`);
- the decode reader, compute and writer kernel sources;
- the `rwkv-layer-harness` boundary fixture (head count 12, head size 64,
  hidden 768, one sequence, one token).

It does not execute a kernel, open a device, reset a card, or claim hardware
behavior.

## Checked facts

The validator reads the actual sources and emits a receipt with:

- ordered reader (18), compute (8), writer (12) argument values and meanings;
- tile width/height, element size, page size and data format;
- circular-buffer indices, capacities and aliases (staging 21, scratch 22,
  output 16);
- bounded tile flow: reader staging 16 → compute consumed 16 → compute output
  6 → writer consumed 6, state tiles per instance 4;
- ordered per-CB phase-window extraction and validation across `decode_reader`,
  `decode_compute`, and `decode_writer`, plus a `cb_phase_readback` section
  re-emitting every extracted CB event alongside its matched source line,
  resolved CB target(s), loop depth, loop scope, and branch scope;
- FNV-1a change detectors for the ABI, host, reader, compute and writer stages.

### Circular-buffer phase-order rules

The validator tokenizes each kernel source (stripping comments and literals),
walks `kernel_main()` with loop and conditional branch scopes, inlines local
lambdas (such as `gc`) at their call sites, and extracts every CB handshake
(`cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`), tile or
pointer access (`pack_tile`, `get_write_ptr`, `get_read_ptr`, `copy_tile`,
`transpose_wh_tile`, `mul_tiles`, `matmul_tiles`, `mul_tiles_bcast_rows`), and
CB-indexed init/query/startup call in execution order.

It enforces:

- **Strict producer/consumer window pairing and scope matching**: every
  `cb_reserve_back(cb, n)` closes with a matching `cb_push_back(cb, n)` at the
  same loop and branch scope, and every `cb_wait_front(cb, n)` closes with a
  matching `cb_pop_front(cb, n)` at the same scope (no push before reserve, no
  pop before wait, no nested duplicate open window, no unclosed window across a
  loop, branch, early control jump, or kernel exit). The three intentional
  non-standard lifetimes in the decodeL kernels are checked explicitly rather
  than skipped: reader-local `c_dram_read_scratch` (single top-level
  `reserve(1)` + `get_write_ptr`, no consumer), compute `cb_ident` (single
  top-level `reserve(4)..pack..push(4)` followed by resident `wait(4)`), and
  compute `cb_state` (loop-carried consumer window popped inside `tok` and
  re-established only under `if (!last)`).
- **Access-in-window containment**: every producer access (`pack_tile`,
  `get_write_ptr`) and producer init falls inside an open `reserve..push` window
  within a nested or matching scope; every consumer access (`copy_tile`,
  `get_read_ptr`, `mul_tiles`, `transpose_wh_tile`, `matmul_tiles`,
  `mul_tiles_bcast_rows`) and consumer init falls inside an open `wait..pop`
  window within a nested or matching scope.
- **Ordered inter-stage decodeL protocol**:
  - `decode_reader` produces `c_natstage` state seed (`NS`) at `[inst]` strictly
    before per-token inputs (`St`) at `[inst, t < L, inp < 6]`.
  - `decode_compute` consumes `c_natstage` state seed (`NS`) at `[inst]`
    strictly before per-token inputs (`St`) at `[inst, tok < L, inp < 6]`, and
    produces `cb_out` in one `St + NS` window at `[inst, tok < L]` with
    region-1 `pack_tile(it, cb_out, it)` strictly before region-2
    `pack_tile(t, cb_out, St + t)`.
  - `decode_writer` consumes `cb_out` at `[inst0, cc < nc, g < nb]` in two
    strictly ordered windows: region-1 (`St`) wait/read/pop strictly before
    region-2 (`NS`) wait/read/pop.
- **Refusal by name (never silent skip)**: any `#define` macro in a kernel,
  unrecognized CB/macro call, CB call outside `kernel_main()`, uninvoked local
  lambda containing CB operations, local shadowing/reassignment of a CB symbol,
  or CB index expression that is not a declared namespace-scope `constexpr
  uint32_t` CB symbol (or `cb_symbol + loop_var` with a compile-time constant
  `for`-loop bound whose offsets all resolve to declared CBs) is rejected
  explicitly by name and line number.

### Negative mutations

The fixture is accepted only when all of those match. The check also rejects:

- changed CB index;
- changed tile page size/format;
- under-capacity staging buffer;
- changed ordered argument assignment or enum meaning;
- changed runtime-argument index in the reader;
- compute output tile-count mismatch;
- writer swapping region-1 and region-2 `cb_wait_front`/`cb_pop_front` order or
  swapping `cb_wait_front` order alone;
- compute moving `cb_pop_front(c_natstage, St)` above the tile-read loop;
- compute moving `pack_tile(..., cb_out, ...)` after `cb_push_back(cb_out, St + NS)`;
- reader acquiring `get_write_ptr(c_natstage)` before `cb_reserve_back` or
  moving `cb_push_back` across its enclosing loop boundary;
- unrecognized `#define` CB wrapper macro, unrecognized external CB/macro call,
  runtime-variable CB index (`dyn_cb`), and runtime-bounded loop offset
  (`c_natstage + tok`);
- stale producer/consumer version (any semantic source change);
- missing/extra validator arguments and unreadable fixtures.

Before the phase-order check replaced the earlier substring-count check, all
three reordering mutations (writer region-1/region-2 swap, compute
`cb_pop_front(c_natstage, St)` before reads, and compute `pack_tile` into
`cb_out` after `cb_push_back(cb_out, St + NS)`) passed the tile-flow validation
completely — failing only at the trailing FNV-1a stage fingerprint check against
the pinned contract, and exiting `0` when stage fingerprints were refreshed.

## Command

```sh
nix build .#checks.x86_64-linux.rwkv-ttwkv7-decode-composition -L
```

The receipt is `receipt.json` in the check output. It preserves all existing
fields byte-identical (including `device_initialized: false` and
`hardware_execution_claimed: false`) and appends `cb_phase_readback` with the
ordered extracted event stream next to each matched source line for reviewer
diffing.

## Boundaries

- Only the decodeL composition is covered; the chunked path and the existing
  architecture-check/CB22 controls are unchanged.
- The fingerprints are change detectors, not cryptographic attestations.
- Source-level composition is not compiled-kernel or device evidence.
- A real generated descriptor/program fact remains a future verification gap;
  the current host binding is the source path this repository actually builds.
