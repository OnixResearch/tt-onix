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
- FNV-1a change detectors for the ABI, host, reader, compute and writer stages.

The fixture is accepted only when all of those match. The check also rejects:

- changed CB index;
- changed tile page size/format;
- under-capacity staging buffer;
- changed ordered argument assignment or enum meaning;
- changed runtime-argument index in the reader;
- compute output tile-count mismatch;
- stale producer/consumer version (any semantic source change).
- missing/extra validator arguments and unreadable fixtures.

## Command

```sh
nix build .#checks.x86_64-linux.rwkv-ttwkv7-decode-composition -L
```

The receipt is `receipt.json` in the check output. It records
`device_initialized: false` and `hardware_execution_claimed: false`.

## Boundaries

- Only the decodeL composition is covered; the chunked path and the existing
  architecture-check/CB22 controls are unchanged.
- The fingerprints are change detectors, not cryptographic attestations.
- Source-level composition is not compiled-kernel or device evidence.
- A real generated descriptor/program fact remains a future verification gap;
  the current host binding is the source path this repository actually builds.
