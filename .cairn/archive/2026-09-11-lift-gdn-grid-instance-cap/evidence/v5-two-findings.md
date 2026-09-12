# Two findings from task V5

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`.

V5 is the device-free check against the variant runtime. Diagnosing it produced
one definite defect and one unresolved build failure.

## Finding 1 (definite): the package overlay reverts the prefill patch

`pkgs/qwen36/default.nix` reconstructs the installed runtime from
`${tt-metal.src}` and then restores one directory from the installed runtime:

```nix
ttcp_src=${tt-metal.src}/ttnn/cpp
ttcp_dst="$out/libexec/tt-metalium/ttnn/cpp"
rm -rf "$ttcp_dst"
cp -r --no-preserve=mode "$ttcp_src" "$ttcp_dst"
# `${tt-metal.src}` is pre-patch. Restore the package-added GDN decode
# program and kernels from the installed runtime after the full-tree copy.
gdn_decode_src=${tt-metal}/libexec/tt-metalium/ttnn/cpp/ttnn/operations/transformer/gated_delta_decode
gdn_decode_dst="$ttcp_dst/ttnn/operations/transformer/gated_delta_decode"
rm -rf "$gdn_decode_dst"
cp -r --no-preserve=mode "$gdn_decode_src" "$gdn_decode_dst"
```

`overrideAttrs` changes `patches` but not `src`. For the variant, `${tt-metal.src}`
is therefore the **same pre-patch v0.77.0 tree**, so the full-tree copy writes
the unpatched
`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`
into the overlay.

The restore step covers `gated_delta_decode` only. It does not cover
`chunk_gated_delta_rule`.

**Consequence:** in the installed variant package the prefill patch would be
silently reverted, while the decode patch would survive. The prefill guard would
still be present at run time, the scan kernels would ship chunked, and the
factory would pass `wi_start`/`wi_count` while it declares `head`/`vblk`. That
is a mismatch, and it is exactly the kind that surfaces as a confusing runtime
error rather than a clear one.

This is independent of the build failure below and must be fixed either way. The
fix is to restore every patched path, not one hard-coded directory. A robust
form is to copy the full C++ tree from `${tt-metal}` after the `${tt-metal.src}`
copy, or to enumerate each patched path in one place so a new patch cannot be
forgotten.

## Finding 2 (unresolved): the variant package build fails silently

| Item | Value |
| --- | --- |
| Shared drv | `/nix/store/bq8xd3xyq9pk9jga30kj1bslrdqrrd23-qwen38-0.1.0.drv` |
| Variant drv | `/nix/store/xlxcgxf2kn96s6jxjrrn3yfl7pn1l127-qwen38-0.1.0.drv` |
| Variant unbuilt output | `/nix/store/f6d87j4nvcy56gd7v1s3wy8z89i9ayhf-qwen38-0.1.0` |

Facts established:

- The two derivations differ, so the `tt-metal` override does take effect.
- `.#qwen38` succeeds and `.#qwen38-gdn-grid` fails, so the variant runtime is
  the differentiator.
- Every model-tree patch applies, including hunks with offsets.
- The log runs `unpackPhase`, `patchPhase`, `updateAutotoolsGnuConfigScriptsPhase`,
  then `installPhase`, and **stops**. `installPhase` emits nothing.
- Total log is 98 lines. `nix log` shows the same content, so the streaming
  truncation is not the cause.
- The failure is not the Tracy tools. Overriding them two different ways
  produced the identical derivation hash, so they are not an input that
  distinguishes the two derivations.
- `installPhase` does contain a step that can fail on a large tree:
  `cp -r` of the whole `ttnn/cpp` source. Disk exhaustion on the remote builder
  is a candidate that has not been excluded.

Not established: the cause. The next step is to build the same derivation
locally rather than on `ssh-ng://root@aspen1.local`, with `--keep-going`, so the
failing command and its stderr are visible.

## Order of work

Finding 1 should be fixed first. It is definite, it is small, and while it
stands the variant package cannot produce trustworthy evidence even if the
build starts passing.
