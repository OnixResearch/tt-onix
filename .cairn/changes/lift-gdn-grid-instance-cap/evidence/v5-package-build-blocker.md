# Blocker: the variant package build fails at the check phase

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V5.

## What was attempted

Task V5 requires the device-free check to run against the variant runtime. The
package check runs in the derivation's `installCheckPhase`, so the vehicle is a
second package instance:

```nix
qwen38GdnGrid = pkgs.callPackage ../pkgs/qwen36 {
  "tt-metal-tracy-tools" = ttMetalGdnGridTracyTools;
  tt-metal = pkgs.tt-metal-gdn-grid;
};
```

exposed as `.#qwen38-gdn-grid`. The Tracy CLI tools are rebuilt from the variant
too, so the package carries a runtime and a profiler built from one revision.

## Result

`nix build .#qwen38-gdn-grid` fails with builder exit code `1`.

| Item | Value |
| --- | --- |
| Failing derivation | `/nix/store/xlxcgxf2kn96s6jxjrrn3yfl7pn1l127-qwen38-0.1.0.drv` |
| Unbuilt output | `/nix/store/f6d87j4nvcy56gd7v1s3wy8z89i9ayhf-qwen38-0.1.0` |

For comparison, the same package on the shared runtime builds and its check
passes. The variant is what differs.

## What is known about the failure

The package's own model-tree patches all apply cleanly, including every hunk
with an offset. The build reaches `installPhase` and the log ends there. The
`installCheckPhase` never appears.

`nix log` on the failing derivation returns no `error`, `Traceback`,
`Exception`, or `assert` line anywhere in its output, so the log does not expose
the cause. This is a log-availability limit, not an absence of failure.

## Not a conclusion

The failure cause is **not established**. Candidates that a next step should
test, in order of cheapness:

1. The `installCheckPhase` output is not captured in the available log. Re-run
   the build with `--keep-going` and full log capture, or build the check
   derivation directly, to see the phase that fails.
2. The variant's `ttnn` may fail an import or ABI expectation the shared build
   satisfies. The check imports `ttnn` in its first two probes.
3. The Tracy tools rebuilt from the variant may not satisfy the package's
   expectation of them.

## Consequence

The variant C++ compiles and ships (see `both-sites-build.md`), but the package
has not been shown to work with it. Tasks `V5` onward stay open. No physical
run should be attempted until this build passes, because a package that cannot
complete its own device-free check cannot produce trustworthy physical
evidence.
