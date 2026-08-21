# P150x2 prefill profile result

## Scope

- Date: 2026-08-21
- Host: `britton-desktop`
- Physical PCIe and UMD device IDs: `0`, `1`
- Exact mesh: `1x2`
- Candidate commit: `e289a86a11f7c1835c52cecdc52f1f13ed52c4d7`
- Candidate package: `/nix/store/w0c80mbyz6lzvqih40ark0j0c1fxqhyw-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Profile ID: `qwen38-p150x2-traced-decode-v1`

The local `ttsim` reference has no real-time program-profiler API or record implementation.
The simulator cannot validate this callback path, so the run used one bounded physical outage.

## Activation

The package-matched runtime opened both devices and reported:

```text
realtime_profiler_active=True
mesh_devices=2
```

Both device receiver threads also completed their clock synchronization.

## Balanced corpus

The valid root order was exactly:

```text
eager,candidate,candidate,eager
```

Each root used the existing four prompts, one warm-up, three measured requests, and `64` generated tokens.
The receipt validated `12` measured responses per root and `48` measured responses in total.
All responses were physical diagnostic data with `admission_eligible = false`.

An earlier eager warm-up exceeded an initial `120`-second client timeout during first-use compilation.
That incomplete attempt was excluded before the balanced order restarted.

## Attribution

The comparator found `100` complete common signature-and-chip rows.
It found no eager-only or candidate-only program signatures.

The diagnostic wall-prefill medians were:

- eager: `1.563908943993738` seconds
- candidate: `2.587822357978439` seconds
- candidate/eager ratio: `1.6547142133287787`
- delta: `1.023913413984701` seconds

These wall values include profiler effects and cannot enter TTFT admission.

The sums of per-signature, per-chip median program durations were:

- eager: `357825370.12429523` nanoseconds
- candidate: `360131420.692155` nanoseconds
- candidate/eager ratio: `1.0064446256760908`
- delta: `2306050.567859769` nanoseconds

The checked program-duration delta explains only `0.0022521929455787217` of the profiled wall delta.
The strongest program difference was the MLP prefill reduce-scatter on chip `0`:

- eager: `5492647.023769595` nanoseconds
- candidate: `6592343.838415919` nanoseconds
- candidate/eager ratio: `1.200212540490469`
- delta: `1099696.8146463241` nanoseconds

The second difference was the matching MLP prefill all-gather on chip `0`:

- eager: `4244755.854348572` nanoseconds
- candidate: `5273455.276122063` nanoseconds
- candidate/eager ratio: `1.2423459574758893`
- delta: `1028699.4217734914` nanoseconds

These two CCL differences account for most of the checked program-duration delta.
They do not explain the wall-prefill regression.

## Disposition

No immutable optimization candidate was selected.
Tuning the two CCL programs would target less than one percent of the profiled wall difference.
The package-matched callback does not expose host enqueue gaps, command-queue idle time, or non-program barriers.
This is the exact bounded attribution blocker.

A later change must measure host dispatch and synchronization spans around candidate prefill.
It must not tune MLP CCL settings from this result alone.
Production remains eager, and traced decode remains physically unadmitted.

## Teardown observation

All four service processes completed their measured responses before faulting during final Python and TTNN teardown.
The response comparator accepted every root before any evidence was used.
The fault occurred after the `stopping` event, inside Python garbage collection for a TTNN object.

The service retained the HTTP handler closure and engine until after mesh closure.
The implementation now drops the server, engine, and model references before it closes the mesh.
The bounded physical budget did not include a second full profile corpus for this teardown-only fix.

## Evidence

- Receipt: `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-prefill-profile-attribution-receipt.json`
- Receipt BLAKE3: `bc0d8ca7cc65fe050a5aeac9e7ec62e45421ff5e2279ef9a00689ff1938be20c`
- Raw local evidence: `.pi/qwen36-prefill-profile-20260821/`

The raw directory contains the activation log, four response roots, service logs, the excluded first attempt, and input hashes.
It is not tracked because it contains machine-local operational data.
