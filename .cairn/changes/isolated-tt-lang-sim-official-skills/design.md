# Design

Use the verified pure-Python wheel in an isolated Python 3 environment, with explicit dependencies and wrappers that discard ambient PYTHONPATH and user sites. Expose only simulator and statistics entry points, not upstream setup/download commands. Verify arithmetic and traced traffic using installed upstream simulator APIs.

Fetch official skills at b6752095f329a04bf6573fb4022b13bc45799167 with a fixed recursive hash. Install selected review-core, precision, trace, L1 and kernel domains with their complete relative references, matched docs and license/NOTICE. A read-only reference CLI automatically loads review-core first and accepts at most two additional distinct domain skills per call. It never installs marketplace plugins, executes hooks or mutates home directories. Preserve `.pi/skills`.

Shared integration belongs to this change owner. Sibling packages remain independently owned. The combined validation completed after source stabilization; the final 2026-10-23 acceptance and its later delivery-record reconciliation are distinguished below.

## Final acceptance, 2026-10-23

The normal native package check now passes with optimized host test oracles,
debug assertions and overflow checks retained: 1,556 passed, zero failed,
46 existing ignored and zero filtered. The [installed evidence](evidence/installed-tooling-2026-10-23.json)
preserves both earlier timeouts and the exact successful output separately in
`native_package_resolution`. No native algorithm, test case or bound was relaxed.

The exact `nix run .#quality` aggregate also passed using the matched Nix 2.36
CLI and hook. Its [portable acceptance receipt](evidence/final-aggregate-2026-10-23.json)
was reconstructed from the existing command output after that run; it was not
inside the historically frozen 893-entry manifest. Delivery-record changes and
their artifact/Cairn checks are subsequent documentation-only evidence, not a
rerun of native, simulator or hardware qualification. All default-off and
no-performance/serving-promotion boundaries remain in force.

