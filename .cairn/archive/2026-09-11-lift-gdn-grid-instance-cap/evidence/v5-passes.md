# V5 passes on the variant runtime

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V5.

## Result

The variant package builds and its device-free check passes.

| Item | Value |
| --- | --- |
| Command | `nix build .#qwen38-gdn-grid --builders ""` |
| Exit code | `0` |
| Log length | 269 lines |
| Check result | `qwen36 device-free check: PASS` |
| PASS lines | 43 |
| `installCheckPhase` | completed in 1 minutes 27 seconds |

The grid-cap probe ran inside the gate:

```
PASS GDN grid instance cap assignment
```

Local build with `--builders ""`, so no remote builder was involved.

## Root cause of the earlier failure

The package's installPhase contains a contents assertion over the overlay:

```sh
gdn_decode_compute_expected_lines=208
test "$(wc -l < "$gdn_decode_compute")" -eq "$gdn_decode_compute_expected_lines"
```

The decode chunking took the compute kernel from 208 to 212 lines, so the
assertion failed and the phase exited non-zero.

This explains every observation that the earlier diagnosis could not:

- the shared package passed, because its runtime carries the 208-line kernel;
- the variant failed, because its runtime carries the 212-line kernel;
- the failure was silent, because the assertion emits nothing on success and the
  `set -x` trace shows the failure only as a bare `test 212 -eq 208`;
- the Tracy override changed nothing, because the assertion does not read it.

The `set -x` trace is what found it. The phase body **did** run; the earlier note
that claimed it did not was wrong. The trace simply ends with the failing `test`
and the exit handler.

## The fix

The line count is replaced by assertions on what identifies a patched kernel:

```sh
grep -Fq 'instance < head_count' "$gdn_decode_compute"
for gdn_scan_rel in dataflow/reader_chunk_gdn_scan.cpp \
  dataflow/writer_chunk_gdn_scan.cpp compute/chunk_gdn_scan.cpp
do
  grep -Fq 'instance < wi_count' "$gdn_scan_path"
done
```

A line count breaks on every edit and, more importantly, cannot tell a patched
kernel from one the full-tree copy reverted. The content markers only exist
post-patch, so they assert the overlay outcome that the line count was
approximating.

## What this does and does not establish

Establishes: the variant runtime imports, the package installs against it, and
all 43 device-free probes pass, including the new grid-cap probe.

Does **not** establish: that either chunked kernel executes correctly. The
device-free check exercises no device op. The argument agreement between the
factories and the kernels is still verified only by inspection, and a mismatch
would appear when an op first runs.

No admission binding moved. No physical run has happened.
