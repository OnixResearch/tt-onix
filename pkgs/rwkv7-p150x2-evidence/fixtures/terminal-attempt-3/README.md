# RWKV-7 P150x2 attempt 3

This bounded fixture retains the decisive files from
`/var/tmp/rwkv7-p150x2-full-device-attempt-3-20260718`.

The attempt ran one hardware process. It isolated both reviewed owners and opened
physical devices 0 and 1. The process failed before verified model execution.
TT-Metal compiled firmware with both device profiling and Watcher enabled. The
Blackhole BRISC image then exceeded its code region.

The source root remains unchanged. This fixture omits reproducible CPU-reference
artifacts and empty orchestration files. `package-files.blake3` binds every copied
raw file. The fixture does not claim successful generation, healthy restoration,
or performance.
