#!/usr/bin/env sh
# Device entry point: serve a writable TT-Metal home from the packaged overlay.
#
# The packaged runtime is read-only and the TT-Metal JIT cache lives under
# TT_METAL_HOME, so the first run materializes a per-user symlink tree (the
# overlay is a symlink tree already; real directories are made writable).
set -eu
root="${LFM3_TT_METAL_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/lfm3/tt-metal-runtime}"
if [ ! -e "$root/.ready" ]; then
  mkdir -p "$root"
  cp -as "@overlay@/." "$root/"
  find "$root" -type d -exec chmod u+w {} +
  : > "$root/.ready"
fi
export PYTHONPATH="@sitePackages@:@ttnn@"
export TT_METAL_HOME="$root"
export TT_METAL_RUNTIME_ROOT="$root"
unset MESH_DEVICE
exec "@python@" -P -m "lfm3_runner.@module@" "$@"
