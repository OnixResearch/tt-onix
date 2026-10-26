# SPDX-License-Identifier: LGPL-2.1-only
# Shell fragment that builds the package-owned TT-Metal runtime overlay at
# "$out/libexec/tt-metalium": a symlink tree of the packaged runtime, the
# package-matched api/ and ttnn/cpp trees for JIT-compiled kernels, the
# runtime's patched JIT kernel directories, and the Tracy CLI tools.
#
# Factored out of pkgs/qwen36 so every TT-Metal model package uses one overlay.
# The fragment text is byte-identical to the original qwen36 installPhase, so
# the qwen38 derivation is unchanged; keep it that way (or accept a qwen38
# rebuild) when editing. The patched JIT path list follows pkgs/tt-metal's
# runtime patches.
{
  lib,
  tt-metal,
  tt-metal-tracy-tools,
}:
let
  metaliumRoot = "${tt-metal}/libexec/tt-metalium";
in
lib.removeSuffix "\n" ''
  # The packaged tracy python module mkdirs $TT_METAL_HOME/build/profiler/
  # build_wasm/traces at import time (tracy/common.py). The packaged runtime
  # is read-only, so serve TT_METAL_HOME from a package-owned overlay that
  # already contains those directories; mkdir(exist_ok=True) then no-ops.
  # The overlay is a symlink tree of the packaged runtime, so every real
  # Metalium file stays reachable without copying.
  mkdir -p "$out/libexec" "$out/libexec/tt-metalium/build/profiler/build_wasm/traces"
  cp -as ${metaliumRoot}/. "$out/libexec/tt-metalium/"
  # cp -as preserved the store's read-only dir modes; make every REAL
  # directory writable (find -type d never matches symlinks) so the api
  # subtree can be replaced below.
  find "$out/libexec/tt-metalium" -type d -exec chmod u+w {} +
  # The packaged 0.74 runtime include tree ships only part of
  # tt_metal/hw/inc/api (e.g. compute/experimental lacks matmul_custom.h,
  # which the qwen36 SDPA kernels JIT-include). Replace the api subtree with
  # real files from the package-matched tt-metal source — the same overlay
  # pattern the rwkv7-p150x2 runtime uses for the unary headers. (A plain
  # merge fails: parts of api, e.g. debug/, are symlinks into the store.)
  api_src=${tt-metal.src}/tt_metal/hw/inc/api
  api_dst="$out/libexec/tt-metalium/tt_metal/hw/inc/api"
  rm -rf "$api_dst"
  cp -r --no-preserve=mode "$api_src" "$api_dst"
  # The packaged runtime also ships only part of the ttnn C++ tree (2733 of
  # 4691 operation files; e.g. the experimental CCL all-gather-fused-matmul
  # kernels the qwen36 TP path JITs are absent). Replace ttnn/cpp with the
  # full package-matched source tree so every JIT kernel source resolves.
  ttcp_src=${tt-metal.src}/ttnn/cpp
  ttcp_dst="$out/libexec/tt-metalium/ttnn/cpp"
  rm -rf "$ttcp_dst"
  cp -r --no-preserve=mode "$ttcp_src" "$ttcp_dst"
  # `${tt-metal.src}` is pre-patch: `overrideAttrs` changes `patches`, not
  # `src`. Any file the runtime JIT-compiles therefore has to be restored from
  # the installed runtime after the full-tree copy, or the overlay silently
  # reverts the patch while the host-side half of it stays compiled in.
  #
  # This list must stay in step with the patch list above. A patch that adds or
  # changes a JIT kernel source MUST add its directory here; otherwise the
  # factory would pass arguments the reverted kernel does not read.
  for patched_jit_path in \
    ttnn/operations/transformer/gated_delta_decode \
    ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels
  do
    patched_jit_src="${tt-metal}/libexec/tt-metalium/ttnn/cpp/$patched_jit_path"
    patched_jit_dst="$ttcp_dst/$patched_jit_path"
    test -e "$patched_jit_src" || {
      echo "patched JIT path is absent from the installed runtime: $patched_jit_path" >&2
      exit 1
    }
    rm -rf "$patched_jit_dst"
    mkdir -p "$(dirname "$patched_jit_dst")"
    cp -r --no-preserve=mode "$patched_jit_src" "$patched_jit_dst"
  done
  mkdir -p "$out/libexec/tt-metalium/build/profiler/build_wasm/traces"
  # Host-side Tracy CLI tools matching the packaged runtime. The packaged
  # `python -m tracy` module resolves them under
  # $TT_METAL_HOME/build/tools/profiler/bin (see tracy/common.py).
  mkdir -p "$out/libexec/tt-metalium/build/tools/profiler/bin"
  ln -s ${tt-metal-tracy-tools}/bin/tracy-capture \
    "$out/libexec/tt-metalium/build/tools/profiler/bin/tracy-capture"
  ln -s ${tt-metal-tracy-tools}/bin/tracy-csvexport \
    "$out/libexec/tt-metalium/build/tools/profiler/bin/tracy-csvexport"
''
