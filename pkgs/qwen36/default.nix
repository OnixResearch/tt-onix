{
  lib,
  stdenvNoCC,
  callPackage,
  fetchFromGitHub,
  fetchurl,
  python3,
  makeWrapper,
  nickel,
  b3sum,
  tt-metal,
  tt-metal-tracy-tools,
}:
let
  source = callPackage ./source.nix { inherit fetchFromGitHub; };
  # r[impl tenstorrent.qwen36.upstream_authority]
  # Canonical adopted authority (committed, mirroring model-targets/qwen3-6).
  canonicalAuthority = builtins.fromJSON (builtins.readFile ./generated/authority.json);
  authorityValidator = import ./authority.nix {
    inherit
      lib
      canonicalAuthority
      ;
  };
  admittedAuthority = authorityValidator.admit canonicalAuthority;
  pythonEnvironment = python3.withPackages (packages: [
    # Runtime deps required by the packaged tt-metal 0.74 ttnn python import
    # (mirrors the tt-metal pyproject dependency list plus tracy requirements).
    packages.click
    packages.graphviz
    packages.loguru
    packages.ml-dtypes
    packages.networkx
    packages.numpy
    packages.pandas
    packages.pyyaml
    packages.seaborn
    packages.websockets
    # Model-stack deps (qwen36 + gated DeltaNet + tt_transformers + vLLM-adjacent).
    packages.pydantic
    packages.pytest
    packages.safetensors
    packages.sentencepiece
    packages.tiktoken
    packages.tokenizers
    packages.torch
    packages.transformers
  ]);
  ttnnSitePackages = "${tt-metal}/lib/python${python3.pythonVersion}/site-packages";
  metaliumRoot = "${tt-metal}/libexec/tt-metalium";
in
stdenvNoCC.mkDerivation {
  pname = "qwen36";
  # Adopted upstream state: tt-metal PR 53543 head (Qwen3.8-27B on qwen36).
  version = "0.1.0";

  src = source;
  dontConfigure = true;
  dontBuild = true;
  patches = [
    # r[impl tenstorrent.qwen36.p150x2_tp]
    # TP=2 (P150x2 / P300) support carried on top of the adopted PR: registers
    # the P150x2 mesh maps, the P300 prefill-chunk entries, and the host-sampling
    # advertisement that TP=2 requires (on-device sampling is certified only for
    # 1x4/1x8). Generated from the adopted tree; validates with patch -p1.
    ./p150x2-tp.patch
    # r[impl tenstorrent.qwen36.generate_tp_reset]
    # Hardware-found upstream bug: reset_tp() calls reset_state() on every
    # layer, but single-device full attention (Qwen36GatedAttention) only
    # exposes reset_cache(); only GDN and the TP wrappers expose reset_state.
    # The 9B single-device generate_tp oracle crashes without this dispatch.
    ./generate-tp-reset.patch
    # r[impl tenstorrent.qwen36.gdn_seq_fallback]
    # Hardware-found API drift: the adopted main-line code routes GDN TP
    # prefill through ttnn.transformer.chunk_gated_delta_rule, which the
    # packaged 0.74 bindings do not expose. The tree already carries a seq
    # adapter built on gated_delta_attn_seq (present in 0.74); this patch makes
    # the dispatch check the runtime instead of hardcoding the fused path.
    ./gdn-seq-fallback.patch
    # r[impl tenstorrent.qwen36.p150x2_runtime_compat]
    # Silicon-found 0.74 gaps: P150x2 waits in the TP=4-tuned GDN fused
    # matmul-reduce-scatter, and the MLP AGMM binding rejects fuse_swiglu.
    # Keep the optimized paths on capable runtimes and admitted larger meshes.
    ./tp2-runtime-compat.patch
    # r[impl tenstorrent.qwen38_p150x2.layer_type_authority]
    # The checkpoint layer_types list owns hybrid dispatch. The pure resolver
    # rejects incomplete metadata and removes the unused 32-layer mapper set.
    ./layer-type-authority.patch
  ];
  nativeBuildInputs = [
    b3sum
    makeWrapper
    nickel
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/lib/python${python3.pythonVersion}/site-packages"
    cp -r models "$out/lib/python${python3.pythonVersion}/site-packages/models"
    cp -r ${./service/qwen36_service} \
      "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service"
    # The tt-metal models tree carries upstream-relative symlinks whose targets
    # live outside models/ (e.g. tests/didt/... under the repo root). Prune the
    # dangling links so the installed tree is self-contained.
    find "$out/lib/python${python3.pythonVersion}/site-packages/models" \
      -type l ! -exec test -e {} \; -delete

    mkdir -p "$out/share/qwen36"
    cp ${./check.py} "$out/share/qwen36/check.py"
    cp ${./authority.ncl} "$out/share/qwen36/authority.ncl"
    cp ${./authority-contract.ncl} "$out/share/qwen36/authority-contract.ncl"
    cp ${./generated/authority.json} "$out/share/qwen36/authority.json"
    cp ${./decode-profile.ncl} "$out/share/qwen36/decode-profile.ncl"
    cp ${./decode-contracts.ncl} "$out/share/qwen36/decode-contracts.ncl"
    cp ${./mtp-probe-profile.ncl} "$out/share/qwen36/mtp-probe-profile.ncl"
    cp ${./mtp-probe-contracts.ncl} "$out/share/qwen36/mtp-probe-contracts.ncl"
    cp ${./mtp-drafter-profile.ncl} "$out/share/qwen36/mtp-drafter-profile.ncl"
    cp ${./mtp-drafter-contracts.ncl} "$out/share/qwen36/mtp-drafter-contracts.ncl"
    cp ${./mtp-selection-profile.ncl} "$out/share/qwen36/mtp-selection-profile.ncl"
    cp ${./mtp-selection-contracts.ncl} "$out/share/qwen36/mtp-selection-contracts.ncl"
    cp ${./mtp-trace-profile.ncl} "$out/share/qwen36/mtp-trace-profile.ncl"
    cp ${./mtp-trace-contracts.ncl} "$out/share/qwen36/mtp-trace-contracts.ncl"
    cp ${./mtp-rollback-profile.ncl} "$out/share/qwen36/mtp-rollback-profile.ncl"
    cp ${./mtp-rollback-contracts.ncl} "$out/share/qwen36/mtp-rollback-contracts.ncl"
    cp ${./decode-component-profile.ncl} "$out/share/qwen36/decode-component-profile.ncl"
    cp ${./decode-component-contracts.ncl} "$out/share/qwen36/decode-component-contracts.ncl"
    cp ${./ttft-control-profile.ncl} "$out/share/qwen36/ttft-control-profile.ncl"
    cp ${./ttft-control-contracts.ncl} "$out/share/qwen36/ttft-control-contracts.ncl"
    mkdir -p decode-profile-export
    cp ${./decode-profile.ncl} decode-profile-export/decode-profile.ncl
    cp ${./decode-contracts.ncl} decode-profile-export/decode-contracts.ncl
    nickel export --format json decode-profile-export/decode-profile.ncl \
      > "$out/share/qwen36/decode-profile.json"
    mkdir -p mtp-probe-profile-export
    cp ${./mtp-probe-profile.ncl} mtp-probe-profile-export/mtp-probe-profile.ncl
    cp ${./mtp-probe-contracts.ncl} mtp-probe-profile-export/mtp-probe-contracts.ncl
    nickel export --format json mtp-probe-profile-export/mtp-probe-profile.ncl \
      > "$out/share/qwen36/mtp-probe-profile.json"
    mkdir -p mtp-drafter-profile-export
    cp ${./mtp-drafter-profile.ncl} mtp-drafter-profile-export/mtp-drafter-profile.ncl
    cp ${./mtp-drafter-contracts.ncl} mtp-drafter-profile-export/mtp-drafter-contracts.ncl
    nickel export --format json mtp-drafter-profile-export/mtp-drafter-profile.ncl \
      > "$out/share/qwen36/mtp-drafter-profile.json"
    mkdir -p mtp-selection-profile-export
    cp ${./mtp-selection-profile.ncl} mtp-selection-profile-export/mtp-selection-profile.ncl
    cp ${./mtp-selection-contracts.ncl} mtp-selection-profile-export/mtp-selection-contracts.ncl
    nickel export --format json mtp-selection-profile-export/mtp-selection-profile.ncl \
      > "$out/share/qwen36/mtp-selection-profile.json"
    mkdir -p mtp-trace-profile-export
    cp ${./mtp-trace-profile.ncl} mtp-trace-profile-export/mtp-trace-profile.ncl
    cp ${./mtp-trace-contracts.ncl} mtp-trace-profile-export/mtp-trace-contracts.ncl
    nickel export --format json mtp-trace-profile-export/mtp-trace-profile.ncl \
      > "$out/share/qwen36/mtp-trace-profile.json"
    mkdir -p mtp-rollback-profile-export
    cp ${./mtp-rollback-profile.ncl} mtp-rollback-profile-export/mtp-rollback-profile.ncl
    cp ${./mtp-rollback-contracts.ncl} mtp-rollback-profile-export/mtp-rollback-contracts.ncl
    nickel export --format json mtp-rollback-profile-export/mtp-rollback-profile.ncl \
      > "$out/share/qwen36/mtp-rollback-profile.json"
    mkdir -p decode-component-profile-export
    cp ${./decode-component-profile.ncl} decode-component-profile-export/decode-component-profile.ncl
    cp ${./decode-component-contracts.ncl} decode-component-profile-export/decode-component-contracts.ncl
    nickel export --format json decode-component-profile-export/decode-component-profile.ncl \
      > "$out/share/qwen36/decode-component-profile.json"
    mkdir -p ttft-control-profile-export
    cp ${./ttft-control-profile.ncl} ttft-control-profile-export/ttft-control-profile.ncl
    cp ${./ttft-control-contracts.ncl} ttft-control-profile-export/ttft-control-contracts.ncl
    nickel export --format json ttft-control-profile-export/ttft-control-profile.ncl \
      > "$out/share/qwen36/ttft-control-profile.json"

    mkdir -p "$out/bin"

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
    mkdir -p "$out/libexec/tt-metalium/build/profiler/build_wasm/traces"
    # Host-side Tracy CLI tools matching the packaged runtime. The packaged
    # `python -m tracy` module resolves them under
    # $TT_METAL_HOME/build/tools/profiler/bin (see tracy/common.py).
    mkdir -p "$out/libexec/tt-metalium/build/tools/profiler/bin"
    ln -s ${tt-metal-tracy-tools}/bin/tracy-capture \
      "$out/libexec/tt-metalium/build/tools/profiler/bin/tracy-capture"
    ln -s ${tt-metal-tracy-tools}/bin/tracy-csvexport \
      "$out/libexec/tt-metalium/build/tools/profiler/bin/tracy-csvexport"
    overlayRoot="$out/libexec/tt-metalium"
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot"
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-check" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --add-flags "$out/share/qwen36/check.py"

    # The serialized P150x2 service owns request policy and uses the validated
    # paged prefill/decode path without an external vLLM dependency.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-serve" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.server" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--decode-mode" \
      --add-flags "eager" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.traced_decode]
    # The candidate wrapper is explicit. The production wrapper remains eager
    # until matched physical admission changes the typed profile.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-trace-candidate" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.server" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--decode-mode" \
      --add-flags "candidate" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_verification_probe]
    # This command is explicit and diagnostic-only. Production remains eager.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_probe" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-probe-profile" \
      --add-flags "$out/share/qwen36/mtp-probe-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_drafter_probe]
    # The drafter command loads checkpoint MTP weights only when invoked.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-drafter-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_draft_selection]
    # This paired control changes only diagnostic draft-token selection.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-selection-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-selection-profile" \
      --add-flags "$out/share/qwen36/mtp-selection-profile.json" \
      --add-flags "--mode" \
      --add-flags "selection" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_draft_trace]
    # This paired control traces only the diagnostic fixed-shape draft step.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-trace-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-trace-profile" \
      --add-flags "$out/share/qwen36/mtp-trace-profile.json" \
      --add-flags "--mode" \
      --add-flags "trace" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_kv_rollback]
    # This paired control compares bounded row restore with MTP re-prefill.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-rollback-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-trace-profile" \
      --add-flags "$out/share/qwen36/mtp-trace-profile.json" \
      --add-flags "--mtp-rollback-profile" \
      --add-flags "$out/share/qwen36/mtp-rollback-profile.json" \
      --add-flags "--mode" \
      --add-flags "rollback" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_recovery_prewarmed_control]
    # This command prewarms every prompt before the full persistent control.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-recovery-prewarmed-control" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-trace-profile" \
      --add-flags "$out/share/qwen36/mtp-trace-profile.json" \
      --add-flags "--mtp-rollback-profile" \
      --add-flags "$out/share/qwen36/mtp-rollback-profile.json" \
      --add-flags "--mode" \
      --add-flags "recovery-control" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]
    # This one-route command preserves atomic progress outside timing windows.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-recovery-stage" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-trace-profile" \
      --add-flags "$out/share/qwen36/mtp-trace-profile.json" \
      --add-flags "--mtp-rollback-profile" \
      --add-flags "$out/share/qwen36/mtp-rollback-profile.json" \
      --add-flags "--mode" \
      --add-flags "stage" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]
    # This command prewarms the exact target path before persistent recovery.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-mtp-recovery-prewarm-stage" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.mtp_drafter" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--mtp-drafter-profile" \
      --add-flags "$out/share/qwen36/mtp-drafter-profile.json" \
      --add-flags "--mtp-trace-profile" \
      --add-flags "$out/share/qwen36/mtp-trace-profile.json" \
      --add-flags "--mtp-rollback-profile" \
      --add-flags "$out/share/qwen36/mtp-rollback-profile.json" \
      --add-flags "--mode" \
      --add-flags "stage" \
      --add-flags "--stage-prewarm-target-validation" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.decode_component_attribution]
    # This explicit command measures diagnostic component traces only.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-decode-component-probe" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set HF_HUB_OFFLINE "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.decode_component_probe" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--component-profile" \
      --add-flags "$out/share/qwen36/decode-component-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # r[impl tenstorrent.qwen38_p150x2.traced_ttft_control]
    # This command compares stable prefill controls without changing production.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-ttft-control" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --set MESH_DEVICE "P150x2" \
      --set QWEN35_SCALE_INPUT "1" \
      --set QWEN35_TEMP "0" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --unset TT_METAL_SIMULATOR \
      --add-flags "-m" \
      --add-flags "qwen36_service.ttft_control_probe" \
      --add-flags "--decode-profile" \
      --add-flags "$out/share/qwen36/decode-profile.json" \
      --add-flags "--control-profile" \
      --add-flags "$out/share/qwen36/ttft-control-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags "$out"

    # qwen36-serve remains the adopted vLLM entry point. It fails closed unless
    # a text-only vLLM 0.10 is supplied by a compatible consumer environment.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-serve" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --set TT_METAL_HOME "$overlayRoot" \
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot" \
      --add-flags "-m" \
      --add-flags "vllm.entrypoints.openai.api_server"

    # Positive: the overlay provides the JIT headers and kernel sources the
    # packaged runtime omits (matmul_custom.h for SDPA; the experimental CCL
    # all-gather-fused-matmul kernels for the qwen36 TP path).
    test -f "$out/libexec/tt-metalium/tt_metal/hw/inc/api/compute/experimental/matmul_custom.h"
    test -f "$out/libexec/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp"

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    # Positive: installed layout is complete and hash-consistent.
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/model.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/qwen36_vllm.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/gdn/tp.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/mlp.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/tt_transformers/tt/model_config.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/prefill_profile_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/prefill_profile_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_probe_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_probe_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/native_mtp_verifier_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_probe.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_drafter_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_drafter_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_drafter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_selection_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_trace_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_trace_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_rollback_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_rollback_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_recovery_control_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_stage_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/mtp_target_prewarm_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_component_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_component_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_component_probe.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/ttft_control_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/ttft_control_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/ttft_control_probe.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/server.py"
    test -f "$out/share/qwen36/decode-profile.json"
    test -f "$out/share/qwen36/mtp-probe-profile.json"
    test -f "$out/share/qwen36/mtp-drafter-profile.json"
    test -f "$out/share/qwen36/mtp-selection-profile.json"
    test -f "$out/share/qwen36/mtp-trace-profile.json"
    test -f "$out/share/qwen36/mtp-rollback-profile.json"
    test -f "$out/share/qwen36/decode-component-profile.json"
    test -f "$out/share/qwen36/ttft-control-profile.json"
    test -x "$out/bin/qwen36-p150x2-serve"
    test -x "$out/bin/qwen36-p150x2-trace-candidate"
    test -x "$out/bin/qwen36-p150x2-mtp-probe"
    test -x "$out/bin/qwen36-p150x2-mtp-drafter-probe"
    test -x "$out/bin/qwen36-p150x2-mtp-selection-probe"
    test -x "$out/bin/qwen36-p150x2-mtp-trace-probe"
    test -x "$out/bin/qwen36-p150x2-mtp-rollback-probe"
    test -x "$out/bin/qwen36-p150x2-mtp-recovery-prewarmed-control"
    test -x "$out/bin/qwen36-p150x2-mtp-recovery-stage"
    test -x "$out/bin/qwen36-p150x2-mtp-recovery-prewarm-stage"
    test -x "$out/bin/qwen36-p150x2-decode-component-probe"
    test -x "$out/bin/qwen36-p150x2-ttft-control"
    "$out/bin/qwen36-p150x2-mtp-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-drafter-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-selection-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-trace-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-rollback-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-recovery-prewarmed-control" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-recovery-stage" --help > /dev/null
    "$out/bin/qwen36-p150x2-mtp-recovery-prewarm-stage" --help > /dev/null
    "$out/bin/qwen36-p150x2-decode-component-probe" --help > /dev/null
    "$out/bin/qwen36-p150x2-ttft-control" --help > /dev/null

    # Positive: the adopted 9-file PR scope is present in the installed tree.
    for f in \
      models/demos/blackhole/qwen36/README.md \
      models/demos/blackhole/qwen36/demo/sample_prompts/.gitignore \
      models/demos/blackhole/qwen36/demo/text_demo.py \
      models/demos/blackhole/qwen36/tt/gdn/tp.py \
      models/demos/blackhole/qwen36/tt/gdn/weights.py \
      models/demos/blackhole/qwen36/tt/mlp.py \
      models/demos/blackhole/qwen36/tt/model.py \
      models/demos/blackhole/qwen36/tt/qwen36_vllm.py \
      models/tt_transformers/tt/model_config.py; do
      test -f "$out/lib/python${python3.pythonVersion}/site-packages/$f" || {
        echo "adopted PR file missing: $f" >&2
        exit 1
      }
    done

    # Positive: authority.ncl export is deterministic and equals the committed
    # generated authority used for eval-time admission. The export runs from a
    # working directory that also holds the relative authority-contract.ncl
    # import, so the store-path copy of authority.ncl resolves it.
    mkdir -p authority-check
    cp ${./authority.ncl} authority-check/authority.ncl
    cp ${./authority-contract.ncl} authority-check/authority-contract.ncl
    nickel export --format json authority-check/authority.ncl > authority-first.json
    nickel export --format json authority-check/authority.ncl > authority-second.json
    cmp authority-first.json authority-second.json
    cmp authority-first.json ${./generated/authority.json}

    # r[verify tenstorrent.qwen38_p150x2.traced_decode]
    # Positive: the candidate profile exports deterministically.
    mkdir -p decode-profile-check
    cp ${./decode-profile.ncl} decode-profile-check/decode-profile.ncl
    cp ${./decode-contracts.ncl} decode-profile-check/decode-contracts.ncl
    nickel export --format json decode-profile-check/decode-profile.ncl > decode-profile-first.json
    nickel export --format json decode-profile-check/decode-profile.ncl > decode-profile-second.json
    cmp decode-profile-first.json decode-profile-second.json
    cmp decode-profile-first.json "$out/share/qwen36/decode-profile.json"
    grep -F '"physically_admitted": false' decode-profile-first.json

    # r[verify tenstorrent.qwen38_p150x2.traced_decode.fallback]
    # Negative: an invalid slot count must fail its typed contract.
    mkdir -p decode-profile-negative
    cp ${./decode-profile.ncl} decode-profile-negative/decode-profile.ncl
    cp ${./decode-contracts.ncl} decode-profile-negative/decode-contracts.ncl
    cp ${./fixtures/invalid-decode-profile.ncl} decode-profile-negative/invalid-decode-profile.ncl
    if nickel export --format json decode-profile-negative/invalid-decode-profile.ncl \
        > invalid-decode-profile.json 2> invalid-decode-profile.log; then
      echo "invalid decode profile exported" >&2
      exit 1
    fi
    test -s invalid-decode-profile.log

    # r[verify tenstorrent.qwen38_p150x2.mtp_verification_probe]
    # Positive: the diagnostic target-probe profile exports deterministically.
    mkdir -p mtp-probe-profile-check
    cp ${./mtp-probe-profile.ncl} mtp-probe-profile-check/mtp-probe-profile.ncl
    cp ${./mtp-probe-contracts.ncl} mtp-probe-profile-check/mtp-probe-contracts.ncl
    nickel export --format json mtp-probe-profile-check/mtp-probe-profile.ncl > mtp-probe-profile-first.json
    nickel export --format json mtp-probe-profile-check/mtp-probe-profile.ncl > mtp-probe-profile-second.json
    cmp mtp-probe-profile-first.json mtp-probe-profile-second.json
    cmp mtp-probe-profile-first.json "$out/share/qwen36/mtp-probe-profile.json"
    grep -F '"diagnostic_only": true' mtp-probe-profile-first.json
    authority_blake3="$(b3sum authority-first.json | cut -d ' ' -f 1)"
    grep -F "\"authority_blake3\": \"$authority_blake3\"" decode-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" mtp-probe-profile-first.json

    # Negative: a changed width set or admission mode must fail its typed contract.
    mkdir -p mtp-probe-profile-negative
    cp ${./mtp-probe-profile.ncl} mtp-probe-profile-negative/mtp-probe-profile.ncl
    cp ${./mtp-probe-contracts.ncl} mtp-probe-profile-negative/mtp-probe-contracts.ncl
    cp ${./fixtures/invalid-mtp-probe-profile.ncl} mtp-probe-profile-negative/invalid-mtp-probe-profile.ncl
    if nickel export --format json mtp-probe-profile-negative/invalid-mtp-probe-profile.ncl \
        > invalid-mtp-probe-profile.json 2> invalid-mtp-probe-profile.log; then
      echo "invalid MTP probe profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-probe-profile.log

    # r[verify tenstorrent.qwen38_p150x2.mtp_drafter_probe]
    # Positive: the diagnostic drafter profile exports deterministically.
    mkdir -p mtp-drafter-profile-check
    cp ${./mtp-drafter-profile.ncl} mtp-drafter-profile-check/mtp-drafter-profile.ncl
    cp ${./mtp-drafter-contracts.ncl} mtp-drafter-profile-check/mtp-drafter-contracts.ncl
    nickel export --format json mtp-drafter-profile-check/mtp-drafter-profile.ncl > mtp-drafter-profile-first.json
    nickel export --format json mtp-drafter-profile-check/mtp-drafter-profile.ncl > mtp-drafter-profile-second.json
    cmp mtp-drafter-profile-first.json mtp-drafter-profile-second.json
    cmp mtp-drafter-profile-first.json "$out/share/qwen36/mtp-drafter-profile.json"
    grep -F '"draft_token_count": 3' mtp-drafter-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" mtp-drafter-profile-first.json

    # Negative: weakened drafter bounds must fail the typed contract.
    mkdir -p mtp-drafter-profile-negative
    cp ${./mtp-drafter-profile.ncl} mtp-drafter-profile-negative/mtp-drafter-profile.ncl
    cp ${./mtp-drafter-contracts.ncl} mtp-drafter-profile-negative/mtp-drafter-contracts.ncl
    cp ${./fixtures/invalid-mtp-drafter-profile.ncl} mtp-drafter-profile-negative/invalid-mtp-drafter-profile.ncl
    if nickel export --format json mtp-drafter-profile-negative/invalid-mtp-drafter-profile.ncl \
        > invalid-mtp-drafter-profile.json 2> invalid-mtp-drafter-profile.log; then
      echo "invalid MTP drafter profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-drafter-profile.log

    # r[verify tenstorrent.qwen38_p150x2.mtp_draft_selection]
    # Positive: the paired selector profile exports deterministically.
    mkdir -p mtp-selection-profile-check
    cp ${./mtp-selection-profile.ncl} mtp-selection-profile-check/mtp-selection-profile.ncl
    cp ${./mtp-selection-contracts.ncl} mtp-selection-profile-check/mtp-selection-contracts.ncl
    nickel export --format json mtp-selection-profile-check/mtp-selection-profile.ncl > mtp-selection-profile-first.json
    nickel export --format json mtp-selection-profile-check/mtp-selection-profile.ncl > mtp-selection-profile-second.json
    cmp mtp-selection-profile-first.json mtp-selection-profile-second.json
    cmp mtp-selection-profile-first.json "$out/share/qwen36/mtp-selection-profile.json"
    grep -F '"maximum_selector_ratio": 0.95' mtp-selection-profile-first.json
    grep -F '"shard-local-top1"' mtp-selection-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" mtp-selection-profile-first.json

    # Negative: changed route order must fail the typed selector contract.
    mkdir -p mtp-selection-profile-negative/fixtures
    cp ${./mtp-selection-profile.ncl} mtp-selection-profile-negative/mtp-selection-profile.ncl
    cp ${./mtp-selection-contracts.ncl} mtp-selection-profile-negative/mtp-selection-contracts.ncl
    cp ${./fixtures/invalid-mtp-selection-profile.ncl} mtp-selection-profile-negative/fixtures/invalid-mtp-selection-profile.ncl
    if nickel export --format json mtp-selection-profile-negative/fixtures/invalid-mtp-selection-profile.ncl \
        > invalid-mtp-selection-profile.json 2> invalid-mtp-selection-profile.log; then
      echo "invalid MTP selection profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-selection-profile.log

    # r[verify tenstorrent.qwen38_p150x2.mtp_draft_trace]
    # Positive: the matched trace profile exports deterministically.
    mkdir -p mtp-trace-profile-check
    cp ${./mtp-trace-profile.ncl} mtp-trace-profile-check/mtp-trace-profile.ncl
    cp ${./mtp-trace-contracts.ncl} mtp-trace-profile-check/mtp-trace-contracts.ncl
    nickel export --format json mtp-trace-profile-check/mtp-trace-profile.ncl > mtp-trace-profile-first.json
    nickel export --format json mtp-trace-profile-check/mtp-trace-profile.ncl > mtp-trace-profile-second.json
    cmp mtp-trace-profile-first.json mtp-trace-profile-second.json
    cmp mtp-trace-profile-first.json "$out/share/qwen36/mtp-trace-profile.json"
    grep -F '"minimum_samples_per_route": 6' mtp-trace-profile-first.json
    grep -F '"maximum_traced_draft_ratio": 0.95' mtp-trace-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" mtp-trace-profile-first.json

    # Negative: fewer retained samples must fail the typed trace contract.
    mkdir -p mtp-trace-profile-negative/fixtures
    cp ${./mtp-trace-profile.ncl} mtp-trace-profile-negative/mtp-trace-profile.ncl
    cp ${./mtp-trace-contracts.ncl} mtp-trace-profile-negative/mtp-trace-contracts.ncl
    cp ${./fixtures/invalid-mtp-trace-profile.ncl} mtp-trace-profile-negative/fixtures/invalid-mtp-trace-profile.ncl
    if nickel export --format json mtp-trace-profile-negative/fixtures/invalid-mtp-trace-profile.ncl \
        > invalid-mtp-trace-profile.json 2> invalid-mtp-trace-profile.log; then
      echo "invalid MTP trace profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-trace-profile.log

    # r[verify tenstorrent.qwen38_p150x2.mtp_kv_rollback]
    # r[verify tenstorrent.qwen38_p150x2.mtp_recovery_stability]
    # Positive: the matched persistent recovery profile exports deterministically.
    mkdir -p mtp-rollback-profile-check
    cp ${./mtp-rollback-profile.ncl} mtp-rollback-profile-check/mtp-rollback-profile.ncl
    cp ${./mtp-rollback-contracts.ncl} mtp-rollback-profile-check/mtp-rollback-contracts.ncl
    nickel export --format json mtp-rollback-profile-check/mtp-rollback-profile.ncl > mtp-rollback-profile-first.json
    nickel export --format json mtp-rollback-profile-check/mtp-rollback-profile.ncl > mtp-rollback-profile-second.json
    cmp mtp-rollback-profile-first.json mtp-rollback-profile-second.json
    cmp mtp-rollback-profile-first.json "$out/share/qwen36/mtp-rollback-profile.json"
    grep -F '"schema_version": 2' mtp-rollback-profile-first.json
    grep -F '"recovery_execution_mode": "persistent-traced"' mtp-rollback-profile-first.json
    grep -F '"required_process_trace_capture_count": 1' mtp-rollback-profile-first.json
    grep -F '"snapshot_row_count": 2' mtp-rollback-profile-first.json
    grep -F '"maximum_snapshot_recovery_ratio": 0.8' mtp-rollback-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" mtp-rollback-profile-first.json

    # Negative: a full-cache row count must fail the typed rollback contract.
    mkdir -p mtp-rollback-profile-negative/fixtures
    cp ${./mtp-rollback-profile.ncl} mtp-rollback-profile-negative/mtp-rollback-profile.ncl
    cp ${./mtp-rollback-contracts.ncl} mtp-rollback-profile-negative/mtp-rollback-contracts.ncl
    cp ${./fixtures/invalid-mtp-rollback-profile.ncl} mtp-rollback-profile-negative/fixtures/invalid-mtp-rollback-profile.ncl
    if nickel export --format json mtp-rollback-profile-negative/fixtures/invalid-mtp-rollback-profile.ncl \
        > invalid-mtp-rollback-profile.json 2> invalid-mtp-rollback-profile.log; then
      echo "invalid MTP rollback profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-rollback-profile.log

    # Negative: more than one trace capture must fail the stability contract.
    cp ${./fixtures/invalid-mtp-recovery-stability-profile.ncl} \
      mtp-rollback-profile-negative/fixtures/invalid-mtp-recovery-stability-profile.ncl
    if nickel export --format json mtp-rollback-profile-negative/fixtures/invalid-mtp-recovery-stability-profile.ncl \
        > invalid-mtp-recovery-stability-profile.json 2> invalid-mtp-recovery-stability-profile.log; then
      echo "invalid MTP recovery stability profile exported" >&2
      exit 1
    fi
    test -s invalid-mtp-recovery-stability-profile.log

    # r[verify tenstorrent.qwen38_p150x2.decode_component_attribution]
    # Positive: component policy exports deterministically and binds authority.
    mkdir -p decode-component-profile-check
    cp ${./decode-component-profile.ncl} decode-component-profile-check/decode-component-profile.ncl
    cp ${./decode-component-contracts.ncl} decode-component-profile-check/decode-component-contracts.ncl
    nickel export --format json decode-component-profile-check/decode-component-profile.ncl > decode-component-profile-first.json
    nickel export --format json decode-component-profile-check/decode-component-profile.ncl > decode-component-profile-second.json
    cmp decode-component-profile-first.json decode-component-profile-second.json
    cmp decode-component-profile-first.json "$out/share/qwen36/decode-component-profile.json"
    grep -F '"diagnostic_only": true' decode-component-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" decode-component-profile-first.json

    # Negative: production eligibility must fail the typed component contract.
    mkdir -p decode-component-profile-negative
    cp ${./decode-component-profile.ncl} decode-component-profile-negative/decode-component-profile.ncl
    cp ${./decode-component-contracts.ncl} decode-component-profile-negative/decode-component-contracts.ncl
    cp ${./fixtures/invalid-decode-component-profile.ncl} decode-component-profile-negative/invalid-decode-component-profile.ncl
    if nickel export --format json decode-component-profile-negative/invalid-decode-component-profile.ncl \
        > invalid-decode-component-profile.json 2> invalid-decode-component-profile.log; then
      echo "invalid decode component profile exported" >&2
      exit 1
    fi
    test -s invalid-decode-component-profile.log

    # r[verify tenstorrent.qwen38_p150x2.traced_ttft_control]
    # Positive: the TTFT control profile exports deterministically and binds authority.
    mkdir -p ttft-control-profile-check
    cp ${./ttft-control-profile.ncl} ttft-control-profile-check/ttft-control-profile.ncl
    cp ${./ttft-control-contracts.ncl} ttft-control-profile-check/ttft-control-contracts.ncl
    nickel export --format json ttft-control-profile-check/ttft-control-profile.ncl > ttft-control-profile-first.json
    nickel export --format json ttft-control-profile-check/ttft-control-profile.ncl > ttft-control-profile-second.json
    cmp ttft-control-profile-first.json ttft-control-profile-second.json
    cmp ttft-control-profile-first.json "$out/share/qwen36/ttft-control-profile.json"
    grep -F '"diagnostic_only": true' ttft-control-profile-first.json
    grep -F '"material_regression_ratio": 1.05' ttft-control-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" ttft-control-profile-first.json

    # Negative: a production-eligible TTFT control must fail its typed contract.
    mkdir -p ttft-control-profile-negative
    cp ${./ttft-control-profile.ncl} ttft-control-profile-negative/ttft-control-profile.ncl
    cp ${./ttft-control-contracts.ncl} ttft-control-profile-negative/ttft-control-contracts.ncl
    cp ${./fixtures/invalid-ttft-control-profile.ncl} ttft-control-profile-negative/invalid-ttft-control-profile.ncl
    if nickel export --format json ttft-control-profile-negative/invalid-ttft-control-profile.ncl \
        > invalid-ttft-control-profile.json 2> invalid-ttft-control-profile.log; then
      echo "invalid TTFT control profile exported" >&2
      exit 1
    fi
    test -s invalid-ttft-control-profile.log

    # Negative: a drifted authority (wrong upstream commit) must be rejected by
    # the Nickel contract shape (commit no longer a 40-hex revision).
    printf '%s\n' \
      'let a = import "authority-check/authority.ncl" in' \
      'a with { upstream = a.upstream & { commit = "deadbeef" } }' \
      > authority-drifted.ncl
    if nickel export --format json authority-drifted.ncl > /dev/null 2> drifted.log; then
      echo "drifted authority exported despite a malformed commit" >&2
      exit 1
    fi
    grep -F "commit" drifted.log

    # Negative: an empty artifact list must be rejected by the contract.
    printf '%s\n' \
      'let a = import "authority-check/authority.ncl" in' \
      'a with { release = a.release & { artifacts = [] } }' \
      > authority-no-artifacts.ncl
    if nickel export --format json authority-no-artifacts.ncl > /dev/null 2> no-artifacts.log; then
      echo "authority with no artifacts exported" >&2
      exit 1
    fi
    grep -F "artifacts" no-artifacts.log

    # Positive: the AdmittedAuthority eval gate passed (this derivation only
    # evaluates when the committed authority exactly matches the validator).
    test -n "${toString admittedAuthority.schema_version}"

    # Positive: device-free check of the adopted stack against the packaged
    # tt-metal runtime.
    "$out/bin/qwen36-check"

    # Negative: env-gated BF4 cache names must never collide; the check script
    # itself rejects a missing .bfp4 distinction.
    runHook postInstallCheck
  '';

  passthru = {
    inherit
      authorityValidator
      canonicalAuthority
      pythonEnvironment
      source
      ttnnSitePackages
      ;
  };

  meta = {
    description = "Qwen3.8-27B on the qwen36 Blackhole path (tt-metal PR 53543, adopted)";
    homepage = "https://github.com/tenstorrent/tt-metal/pull/53543";
    license = lib.licenses.asl20;
    mainProgram = "qwen36-check";
    platforms = lib.platforms.linux;
  };
}
