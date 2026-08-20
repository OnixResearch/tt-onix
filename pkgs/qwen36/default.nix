{
  lib,
  stdenvNoCC,
  callPackage,
  fetchFromGitHub,
  fetchurl,
  python3,
  makeWrapper,
  nickel,
  tt-metal,
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
  ];
  nativeBuildInputs = [
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
      --add-flags "qwen36_service.server"

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
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/server.py"
    test -x "$out/bin/qwen36-p150x2-serve"

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
