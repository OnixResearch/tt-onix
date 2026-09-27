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
  # True only for the GDN grid variant runtime, whose kernels carry the
  # per-core instance chunking. The kernel-content assertions below fail
  # closed in both directions, so neither runtime can silently gain or lose
  # the chunked shape without breaking its package build.
  gdnGridRuntime ? false,
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
    packages.blake3
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
  executionPatches = [
    ./p150x2-tp.patch
    ./generate-tp-reset.patch
    ./gdn-seq-fallback.patch
    ./tp2-runtime-compat.patch
    ./layer-type-authority.patch
    ./gdn-decode-layout-candidate.patch
    ./prefill-device-staging.patch
    ./gdn-decode-fusion-candidate.patch
    ./prefill-runtime-admission.patch
    ./grouped-prefill-row-assembly.patch
    ./gdn-decode-step.patch
    ./gdn-fused-prefill-gate.patch
    ./eager-prefill-chunk.patch
    ./grouped-prefill-exact.patch
    ./masked-prefill-traces.patch
    ./structured-read-rows.patch
    ./prefix-cache.patch
  ];
  stableIdentityPath = name: path: builtins.path { inherit name path; };
  # The service source must never carry interpreter bytecode: local path:
  # builds copy __pycache__ verbatim while git flakes filter it, which made
  # locally screened identities diverge from deployed ones.
  stableServiceSource = lib.sources.cleanSourceWith {
    src = ./service/qwen36_service;
    name = "qwen36-service-source";
    filter =
      path: _type: !lib.hasInfix "__pycache__" (toString path) && !lib.hasSuffix ".pyc" (toString path);
  };
  executionIdentityLocalInputs = [
    (stableIdentityPath "qwen36-service" stableServiceSource)
    (stableIdentityPath "qwen36-execution-wrapper-contract.ncl" ./execution-wrapper-contract.ncl)
    (stableIdentityPath "qwen36-authority.ncl" ./authority.ncl)
    (stableIdentityPath "qwen36-authority-contract.ncl" ./authority-contract.ncl)
    (stableIdentityPath "qwen36-authority.json" ./generated/authority.json)
    (stableIdentityPath "qwen36-decode-profile.ncl" ./decode-profile.ncl)
    (stableIdentityPath "qwen36-decode-contracts.ncl" ./decode-contracts.ncl)
    (stableIdentityPath "qwen36-batch-admission-policy.ncl" ./batch-admission-policy.ncl)
    (stableIdentityPath "qwen36-batch-admission-policy-validated.ncl" ./batch-admission-policy-validated.ncl)
    (stableIdentityPath "qwen36-batch-admission-contracts.ncl" ./batch-admission-contracts.ncl)
    (stableIdentityPath "qwen36-batch-admission-profile.ncl" ./batch-admission-profile.ncl)
    (stableIdentityPath "qwen36-batch-admission-b8-policy.ncl" ./batch-admission-b8-policy.ncl)
    (stableIdentityPath "qwen36-batch-admission-b8-policy-validated.ncl" ./batch-admission-b8-policy-validated.ncl)
    (stableIdentityPath "qwen36-batch-admission-b8-contracts.ncl" ./batch-admission-b8-contracts.ncl)
    (stableIdentityPath "qwen36-batch-admission-b8-profile.ncl" ./batch-admission-b8-profile.ncl)
    (stableIdentityPath "qwen36-batch-admission-b16-policy.ncl" ./batch-admission-b16-policy.ncl)
    (stableIdentityPath "qwen36-batch-admission-b16-policy-validated.ncl" ./batch-admission-b16-policy-validated.ncl)
    (stableIdentityPath "qwen36-batch-admission-b16-contracts.ncl" ./batch-admission-b16-contracts.ncl)
    (stableIdentityPath "qwen36-batch-admission-b16-profile.ncl" ./batch-admission-b16-profile.ncl)
    (stableIdentityPath "qwen36-trace-reuse-policy.ncl" ./trace-reuse-policy.ncl)
    (stableIdentityPath "qwen36-trace-reuse-policy-validated.ncl" ./trace-reuse-policy-validated.ncl)
    (stableIdentityPath "qwen36-trace-reuse-contracts.ncl" ./trace-reuse-contracts.ncl)
    (stableIdentityPath "qwen36-trace-reuse-profile.ncl" ./trace-reuse-profile.ncl)
    (stableIdentityPath "qwen36-gdn-decode-fusion-contracts.ncl" ./gdn-decode-fusion-contracts.ncl)
    (stableIdentityPath "qwen36-gdn-decode-fusion-profile.ncl" ./gdn-decode-fusion-profile.ncl)
    (stableIdentityPath "qwen36-gdn-occupancy-policy.ncl" ./gdn-occupancy-policy.ncl)
    (stableIdentityPath "qwen36-continuous-slot-policy.ncl" ./continuous-slot-policy.ncl)
    (stableIdentityPath "qwen36-continuous-slot-contracts.ncl" ./continuous-slot-contracts.ncl)
    (stableIdentityPath "qwen36-continuous-slot-admission-policy.ncl" ./continuous-slot-admission-policy.ncl)
    (stableIdentityPath "qwen36-continuous-slot-admission-contracts.ncl" ./continuous-slot-admission-contracts.ncl)
    (stableIdentityPath "qwen36-prefix-cache-policy.ncl" ./prefix-cache-policy.ncl)
    (stableIdentityPath "qwen36-prefix-cache-contracts.ncl" ./prefix-cache-contracts.ncl)
    (stableIdentityPath "qwen36-prefix-cache-admission-policy.ncl" ./prefix-cache-admission-policy.ncl)
    (stableIdentityPath "qwen36-prefix-cache-admission-contracts.ncl" ./prefix-cache-admission-contracts.ncl)
    (stableIdentityPath "qwen36-kernel-distribution-admission-contracts.ncl" ./kernel-distribution-admission-contracts.ncl)
    (stableIdentityPath "qwen36-kernel-distribution-admission-policy.ncl" ./kernel-distribution-admission-policy.ncl)
    (stableIdentityPath "qwen36-kernel-distribution-admission-policy-validated.ncl" ./kernel-distribution-admission-policy-validated.ncl)
  ];
  executionIdentityPatchInputs = map (
    patch: stableIdentityPath "qwen36-${builtins.baseNameOf (toString patch)}" patch
  ) executionPatches;
  executionIdentityInputs =
    executionIdentityLocalInputs
    ++ [
      source
      pythonEnvironment
      tt-metal
      tt-metal-tracy-tools
      makeWrapper
    ]
    ++ executionIdentityPatchInputs;
  mkExecutionIdentity =
    inputs:
    let
      manifestLines = [
        "schema=onix.qwen36.execution-identity.v1"
      ]
      ++ map (input: "input=${toString input}") inputs;
    in
    stdenvNoCC.mkDerivation {
      pname = "qwen36-execution-identity";
      version = "0.1.0";
      dontUnpack = true;
      installPhase = ''
        runHook preInstall
        mkdir -p "$out/share/qwen36"
        printf '%s\n' ${lib.concatMapStringsSep " " lib.escapeShellArg manifestLines} \
          > "$out/share/qwen36/execution-inputs"
        runHook postInstall
      '';
    };
  executionIdentity = mkExecutionIdentity executionIdentityInputs;
  executionIdentityAfterMetadataPromotion = mkExecutionIdentity executionIdentityInputs;
  executionIdentityWithDrift = mkExecutionIdentity (
    executionIdentityInputs
    ++ [
      (stableIdentityPath "qwen36-execution-identity-drift.txt" ./fixtures/execution-identity-drift.txt)
    ]
  );
in
assert lib.assertMsg (
  toString executionIdentity == toString executionIdentityAfterMetadataPromotion
) "receipt-only promotion changed the Qwen execution identity";
assert lib.assertMsg (
  toString executionIdentity != toString executionIdentityWithDrift
) "an execution input failed to change the Qwen execution identity";
stdenvNoCC.mkDerivation {
  pname = "qwen38";
  # Adopted upstream state: tt-metal PR 53543 head (Qwen3.8-27B on qwen36).
  version = "0.1.0";

  src = source;
  dontConfigure = true;
  dontBuild = true;
  # Declares the expected kernel shape so the install assertions bind the
  # package execution identity to the runtime it actually carries.
  gdnGridRuntimeFlag = if gdnGridRuntime then "1" else "0";
  # Every patch can change measured execution and is part of the stable identity.
  patches = executionPatches;
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
    cp ${./mtp-kv-tail-device-check.py} "$out/share/qwen36/mtp-kv-tail-device-check.py"
    cp ${./authority.ncl} "$out/share/qwen36/authority.ncl"
    cp ${./authority-contract.ncl} "$out/share/qwen36/authority-contract.ncl"
    cp ${./generated/authority.json} "$out/share/qwen36/authority.json"
    cp ${./execution-wrapper-contract.ncl} "$out/share/qwen36/execution-wrapper-contract.ncl"
    cp ${./decode-profile.ncl} "$out/share/qwen36/decode-profile.ncl"
    cp ${./decode-contracts.ncl} "$out/share/qwen36/decode-contracts.ncl"
    cp ${./batch-admission-profile.ncl} "$out/share/qwen36/batch-admission-profile.ncl"
    cp ${./batch-admission-policy.ncl} "$out/share/qwen36/batch-admission-policy.ncl"
    cp ${./batch-admission-policy-validated.ncl} "$out/share/qwen36/batch-admission-policy-validated.ncl"
    cp ${./batch-admission-binding.ncl} "$out/share/qwen36/batch-admission-binding.ncl"
    cp ${./batch-admission-binding-validated.ncl} "$out/share/qwen36/batch-admission-binding-validated.ncl"
    cp ${./batch-admission-contracts.ncl} "$out/share/qwen36/batch-admission-contracts.ncl"
    cp ${./kernel-distribution-admission-contracts.ncl} "$out/share/qwen36/kernel-distribution-admission-contracts.ncl"
    cp ${./kernel-distribution-admission-policy.ncl} "$out/share/qwen36/kernel-distribution-admission-policy.ncl"
    cp ${./kernel-distribution-admission-policy-validated.ncl} "$out/share/qwen36/kernel-distribution-admission-policy-validated.ncl"
    cp ${./batch-admission-b8-profile.ncl} "$out/share/qwen36/batch-admission-b8-profile.ncl"
    cp ${./batch-admission-b16-policy.ncl} "$out/share/qwen36/batch-admission-b16-policy.ncl"
    cp ${./batch-admission-b16-policy-validated.ncl} "$out/share/qwen36/batch-admission-b16-policy-validated.ncl"
    cp ${./batch-admission-b16-binding.ncl} "$out/share/qwen36/batch-admission-b16-binding.ncl"
    cp ${./batch-admission-b16-binding-validated.ncl} "$out/share/qwen36/batch-admission-b16-binding-validated.ncl"
    cp ${./batch-admission-b16-contracts.ncl} "$out/share/qwen36/batch-admission-b16-contracts.ncl"
    cp ${./batch-admission-b16-profile.ncl} "$out/share/qwen36/batch-admission-b16-profile.ncl"
    cp ${./batch-admission-b8-policy.ncl} "$out/share/qwen36/batch-admission-b8-policy.ncl"
    cp ${./batch-admission-b8-policy-validated.ncl} "$out/share/qwen36/batch-admission-b8-policy-validated.ncl"
    cp ${./batch-admission-b8-binding.ncl} "$out/share/qwen36/batch-admission-b8-binding.ncl"
    cp ${./batch-admission-b8-binding-validated.ncl} "$out/share/qwen36/batch-admission-b8-binding-validated.ncl"
    cp ${./batch-admission-b8-contracts.ncl} "$out/share/qwen36/batch-admission-b8-contracts.ncl"
    cp ${./trace-reuse-profile.ncl} "$out/share/qwen36/trace-reuse-profile.ncl"
    cp ${./trace-reuse-policy.ncl} "$out/share/qwen36/trace-reuse-policy.ncl"
    cp ${./trace-reuse-policy-validated.ncl} "$out/share/qwen36/trace-reuse-policy-validated.ncl"
    cp ${./trace-reuse-binding.ncl} "$out/share/qwen36/trace-reuse-binding.ncl"
    cp ${./trace-reuse-binding-validated.ncl} "$out/share/qwen36/trace-reuse-binding-validated.ncl"
    cp ${./trace-reuse-contracts.ncl} "$out/share/qwen36/trace-reuse-contracts.ncl"
    ln -s ${executionIdentity} "$out/share/qwen36/execution-identity"
    # The service derives grouped-prefill eligibility from what it actually
    # links. Only the GDN grid variant carries the chunked scan instance loop,
    # so this record is the one capability fact the package itself knows.
    gdn_scan_chunked=false
    if [ "$gdnGridRuntimeFlag" = "1" ]; then gdn_scan_chunked=true; fi
    printf '%s\n' \
      '{' \
      '  "schema_version": 1,' \
      "  \"gdn_scan_chunked_instances\": $gdn_scan_chunked" \
      '}' > "$out/share/qwen36/runtime-capability.json"
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
    cp ${./gdn-decode-fusion-profile.ncl} "$out/share/qwen36/gdn-decode-fusion-profile.ncl"
    cp ${./gdn-decode-fusion-contracts.ncl} "$out/share/qwen36/gdn-decode-fusion-contracts.ncl"
    cp ${./gdn-occupancy-policy.ncl} "$out/share/qwen36/gdn-occupancy-policy.ncl"
    nickel export --format json ${./gdn-occupancy-policy.ncl} \
      > "$out/share/qwen36/gdn-occupancy-policy.json"
    mkdir -p continuous-slot-policy-export
    cp ${./continuous-slot-policy.ncl} continuous-slot-policy-export/continuous-slot-policy.ncl
    cp ${./continuous-slot-contracts.ncl} continuous-slot-policy-export/continuous-slot-contracts.ncl
    cp continuous-slot-policy-export/*.ncl "$out/share/qwen36/"
    nickel export --format json continuous-slot-policy-export/continuous-slot-policy.ncl \
      > "$out/share/qwen36/continuous-slot-policy.json"
    cp ${./continuous-slot-admission-policy.ncl} continuous-slot-policy-export/continuous-slot-admission-policy.ncl
    cp ${./continuous-slot-admission-contracts.ncl} continuous-slot-policy-export/continuous-slot-admission-contracts.ncl
    cp continuous-slot-policy-export/continuous-slot-admission-*.ncl "$out/share/qwen36/"
    nickel export --format json continuous-slot-policy-export/continuous-slot-admission-policy.ncl \
      > "$out/share/qwen36/continuous-slot-admission-policy.json"
    cp ${./prefix-cache-policy.ncl} continuous-slot-policy-export/prefix-cache-policy.ncl
    cp ${./prefix-cache-contracts.ncl} continuous-slot-policy-export/prefix-cache-contracts.ncl
    cp continuous-slot-policy-export/prefix-cache-*.ncl "$out/share/qwen36/"
    nickel export --format json continuous-slot-policy-export/prefix-cache-policy.ncl \
      > "$out/share/qwen36/prefix-cache-policy.json"
    cp ${./prefix-cache-admission-policy.ncl} continuous-slot-policy-export/prefix-cache-admission-policy.ncl
    cp ${./prefix-cache-admission-contracts.ncl} continuous-slot-policy-export/prefix-cache-admission-contracts.ncl
    cp continuous-slot-policy-export/prefix-cache-admission-*.ncl "$out/share/qwen36/"
    nickel export --format json continuous-slot-policy-export/prefix-cache-admission-policy.ncl \
      > "$out/share/qwen36/prefix-cache-admission-policy.json"
    cp ${./ttft-control-profile.ncl} "$out/share/qwen36/ttft-control-profile.ncl"
    cp ${./ttft-control-contracts.ncl} "$out/share/qwen36/ttft-control-contracts.ncl"
    mkdir -p decode-profile-export
    cp ${./decode-profile.ncl} decode-profile-export/decode-profile.ncl
    cp ${./decode-contracts.ncl} decode-profile-export/decode-contracts.ncl
    nickel export --format json decode-profile-export/decode-profile.ncl \
      > "$out/share/qwen36/decode-profile.json"
    mkdir -p batch-admission-profile-export
    cp ${./batch-admission-profile.ncl} batch-admission-profile-export/batch-admission-profile.ncl
    cp ${./batch-admission-policy.ncl} batch-admission-profile-export/batch-admission-policy.ncl
    cp ${./batch-admission-binding.ncl} batch-admission-profile-export/batch-admission-binding.ncl
    cp ${./batch-admission-contracts.ncl} batch-admission-profile-export/batch-admission-contracts.ncl
    nickel export --format json batch-admission-profile-export/batch-admission-profile.ncl \
      > "$out/share/qwen36/batch-admission-profile.json"
    mkdir -p batch-admission-b8-profile-export
    cp ${./batch-admission-b8-profile.ncl} batch-admission-b8-profile-export/batch-admission-b8-profile.ncl
    cp ${./batch-admission-b8-policy.ncl} batch-admission-b8-profile-export/batch-admission-b8-policy.ncl
    cp ${./batch-admission-b8-policy-validated.ncl} batch-admission-b8-profile-export/batch-admission-b8-policy-validated.ncl
    cp ${./batch-admission-b8-binding.ncl} batch-admission-b8-profile-export/batch-admission-b8-binding.ncl
    cp ${./batch-admission-b8-binding-validated.ncl} batch-admission-b8-profile-export/batch-admission-b8-binding-validated.ncl
    cp ${./batch-admission-b8-contracts.ncl} batch-admission-b8-profile-export/batch-admission-b8-contracts.ncl
    nickel export --format json batch-admission-b8-profile-export/batch-admission-b8-profile.ncl \
      > "$out/share/qwen36/batch-admission-b8-profile.json"
    mkdir -p batch-admission-b16-profile-export
    cp ${./batch-admission-b16-profile.ncl} batch-admission-b16-profile-export/batch-admission-b16-profile.ncl
    cp ${./batch-admission-b16-policy.ncl} batch-admission-b16-profile-export/batch-admission-b16-policy.ncl
    cp ${./batch-admission-b16-policy-validated.ncl} batch-admission-b16-profile-export/batch-admission-b16-policy-validated.ncl
    cp ${./batch-admission-b16-binding.ncl} batch-admission-b16-profile-export/batch-admission-b16-binding.ncl
    cp ${./batch-admission-b16-binding-validated.ncl} batch-admission-b16-profile-export/batch-admission-b16-binding-validated.ncl
    cp ${./batch-admission-b16-contracts.ncl} batch-admission-b16-profile-export/batch-admission-b16-contracts.ncl
    nickel export --format json batch-admission-b16-profile-export/batch-admission-b16-profile.ncl \
      > "$out/share/qwen36/batch-admission-b16-profile.json"
    mkdir -p trace-reuse-profile-export
    cp ${./trace-reuse-profile.ncl} trace-reuse-profile-export/trace-reuse-profile.ncl
    cp ${./trace-reuse-policy.ncl} trace-reuse-profile-export/trace-reuse-policy.ncl
    cp ${./trace-reuse-binding.ncl} trace-reuse-profile-export/trace-reuse-binding.ncl
    cp ${./trace-reuse-contracts.ncl} trace-reuse-profile-export/trace-reuse-contracts.ncl
    nickel export --format json trace-reuse-profile-export/trace-reuse-profile.ncl \
      > "$out/share/qwen36/trace-reuse-profile.json"
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
    cp ${./mtp-drafter-profile-k4.ncl} mtp-drafter-profile-export/mtp-drafter-profile-k4.ncl
    nickel export --format json mtp-drafter-profile-export/mtp-drafter-profile-k4.ncl \
      > "$out/share/qwen36/mtp-drafter-profile-k4.json"
    cp ${./mtp-drafter-profile-k2.ncl} mtp-drafter-profile-export/mtp-drafter-profile-k2.ncl
    nickel export --format json mtp-drafter-profile-export/mtp-drafter-profile-k2.ncl \
      > "$out/share/qwen36/mtp-drafter-profile-k2.json"
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
    cp ${./mtp-trace-profile-k4.ncl} mtp-trace-profile-export/mtp-trace-profile-k4.ncl
    nickel export --format json mtp-trace-profile-export/mtp-trace-profile-k4.ncl \
      > "$out/share/qwen36/mtp-trace-profile-k4.json"
    cp ${./mtp-trace-profile-k2.ncl} mtp-trace-profile-export/mtp-trace-profile-k2.ncl
    nickel export --format json mtp-trace-profile-export/mtp-trace-profile-k2.ncl \
      > "$out/share/qwen36/mtp-trace-profile-k2.json"
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
    mkdir -p gdn-decode-fusion-profile-export
    cp ${./gdn-decode-fusion-profile.ncl} gdn-decode-fusion-profile-export/gdn-decode-fusion-profile.ncl
    cp ${./gdn-decode-fusion-contracts.ncl} gdn-decode-fusion-profile-export/gdn-decode-fusion-contracts.ncl
    nickel export --format json gdn-decode-fusion-profile-export/gdn-decode-fusion-profile.ncl \
      > "$out/share/qwen36/gdn-decode-fusion-profile.json"
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
      --add-flags "candidate" \
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

    # r[impl tenstorrent.qwen38_p150x2.traced_decode]
    # Production serves the traced candidate. Matched physical measurement on
    # P150x2: served eager ~3.7 t/s (ttft 63.7 s) vs traced ~16.2 t/s (ttft 3.8 s);
    # trace replay verified on a second request (no per-request re-capture).
    # Formal decode admission (physically_admitted) remains pending in the
    # typed profile; the candidate path falls back to eager on any failure.
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
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

    # r[impl tenstorrent.qwen38_p150x2.batch_performance_admission]
    # Explicit physical evidence mode. Normal production stays ineligible.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-batch-admission" \
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
      --add-flags "--batch-admission-profile" \
      --add-flags "$out/share/qwen36/batch-admission-profile.json" \
      --add-flags "--batch-admission" \
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

    # r[impl tenstorrent.qwen38_p150x2.batch_b8_admission]
    # Explicit B=8 physical evidence mode. Normal production stays ineligible.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-b8-admission" \
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
      --add-flags "--batch-admission-profile" \
      --add-flags "$out/share/qwen36/batch-admission-b8-profile.json" \
      --add-flags "--batch-admission" \
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

    # r[impl tenstorrent.qwen38_p150x2.batch_b16_admission]
    # Explicit B=16 physical evidence mode. Normal production stays ineligible.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-b16-admission" \
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
      --add-flags "--batch-admission-profile" \
      --add-flags "$out/share/qwen36/batch-admission-b16-profile.json" \
      --add-flags "--batch-admission" \
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

    # r[impl tenstorrent.qwen38_p150x2.same_width_trace_reuse]
    # Explicit physical evidence mode. Normal production follows profile admission.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-p150x2-trace-reuse-admission" \
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
      --add-flags "--trace-reuse-profile" \
      --add-flags "$out/share/qwen36/trace-reuse-profile.json" \
      --add-flags "--trace-reuse-admission" \
      --add-flags "--runtime-identity" \
      --add-flags ${lib.escapeShellArg tt-metal} \
      --add-flags "--package-identity" \
      --add-flags ${lib.escapeShellArg executionIdentity}

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

    # r[impl tenstorrent.qwen38_p150x2.continuous_slot_admission]
    # Evaluates two continuous-slot measurement roots against the predeclared gates.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-continuous-slot-receipt" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --prefix PATH : ${lib.makeBinPath [ b3sum ]} \
      --add-flags "-m" \
      --add-flags "qwen36_service.continuous_slot_receipt" \
      --add-flags "--policy" \
      --add-flags "$out/share/qwen36/continuous-slot-admission-policy.json"

    # r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
    # Evaluates matched cold and cached prefix-cache roots against the predeclared gates.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/qwen36-prefix-cache-receipt" \
      --set PYTHONPATH "$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
      --prefix PATH : ${lib.makeBinPath [ b3sum ]} \
      --add-flags "-m" \
      --add-flags "qwen36_service.prefix_cache_receipt" \
      --add-flags "--policy" \
      --add-flags "$out/share/qwen36/prefix-cache-admission-policy.json"

    # Public commands use the checkpoint version. Preserve old consumers with
    # relative aliases, without renaming upstream modules or receipt schemas.
    for legacy_command in "$out"/bin/qwen36*; do
      canonical_name="qwen38''${legacy_command##*/qwen36}"
      mv "$legacy_command" "$out/bin/$canonical_name"
      ln -s "$canonical_name" "$legacy_command"
    done

    # Positive: the overlay provides the JIT headers and kernel sources the
    # packaged runtime omits (matmul_custom.h for SDPA; the experimental CCL
    # all-gather-fused-matmul kernels for the qwen36 TP path).
    test -f "$out/libexec/tt-metalium/tt_metal/hw/inc/api/compute/experimental/matmul_custom.h"
    test -f "$out/libexec/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp"
    gdn_decode_compute="$out/libexec/tt-metalium/ttnn/cpp/ttnn/operations/transformer/gated_delta_decode/device/kernels/compute/gated_delta_decode.cpp"
    test -f "$gdn_decode_compute"
    test "$(tail -n 1 "$gdn_decode_compute")" = '}'
    grep -Fq 'compute_kernel_hw_startup<SrcOrder::Reverse>' "$gdn_decode_compute"
    grep -Fq 'mul_tiles_bcast_scalar_init_short' "$gdn_decode_compute"
    grep -Fq 'sub_tiles_init' "$gdn_decode_compute"
    grep -Fq 'add_tiles_init' "$gdn_decode_compute"
    # The grouped-prefill assembly patch is shape-checked here (one file, the
    # model's state assembly): an assertion under the variant derivation would
    # cost a full tt-metal rebuild.
    test "$(grep -c '^--- ' ${./grouped-prefill-row-assembly.patch})" -eq 1

    # The occupancy policy is identity-bound and fail-closed: a split cannot be
    # driven from it until the fixed-cost share is measured, because the planner
    # refuses a non-numeric share. Assert both halves here.
    occupancy_policy="$out/share/qwen36/gdn-occupancy-policy.json"
    test -f "$occupancy_policy"
    grep -Fq '"fixed_cost_share": "not-measured"' "$occupancy_policy"
    grep -Fq '"minimum_occupancy": 0.5' "$occupancy_policy"
    grep -Fq '"value_tiles": 4' "$occupancy_policy"

    # Assert what identifies the PATCHED kernels, not how many lines they have.
    # A line count breaks on every edit and, worse, cannot tell a patched kernel
    # from one the full-tree copy reverted. These markers only exist post-patch.
    # The block fails closed in BOTH directions: the variant runtime must carry
    # the chunked markers and must have lost the decode grid guard, and the
    # shared runtime must carry neither marker and must keep the guard. A
    # reverted patch can therefore never build quietly on either side.
    gdn_runtime_lib="${tt-metal}/lib/_ttnncpp.so"
    test -f "$gdn_runtime_lib"
    gdn_decode_guard_marker='num_heads exceeds the available compute grid'
    for gdn_scan_rel in \
      dataflow/reader_chunk_gdn_scan.cpp \
      dataflow/writer_chunk_gdn_scan.cpp \
      compute/chunk_gdn_scan.cpp
    do
      gdn_scan_path="$out/libexec/tt-metalium/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/$gdn_scan_rel"
      test -f "$gdn_scan_path"
      if [ "$gdnGridRuntimeFlag" = "1" ]; then
        grep -Fq 'instance < wi_count' "$gdn_scan_path" || {
          echo "GDN grid variant runtime lost the chunked scan kernel marker in $gdn_scan_rel" >&2
          exit 1
        }
      elif grep -Fq 'instance < wi_count' "$gdn_scan_path"; then
        echo "shared runtime carries a chunked scan kernel marker in $gdn_scan_rel" >&2
        exit 1
      fi
    done
    if [ "$gdnGridRuntimeFlag" = "1" ]; then
      grep -Fq 'instance < head_count' "$gdn_decode_compute" || {
        echo "GDN grid variant runtime lost the chunked decode kernel marker" >&2
        exit 1
      }
      # The destination-register batching is shape-checked directionally. The
      # batched form moves the sync out of the tile loop, so the unbatched
      # destination-zero operand call is what must be gone, and the batched
      # per-tile destination calls are what must be present. Counting
      # `tile_regs_acquire()` cannot tell them apart: both forms acquire once
      # per loop, the batched form simply does it outside the loop.
      grep -Fq 'pack_tile(tile, cb_state_decay, tile)' "$gdn_decode_compute" || {
        echo "GDN grid variant runtime lost the batched destination registers" >&2
        exit 1
      }
      grep -Fq 'matmul_tiles(cb_k, cb_state_decay, key_tile, key_tile * val_tiles + val_tile, val_tile)' "$gdn_decode_compute" || {
        echo "GDN grid variant runtime lost the batched accumulate destination" >&2
        exit 1
      }
      if grep -Fq 'mul_tiles_bcast_scalar(cb_state, cb_g_exp, tile, 0, 0);' "$gdn_decode_compute"; then
        echo "GDN grid variant runtime still carries a per-tile destination-zero operand" >&2
        exit 1
      fi
      test "$(grep -c '^--- ' ${../tt-metal/gdn-decode-dst-batch.patch})" -eq 1
      grep -Fq 'gated_delta_decode/device/kernels/compute/gated_delta_decode.cpp' ${../tt-metal/gdn-decode-dst-batch.patch}
      if grep -Fq "$gdn_decode_guard_marker" "$gdn_runtime_lib"; then
        echo "GDN grid variant runtime still carries the one-instance-per-core decode guard" >&2
        exit 1
      fi
    else
      if grep -Fq 'instance < head_count' "$gdn_decode_compute"; then
        echo "shared runtime carries a chunked decode kernel marker" >&2
        exit 1
      fi
      if grep -Fq 'pack_tile(tile, cb_state_decay, tile)' "$gdn_decode_compute"; then
        echo "shared runtime carries the batched destination registers" >&2
        exit 1
      fi
      grep -Fq 'mul_tiles_bcast_scalar(cb_state, cb_g_exp, tile, 0, 0);' "$gdn_decode_compute" || {
        echo "shared runtime lost its per-tile destination-zero operand" >&2
        exit 1
      }
      grep -Fq "$gdn_decode_guard_marker" "$gdn_runtime_lib" || {
        echo "shared runtime lost the pre-change decode grid guard its receipts rely on" >&2
        exit 1
      }
    fi

    # Negative: reject newer API names that package-matched TRISC headers do not declare.
    if grep -Fq 'mul_bcast_scalar_init(' "$gdn_decode_compute" || \
       grep -Fq 'sub_init(' "$gdn_decode_compute" || \
       grep -Fq 'add_init(' "$gdn_decode_compute"; then
      echo "GDN decode compute source contains an unsupported TT-Metal init name" >&2
      exit 1
    fi

    gdn_delta_rule="$out/lib/python${python3.pythonVersion}/site-packages/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py"
    test -f "$gdn_delta_rule"
    grep -Fq 'q_head_major = ttnn.permute' "$gdn_delta_rule"
    grep -Fq 'beta_vector = ttnn.unsqueeze(beta, -1)' "$gdn_delta_rule"
    grep -Fq 'output = ttnn.permute(output_head_major' "$gdn_delta_rule"
    if grep -Fq 'q_flat = ttnn.reshape(q,' "$gdn_delta_rule"; then
      echo "GDN fused recurrence contains an unsafe metadata-only Q reshape" >&2
      exit 1
    fi

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    # Positive: each legacy command resolves to one executable canonical command.
    test -x "$out/bin/qwen38-check"
    test -x "$out/bin/qwen38-p150x2-serve"
    for legacy_command in "$out"/bin/qwen36*; do
      canonical_name="qwen38''${legacy_command##*/qwen36}"
      test -L "$legacy_command"
      test "$(readlink "$legacy_command")" = "$canonical_name"
      test -x "$out/bin/$canonical_name"
      test ! -L "$out/bin/$canonical_name"
    done
    "$out/bin/qwen38-p150x2-serve" --help > canonical-help.log
    "$out/bin/qwen36-p150x2-serve" --help > legacy-help.log
    cmp canonical-help.log legacy-help.log

    # Negative: both command names reject malformed arguments before device I/O.
    for command_name in qwen38-p150x2-serve qwen36-p150x2-serve; do
      if "$out/bin/$command_name" --model-path /nonexistent-qwen-model \
          --unknown-package-option > "$command_name-invalid.log" 2>&1; then
        echo "$command_name accepted an unknown option" >&2
        exit 1
      fi
      grep -F 'unrecognized arguments: --unknown-package-option' "$command_name-invalid.log"
    done

    # Positive: installed layout is complete and hash-consistent.
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/model.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/qwen36_vllm.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/gdn/tp.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/gdn/decode_layout_candidate.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/gdn/decode_fusion_candidate.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/demos/blackhole/qwen36/tt/mlp.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/models/tt_transformers/tt/model_config.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/decode_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/batch_admission_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/trace_reuse_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/prefill_profile_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/prefill_profile_adapter.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/readiness_core.py"
    test -f "$out/lib/python${python3.pythonVersion}/site-packages/qwen36_service/systemd_notify_adapter.py"
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
    test -f "$out/share/qwen36/execution-wrapper-contract.ncl"
    nickel export --format json "$out/share/qwen36/execution-wrapper-contract.ncl" \
      > execution-wrapper-contract.json
    grep -F '"package_identity_kind": "nix-execution-artifact"' execution-wrapper-contract.json
    test -f "$out/share/qwen36/batch-admission-profile.json"
    test -f "$out/share/qwen36/batch-admission-b8-profile.json"
    test -f "$out/share/qwen36/batch-admission-b16-profile.json"
    b16_receipt=${./evidence/2026-09-12/qwen38-27b-batch-admission-b16-receipt.json}
    b16_receipt_blake3=$(b3sum "$b16_receipt" | cut -d' ' -f1)
    ${pythonEnvironment}/bin/python - <<PROFILE_CHECK
import json
with open("$out/share/qwen36/batch-admission-b16-profile.json") as handle:
    profile = json.load(handle)
with open("$b16_receipt") as handle:
    receipt = json.load(handle)
assert profile["workload"]["candidate_width"] == 16
assert len(profile["workload"]["prompts"]) == 16
assert profile["admission"]["minimum_aggregate_decode_speedup_ratio"] == 6.0
assert profile["admission"]["minimum_per_user_decode_retention_ratio"] == 0.75
assert profile["admission"]["maximum_ttft_ratio"] == 9.0
assert profile["admission"]["maximum_tail_latency_ratio"] == 1.5
assert profile["identities"]["receipt_blake3"] == "$b16_receipt_blake3"
assert profile["physically_admitted"] == receipt["admission"]["admitted"]
PROFILE_CHECK
    test -f "$out/share/qwen36/batch-admission-policy.ncl"
    test -f "$out/share/qwen36/batch-admission-binding.ncl"
    test -f "$out/share/qwen36/trace-reuse-profile.json"
    test -f "$out/share/qwen36/trace-reuse-policy.ncl"
    test -f "$out/share/qwen36/trace-reuse-binding.ncl"
    test -L "$out/share/qwen36/execution-identity"
    test "$(readlink "$out/share/qwen36/execution-identity")" = ${lib.escapeShellArg executionIdentity}
    test -f "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"
    grep -F ${lib.escapeShellArg (toString (stableIdentityPath "qwen36-batch-admission-policy.ncl" ./batch-admission-policy.ncl))} \
      "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"
    if grep -F 'batch-admission-binding.ncl' \
        "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"; then
      echo "mutable batch admission binding entered execution identity" >&2
      exit 1
    fi
    grep -F ${lib.escapeShellArg (toString (stableIdentityPath "qwen36-trace-reuse-policy.ncl" ./trace-reuse-policy.ncl))} \
      "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"
    grep -F ${lib.escapeShellArg (toString (stableIdentityPath "qwen36-kernel-distribution-admission-policy.ncl" ./kernel-distribution-admission-policy.ncl))} \
      "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"
    if grep -F 'trace-reuse-binding.ncl' \
        "$out/share/qwen36/execution-identity/share/qwen36/execution-inputs"; then
      echo "mutable trace reuse binding entered execution identity" >&2
      exit 1
    fi
    grep -F ${lib.escapeShellArg executionIdentity} "$out/bin/qwen36-p150x2-serve"
    grep -F ${lib.escapeShellArg executionIdentity} "$out/bin/qwen36-p150x2-batch-admission"
    grep -F ${lib.escapeShellArg executionIdentity} "$out/bin/qwen36-p150x2-b8-admission"
    grep -F ${lib.escapeShellArg executionIdentity} "$out/bin/qwen36-p150x2-b16-admission"
    grep -F ${lib.escapeShellArg executionIdentity} "$out/bin/qwen36-p150x2-trace-reuse-admission"
    test -f "$out/share/qwen36/mtp-probe-profile.json"
    test -f "$out/share/qwen36/mtp-drafter-profile.json"
    test -f "$out/share/qwen36/mtp-selection-profile.json"
    test -f "$out/share/qwen36/mtp-trace-profile.json"
    test -f "$out/share/qwen36/mtp-rollback-profile.json"
    test -f "$out/share/qwen36/decode-component-profile.json"
    test -f "$out/share/qwen36/gdn-decode-fusion-profile.json"
    test -f "$out/share/qwen36/ttft-control-profile.json"
    test -x "$out/bin/qwen36-p150x2-serve"
    test -x "$out/bin/qwen36-p150x2-trace-candidate"
    test -x "$out/bin/qwen36-p150x2-batch-admission"
    test -x "$out/bin/qwen36-p150x2-b8-admission"
    test -x "$out/bin/qwen36-p150x2-trace-reuse-admission"
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
    "$out/bin/qwen36-p150x2-batch-admission" --help > /dev/null
    "$out/bin/qwen36-p150x2-b8-admission" --help > /dev/null
    "$out/bin/qwen36-p150x2-trace-reuse-admission" --help > /dev/null
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

    # r[verify tenstorrent.qwen38_p150x2.batch_performance_admission]
    # Positive: the B=4 admission profile exports deterministically.
    mkdir -p batch-admission-profile-check
    cp ${./batch-admission-profile.ncl} batch-admission-profile-check/batch-admission-profile.ncl
    cp ${./batch-admission-policy.ncl} batch-admission-profile-check/batch-admission-policy.ncl
    cp ${./batch-admission-policy-validated.ncl} batch-admission-profile-check/batch-admission-policy-validated.ncl
    cp ${./batch-admission-binding.ncl} batch-admission-profile-check/batch-admission-binding.ncl
    cp ${./batch-admission-binding-validated.ncl} batch-admission-profile-check/batch-admission-binding-validated.ncl
    cp ${./batch-admission-contracts.ncl} batch-admission-profile-check/batch-admission-contracts.ncl
    nickel export --format json batch-admission-profile-check/batch-admission-policy-validated.ncl > batch-admission-policy.json
    nickel export --format json batch-admission-profile-check/batch-admission-binding-validated.ncl > batch-admission-binding.json
    nickel export --format json batch-admission-profile-check/batch-admission-profile.ncl > batch-admission-profile-first.json
    nickel export --format json batch-admission-profile-check/batch-admission-profile.ncl > batch-admission-profile-second.json
    cmp batch-admission-profile-first.json batch-admission-profile-second.json
    cmp batch-admission-profile-first.json "$out/share/qwen36/batch-admission-profile.json"
    grep -F '"candidate_width": 4' batch-admission-profile-first.json
    grep -F '"physically_admitted": true' batch-admission-profile-first.json
    grep -F '"receipt_blake3": "489f2fbcb414ae04e536fc055857864c3255d727e4c12b79dc701a86ded3d028"' \
      batch-admission-profile-first.json

    # r[verify tenstorrent.qwen38_p150x2.batch_b8_admission]
    # Positive: the B=8 admission profile exports deterministically and stays unadmitted.
    mkdir -p batch-admission-b8-profile-check
    cp ${./batch-admission-b8-profile.ncl} batch-admission-b8-profile-check/batch-admission-b8-profile.ncl
    cp ${./batch-admission-b8-policy.ncl} batch-admission-b8-profile-check/batch-admission-b8-policy.ncl
    cp ${./batch-admission-b8-policy-validated.ncl} batch-admission-b8-profile-check/batch-admission-b8-policy-validated.ncl
    cp ${./batch-admission-b8-binding.ncl} batch-admission-b8-profile-check/batch-admission-b8-binding.ncl
    cp ${./batch-admission-b8-binding-validated.ncl} batch-admission-b8-profile-check/batch-admission-b8-binding-validated.ncl
    cp ${./batch-admission-b8-contracts.ncl} batch-admission-b8-profile-check/batch-admission-b8-contracts.ncl
    nickel export --format json batch-admission-b8-profile-check/batch-admission-b8-policy-validated.ncl > batch-admission-b8-policy.json
    nickel export --format json batch-admission-b8-profile-check/batch-admission-b8-binding-validated.ncl > batch-admission-b8-binding.json
    nickel export --format json batch-admission-b8-profile-check/batch-admission-b8-profile.ncl > batch-admission-b8-profile-first.json
    nickel export --format json batch-admission-b8-profile-check/batch-admission-b8-profile.ncl > batch-admission-b8-profile-second.json
    cmp batch-admission-b8-profile-first.json batch-admission-b8-profile-second.json
    cmp batch-admission-b8-profile-first.json "$out/share/qwen36/batch-admission-b8-profile.json"
    grep -F '"candidate_width": 8' batch-admission-b8-profile-first.json
    grep -F '"minimum_aggregate_decode_speedup_ratio": 6' batch-admission-b8-profile-first.json
    grep -F '"maximum_ttft_ratio": 9' batch-admission-b8-profile-first.json
    # The B=8 admission state is bound to the tracked receipt: the exported
    # profile must name the tracked receipt's digest, and its admitted flag must
    # equal what that receipt's own admission block says. A rejection receipt
    # cannot sit behind an admitted binding, and a pass cannot be exported while
    # the tracked receipt records a failure.
    b8_receipt=${./evidence/2026-09-12/qwen38-27b-batch-admission-b8-receipt.json}
    b8_receipt_blake3=$(b3sum "$b8_receipt" | cut -d' ' -f1)
    grep -F "\"receipt_blake3\": \"$b8_receipt_blake3\"" batch-admission-b8-profile-first.json
    if grep -F '"admitted": true' "$b8_receipt"; then
      grep -F '"physically_admitted": true' batch-admission-b8-profile-first.json
    elif grep -F '"admitted": false' "$b8_receipt"; then
      grep -F '"physically_admitted": false' batch-admission-b8-profile-first.json
    else
      echo "tracked B=8 receipt carries no admission verdict" >&2
      exit 1
    fi

    # Negative: admitted B=8 without a receipt must fail.
    mkdir -p batch-admission-b8-profile-negative
    cp ${./batch-admission-b8-profile.ncl} batch-admission-b8-profile-negative/batch-admission-b8-profile.ncl
    cp ${./batch-admission-b8-policy.ncl} batch-admission-b8-profile-negative/batch-admission-b8-policy.ncl
    cp ${./batch-admission-b8-binding.ncl} batch-admission-b8-profile-negative/batch-admission-b8-binding.ncl
    cp ${./batch-admission-b8-contracts.ncl} batch-admission-b8-profile-negative/batch-admission-b8-contracts.ncl
    cp ${./fixtures/invalid-batch-admission-b8-profile.ncl} batch-admission-b8-profile-negative/invalid-batch-admission-b8-profile.ncl
    if nickel export --format json batch-admission-b8-profile-negative/invalid-batch-admission-b8-profile.ncl \
        > invalid-batch-admission-b8-profile.json 2> invalid-batch-admission-b8-profile.log; then
      echo "invalid B=8 admission profile exported" >&2
      exit 1
    fi
    test -s invalid-batch-admission-b8-profile.log

    # Negative: a candidate width outside the admitted set must fail its closed contract.
    cp ${./fixtures/invalid-batch-admission-b8-width.ncl} batch-admission-b8-profile-negative/invalid-batch-admission-b8-width.ncl
    if nickel export --format json batch-admission-b8-profile-negative/invalid-batch-admission-b8-width.ncl \
        > invalid-batch-admission-b8-width.json 2> invalid-batch-admission-b8-width.log; then
      echo "invalid B=8 width exported" >&2
      exit 1
    fi
    test -s invalid-batch-admission-b8-width.log

    # r[verify tenstorrent.qwen38_p150x2.same_width_trace_reuse]
    # Positive: the trace-reuse profile exports deterministically and stays unadmitted.
    mkdir -p trace-reuse-profile-check
    cp ${./trace-reuse-profile.ncl} trace-reuse-profile-check/trace-reuse-profile.ncl
    cp ${./trace-reuse-policy.ncl} trace-reuse-profile-check/trace-reuse-policy.ncl
    cp ${./trace-reuse-policy-validated.ncl} trace-reuse-profile-check/trace-reuse-policy-validated.ncl
    cp ${./trace-reuse-binding.ncl} trace-reuse-profile-check/trace-reuse-binding.ncl
    cp ${./trace-reuse-binding-validated.ncl} trace-reuse-profile-check/trace-reuse-binding-validated.ncl
    cp ${./trace-reuse-contracts.ncl} trace-reuse-profile-check/trace-reuse-contracts.ncl
    nickel export --format json trace-reuse-profile-check/trace-reuse-policy-validated.ncl > trace-reuse-policy.json
    nickel export --format json trace-reuse-profile-check/trace-reuse-binding-validated.ncl > trace-reuse-binding.json
    nickel export --format json trace-reuse-profile-check/trace-reuse-profile.ncl > trace-reuse-profile-first.json
    nickel export --format json trace-reuse-profile-check/trace-reuse-profile.ncl > trace-reuse-profile-second.json
    cmp trace-reuse-profile-first.json trace-reuse-profile-second.json
    cmp trace-reuse-profile-first.json "$out/share/qwen36/trace-reuse-profile.json"
    grep -F '"width": 4' trace-reuse-profile-first.json
    grep -F '"maximum_resident_trace_count": 1' trace-reuse-profile-first.json
    grep -F '"physically_admitted": false' trace-reuse-profile-first.json
    grep -F '"receipt_blake3": "not-measured"' trace-reuse-profile-first.json

    # Negative: admitted trace reuse without a receipt must fail.
    mkdir -p trace-reuse-profile-negative
    cp ${./trace-reuse-profile.ncl} trace-reuse-profile-negative/trace-reuse-profile.ncl
    cp ${./trace-reuse-policy.ncl} trace-reuse-profile-negative/trace-reuse-policy.ncl
    cp ${./trace-reuse-binding.ncl} trace-reuse-profile-negative/trace-reuse-binding.ncl
    cp ${./trace-reuse-contracts.ncl} trace-reuse-profile-negative/trace-reuse-contracts.ncl
    cp ${./fixtures/invalid-trace-reuse-profile.ncl} trace-reuse-profile-negative/invalid-trace-reuse-profile.ncl
    if nickel export --format json trace-reuse-profile-negative/invalid-trace-reuse-profile.ncl \
        > invalid-trace-reuse-profile.json 2> invalid-trace-reuse-profile.log; then
      echo "invalid trace reuse profile exported" >&2
      exit 1
    fi
    test -s invalid-trace-reuse-profile.log

    # Negative: an admitted batch profile without a receipt must fail.
    mkdir -p batch-admission-profile-negative
    cp ${./batch-admission-profile.ncl} batch-admission-profile-negative/batch-admission-profile.ncl
    cp ${./batch-admission-policy.ncl} batch-admission-profile-negative/batch-admission-policy.ncl
    cp ${./batch-admission-binding.ncl} batch-admission-profile-negative/batch-admission-binding.ncl
    cp ${./batch-admission-contracts.ncl} batch-admission-profile-negative/batch-admission-contracts.ncl
    cp ${./fixtures/invalid-batch-admission-profile.ncl} batch-admission-profile-negative/invalid-batch-admission-profile.ncl
    if nickel export --format json batch-admission-profile-negative/invalid-batch-admission-profile.ncl \
        > invalid-batch-admission-profile.json 2> invalid-batch-admission-profile.log; then
      echo "invalid batch admission profile exported" >&2
      exit 1
    fi
    test -s invalid-batch-admission-profile.log

    # r[verify tenstorrent.qwen38_p150x2.continuous_slot_policy]
    # Negative: more refills per boundary than slots must fail the typed contract.
    mkdir -p continuous-slot-policy-negative
    cp ${./continuous-slot-contracts.ncl} continuous-slot-policy-negative/continuous-slot-contracts.ncl
    cp ${./fixtures/invalid-continuous-slot-policy.ncl} continuous-slot-policy-negative/invalid-continuous-slot-policy.ncl
    if nickel export --format json continuous-slot-policy-negative/invalid-continuous-slot-policy.ncl \
        > invalid-continuous-slot-policy.json 2> invalid-continuous-slot-policy.log; then
      echo "invalid continuous-slot policy exported" >&2
      exit 1
    fi
    test -s invalid-continuous-slot-policy.log

    # r[verify tenstorrent.qwen38_p150x2.continuous_slot_admission]
    # Negative: an admission without exact candidate parity must fail its contract.
    cp ${./continuous-slot-admission-contracts.ncl} continuous-slot-policy-negative/continuous-slot-admission-contracts.ncl
    cp ${./fixtures/invalid-continuous-slot-admission-policy.ncl} continuous-slot-policy-negative/invalid-continuous-slot-admission-policy.ncl
    if nickel export --format json continuous-slot-policy-negative/invalid-continuous-slot-admission-policy.ncl \
        > invalid-continuous-slot-admission-policy.json 2> invalid-continuous-slot-admission-policy.log; then
      echo "invalid continuous-slot admission policy exported" >&2
      exit 1
    fi
    test -s invalid-continuous-slot-admission-policy.log

    # r[verify tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
    # Negative: a minimum prefix off the chunk boundary must fail its contract.
    cp ${./prefix-cache-contracts.ncl} continuous-slot-policy-negative/prefix-cache-contracts.ncl
    cp ${./fixtures/invalid-prefix-cache-policy.ncl} continuous-slot-policy-negative/invalid-prefix-cache-policy.ncl
    if nickel export --format json continuous-slot-policy-negative/invalid-prefix-cache-policy.ncl \
        > invalid-prefix-cache-policy.json 2> invalid-prefix-cache-policy.log; then
      echo "invalid prefix cache policy exported" >&2
      exit 1
    fi
    test -s invalid-prefix-cache-policy.log
    cp ${./prefix-cache-admission-contracts.ncl} continuous-slot-policy-negative/prefix-cache-admission-contracts.ncl
    cp ${./fixtures/invalid-prefix-cache-admission-policy.ncl} continuous-slot-policy-negative/invalid-prefix-cache-admission-policy.ncl
    if nickel export --format json continuous-slot-policy-negative/invalid-prefix-cache-admission-policy.ncl \
        > invalid-prefix-cache-admission-policy.json 2> invalid-prefix-cache-admission-policy.log; then
      echo "invalid prefix cache admission policy exported" >&2
      exit 1
    fi
    test -s invalid-prefix-cache-admission-policy.log

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

    # r[verify tenstorrent.qwen38_p150x2.gdn_decode_fusion]
    # Positive: fusion policy exports deterministically with fixed gates.
    mkdir -p gdn-decode-fusion-profile-check
    cp ${./gdn-decode-fusion-profile.ncl} gdn-decode-fusion-profile-check/gdn-decode-fusion-profile.ncl
    cp ${./gdn-decode-fusion-contracts.ncl} gdn-decode-fusion-profile-check/gdn-decode-fusion-contracts.ncl
    nickel export --format json gdn-decode-fusion-profile-check/gdn-decode-fusion-profile.ncl > gdn-decode-fusion-profile-first.json
    nickel export --format json gdn-decode-fusion-profile-check/gdn-decode-fusion-profile.ncl > gdn-decode-fusion-profile-second.json
    cmp gdn-decode-fusion-profile-first.json gdn-decode-fusion-profile-second.json
    cmp gdn-decode-fusion-profile-first.json "$out/share/qwen36/gdn-decode-fusion-profile.json"
    grep -F '"diagnostic_only": true' gdn-decode-fusion-profile-first.json
    grep -F '"maximum_gdn_median_ratio": 0.95' gdn-decode-fusion-profile-first.json
    grep -F '"maximum_full_step_median_ratio": 0.97' gdn-decode-fusion-profile-first.json
    grep -F "\"authority_blake3\": \"$authority_blake3\"" gdn-decode-fusion-profile-first.json

    # Negative: retain valid imports and require a contract error, not a parser error.
    mkdir -p gdn-decode-fusion-profile-negative/fixtures
    cp ${./gdn-decode-fusion-profile.ncl} gdn-decode-fusion-profile-negative/gdn-decode-fusion-profile.ncl
    cp ${./gdn-decode-fusion-contracts.ncl} gdn-decode-fusion-profile-negative/gdn-decode-fusion-contracts.ncl
    cp ${./fixtures/invalid-gdn-decode-fusion-profile.ncl} gdn-decode-fusion-profile-negative/fixtures/profile.ncl
    cp ${./fixtures/invalid-gdn-decode-fusion-modes.ncl} gdn-decode-fusion-profile-negative/fixtures/modes.ncl
    cp ${./fixtures/invalid-gdn-decode-fusion-parity.ncl} gdn-decode-fusion-profile-negative/fixtures/parity.ncl
    for fixture in profile modes parity; do
      if nickel export --format json "gdn-decode-fusion-profile-negative/fixtures/$fixture.ncl" \
          > "invalid-gdn-decode-fusion-$fixture.json" 2> "invalid-gdn-decode-fusion-$fixture.log"; then
        echo "invalid GDN decode fusion $fixture exported" >&2
        exit 1
      fi
      grep -Fi "contract broken" "invalid-gdn-decode-fusion-$fixture.log"
    done

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
    for probe in ${./sequence-probe-check.py} ${./warmup-tail-check.py} ${./mtp-serving-check.py} ${./mtp-serving-round-check.py} ${./mtp-convolution-window-check.py} ${./mtp-prefix-trace-check.py} ${./mtp-kv-tail-check.py}; do
      PYTHONPATH="$out/lib/python${python3.pythonVersion}/site-packages:${ttnnSitePackages}" \
        TT_METAL_HOME="$out/libexec/tt-metalium" \
        TT_METAL_RUNTIME_ROOT="$out/libexec/tt-metalium" \
        ${pythonEnvironment}/bin/python "$probe"
    done

    PYTHONPATH="$out/lib/python${python3.pythonVersion}/site-packages" \
      ${pythonEnvironment}/bin/python ${./mtp-width-profile-check.py} "$out/share/qwen36"

    # Negative: env-gated BF4 cache names must never collide; the check script
    # itself rejects a missing .bfp4 distinction.
    runHook postInstallCheck
  '';

  passthru = {
    inherit
      authorityValidator
      canonicalAuthority
      executionIdentity
      pythonEnvironment
      source
      ttnnSitePackages
      ;
  };

  meta = {
    description = "Qwen3.8-27B on the qwen36 Blackhole path (tt-metal PR 53543, adopted)";
    homepage = "https://github.com/tenstorrent/tt-metal/pull/53543";
    license = lib.licenses.asl20;
    mainProgram = "qwen38-check";
    platforms = lib.platforms.linux;
  };
}
