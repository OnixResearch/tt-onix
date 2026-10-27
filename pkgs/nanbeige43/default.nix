# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.mesh]
# Nanbeige4.3-33B (plain LlamaForCausalLM) on the Blackhole P150x2 mesh through
# the runtime-matched TT-Metal tt_transformers generic HF path. The models/
# tree is installed unpatched from ${tt-metal.src}; the repo-authored runner
# drives create_tt_model/Generator directly. See
# .cairn/changes/bring-up-nanbeige43-33b-p150x2/.
{
  lib,
  stdenvNoCC,
  callPackage,
  fetchurl,
  linkFarm,
  python3,
  makeWrapper,
  tt-metal,
  tt-metal-tracy-tools,
  ttsim,
}:
let
  canonicalAuthority = builtins.fromJSON (builtins.readFile ./generated/authority.json);
  authorityValidator = import ./authority.nix { inherit lib canonicalAuthority; };
  admittedAuthority = authorityValidator.admit canonicalAuthority;
  # r[verify tenstorrent.nanbeige43_p150x2.checkpoint_authority.invalid]
  # Evaluation-time negative fixtures: every mutation must be rejected.
  mutate = f: canonicalAuthority // f canonicalAuthority;
  mapFile =
    name: g: a:
    map (file: if file.name == name then file // g file else file) a.files;
  rejectedFixtures = {
    revision = mutate (_: {
      revision = "0000000000000000000000000000000000000000";
    });
    shard-hash = mutate (a: {
      files = mapFile "model-00007-of-00014.safetensors" (_: {
        sha256 = lib.fixedWidthString 64 "0" "";
      }) a;
    });
    shard-count = mutate (a: {
      files = builtins.filter (file: file.name != "model-00014-of-00014.safetensors") a.files;
    });
    shard-bytes = mutate (a: {
      checkpoint_shard_bytes = a.checkpoint_shard_bytes + 1;
    });
    role = mutate (a: {
      files = mapFile "tokenizer.model" (_: { role = "checkpoint_shard"; }) a;
    });
  };
  fixturesRejected = lib.all (fixture: !authorityValidator.accepts fixture) (
    builtins.attrValues rejectedFixtures
  );
  pinnedMetadata =
    linkFarm "nanbeige43-metadata-${builtins.substring 0 12 admittedAuthority.revision}"
      (
        map (file: {
          inherit (file) name;
          path = fetchurl {
            url = "https://huggingface.co/${admittedAuthority.repository}/resolve/${admittedAuthority.revision}/${file.name}";
            hash = builtins.convertHash {
              hash = file.sha256;
              hashAlgo = "sha256";
              toHashFormat = "sri";
            };
          };
        }) (builtins.filter (file: file.role != "checkpoint_shard") admittedAuthority.files)
      );
  pythonEnvironment = python3.withPackages (packages: [
    # Runtime deps of the packaged ttnn import (same set as pkgs/qwen36).
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
    # tt_transformers model stack and the CPU reference (same set as pkgs/qwen36).
    packages.blake3
    packages.pydantic
    packages.pillow
    packages.pytest
    packages.safetensors
    packages.sentencepiece
    packages.tiktoken
    packages.tokenizers
    packages.torch
    packages.torchvision
    packages.transformers
  ]);
  ttnnSitePackages = "${tt-metal}/lib/python${python3.pythonVersion}/site-packages";
  metaliumOverlay = callPackage ../tt-metal/metalium-overlay.nix {
    inherit tt-metal tt-metal-tracy-tools;
  };
  sitePackages = "lib/python${python3.pythonVersion}/site-packages";
in
assert lib.assertMsg fixturesRejected "nanbeige43 authority validator admitted a drifted fixture";
stdenvNoCC.mkDerivation {
  pname = "nanbeige43";
  version = "0.1.0";

  # The runtime-matched source: the exact tree tt-metal was built from.
  src = tt-metal.src;
  # The batch-8 defect is in the packaged fused QK path; the fix is the non-fused
  # path for that batch size only. See docs/nanbeige-batch8.md.
  patches = [ ./patches/0001-batch-8-non-fused-qk.patch ];
  dontConfigure = true;
  dontBuild = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/${sitePackages}" "$out/share/nanbeige43" "$out/bin"
    cp -r models "$out/${sitePackages}/models"
    # Upstream-relative symlinks that point outside models/ cannot resolve here.
    find "$out/${sitePackages}/models" -type l ! -exec test -e {} \; -delete
    cp -r ${./runner/nanbeige43_runner} "$out/${sitePackages}/nanbeige43_runner"
    chmod -R u+w "$out/${sitePackages}/nanbeige43_runner"
    if find "$out/${sitePackages}/nanbeige43_runner" \( -name '__pycache__' -o -name '*.pyc' \) -print -quit | grep -q .; then
      echo "runner output contains interpreter bytecode" >&2
      exit 1
    fi
    cp ${./generated/authority.json} "$out/share/nanbeige43/authority.json"
    ln -s ${pinnedMetadata} "$out/share/nanbeige43/metadata"

    ${metaliumOverlay}
    overlayRoot="$out/libexec/tt-metalium"

    common=(
      --set PYTHONPATH "$out/${sitePackages}:${ttnnSitePackages}"
      --set TT_METAL_HOME "$overlayRoot"
      --set TT_METAL_RUNTIME_ROOT "$overlayRoot"
      --set NANBEIGE43_AUTHORITY "$out/share/nanbeige43/authority.json"
      --set NANBEIGE43_RUNTIME_IDENTITY "${tt-metal}"
      --set NANBEIGE43_PACKAGE_IDENTITY "$out"
      --set HF_HUB_OFFLINE 1
      --unset MESH_DEVICE
    )
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/nanbeige43-check" "''${common[@]}" \
      --add-flags "-P -m nanbeige43_runner.check --metadata $out/share/nanbeige43/metadata"
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/nanbeige43-reference" "''${common[@]}" \
      --add-flags "-P -m nanbeige43_runner.reference"
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/nanbeige43-reduce-checkpoint" "''${common[@]}" \
      --add-flags "-P -m nanbeige43_runner.reduce"
    # Production: fail-closed exact 1x2 mesh on PCIe devices 0 and 1.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/nanbeige43-p150x2" "''${common[@]}" \
      --unset TT_METAL_SIMULATOR \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --add-flags "-P -m nanbeige43_runner.run" \
      --append-flags "--mesh p150x2"
    # Hardware-neutral 1x1 smoke (reduced-layer checkpoints only); the same
    # runner and model code as production.
    makeWrapper ${pythonEnvironment}/bin/python "$out/bin/nanbeige43-smoke-1x1" "''${common[@]}" \
      --add-flags "-P -m nanbeige43_runner.run" \
      --append-flags "--mesh smoke-1x1"
    # The 1x1 smoke on the packaged Blackhole simulator (slow dispatch,
    # SFPLOADMACRO disabled), per the AGENTS.md simulator-before-hardware flow.
    makeWrapper ${ttsim}/bin/ttsim-blackhole-sfpu "$out/bin/nanbeige43-ttsim-smoke-1x1" \
      --add-flags "$out/bin/nanbeige43-smoke-1x1"

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    export HOME="$TMPDIR"
    # Exits non-zero on the first failed check.
    "$out/bin/nanbeige43-check" > "$TMPDIR/check.json"
    # The runner modules import against the packaged runtime.
    for runner in nanbeige43-p150x2 nanbeige43-smoke-1x1; do
      "$out/bin/$runner" --help > /dev/null
    done
    runHook postInstallCheck
  '';

  passthru = {
    inherit
      authorityValidator
      canonicalAuthority
      pinnedMetadata
      pythonEnvironment
      rejectedFixtures
      ;
  };

  meta = {
    description = "Nanbeige4.3-33B on Blackhole P150x2 through TT-Metal tt_transformers";
    license = lib.licenses.lgpl21Only;
    platforms = [ "x86_64-linux" ];
  };
}
