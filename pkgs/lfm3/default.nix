# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.lfm3.p150.gpu]
# LFM3-1.7B (tri-hybrid: 13 gated short convolutions, 5 sliding-window attention
# layers, 6 global attention layers) on one Blackhole P150 through TT-Metal 0.77.
# The runner is repo-authored TT-NN code: the checkpoint's architecture has no
# tt_transformers path. See docs/lfm3-1.7b-p150.md.
{
  lib,
  stdenvNoCC,
  callPackage,
  python3,
  tt-metal,
  tt-metal-tracy-tools,
}:
let
  pythonEnvironment = python3.withPackages (packages: [
    # ttnn runtime deps (same set as pkgs/qwen36 and pkgs/nanbeige43).
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
    # Checkpoint loading and tokenization.
    packages.safetensors
    packages.tokenizers
    packages.torch
  ]);
  ttnnSitePackages = "${tt-metal}/lib/python${python3.pythonVersion}/site-packages";
  metaliumOverlay = callPackage ../tt-metal/metalium-overlay.nix {
    inherit tt-metal tt-metal-tracy-tools;
  };
  sitePackages = "lib/python${python3.pythonVersion}/site-packages";
  installEntry =
    {
      name,
      module,
      template,
    }:
    ''
      install -m755 ${template} "$out/bin/${name}"
      substituteInPlace "$out/bin/${name}" \
        --replace-fail @python@ "${pythonEnvironment}/bin/python" \
        --replace-fail @sitePackages@ "$out/${sitePackages}" \
        --replace-fail @module@ "${module}"
    '';
in
stdenvNoCC.mkDerivation {
  pname = "lfm3";
  version = "0.1.0";

  # No upstream source tree: the runner is this repository's code.
  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  # The install check imports ttnn, so the runtime closure must be in the sandbox.
  buildInputs = [ tt-metal ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/${sitePackages}" "$out/bin"
    cp -r ${./runner/lfm3_runner} "$out/${sitePackages}/lfm3_runner"
    chmod -R u+w "$out/${sitePackages}/lfm3_runner"
    if find "$out/${sitePackages}/lfm3_runner" \( -name '__pycache__' -o -name '*.pyc' \) -print -quit | grep -q .; then
      echo "runner output contains interpreter bytecode" >&2
      exit 1
    fi

    ${metaliumOverlay}

    ${installEntry {
      name = "lfm3-run";
      module = "cli";
      template = ./launch-device.sh;
    }}
    ${installEntry {
      name = "lfm3-check";
      module = "check";
      template = ./launch-device.sh;
    }}
    substituteInPlace "$out/bin/lfm3-run" "$out/bin/lfm3-check" \
      --replace-fail @overlay@ "$out/libexec/tt-metalium" \
      --replace-fail @ttnn@ "${ttnnSitePackages}"

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    export HOME="$TMPDIR"
    for entry in lfm3-run lfm3-check; do
      "$out/bin/$entry" --help > /dev/null
    done
    "$out/bin/lfm3-check" --help | grep -q -- --reference
    runHook postInstallCheck
  '';

  passthru = { inherit pythonEnvironment; };

  meta = {
    description = "LFM3-1.7B on one Blackhole P150 through TT-NN";
    license = lib.licenses.lgpl21Only;
    platforms = [ "x86_64-linux" ];
  };
}
