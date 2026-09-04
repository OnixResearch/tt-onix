{
  cairn,
  cairnPolicy,
  evidenceEnvironment,
  lib,
  pkgs,
}:
let
  repositoryRoot = ../.;
  repositorySource = lib.cleanSource repositoryRoot;
  qualityVersion = "0.1.0";

  mkCargoQuality =
    {
      name,
      cargoRoot,
      environment ? { },
    }:
    pkgs.rustPlatform.buildRustPackage (
      {
        pname = "${name}-quality";
        version = qualityVersion;
        src = repositorySource;
        sourceRoot = "source";
        inherit cargoRoot;
        cargoLock.lockFile = repositoryRoot + "/${cargoRoot}/Cargo.lock";
        nativeBuildInputs = [
          pkgs.clippy
          pkgs.rustfmt
        ];
        buildPhase = ''
          runHook preBuild
          cd ${lib.escapeShellArg cargoRoot}
          cargo fmt --check
          cargo clippy --all-targets -- -D warnings
          cargo test --all-targets
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          touch "$out"
          runHook postInstall
        '';
        doCheck = false;
      }
      // environment
    );

  cargoChecks = map mkCargoQuality [
    {
      name = "repository-quality";
      cargoRoot = "crates/repository-quality";
    }
    {
      name = "model-target-consumer-generator";
      cargoRoot = "pkgs/model-target-consumer-generator";
    }
    {
      name = "model-target-core";
      cargoRoot = "pkgs/model-target-core";
    }
    {
      name = "model-target-report";
      cargoRoot = "pkgs/model-target-report";
    }
    {
      name = "rwkv-lab";
      cargoRoot = "pkgs/rwkv-lab";
    }
    {
      name = "rwkv-layer-harness";
      cargoRoot = "pkgs/rwkv-layer-harness";
    }
    {
      name = "persistent-receipt-validator";
      cargoRoot = "pkgs/rwkv-ttwkv7-persistent-receipt-validator";
    }
    {
      name = "rwkv7-lora-dual-p150a-core";
      cargoRoot = "pkgs/rwkv7-lora-dual-p150a/core";
    }
    {
      name = "rwkv7-lora-p150a-core";
      cargoRoot = "pkgs/rwkv7-lora-p150a/core";
    }
    {
      name = "rwkv7-p150x2-evidence";
      cargoRoot = "pkgs/rwkv7-p150x2-evidence";
      environment = evidenceEnvironment;
    }
    {
      name = "training-platform-core";
      cargoRoot = "pkgs/training-platform/core";
    }
    {
      name = "ttsim-runtime-validator";
      cargoRoot = "pkgs/ttsim-runtime-validator";
    }
    {
      name = "tt-vibethinker-bench";
      cargoRoot = "pkgs/tt-vibethinker-bench";
    }
  ];

  repositoryQualityPackage = pkgs.rustPlatform.buildRustPackage {
    pname = "repository-quality";
    version = qualityVersion;
    src = repositorySource;
    sourceRoot = "source";
    cargoRoot = "crates/repository-quality";
    cargoLock.lockFile = ../crates/repository-quality/Cargo.lock;
    nativeBuildInputs = [ pkgs.clippy ];
    buildPhase = ''
      runHook preBuild
      cd crates/repository-quality
      cargo build --release
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/bin"
      cp target/release/repository-quality "$out/bin/repository-quality"
      runHook postInstall
    '';
    doCheck = false;
  };

  nixQuality =
    pkgs.runCommand "tenstorrent-nix-quality"
      {
        nativeBuildInputs = [ pkgs.nixfmt ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        mapfile -d "" nix_sources < <(
          find "$src" -type f -name '*.nix' \
            ! -path "$src/quality/fixtures/language/*" -print0 | sort -z
        )
        if [ "''${#nix_sources[@]}" -eq 0 ]; then
          echo "Nix quality found no maintained sources" >&2
          exit 1
        fi
        nixfmt --check "''${nix_sources[@]}"
        touch "$out"
      '';

  shellQuality =
    pkgs.runCommand "tenstorrent-shell-quality"
      {
        nativeBuildInputs = [ pkgs.shellcheck ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        mapfile -d "" scripts < <(
          find "$src" -type f -name '*.sh' \
            ! -path "$src/quality/fixtures/language/*" -print0 | sort -z
        )
        if [ "''${#scripts[@]}" -eq 0 ]; then
          echo "shell quality found no maintained scripts" >&2
          exit 1
        fi
        shellcheck --severity=warning "''${scripts[@]}"
        for script in "''${scripts[@]}"; do
          head -n 1 "$script" | grep -Eq '^#!(/usr/bin/env (bash|sh)|.*/(bash|sh))$'
          bash -n "$script"
        done
        touch "$out"
      '';

  pythonQuality =
    pkgs.runCommand "tenstorrent-python-quality"
      {
        nativeBuildInputs = [ pkgs.ruff ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        cd "$src"
        ruff check --no-cache .
        ruff format --no-cache --check .
        touch "$out"
      '';

  negativeLanguageQuality =
    pkgs.runCommand "tenstorrent-negative-language-quality"
      {
        nativeBuildInputs = [
          pkgs.cargo
          pkgs.clippy
          pkgs.gcc
          pkgs.nickel
          pkgs.nixfmt
          pkgs.ruff
          pkgs.rustc
          pkgs.shellcheck
        ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        cp -r "$src/quality/fixtures/language/rust" "$TMPDIR/rust-negative"
        chmod -R u+w "$TMPDIR/rust-negative"
        if cargo clippy --manifest-path "$TMPDIR/rust-negative/Cargo.toml" \
          --all-targets -- -D warnings >"$TMPDIR/rust.log" 2>&1; then
          echo "negative Rust fixture passed Clippy" >&2
          exit 1
        fi
        grep -F 'too many arguments' "$TMPDIR/rust.log"

        if shellcheck "$src/quality/fixtures/language/shell/missing-shebang.sh" \
          >"$TMPDIR/shell.log" 2>&1; then
          echo "negative shell fixture passed ShellCheck" >&2
          exit 1
        fi
        grep -F 'SC2148' "$TMPDIR/shell.log"

        if ruff check --isolated --no-cache "$src/quality/fixtures/language/python/unused.py" \
          >"$TMPDIR/python.log" 2>&1; then
          echo "negative Python fixture passed Ruff" >&2
          exit 1
        fi
        grep -F 'F401' "$TMPDIR/python.log"

        if nixfmt --check "$src/quality/fixtures/language/nix/malformed.nix" \
          >"$TMPDIR/nix.log" 2>&1; then
          echo "negative Nix fixture passed nixfmt" >&2
          exit 1
        fi
        grep -F 'expecting expression' "$TMPDIR/nix.log" || {
          cat "$TMPDIR/nix.log" >&2
          exit 1
        }

        if nickel typecheck "$src/quality/fixtures/language/nickel/invalid.ncl" \
          >"$TMPDIR/nickel.log" 2>&1; then
          echo "negative Nickel fixture typechecked successfully" >&2
          exit 1
        fi
        grep -F 'unexpected token' "$TMPDIR/nickel.log" || {
          cat "$TMPDIR/nickel.log" >&2
          exit 1
        }
        touch "$out"
      '';

  policyQuality =
    pkgs.runCommand "tenstorrent-quality-policy"
      {
        nativeBuildInputs = [ pkgs.nickel ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        cd "$src"
        nickel export --format json quality/policy.ncl > "$TMPDIR/repository-quality.json"
        cmp "$TMPDIR/repository-quality.json" quality/generated/repository-quality.json
        touch "$out"
      '';

  artifactQuality =
    pkgs.runCommand "tenstorrent-artifact-quality"
      {
        nativeBuildInputs = [ repositoryQualityPackage ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        repository-quality check "$src" "$src/quality/generated/artifact-manifest.json"
        touch "$out"
      '';

  cairnQuality =
    pkgs.runCommand "tenstorrent-cairn-quality"
      {
        nativeBuildInputs = [
          cairn
          repositoryQualityPackage
        ];
        src = repositorySource;
      }
      ''
        set -euo pipefail
        cairn validate --root "$src" --policy ${lib.escapeShellArg cairnPolicy} > "$TMPDIR/cairn.json"
        repository-quality cairn-verdict "$TMPDIR/cairn.json"
        touch "$out"
      '';
in
{
  inherit
    artifactQuality
    cairnQuality
    cargoChecks
    negativeLanguageQuality
    nixQuality
    policyQuality
    pythonQuality
    repositoryQualityPackage
    shellQuality
    ;
}
