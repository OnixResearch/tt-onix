{
  lib,
  rustPlatform,
  fetchFromGitHub,
  bash,
  coreutils,
}:
rustPlatform.buildRustPackage rec {
  pname = "tt-toplike";
  version = "0.13.17";

  # `sha256-KLMfJlg7ShIQi2cPVvhOMEdzwhTVVRFzWfQ4qGmYdRM=` is the *flat* sha256 of
  # the release tarball (`nix hash file` on
  # .../tenstorrent/tt-toplike/archive/refs/tags/v0.13.17.tar.gz). fetchFromGitHub
  # hashes the unpacked tree instead, so it needs the tree hash below; the flat
  # value fails the fixed-output check in this fetcher.
  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-toplike";
    tag = "v${version}";
    hash = "sha256-u2fv5kzVTo0GuuxVOvb8rkvYOo6gfr1ljzdXNeUFl9o=";
  };

  # Vendored from the pinned v0.13.17 tag. Keeping it in the repository avoids
  # an eval-time fetchurl dependency (aarch64 flake evaluation runs --no-build).
  cargoLock.lockFile = ./Cargo.lock;

  # TUI only: the `tui` feature pulls ratatui/crossterm and nothing else. The
  # gui/egui/app frontends need a GPU-accelerated window stack, and the default
  # feature set adds the JSON (tt-smi) and remote WebSocket backends, which an
  # overlay that ships one monitoring binary does not need.
  buildNoDefaultFeatures = true;
  buildFeatures = [ "tui" ];

  # Only the TUI target: every other [[bin]] is feature-gated off and src/main.rs
  # is an exec shim for `cargo install`, not a program to ship.
  cargoBuildFlags = [
    "--bin"
    "tt-toplike-tui"
  ];

  # The upstream tests spawn `/bin/sleep` and `/bin/sh` to inspect a real
  # process's cwd and stdout. Those paths do not exist in the Nix sandbox.
  # This is a test-only substitution; the shipped TUI never uses them.
  postPatch = ''
    substituteInPlace \
      src/workload/train/monitor.rs \
      src/workload/train/logsrc.rs \
      src/workload/inference_server/probe.rs \
      --replace-fail '"/bin/sleep"' '"${coreutils}/bin/sleep"'
    substituteInPlace \
      src/workload/train/monitor.rs \
      src/workload/inference_server/probe.rs \
      --replace-fail '"/bin/sh"' '"${bash}/bin/sh"'
  '';

  nativeCheckInputs = [
    bash
    coreutils
  ];

  postInstall = ''
    mv $out/bin/tt-toplike-tui $out/bin/tt-toplike
  '';

  # `cargo test` also compiles examples/, and upstream's
  # examples/gui_concept.rs is written against the `gui` feature (iced), which a
  # TUI-only build disables. Scope the run to the library, binaries and tests/;
  # the postPatch above makes the process-spawning tests work without /bin.
  doCheck = true;
  cargoTestFlags = [
    "--lib"
    "--bins"
    "--tests"
  ];

  meta = {
    description = "Real-time hardware telemetry TUI for Tenstorrent silicon";
    homepage = "https://github.com/tenstorrent/tt-toplike";
    changelog = "https://github.com/tenstorrent/tt-toplike/blob/${src.tag}/CHANGELOG.md";
    mainProgram = "tt-toplike";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
