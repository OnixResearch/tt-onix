{
  lib,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  pname = "ttsim-runtime-validator";
  version = "0.1.0";

  src = lib.cleanSource ./.;
  cargoLock.lockFile = ./Cargo.lock;
  RUSTFLAGS = "-D warnings";

  meta = {
    description = "Bounded TTsim smoke classifier and simulated-receipt validator";
    license = lib.licenses.lgpl21Only;
    mainProgram = "ttsim-runtime-validator";
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
  };
}
