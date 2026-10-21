{
  lib,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  pname = "model-target-consumer-generator";
  version = "0.1.0";

  src = lib.cleanSource ./.;
  cargoLock.lockFile = ./Cargo.lock;

  doCheck = true;
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/model-target-consumer-generator" --self-test
    runHook postInstallCheck
  '';

  meta = {
    description = "Deterministic model-target consumer identity generator";
    license = lib.licenses.lgpl21Only;
    mainProgram = "model-target-consumer-generator";
    platforms = lib.platforms.linux;
  };
}
