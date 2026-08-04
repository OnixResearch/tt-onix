{
  lib,
  rustPlatform,
  tokenizerVocabulary,
  deviceRuntime,
}:
rustPlatform.buildRustPackage {
  pname = "rwkv7-p150x2-evidence";
  version = "0.1.0";

  src = lib.cleanSource ./.;
  cargoLock.lockFile = ./Cargo.lock;

  RWKV7_TOKENIZER_VOCABULARY = tokenizerVocabulary;
  RWKV7_DEVICE_RUNTIME = deviceRuntime;

  doCheck = true;
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-evidence" --self-test
    "$out/bin/rwkv7-p150x2-command" --self-test
    runHook postInstallCheck
  '';

  meta = {
    description = "Fail-closed raw evidence classifier for RWKV-7 generation on P150x2";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "rwkv7-p150x2-evidence";
  };
}
