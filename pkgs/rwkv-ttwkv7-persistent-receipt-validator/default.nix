{
  lib,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  pname = "rwkv-ttwkv7-persistent-receipt-validator";
  version = "0.1.0";

  src = lib.cleanSource ./.;
  cargoLock.lockFile = ./Cargo.lock;
  RUSTFLAGS = "-D warnings";

  meta = {
    description = "Device-free validator for persistent ttWKV7 host receipts";
    license = lib.licenses.lgpl21Only;
    mainProgram = "rwkv-ttwkv7-persistent-receipt-validator";
    platforms = [ "x86_64-linux" ];
  };
}
