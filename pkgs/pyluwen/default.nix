{
  lib,
  buildPythonPackage,
  runCommand,
  fetchFromGitHub,
  rustPlatform,
  maturin,
  protobuf_30,
}:
buildPythonPackage rec {
  pname = "pyluwen";
  version = "0.10.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "luwen";
    tag = "v${version}";
    hash = "sha256-J0SvCBsDi3GMvnwqgqGMUDvrSO+baznDhsfuvF44Ens=";
  };

  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit src;
    hash = "sha256-vEuVxBGIJAyc8POy3EPTzTK5g5SE3QDZcnTnAb3b5k0=";
  };

  sourceRoot = "${src.name}/bind/${pname}";

  prePatch = ''
    chmod -R u+w ../../
    cd ../../
  '';

  postPatch = ''
    cd ../$sourceRoot
    cp --no-preserve=ownership,mode ../../Cargo.lock .
    sed -i '0,/version = /{s/version = "*.*.*"/version = "${version}"/g}' Cargo.toml
  '';

  nativeBuildInputs = with rustPlatform; [
    cargoSetupHook
    maturinBuildHook
    protobuf_30
  ];

  build-system = [ maturin ];

  meta = {
    description = "Tenstorrent system interface library";
    homepage = "https://github.com/tenstorrent/luwen";
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    license = with lib.licenses; [ asl20 ];
  };
}
