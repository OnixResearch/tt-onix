{ stdenv, python3 }:
let
  # Match the deployed daemon and nix-grpc-store C plugin ABI, including its
  # libraries and toolchain. These pins reproduce the unmodified CLI derivation
  # bbysj1afh25692x2438bs1dxjj504k27-nix-2.36.0.drv, not nixpkgs' default Nix.
  source = builtins.fetchTree {
    type = "github";
    owner = "onixcomputer";
    repo = "nix";
    rev = "388dea6acc4d45c7d47c9debcc105435ed0a995e";
    narHash = "sha256-2ZkHYHN1YGOjsw7M/l03Wi60WROTaCgOhshT14ZnOYs=";
  };
  nixpkgs = builtins.fetchTree {
    type = "tarball";
    url = "https://releases.nixos.org/nixos/26.05/nixos-26.05.4193.a50de1b7d8a5/nixexprs.tar.xz";
    narHash = "sha256-H9+N+GFtsbVC8ZniHliChM7ndizxtqVZs6bnGOLM3WQ=";
  };
  upstream = (import (source + "/flake.nix")).outputs {
    self = source;
    nixpkgs = nixpkgs // {
      lib = (import nixpkgs { system = stdenv.hostPlatform.system; }).lib;
    };
    # Only the internal packaging overlay is evaluated; regression-flake inputs
    # are irrelevant to that overlay and need not be fetched.
    nixpkgs-regression = nixpkgs;
  };
  matchedPkgs = import nixpkgs {
    system = stdenv.hostPlatform.system;
    overlays = [ upstream.overlays.internal ];
  };
in
matchedPkgs.nixComponents2.nix-cli.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ [ ./nix-2.36-bounded-lock-filename.patch ];
  doInstallCheck = true;
  nativeInstallCheckInputs = [ python3 ];
  installCheckPhase = ''
    runHook preInstallCheck
    python3 ${./test-slot-locks.py} "$out/bin/nix"
    runHook postInstallCheck
  '';
  meta = old.meta // {
    description = "Nix 2.36.0 CLI with filesystem-bounded remote builder slot locks";
  };
})
