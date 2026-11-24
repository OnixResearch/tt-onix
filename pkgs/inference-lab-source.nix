{
  fetchFromGitHub,
}:
let
  rev = "7ec5f6aef4bc64782cb4c4a504832246b5366f1c";
in
{
  inherit rev;
  repo = "OnixResearch/inference-lab";
  rollbackDonorCommit = "5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a";
  src = fetchFromGitHub {
    owner = "OnixResearch";
    repo = "inference-lab";
    inherit rev;
    hash = "sha256-Yc9xhnEDs3XZ0y9k9dp2Tj0jbBn9zeN7SSI/if/QjHo=";
  };
}
