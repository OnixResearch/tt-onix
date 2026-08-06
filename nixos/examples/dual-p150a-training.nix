# r[impl tenstorrent.training_platform.configuration]
# r[impl tenstorrent.training_platform.host_module]
# Dual-P150A TT-XLA training platform example.
#
# Import this module with:
#
#   imports = [
#     tenstorrent-nix.nixosModules.default
#     tenstorrent-nix.nixosModules.dual-p150a-training
#   ];
#
# This example makes no training correctness, convergence, or performance claim.
{ ... }:
{
  hardware.tenstorrent = {
    enable = true;
    meshName = "p150_x2";
    accessPolicy = "group-restricted";
    training.enable = true;
  };
}
