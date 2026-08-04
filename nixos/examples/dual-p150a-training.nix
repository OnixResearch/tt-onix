# r[impl tenstorrent.training_platform.configuration]
# Dual-P150A TT-XLA training platform example.
#
# Import this module alongside the repository NixOS module:
#
#   imports = [
#     tenstorrent-nix.nixosModules.default
#     tenstorrent-nix.nixosModules.examples.dual-p150a-training
#   ];
#
# It declares the p150_x2 mesh for two Blackhole P150A cards and the 1G
# hugepage allocation that the pinned TT-XLA runtime image requires.
# It makes no training correctness, convergence, or performance claim.
{ ... }:
{
  hardware.tenstorrent = {
    enable = true;
    # Two P150A cards cabled as one 1x2 Blackhole mesh.
    meshName = "p150_x2";
  };

  # The upstream TT-XLA runtime images mount /dev/hugepages-1G and require
  # reserved 1G pages. Eight pages match the validated baseline host; reduce
  # only with a reviewed memory budget.
  boot.kernelParams = [
    "hugepagesz=1G"
    "hugepages=8"
  ];

  # /dev/hugepages-1G must exist before any training container starts.
  systemd.mounts = [
    {
      what = "hugetlbfs";
      where = "/dev/hugepages-1G";
      type = "hugetlbfs";
      options = "pagesize=1024M,mode=777";
      wantedBy = [ "sysinit.target" ];
    }
  ];

  # Docker runs the digest-pinned runtime image. Rootless or alternative
  # container runtimes are outside the validated platform boundary.
  virtualisation.docker.enable = true;
}
