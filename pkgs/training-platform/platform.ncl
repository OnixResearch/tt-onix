# r[impl tenstorrent.training_platform.configuration]
# Dual-P150A training platform configuration. Exports deterministic JSON for
# the launcher and the admission core.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  bundle_id = "tt-xla-training-2026-08-04",
  devices = {
    count = 2,
    architecture = "blackhole",
    board_class = "p150",
    device_root = "/dev/tenstorrent",
  },
  mesh = {
    name = "p150_x2",
    rows = 1,
    columns = 2,
    fabric_channels = 4,
    descriptor_relative_path =
      "libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto",
  },
  hugepages = {
    mount = "/dev/hugepages-1G",
    page_size = "1G",
    required_free_pages = 1,
  },
  cache = {
    root = "/var/lib/tenstorrent-training/cache",
  },
  output = {
    evidence_root = "/var/lib/tenstorrent-training/evidence",
  },
  evidence = {
    hash_algorithm = "blake3",
    exclude = ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"],
  },
  smoke = {
    timeout_seconds = 900,
    collective_free = true,
  },
} | contracts.PlatformConfigClosed | contracts.PlatformConfigDeepClosed | contracts.PlatformConfig
