# r[impl tenstorrent.harness.measurement_profile]
# Measurement extends the admitted parity profile without changing it.
let contracts = import "contracts.ncl" in
let parity = import "profile.ncl" in
let measurement_repeat_count = 5 in
let measurement = {
  profile_id = "rwkv7-lora-dual-p150a-measurement",
  parity_profile_id = "rwkv7-lora-dual-p150a",
  repeat_count = measurement_repeat_count,
  metric_set = ["stage_time_seconds", "loss"],
  machine_fact_set = [
    "board_identity",
    "firmware_version",
    "mesh_descriptor_identity",
    "pcie_link",
    "power_state",
  ],
} | contracts.MeasurementProfile in
parity & {
  parity_profile = parity,
  measurement_profile = measurement,
}
