let contracts = import "../config.ncl" in
let base = import "../runner.ncl" in
(base & { lora.target_roles | force = [] }) | contracts.ReferenceConfig
