# r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
# Named single-P150A acceptance profile for RWKV-7 LoRA bring-up.
# Model, tokenizer, LoRA, and optimizer identities bind the CPU reference
# authority in pkgs/rwkv7-lora-reference/runner.ncl. Limits, tolerances,
# criteria, and timeouts are named here; source code carries no literals.
let contracts = import "contracts.ncl" in
(import "fixtures/base-profile.ncl") | contracts.ProfileClosed | contracts.Profile
