# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.registry]
#
# The runtime-profile registry: one record per switchable Tenstorrent model
# runtime. Every value below is transcribed from a source in this tree or from a
# recorded receipt; the comment on each block names that source. Where a field
# does not apply, the profile omits it (the contract's explicit absence) rather
# than recording an empty string.
#
# This file is NOT the model-target registry. `model-targets/profiles.ncl` keeps
# its own `profile_identity`, and `pkgs/rwkv7-p150x2/persistent-decode-profile.ncl`
# pins the RWKV profile identity `d33ed105…`. Adding launch data there would move
# every profile identity and invalidate that admission, so launch data lives here
# and references the model-target identity instead.
#
# `@package@` expands to the profile package's store path. It is the only
# placeholder; the contract refuses any other use.
let contracts = import "runtime-contract.ncl" in

{
  schema_version = 1,
  profiles = [
    # r[impl tenstorrent.runtime_profile.registry.qwen38]
    # The served Qwen3.8-27B runtime. Package and canonical command:
    # flake-modules/onix-packages.nix:883 (attribute `qwen38`) and
    # pkgs/qwen36/default.nix:1159-1170 (`qwen38-p150x2-serve` is the canonical
    # executable; `qwen36-*` are symlinks). The argv is the served command from
    # scripts/qwen38-profile-99.conf, which the NixOS module's own ExecStart
    # agrees with at nixos/module.nix:415-440. The environment is the reviewed
    # production set in scripts/qwen38-profile-service:49-63 ("model identity or
    # production Qwen environment"); the unit additionally sets HOME,
    # MPLCONFIGDIR, XDG_CACHE_HOME, and PYTHONUNBUFFERED for service plumbing
    # (nixos/module.nix:392-405) and those are deliberately not model data.
    {
      schema = "onix.runtime-profile.v1",
      name = "qwen38",
      mode = "serve",
      package = {
        attribute = "qwen38",
        # scripts/qwen38-profile-service:113-118 and scripts/qwen38-deploy:161-163
        # both require this exact store-name suffix.
        store_name = "-qwen38-0.1.0",
      },
      entry = {
        binary = "bin/qwen38-p150x2-serve",
        argv_prefix = [
          "--host",
          "127.0.0.1",
          "--port",
          "8000",
          "--model-path",
          "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
          "--model-alias",
          "Qwen3.8-27B",
          "--max-sequence-length",
          "131072",
          "--max-generation-tokens",
          "8192",
          "--require-systemd-notify",
          "--continuous-slot-policy",
          "@package@/share/qwen36/continuous-slot-policy-b8.json",
          "--prefix-cache-policy",
          "@package@/share/qwen36/prefix-cache-policy.json",
        ],
      },
      env = {
        HF_HUB_OFFLINE = "1",
        HF_MODEL = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        MESH_DEVICE = "P150x2",
        QWEN35_TEMP = "0",
        QWEN35_GDN_DECODE_FUSION = "step",
        QWEN35_GROUPED_PREFILL = "1",
        QWEN36_TRACED_PREFILL = "1",
        QWEN36_SAMPLED_FUSION = "1",
        QWEN36_GDN_FUSED_PREFILL = "0",
        QWEN36_GDN_BF4 = "0",
        QWEN36_MTP_SERVING = "0",
        # nixos/module.nix:404: the TT tensor cache is the checkpoint directory.
        TT_CACHE_PATH = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      },
      mesh = {
        # pkgs/qwen36/service/qwen36_service/server.py:1508-1538 opens
        # ttnn.MeshShape(1, 2) with FABRIC_1D and requires both admitted devices.
        name = "P150x2",
        shape = [ 1, 2 ],
        physical_device_ids = [ 0, 1 ],
        fabric = "FABRIC_1D",
      },
      port = 8000,
      limits = {
        max_sequence_length = 131072,
        max_generation_tokens = 8192,
      },
      checkpoint = {
        path = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        # pkgs/qwen36/decode-profile.ncl:31 and pkgs/qwen36/authority.ncl:10.
        revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        # The package's canonical authority export is hashed, and the value below
        # is that file's BLAKE3 (it is the same value the package records as
        # `identities.authority_blake3` in pkgs/qwen36/decode-profile.ncl:27).
        authority_kind = "generated-authority-blake3",
        authority_path = "pkgs/qwen36/generated/authority.json",
        authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
      },
      cache = {
        # nixos/module.nix:404: no cache key beyond the checkpoint directory.
        root = "/home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      },
      precision = {
        # The service has no named precision preset; these environment keys are
        # the precision and fusion selectors the reviewed production set pins.
        flags = [
          "QWEN35_GDN_DECODE_FUSION",
          "QWEN35_GROUPED_PREFILL",
          "QWEN36_TRACED_PREFILL",
          "QWEN36_SAMPLED_FUSION",
          "QWEN36_GDN_FUSED_PREFILL",
          "QWEN36_GDN_BF4",
        ],
      },
      admission = {
        # scripts/qwen38-deploy:34,93-112: both receipt kinds must name the exact
        # package. A newer admission supersedes these; updating this list is part
        # of that admission change.
        kind = "receipt-set",
        receipts = [
          {
            kind = "continuous-slot-admission",
            path = "pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json",
            blake3 = "4573cea118453a339ad86b73fe67f729e38dd1c5187b9bb09a4ba527e66af11e",
          },
          {
            kind = "prefix-cache-admission",
            path = "pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json",
            blake3 = "eb09b2cc9ec785b223200bbe67b11b6e2665dd25214ce20ca41500220b136f91",
          },
        ],
      },
      readiness = {
        # server.py:1076-1098; the readiness timeout is
        # scripts/qwen38-profile-service:42 (START_TIMEOUT).
        kind = "http-healthz",
        path = "/healthz",
        timeout_seconds = 1250,
      },
      reset = {
        # AGENTS.md:245-247: a reset is required after a TT-Metal owner stops and
        # before native-stack execution.
        on_release = "required-before-native",
        device_paths = [ "/dev/tenstorrent/0", "/dev/tenstorrent/1" ],
      },
      # pkgs/qwen36/default.nix builds libexec/tt-metalium beside the wrappers and
      # exports share/qwen36/execution-identity as a symlink to the identity
      # derivation.
      runtime = {
        root = "libexec/tt-metalium",
        execution_identity = "share/qwen36/execution-identity",
      },
      non_claims = [
        "the served package carries only the receipts that name it",
        "no B=8 v3 interleaved receipt is carried by a servable package",
        "serving readiness is not a throughput, quality, or concurrency admission",
      ],
    },

    # r[impl tenstorrent.runtime_profile.registry.nanbeige43]
    # The Nanbeige4.3-33B runner. Package and wrappers:
    # flake-modules/onix-packages.nix:887 and pkgs/nanbeige43/default.nix:144-150.
    # There is no service: the wrapper appends `--mesh p150x2`, and the caller
    # supplies the mode plus `--checkpoint`, `--cache-root`, and `--receipt`
    # (pkgs/nanbeige43/runner/nanbeige43_runner/run.py:712-717).
    {
      schema = "onix.runtime-profile.v1",
      name = "nanbeige43",
      mode = "job",
      package = {
        attribute = "nanbeige43",
        store_name = "-nanbeige43-0.1.0",
      },
      entry = {
        binary = "bin/nanbeige43-p150x2",
        # pkgs/nanbeige43/default.nix:145-150 appends the production mesh flag.
        argv_prefix = [ "--mesh", "p150x2" ],
      },
      env = {
        # pkgs/nanbeige43/default.nix:135 and run.py:811. The padding value is
        # part of the weight-cache key, so it is launch data, not a detail.
        HF_HUB_OFFLINE = "1",
        PAD_MLP_CORES = "16",
      },
      mesh = {
        # pkgs/nanbeige43/runner/nanbeige43_runner/mesh.py:34-57.
        name = "p150x2",
        shape = [ 1, 2 ],
        physical_device_ids = [ 0, 1 ],
        fabric = "FABRIC_1D",
      },
      limits = {
        # run.py:721-722 defaults: --max-seq-len 8192, --page-block 32,
        # --page-blocks 256; --max-tokens defaults to 128 (run.py:735).
        max_sequence_length = 8192,
        max_generation_tokens = 128,
      },
      checkpoint = {
        # The recorded full-checkpoint directory from
        # .cairn/changes/bring-up-nanbeige43-33b-p150x2/evidence/receipts/L3b-p150x2-full-run2.json.
        path = "/home/brittonr/models/nanbeige4.3-33b-c1093c501cc6",
        # pkgs/nanbeige43/authority.nix:10 and generated/authority.json.
        revision = "c1093c501cc6460427c0ac634db7223d0450c0ee",
        # BLAKE3 of pkgs/nanbeige43/generated/authority.json, the document whose
        # per-file SHA-256 and sizes `verify_checkpoint` enforces before any
        # device touch (runner/nanbeige43_runner/authority.py:1-15,71-104).
        authority_kind = "generated-authority-blake3",
        authority_path = "pkgs/nanbeige43/generated/authority.json",
        authority_blake3 = "4cc332da8d2f2626ff03a60f68ee2f2805799411c2b6cb67444571f2316d345d",
      },
      cache = {
        # run.py:812-815 builds `<cache-root>/<checkpoint.name>/mlp-pad-16`; the
        # observed host root and key are recorded here, and the checkpoint-name
        # segment stays derived at run time.
        root = "/home/brittonr/models/nanbeige43-cache",
        key = "mlp-pad-16",
      },
      precision = {
        # run.py:720: `--preset` defaults to `accuracy`.
        preset = "accuracy",
        flags = [],
      },
      admission = {
        # The checkpoint authority admits the weights before any device open
        # (run.py:788-792; authority.py docstring r[...checkpoint_authority]).
        kind = "authority-digest",
      },
      readiness = {
        # No endpoint. The runner exits non-zero on the first failed gate and
        # prints `[result] all gates passed` (run.py:639-707).
        kind = "exit-code",
      },
      reset = {
        on_release = "required-before-native",
        device_paths = [ "/dev/tenstorrent/0", "/dev/tenstorrent/1" ],
      },
      # pkgs/nanbeige43/default.nix sets TT_METAL_HOME to $out/libexec/tt-metalium.
      runtime = {
        root = "libexec/tt-metalium",
      },
      non_claims = [
        "no serving endpoint and no serving benchmark",
        "batch 8 is refused; batch 32 is accepted for aggregate throughput only",
        "host-wall timings are smoke measurements, not benchmarks",
      ],
    },

    # r[impl tenstorrent.runtime_profile.registry.rwkv7_ttnn]
    # The pinned TT-NN RWKV-7 runtime. Package attribute
    # `rwkv7-p150x2-runtime` (flake-modules/onix-packages.nix:1285), pname
    # `rwkv7-p150x2-runtime` (pkgs/rwkv7-p150x2/default.nix:35-39,61). The
    # wrapper sets TT_METAL_HOME, TT_METAL_RUNTIME_ROOT, TT_MESH_GRAPH_DESC_PATH,
    # and RWKV7_FUSED_WKV_KERNEL_ROOT (default.nix:125-136), so the launcher adds
    # no environment of its own.
    {
      schema = "onix.runtime-profile.v1",
      name = "rwkv7-ttnn",
      mode = "job",
      package = {
        attribute = "rwkv7-p150x2-runtime",
        store_name = "-rwkv7-p150x2-runtime",
      },
      entry = {
        binary = "bin/rwkv7-p150x2-runtime",
        # Every mode is an explicit flag: --self-test, --fused-wkv-*, or
        # --persistent-decode-* (pkgs/rwkv7-p150x2/src/main.cpp:155-188).
        argv_prefix = [],
      },
      env = {},
      mesh = {
        # RWKV7_MESH_DESCRIPTOR_PATH is the p150_x2 mesh-graph descriptor
        # (default.nix:45-52,104-108); the name is that descriptor's basename
        # without the `.textproto` suffix.
        name = "p150_x2_mesh_graph_descriptor",
        shape = [ 1, 2 ],
        physical_device_ids = [ 0, 1 ],
        fabric = "p150_x2_mesh_graph_descriptor",
      },
      limits = {
        # model-targets/profiles.ncl resources.maximum_prompt_tokens (32) and the
        # package's maximumGeneratedTokens default (8,
        # pkgs/rwkv7-p150x2/default.nix:27).
        max_sequence_length = 32,
        max_generation_tokens = 8,
      },
      checkpoint = {
        # No host path: the checkpoint is a Nix-materialized artifact pinned by
        # model-targets/profiles.ncl (role `checkpoint`). Absence is explicit.
        revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216",
        # The runtime bakes the model-target profile identity and the checkpoint
        # BLAKE3 (pkgs/rwkv7-p150x2/default.nix:100-118,128-129).
        authority_kind = "model-target-profile-identity",
        authority_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
      },
      precision = {
        # model-targets/profiles.ncl model.precision for this target.
        preset = "bf16_checkpoint_fp32_reference",
        flags = [],
      },
      admission = {
        kind = "authority-digest",
      },
      readiness = {
        # pkgs/rwkv7-p150x2/default.nix:170-171: `--self-test` must print
        # `rwkv7 P150x2 runtime self-test: PASS`.
        kind = "self-test",
      },
      reset = {
        on_release = "required-before-native",
        device_paths = [ "/dev/tenstorrent/0", "/dev/tenstorrent/1" ],
      },
      model_target_name = "rwkv7-goose-world2.8-0.1b-p150x2",
      model_target_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
      non_claims = [
        "no serving endpoint; generation is a bounded gate",
        "persistent decode is admitted only on the recorded profile",
      ],
    },

    # r[impl tenstorrent.runtime_profile.registry.metalium_rs_rwkv7]
    # The native Rust RWKV-7 gates. Package attribute `metalium-rs-rwkv7-model`
    # (flake-modules/onix-packages.nix:1251), which wraps
    # `<metalium-rs>/bin/rwkv7-model-gate --checkpoint <pinned artifact>`
    # (onix-packages.nix:48-52). `entry.binary` names the whole-model gate;
    # `admission.gates` names every gate that must pass.
    {
      schema = "onix.runtime-profile.v1",
      name = "metalium-rs-rwkv7",
      mode = "job",
      package = {
        attribute = "metalium-rs-rwkv7-model",
        store_name = "-rwkv7-model-checkpoint",
      },
      entry = {
        binary = "bin/rwkv7-model-checkpoint",
        # Gate selection is a flag on the gate binary, not a fixed argument
        # (metalium-rs/crates/rwkv7/src/bin/rwkv7-model-gate.rs:277).
        argv_prefix = [],
      },
      env = {},
      mesh = {
        # metalium-rs/AGENTS.md:128-130: validate on card 0; `--device all` only
        # for mesh, Ethernet, or larger-model work.
        name = "card-0",
        shape = [ 1, 1 ],
        physical_device_ids = [ 0 ],
        fabric = "none (single-card compute; no fabric)",
      },
      limits = {
        # model-targets/profiles.ncl resources.maximum_prompt_tokens (32) and the
        # generate gate's EXPECTED_STEPS (8,
        # metalium-rs/crates/rwkv7/src/bin/rwkv7-generate-gate.rs:28).
        max_sequence_length = 32,
        max_generation_tokens = 8,
      },
      checkpoint = {
        # No host path: the artifact is a fixed-output fetch pinned by URL and
        # SHA-256 at flake-modules/onix-packages.nix:20-24 and by BLAKE3 in
        # model-targets/profiles.ncl.
        revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216",
        authority_kind = "model-target-profile-identity",
        authority_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
      },
      precision = {
        # No named preset: the accurate and `--fast` matmul paths are per-gate
        # choices recorded beside each gate's bounds
        # (metalium-rs/AGENTS.md:210-216).
        flags = [],
      },
      admission = {
        # Each gate is its own writeShellApplication derivation
        # (flake-modules/onix-packages.nix:43-67), so only the whole-model gate
        # lives in this profile's package. The other four are named without an
        # artifact: a package-relative path for them does not exist, and an
        # absolute store path would go stale on every metalium-rs rebuild. `check`
        # verifies the artifact and digest of any gate that records one.
        kind = "gate-exit",
        gates = [
          { name = "rwkv7-model-checkpoint" },
          { name = "rwkv7-layer-checkpoint" },
          { name = "rwkv7-generate-checkpoint" },
          { name = "rwkv7-prompt-checkpoint" },
          { name = "rwkv7-tokenizer" },
        ],
      },
      readiness = {
        kind = "exit-code",
      },
      reset = {
        # A native run is the successor that needs the reset, not the predecessor.
        on_release = "never",
        device_paths = [ "/dev/tenstorrent/0" ],
      },
      model_target_name = "rwkv7-goose-world2.8-0.1b-p150x2",
      model_target_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
      non_claims = [
        "gates do not establish a sampler, stop-policy-complete endpoint, or serving parity",
        "prompt generation uses fast matmuls; the accurate path fragments GDDR after eight tokens",
      ],
    },
  ],
} | contracts.Profiles | contracts.Semantics | contracts.Unique
