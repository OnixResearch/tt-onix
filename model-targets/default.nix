{
  lib,
  pkgs,
}:
let
  expectedSchemaVersion = 1;
  profileData = builtins.fromJSON (builtins.readFile ./generated/profiles.json);
  qwenAuthorityData = builtins.fromJSON (builtins.readFile ./qwen3-6/generated/authority.json);
  qwenAuthorityValidator = import ./qwen3-6/authority.nix {
    inherit lib;
    canonicalAuthority = qwenAuthorityData;
  };
  generatedConsumerProfiles = import ./generated/consumer-identities.nix;
  rawProfilesByName = builtins.listToAttrs (
    map (profile: {
      name = profile.target_name;
      value = profile;
    }) profileData.profiles
  );
  consumerProfilesByName = builtins.listToAttrs (
    map (profile: {
      name = profile.target_name;
      value = profile;
    }) generatedConsumerProfiles
  );
  profileNamesAreUnique =
    builtins.length profileData.profiles == builtins.length (builtins.attrNames rawProfilesByName);
  targetLibrary = import ./mk-model-target.nix { inherit lib; };

  getProfile =
    targetName:
    if !builtins.hasAttr targetName rawProfilesByName then
      throw "unknown model target profile '${targetName}'"
    else if !builtins.hasAttr targetName consumerProfilesByName then
      throw "missing generated consumer identity for model target profile '${targetName}'"
    else
      let
        rawProfile = rawProfilesByName.${targetName};
        consumerIdentity = consumerProfilesByName.${targetName};
        projectedProfile = builtins.removeAttrs consumerIdentity [ "profile_blake3" ];
      in
      if projectedProfile != rawProfile then
        throw "generated consumer identity mismatch for model target profile '${targetName}'"
      else
        rawProfile
        // {
          profile_identity = consumerIdentity.profile_blake3;
          consumer_identity = consumerIdentity;
        };

  profilesByName = lib.mapAttrs (targetName: _profile: getProfile targetName) rawProfilesByName;

  fetchProfileArtifacts =
    profile:
    builtins.listToAttrs (
      map (artifact: {
        name = artifact.role;
        value = {
          path = pkgs.fetchurl {
            inherit (artifact) name url;
            hash = artifact.nix_hash;
          };
          inherit (artifact) blake3 nix_hash;
        };
      }) profile.artifacts
    );

  artifactPaths = artifacts: lib.mapAttrs (_role: artifact: artifact.path) artifacts;

  rwkvRequiredOperations = [
    "embedding"
    "layer-norm"
    "layer-norm-unaffined"
    "matmul"
    "add"
    "multiply"
    "subtract"
    "divide"
    "sigmoid"
    "tanh"
    "exp"
    "relu"
    "square"
    "sqrt"
    "sum-last"
    "maximum-scalar"
    "transpose"
    "argmax"
  ];
  transformerFixtureRequiredOperations = [
    "embedding"
    "rope"
    "scaled-dot-product-attention"
    "kv-cache-update"
    "matmul"
    "argmax"
  ];
  qwenRequiredOperations = [
    "add"
    "argmax"
    "as_tensor"
    "clone"
    "concat"
    "copy"
    "copy_host_to_device_tensor"
    "cumsum"
    "divide"
    "embedding"
    "eq"
    "exp"
    "fill_cache"
    "from_torch"
    "linear"
    "matmul"
    "maximum"
    "multiply"
    "neg"
    "pad"
    "paged_update_cache"
    "permute"
    "plus_one"
    "repeat"
    "repeat_interleave"
    "reshape"
    "rms_norm"
    "rsqrt"
    "sampling"
    "scatter"
    "sharded_to_interleaved"
    "sigmoid"
    "silu"
    "slice"
    "softmax"
    "softplus"
    "sparse_matmul"
    "subtract"
    "sum"
    "topk"
    "to_layout"
    "to_memory_config"
    "transformer.chunked_scaled_dot_product_attention"
    "transformer.scaled_dot_product_attention"
    "transformer.scaled_dot_product_attention_decode"
    "transpose"
    "tril"
    "typecast"
    "untilize"
    "zeros"
    "zeros_like"
    "ttl.gated_delta_chunk_state"
    "ttl.gated_delta_decode_step"
  ];
  qwenSimulatorAudit = {
    ttsimVersion = "1.9.2";
    runtime = "libttsim_bh.so";
    topology = "blackhole-1x1";
    disposition = "eligible_for_bounded_change";
    requiredOperatorInventoryComplete = true;
    packageMatched = false;
    boundedPathAvailable = true;
    requiredEnvironment = {
      dispatch = "slow";
      disableSfploadmacro = true;
    };
    staticEvidence = [
      "ttsim source includes Blackhole BF16, BFP8, BFP4, matrix, SFPU, NoC, and RISC-V paths"
      "ttsim fails closed for unsupported instruction modes"
    ];
    operatorCoverage = [
      {
        class = "tensor_movement_elementwise_and_normalization";
        status = "inventory_complete_run_not_attempted";
      }
      {
        class = "bf16_bfp8_bfp4_matrix_operations";
        status = "static_format_evidence_only";
      }
      {
        class = "sdpa_and_cache_updates";
        status = "inventory_complete_run_not_attempted";
      }
      {
        class = "sparse_moe_topk_and_sampling";
        status = "inventory_complete_run_not_attempted";
      }
      {
        class = "custom_ttl_gated_delta_kernels";
        status = "blocked_without_package_matched_proof";
      }
    ];
    notRunReasons = [
      "Qwen bring-up requires tt-metal base 3218270556c5ff58eba1e68d112d09a11b37dc0e"
      "repository tt-metal package does not match the Qwen bring-up base"
      "custom ttl Gated DeltaNet kernels lack package-matched TTsim evidence"
      "metadata-only change excludes the 26 checkpoint shards"
    ];
    nonClaims = [
      "no simulator success"
      "no silicon success"
      "no physical placement"
      "no profiling or performance"
      "no thermals or device health"
      "no Qwen3.6 support"
    ];
  };

  mkArchitectureAdapter =
    profile:
    if profile.model.architecture_family == "rwkv7" then
      {
        id = profile.adapters.architecture;
        architectureFamilies = [ "rwkv7" ];
        stateKinds = [ "recurrent_matrix" ];
        checkpointLayouts = [ profile.model.checkpoint_layout ];
        precisions = [ profile.model.precision ];
        topologies = [ profile.topology.id ];
        equations = "rwkv7_generalized_delta_rule";
        tensorLayout = "rwkv7_hf_world2_8";
        requiredOperations = rwkvRequiredOperations;
        referenceAuthority = "rwkv7_paper_and_package_matched_cpu_fp32";
        tolerances = {
          finalHidden = profile.consumer.tolerances.final_hidden;
          logits = profile.consumer.tolerances.logits;
          recurrentState = profile.consumer.tolerances.recurrent_state;
          controlDivergenceFloor = profile.consumer.tolerances.control_divergence_floor;
        };
      }
    else if profile.model.architecture_family == "synthetic_transformer" then
      {
        id = profile.adapters.architecture;
        architectureFamilies = [ "synthetic_transformer" ];
        stateKinds = [ "kv_cache" ];
        checkpointLayouts = [ profile.model.checkpoint_layout ];
        precisions = [ profile.model.precision ];
        topologies = [ profile.topology.id ];
        equations = "scaled_dot_product_attention_fixture";
        tensorLayout = "transformer_qkv_fixture";
        requiredOperations = transformerFixtureRequiredOperations;
        referenceAuthority = "device_free_synthetic_fixture";
        tolerances = {
          logits = profile.consumer.tolerances.logits;
          kvCache = profile.consumer.tolerances.recurrent_state;
        };
      }
    # r[impl tenstorrent.model_targets.qwen3_6_candidate_profile]
    else if profile.model.architecture_family == "qwen3_6_hybrid_moe" then
      {
        id = profile.adapters.architecture;
        architectureFamilies = [ "qwen3_6_hybrid_moe" ];
        stateKinds = [ "hybrid_kv_recurrent" ];
        checkpointLayouts = [ "safetensors_26_shard_index_metadata_only" ];
        precisions = [ "bf4_experts_bf8_attention_bf16_activations_candidate" ];
        topologies = [ "p150x1-candidate" ];
        equations = "qwen3_5_moe_reference_gated_delta_and_scaled_dot_product_attention";
        equationsSource = {
          repository = qwenAuthorityData.upstream.repository_url;
          commit = qwenAuthorityData.upstream.commit;
          path = "models/demos/qwen3_6_a3b/reference/qwen3_5_moe.py";
        };
        tensorLayout = "qwen3_6_text_only_hybrid_moe_candidate";
        stateLifecycle = {
          fullAttention = "fixed-shape KV cache is filled during prefill and updated at each decode position";
          linearAttention = "causal-convolution history and Gated DeltaNet matrix state carry across tokens";
        };
        requiredOperations = qwenRequiredOperations;
        simulatorAudit = qwenSimulatorAudit;
        referenceAuthority = "pinned_knight_ops_equations_and_hugging_face_metadata";
        tolerances = {
          finalHidden = profile.consumer.tolerances.final_hidden;
          logits = profile.consumer.tolerances.logits;
          recurrentState = profile.consumer.tolerances.recurrent_state;
          controlDivergenceFloor = profile.consumer.tolerances.control_divergence_floor;
        };
      }
    else
      throw "unsupported architecture family '${profile.model.architecture_family}'";

  mkEvidenceAdapter = profile: {
    id = profile.adapters.evidence;
    architectureFamilies = [ profile.model.architecture_family ];
    stateKinds = [ profile.state.kind ];
    commonEnvelopeFields = [
      "target_identity"
      "artifact_identities"
      "topology"
      "process_policy"
      "profiler_attribution"
      "cpu_fallback"
      "publication_state"
    ];
  };

  mkTopology = profile: {
    inherit (profile.topology) id replication;
    meshShape = profile.topology.mesh_shape;
    physicalDeviceIds = profile.topology.physical_device_ids;
  };
in
assert lib.assertMsg (
  profileData.schema_version == expectedSchemaVersion
) "model target profile schema version mismatch";
assert lib.assertMsg profileNamesAreUnique "model target profile names must be unique";
{
  inherit
    artifactPaths
    fetchProfileArtifacts
    getProfile
    mkArchitectureAdapter
    mkEvidenceAdapter
    mkTopology
    profilesByName
    ;
  # r[verify tenstorrent.model_targets.qwen3_6_upstream_authority.valid]
  qwenAuthority = qwenAuthorityValidator.admit qwenAuthorityData;
  admitQwenAuthority = qwenAuthorityValidator.admit;
  inherit (targetLibrary) mkModelTarget;
}
