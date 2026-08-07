{
  lib,
  pkgs,
}:
let
  expectedSchemaVersion = 1;
  profileData = builtins.fromJSON (builtins.readFile ./generated/profiles.json);
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
  inherit (targetLibrary) mkModelTarget;
}
