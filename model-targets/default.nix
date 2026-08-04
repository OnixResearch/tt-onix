{
  lib,
  pkgs,
}:
let
  expectedSchemaVersion = 1;
  profileData = builtins.fromJSON (builtins.readFile ./generated/profiles.json);
  profilesByName = builtins.listToAttrs (
    map (profile: {
      name = profile.target_name;
      value = profile;
    }) profileData.profiles
  );
  profileNamesAreUnique =
    builtins.length profileData.profiles == builtins.length (builtins.attrNames profilesByName);
  targetLibrary = import ./mk-model-target.nix { inherit lib; };

  getProfile = targetName:
    if builtins.hasAttr targetName profilesByName then
      profilesByName.${targetName}
    else
      throw "unknown model target profile '${targetName}'";

  fetchProfileArtifacts = profile:
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

  mkArchitectureAdapter = profile:
    if profile.model.architecture_family == "rwkv7" then
      {
        id = profile.adapters.architecture;
        architectureFamilies = [ "rwkv7" ];
        stateKinds = [ "recurrent_matrix" ];
        checkpointLayouts = [ "safetensors_single_file" ];
        precisions = [ "bf16_checkpoint_fp32_reference" ];
        topologies = [ "p150x2-exact-physical-0-1" ];
        equations = "rwkv7_generalized_delta_rule";
        tensorLayout = "rwkv7_hf_world2_8";
        requiredOperations = rwkvRequiredOperations;
        referenceAuthority = "rwkv7_paper_and_package_matched_cpu_fp32";
        tolerances = {
          finalHidden = 0.5;
          logits = 1.0;
          recurrentState = 1.0;
          controlDivergenceFloor = 0.5;
        };
      }
    else if profile.model.architecture_family == "synthetic_transformer" then
      {
        id = profile.adapters.architecture;
        architectureFamilies = [ "synthetic_transformer" ];
        stateKinds = [ "kv_cache" ];
        checkpointLayouts = [ "synthetic_single_file" ];
        precisions = [ "bf16_fixture" ];
        topologies = [ "device-free-fixture" ];
        equations = "scaled_dot_product_attention_fixture";
        tensorLayout = "transformer_qkv_fixture";
        requiredOperations = transformerFixtureRequiredOperations;
        referenceAuthority = "device_free_synthetic_fixture";
        tolerances = {
          logits = 0.01;
          kvCache = 0.01;
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
assert lib.assertMsg (profileData.schema_version == expectedSchemaVersion)
  "model target profile schema version mismatch";
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
