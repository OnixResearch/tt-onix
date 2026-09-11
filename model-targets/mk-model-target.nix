{ lib }:
let
  require = condition: diagnostic: if condition then true else throw diagnostic;

  sortedNames = value: builtins.sort builtins.lessThan (builtins.attrNames value);

  profileArtifacts =
    profile:
    builtins.listToAttrs (
      map (artifact: {
        name = artifact.role;
        value = {
          inherit (artifact) blake3 nix_hash;
        };
      }) profile.artifacts
    );

  artifactIdentityMatches =
    expected: actual: expected.blake3 == actual.blake3 && expected.nix_hash == actual.nix_hash;

  validateConsumerIdentity =
    profile:
    let
      projectedProfile = builtins.removeAttrs profile.consumer_identity [ "profile_blake3" ];
      selectedProfile = builtins.removeAttrs profile [
        "consumer_identity"
        "profile_identity"
      ];
    in
    [
      (require (
        profile.profile_identity == profile.consumer_identity.profile_blake3
      ) "model target profile identity mismatch")
      (require (
        selectedProfile == projectedProfile
      ) "model target generated consumer projection mismatch")
    ];

  validateArtifacts =
    profile: artifacts:
    let
      expected = profileArtifacts profile;
      expectedRoles = map (artifact: artifact.role) profile.artifacts;
      rolesAreUnique = builtins.length expectedRoles == builtins.length (builtins.attrNames expected);
      namesMatch = sortedNames expected == sortedNames artifacts;
      identitiesMatch =
        namesMatch
        && lib.all (role: artifactIdentityMatches expected.${role} artifacts.${role}) (
          builtins.attrNames expected
        );
    in
    [
      (require rolesAreUnique "model target profile contains duplicate artifact roles")
      (require namesMatch "model target artifact role mismatch")
      (require identitiesMatch "model target artifact identity mismatch")
    ];

  validateAdapters = profile: architectureAdapter: evidenceAdapter: topology: [
    (require (
      architectureAdapter.id == profile.adapters.architecture
    ) "model target architecture adapter identity mismatch")
    (require (
      evidenceAdapter.id == profile.adapters.evidence
    ) "model target evidence adapter identity mismatch")
    (require (builtins.elem profile.model.architecture_family architectureAdapter.architectureFamilies) "model target architecture family mismatch")
    (require (builtins.elem profile.state.kind architectureAdapter.stateKinds) "model target state kind mismatch")
    (require (builtins.elem profile.model.checkpoint_layout architectureAdapter.checkpointLayouts) "model target checkpoint layout mismatch")
    (require (builtins.elem profile.model.precision architectureAdapter.precisions) "model target precision mismatch")
    (require (builtins.elem profile.topology.id architectureAdapter.topologies) "model target architecture topology mismatch")
    (require (builtins.elem profile.model.architecture_family evidenceAdapter.architectureFamilies) "model target evidence architecture mismatch")
    (require (builtins.elem profile.state.kind evidenceAdapter.stateKinds) "model target evidence state mismatch")
    (require (topology.id == profile.topology.id) "model target topology identity mismatch")
    (require (topology.meshShape == profile.topology.mesh_shape) "model target mesh shape mismatch")
    (require (
      topology.physicalDeviceIds == profile.topology.physical_device_ids
    ) "model target physical device identity mismatch")
    (require (topology.replication == profile.topology.replication) "model target replication mismatch")
  ];

  mkModelTarget =
    {
      profile,
      artifacts,
      architectureAdapter,
      evidenceAdapter,
      topology,
      builders,
    }:
    let
      validation =
        validateConsumerIdentity profile
        ++ validateArtifacts profile artifacts
        ++ validateAdapters profile architectureAdapter evidenceAdapter topology;
      context = {
        inherit
          architectureAdapter
          artifacts
          evidenceAdapter
          profile
          topology
          ;
      };
      runtime = if profile.runnable then builders.runtime context else null;
      evidence = if profile.runnable then builders.evidence (context // { inherit runtime; }) else null;
      command =
        if profile.runnable then builders.command (context // { inherit evidence runtime; }) else null;
      sourceBoundary =
        if profile.runnable then
          builders.sourceBoundary (context // { inherit command evidence runtime; })
        else
          null;
      closure =
        if profile.runnable then
          builders.closure (
            context
            // {
              inherit
                command
                evidence
                runtime
                sourceBoundary
                ;
            }
          )
        else
          null;
      outputValidation = [
        (require (!profile.runnable || runtime != null) "runnable model target lacks a runtime output")
        (require (!profile.runnable || evidence != null) "runnable model target lacks an evidence output")
        (require (!profile.runnable || command != null) "runnable model target lacks a command output")
        (require (
          !profile.runnable || sourceBoundary != null
        ) "runnable model target lacks a source-boundary output")
        (require (!profile.runnable || closure != null) "runnable model target lacks a closure output")
      ];
      result = {
        inherit
          architectureAdapter
          artifacts
          closure
          command
          evidence
          evidenceAdapter
          profile
          runtime
          sourceBoundary
          topology
          ;
        targetName = profile.target_name;
        runnableOutputExposed = profile.runnable;
      };
    in
    builtins.deepSeq (validation ++ outputValidation) result;
in
{
  inherit mkModelTarget;
}
