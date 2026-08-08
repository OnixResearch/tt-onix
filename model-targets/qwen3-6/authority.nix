{ lib, canonicalAuthority }:
let
  require = condition: diagnostic: if condition then true else throw diagnostic;
  expectedCheckpointShardCount = canonicalAuthority.release.checkpoint_shard_count;
  expectedArtifactRoles = [
    "chat_template"
    "checkpoint_index"
    "generation_config"
    "model_config"
    "model_license"
    "tokenizer"
    "tokenizer_config"
    "tokenizer_merges"
    "tokenizer_vocabulary"
  ];
  artifactRoles =
    authority:
    builtins.sort builtins.lessThan (map (artifact: artifact.role) authority.release.artifacts);
  artifactNames = authority: map (artifact: artifact.name) authority.release.artifacts;
  artifactIdentities =
    authority:
    builtins.listToAttrs (
      map (artifact: {
        name = artifact.role;
        value = {
          inherit (artifact) blake3 nix_hash;
        };
      }) authority.release.artifacts
    );
  checkpointShardNamesAreUnique =
    authority:
    builtins.length authority.release.checkpoint_shard_names
    == builtins.length (lib.unique authority.release.checkpoint_shard_names);
  # r[verify tenstorrent.model_targets.qwen3_6_candidate_profile.weights_excluded]
  containsWeightShard =
    authority:
    lib.any (name: builtins.match "model-[0-9]+-of-[0-9]+\\.safetensors" name != null) (
      artifactNames authority
    );
  validation = authority: [
    (require (
      authority.upstream.commit == canonicalAuthority.upstream.commit
    ) "Qwen3.6 upstream commit mismatch")
    (require (
      authority.upstream.tree_blake3 == canonicalAuthority.upstream.tree_blake3
    ) "Qwen3.6 upstream tree BLAKE3 mismatch")
    (require (
      authority.release.revision == canonicalAuthority.release.revision
    ) "Qwen3.6 release revision mismatch")
    (require (
      artifactIdentities authority == artifactIdentities canonicalAuthority
    ) "Qwen3.6 metadata artifact identity mismatch")
    (require (authority.upstream.branch == "qwen3_6_a3b-bringup") "Qwen3.6 authority branch mismatch")
    (require (
      authority.upstream.subtree == "models/demos/qwen3_6_a3b"
    ) "Qwen3.6 authority subtree mismatch")
    (require (authority.release.license == "Apache-2.0") "Qwen3.6 release license mismatch")
    (require (
      authority.release.checkpoint_shard_count == expectedCheckpointShardCount
    ) "Qwen3.6 checkpoint shard count mismatch")
    (require (
      builtins.length authority.release.checkpoint_shard_names == expectedCheckpointShardCount
    ) "Qwen3.6 checkpoint shard-name count mismatch")
    (require (checkpointShardNamesAreUnique authority) "Qwen3.6 checkpoint shard names must be unique")
    (require (
      artifactRoles authority == expectedArtifactRoles
    ) "Qwen3.6 metadata artifact role mismatch")
    (require (
      !containsWeightShard authority
    ) "Qwen3.6 authority must not contain checkpoint shard bytes")
    (require (authority == canonicalAuthority) "Qwen3.6 authority drift")
  ];
  admit = authority: builtins.deepSeq (validation authority) authority;
in
{
  inherit admit;
}
