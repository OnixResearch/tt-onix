# r[verify tenstorrent.qwen36.upstream_authority.valid]
# Admission validator for the Qwen3.8-27B/qwen36 adoption authority. A
# canonical (checked-in) authority is admitted only when it matches every
# recorded PR and release fact. The package installCheck exercises the same
# validator on mutated fixtures to prove drift is rejected.
{ lib, canonicalAuthority }:
let
  require = condition: diagnostic: if condition then true else throw diagnostic;
  shardCount = canonicalAuthority.release.checkpoint_shard_count;
  expectedFullAttentionInterval = 4;
  expectedFullAttentionIndices = lib.genList (
    groupIndex: ((groupIndex + 1) * expectedFullAttentionInterval) - 1
  ) canonicalAuthority.release.full_attention_layer_count;
  artifactRoles =
    authority:
    builtins.sort builtins.lessThan (map (artifact: artifact.role) authority.release.artifacts);
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
  shardNamesUnique =
    authority:
    builtins.length authority.release.checkpoint_shard_names
    == builtins.length (lib.unique authority.release.checkpoint_shard_names);
  shardNamesMatchCount =
    authority:
    builtins.length authority.release.checkpoint_shard_names
    == authority.release.checkpoint_shard_count;
  expectedArtifactRoles = [
    "chat_template"
    "checkpoint_index"
    "generation_config"
    "image_preprocessor_config"
    "model_config"
    "model_license"
    "tokenizer"
    "tokenizer_config"
    "tokenizer_merges"
    "tokenizer_vocabulary"
    "video_preprocessor_config"
  ];
  validation = authority: [
    (require (
      authority.upstream.commit == canonicalAuthority.upstream.commit
    ) "qwen36 adoption commit mismatch")
    (require (
      authority.upstream.repository_url == "https://github.com/tenstorrent/tt-metal.git"
    ) "qwen36 adoption repository mismatch")
    (require (
      authority.upstream.merged_upstream == false
    ) "qwen36 adoption records an upstream merge for a closed-but-unmerged PR")
    (require (builtins.elem "p150x2-tp" authority.upstream.local_patches) "qwen36 adoption lost the carried p150x2-tp patch")
    (require (builtins.elem "generate-tp-reset" authority.upstream.local_patches) "qwen36 adoption lost the carried generate-tp-reset patch (hardware-found reset_tp bug)")
    (require (builtins.elem "gdn-seq-fallback" authority.upstream.local_patches) "qwen36 adoption lost the carried gdn-seq-fallback patch (0.74 lacks chunk_gated_delta_rule)")
    (require (builtins.elem "tp2-runtime-compat" authority.upstream.local_patches) "qwen36 adoption lost the carried tp2-runtime-compat patch (0.74 TP=2 fusion gaps)")
    (require (builtins.elem "layer-type-authority" authority.upstream.local_patches) "qwen36 adoption lost the carried layer-type-authority patch (checkpoint layer_types own dispatch)")
    (require (
      authority.pr.state == "closed" && authority.pr.merged == false
    ) "qwen36 PR state does not match closed-unmerged")
    (require (
      authority.pr.head_commit == authority.upstream.commit
    ) "qwen36 PR head commit does not equal the adopted pin")
    (require (authority.release.model_id == "Qwen/Qwen3.8-27B") "qwen36 release model id mismatch")
    (require (
      authority.release.architecture == "Qwen3_5ForConditionalGeneration"
    ) "qwen36 release architecture mismatch")
    (require (
      authority.release.revision == canonicalAuthority.release.revision
    ) "qwen36 release revision mismatch")
    (require (
      authority.release.checkpoint_shard_count == shardCount
    ) "qwen36 checkpoint shard count mismatch")
    (require (shardNamesMatchCount authority) "qwen36 checkpoint shard-name count mismatch")
    (require (shardNamesUnique authority) "qwen36 checkpoint shard names must be unique")
    (require (
      artifactRoles authority == expectedArtifactRoles
    ) "qwen36 metadata artifact role mismatch")
    (require (
      artifactIdentities authority == artifactIdentities canonicalAuthority
    ) "qwen36 metadata artifact identity mismatch")
    (require (
      authority.release.full_attention_interval == expectedFullAttentionInterval
    ) "qwen36 full-attention interval mismatch")
    (require (
      authority.release.full_attention_layer_indices == expectedFullAttentionIndices
    ) "qwen36 full-attention layer indices mismatch")
    (require (
      authority.release.full_attention_layer_count + authority.release.linear_attention_layer_count
      == authority.release.layer_count
    ) "qwen36 hybrid layer counts do not match the declared layer count")
    (require (
      authority.release.weights_excluded == true
    ) "qwen36 authority must not include weight shard bytes")
    (require (authority == canonicalAuthority) "qwen36 adoption authority drift")
  ];
  admit = authority: builtins.deepSeq (validation authority) authority;
in
{
  inherit admit;
}
