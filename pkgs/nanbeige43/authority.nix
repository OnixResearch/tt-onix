# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.checkpoint_authority]
# Evaluation-time validator for the pinned Nanbeige4.3-33B checkpoint authority.
# The canonical authority is admitted only when it matches the recorded
# revision, shard layout, and artifact roles; the package installCheck runs the
# same validator on mutated fixtures to prove drift is rejected.
{ lib, canonicalAuthority }:
let
  require = condition: diagnostic: if condition then true else throw diagnostic;
  expectedRevision = "c1093c501cc6460427c0ac634db7223d0450c0ee";
  expectedShardCount = 14;
  expectedMetadataRoles = [
    "added_tokens"
    "checkpoint_index"
    "generation_config"
    "model_config"
    "special_tokens_map"
    "tokenizer"
    "tokenizer_config"
    "tokenizer_model"
  ];
  isHex = length: value: builtins.match "[0-9a-f]{${toString length}}" value != null;
  shards = authority: builtins.filter (file: file.role == "checkpoint_shard") authority.files;
  metadata = authority: builtins.filter (file: file.role != "checkpoint_shard") authority.files;
  names = authority: map (file: file.name) authority.files;
  shardBytes = authority: lib.foldl' (sum: file: sum + file.size) 0 (shards authority);
  validation = authority: [
    (require (
      authority.schema == "nanbeige43-checkpoint-authority/v1"
    ) "nanbeige43 authority schema mismatch")
    (require (
      authority.repository == "Nanbeige/Nanbeige4.3-33B"
    ) "nanbeige43 authority repository mismatch")
    (require (authority.revision == expectedRevision) "nanbeige43 authority revision mismatch")
    (require (
      authority.revision == canonicalAuthority.revision
    ) "nanbeige43 authority drifted from canonical revision")
    (require (
      authority.checkpoint_shard_count == expectedShardCount
      && builtins.length (shards authority) == expectedShardCount
    ) "nanbeige43 authority shard count mismatch")
    (require (
      builtins.length (names authority) == builtins.length (lib.unique (names authority))
    ) "nanbeige43 authority file names are not unique")
    (require (
      builtins.sort builtins.lessThan (map (file: file.role) (metadata authority))
      == expectedMetadataRoles
    ) "nanbeige43 authority metadata roles mismatch")
    (require (builtins.all (
      file: isHex 64 file.sha256
    ) authority.files) "nanbeige43 authority sha256 is not 64 lowercase hex digits")
    (require (builtins.all (
      file: file.size > 0
    ) authority.files) "nanbeige43 authority file size must be positive")
    (require (
      shardBytes authority == authority.checkpoint_shard_bytes
    ) "nanbeige43 authority shard sizes do not sum to checkpoint_shard_bytes")
    (require (
      authority.checkpoint_tensor_bytes == 2 * authority.architecture.parameter_count
    ) "nanbeige43 authority tensor bytes are not BF16 of every parameter")
    (require (
      builtins.listToAttrs (map (file: lib.nameValuePair file.name file.sha256) authority.files)
      == builtins.listToAttrs (
        map (file: lib.nameValuePair file.name file.sha256) canonicalAuthority.files
      )
    ) "nanbeige43 authority file identity drifted from canonical")
  ];
  admit = authority: builtins.deepSeq (validation authority) authority;
in
{
  inherit admit expectedRevision;
  accepts = authority: (builtins.tryEval (admit authority)).success;
}
