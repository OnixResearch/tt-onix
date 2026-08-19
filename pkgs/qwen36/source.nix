# r[impl tenstorrent.qwen36.adopted_source]
# Adopted tt-metal qwen36 stack: tenstorrent/tt-metal at the PR-53543 head.
#
# tt-metal PR 53543 ("Enable Qwen3.8-27B on the existing qwen36 Blackhole
# path") is closed but NOT merged upstream. Pinning the PR head commit makes
# the adopted tree deterministic: it already contains Qwen3.8-27B support on
# top of the main-line qwen36 base. No local patch file is required.
#
# The package installs only the models/ subtree (the qwen36 stack and its
# models/* dependencies) on top of the packaged tt-metal 0.74 runtime.
{
  fetchFromGitHub,
}:
fetchFromGitHub {
  owner = "tenstorrent";
  repo = "tt-metal";
  rev = "d102e7f4512060e09e580e40cf26a870bfa88c55";
  # SHA-256 (fetchzip/fetchFromGitHub semantics: extracted tree content).
  hash = "sha256-jxoNGFOPUHbF2eOvkduBThCOJ7JjUtEAx0pLyusVt9Y=";
}
