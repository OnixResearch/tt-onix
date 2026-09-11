# TT-Metal 0.77.0 with the GDN grid instance cap lifted.
#
# r[impl tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
#
# `pkgs/tt-metal/gdn-decode-fp32.patch` vendors the fused GDN decode op, whose
# program factory opens exactly one instance per compute core and refuses any
# head count above the core count. That bounds the admitted fast decode path at
# B=4 on a P150x2 mesh, because the decode head count is `B * local_heads`.
#
# This derivation appends a second patch that assigns each active core a
# contiguous chunk of `ceil(num_heads / ncores)` instances. When the head count
# is at or below the core count the chunk is one and the produced program
# mapping is identical to the unpatched mapping, so every receipt produced under
# that mapping stays valid.
#
# The patch is appended rather than folded into `gdn-decode-fp32.patch` so the
# shared `tt-metal` output stays byte-identical for RWKV-7 and the training
# platform. Only the Qwen3.8 decode candidate consumes this variant.
{
  tt-metal,
}:
tt-metal.overrideAttrs (old: {
  # `patches` are applied in list order, so this one sees the tree after
  # gdn-decode-fp32.patch has created the gated_delta_decode sources.
  patches = (old.patches or [ ]) ++ [ ../tt-metal/gdn-decode-grid-chunk.patch ];
})
