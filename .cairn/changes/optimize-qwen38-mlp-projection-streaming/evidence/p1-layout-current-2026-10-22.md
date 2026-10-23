# Current TP2 decode MLP weight layout — 2026-10-22

Scope: source inspection only, starting from repository HEAD `9f4a819`; no hardware, simulator, kernel execution, or code changes. Citations below use `mlp.py`, `tp_common.py`, and `model_config.py` to mean the files under `.pi/references/tenstorrent/tt-metal-latest/models/demos/blackhole/qwen36/tt/`. Patch citations are repository-relative. Grid dimensions are `(x, y)` (columns, rows), not a bank-ownership claim.

## 1. For the TP2 decode MLP, what layout does each of gate, up, down actually use today: which shard grid (`shard_w`/`shard_h` arguments), what `per_core_N`, which memory config (DRAM-sharded vs interleaved/L1), and how many cores per matmul?

The configured default is **interleaved DRAM weights with 1D multicast decode**, not the separately constructed DRAM-sharded configurations: `self.mlp_1d_decode = True` (`model_config.py:176-202`), while the weight loader requires `and not getattr(args, "mlp_1d_decode", False)` to select DRAM-sharded weights (`mlp.py:54-62`). For TP2, let `D = args.dim`, `H = args.hidden_dim`, and `W = mesh_device.compute_with_storage_grid_size().x`; the specified sources do not supply numeric checkpoint values for `D` or `H` (`model_config.py:37-45,105,183`).

| Projection | Weight mesh sharding and per-device matrix | Actual default worker grid and cores on BH with `W=11` | `per_core_N` (32-element tiles) | Weight / activation-output memory |
| --- | --- | --- | --- | --- |
| Gate (`w1`) | `tpc.shard_w(..., dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat4_b)`; `[D, H/2]`, column-parallel | `(11,4)`, 44 configured cores | `ceil(ceil((H/2)/32)/44)` | DRAM-interleaved weight; L1-interleaved input and output |
| Up (`w3`) | `tpc.shard_w(..., dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat4_b)`; `[D, H/2]`, column-parallel | `(11,4)`, 44 configured cores | `ceil(ceil((H/2)/32)/44)` | DRAM-interleaved weight; L1-interleaved input and output |
| Down (`w2`) | `tpc.shard_w(..., dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)`; `[H/2, D]`, row-parallel | `(11,3)`, 33 configured cores | `ceil(ceil(D/32)/33)` | DRAM-interleaved weight; L1-interleaved gated input and partial output |

Weight arguments and dtypes: `mlp.py:109-135`. `shard_w` transposes `[out,in]` to `[in,out]`, uses `ShardTensorToMesh(mesh, dim=dim)`, `TILE_LAYOUT`, and the passed memory config (`tp_common.py:234-244`). **There are no `shard_w`/`shard_h` grid-dimension arguments in these sources**: `shard_w` is the mesh-weight loader, not a worker-grid helper, and no `shard_h` is defined or used in the three inspected files.

The grid builder uses `cols = min(grid_w, num_cores)`, `rows = math.ceil(num_cores / cols)`, and `per_core_n = math.ceil(n_tiles / (cols * rows))`, then sets `compute_with_storage_grid_size=(cols, rows)`, `per_core_N=per_core_n`, and `mcast_in0=True` (`tp_common.py:102-130`). The calls request 44 cores for gate/up and 33 for down, passing `grid_w=self.decode_grid_w` (`model_config.py:183-202`). For other worker widths the exact configured core count is `min(W,C) * ceil(C/min(W,C))`, with `C=44` or `33`, not necessarily the requested budget. The table's 44/33 counts are configured worker-grid capacity, not a measurement of reader utilization.

Decode explicitly converts gate/up input to `ttnn.L1_MEMORY_CONFIG` and sends both outputs there (`mlp.py:254-271`). The gated activation is L1 (`mlp.py:299-312`); down selects `args.mlp_w2_decode_1d_progcfg` and L1 partial output (`mlp.py:315-329`). The subsequent collective requests DRAM output (`mlp.py:332-340`); that is not down's weight placement.

The alternative DRAM-sharded gate/up branch exists (`mlp.py:80-105,233-253`), but is not the configured default. Merely seeing `mlp_w1_weight_memcfg`, `mlp_w3_weight_memcfg`, or `mlp_w2_weight_memcfg` constructed (`model_config.py:151-156`) does not establish their use; the loader uses interleaved down weights even in the alternative branch (`mlp.py:98-105`).

Packaged-patch check: the execution patch list is `pkgs/qwen36/default.nix:67-96`. Relevant hunks do not change these decode grids or placement: `pkgs/qwen36/layer-type-authority.patch:5-71` changes layer-type authority; `pkgs/qwen36/tp2-runtime-compat.patch:28-50` gates fused **prefill** SwiGLU on runtime support; `pkgs/qwen36/cache-hit-shard-load.patch:12-30` adds cache loading, subsequently refactored by `pkgs/qwen36/cache-hit-fused-projections.patch:4-35`; `pkgs/qwen36/cache-hit-single-projections.patch:1-24` changes the single-device MLP loader. The final TP cache-hit path returns `ttnn.load_tensor(cache_file, device=mesh)` as-is (`pkgs/qwen36/cache-hit-fused-projections.patch:6-16,19-35`), so the table describes configured fresh/matching-cache layout, not an inspection of any on-disk cache.

## 2. Do the matmul reader cores already own DRAM banks one-to-one, or is the access round-robin/interleaved across banks? Cite the bank constants and the sharding helper that decide this.

**The default path uses interleaved DRAM weights, not a configured one-reader-per-bank assignment.** Gate/up/down all pass `ttnn.DRAM_MEMORY_CONFIG` (`mlp.py:109-135`); the 1D builder explicitly requires interleaved weights (`tp_common.py:102-109`). Thus the source-level placement is interleaved across banks, rather than bank-owned worker shards. The exact round-robin page-to-bank order and reader issue schedule are not implemented in these Python files and cannot be asserted from them.

The available bank-related constants are `DRAM_CORES = 8` and `DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})` (`tp_common.py:15-18`). They define an eight-coordinate `(8,1)` logical DRAM shard grid, **not** a mapping from the 44/33 default worker cores to eight banks. The alternative helper `create_dram_sharded_mem_config(k, n)` pads `n` to a multiple of `TILE_SIZE * DRAM_CORES`, constructs `ShardSpec(DRAM_GRID, (k, padded_n // DRAM_CORES), ROW_MAJOR)`, and returns `MemoryConfig(WIDTH_SHARDED, DRAM, shard_spec)` (`tp_common.py:60-73`). That helper establishes eight width shards for the alternative layout; it does not prove physical reader-core/bank ownership, and default MLP loading bypasses it (`mlp.py:58-62,109-135`).

## 3. Would re-sharding to a bank-matched 1×8 shard grid with `per_core_N` derived from that grid reduce weight bytes read or rereads per decode step versus today? Answer exactly one of **yes / no / cannot tell from the source**, and give the reason in ≤6 lines.

**cannot tell from the source**
The Python specifies placement and work partitioning, not weight-reader transactions or per-step reread counts (`tp_common.py:102-130`; `mlp.py:254-271,315-329`).
Its comment that “every core streams the full K” constrains `in0_block_w`; it does not prove duplicate weight reads (`tp_common.py:115-117`).
Changing to eight shards does not itself reduce the logical weight matrix or dtype, and the existing DRAM-sharded helper already defines eight width shards (`tp_common.py:60-73`).
No allowed source establishes either current avoidable weight rereads or fewer transactions after the proposed grid change.

## Limitations

- No hardware or simulator was used; no performance, transaction count, bank bandwidth, or actual reader utilization was measured.
- The allowed sources do not include matmul reader kernels, allocator/page-to-bank mapping, or the runtime implementation of `DRAM_MEMORY_CONFIG`; exact round-robin addressing and physical bank ownership remain unsettled.
- Numeric checkpoint `dim`/`hidden_dim` values and the actual device worker width are outside the supplied sources. Accordingly, `per_core_N` is given exactly as a formula; 44/33-core counts are conditional on the documented BH width of 11 (`model_config.py:178-201`).
- Existing cached tensor metadata was not inspected. Packaged cache hits return stored tensors as-is, so a mismatching cache can differ from the configured creation path (`pkgs/qwen36/cache-hit-fused-projections.patch:6-16,19-35`; `mlp.py:54-56`).
- No alternate bank-matched implementation was executed, and `(1,8)` versus `(8,1)` coordinate conventions cannot be conflated: the existing logical DRAM grid spans x=0..7 at y=0 (`tp_common.py:17-18`).
