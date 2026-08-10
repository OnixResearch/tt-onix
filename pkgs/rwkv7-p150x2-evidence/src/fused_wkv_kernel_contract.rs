//! Static contract guard for the fused RWKV-7 WKV kernel sources.
//!
//! This guard does not compile or run the device kernels. It rejects source
//! drift that removes a recurrence term, changes fixed geometry, aliases the
//! output boundaries, breaks CB balance, or signals completion before writes.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelContractError {
    pub code: &'static str,
    pub detail: String,
}

impl KernelContractError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for KernelContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for KernelContractError {}

const HEAD_GEOMETRY: &str = "constexpr std::uint32_t kHeadTileCount = 2;";
const STATE_GEOMETRY: &str = "kStateTileCount =\n    kHeadTileCount * kHeadTileCount;";
const BLACKHOLE_GUARD: &str = "#if !defined(ARCH_BLACKHOLE)";
const WORMHOLE_BRANCH: &str = "ARCH_WORMHOLE";
const OUTPUT_CB: &str = "tt::CBIndex::c_16";
const NEXT_STATE_OUTPUT_CB: &str = "tt::CBIndex::c_17";
const BF16_PACK_BOUNDARY: &str = "pack_reconfig_data_format(kOutputCb);";
const FP32_PACK_BOUNDARY: &str = "pack_reconfig_data_format(kOutputCb, kNextStateOutputCb);";
const READ_BARRIER: &str = "noc_async_read_barrier();";
const PADDING_CLEAR: &str = "clear_vector_padding(first_write_address + tile * tile_bytes);";
const VECTOR_CREDIT: &str = "cb_push_back(cb, kHeadTileCount);";
const WRITE_BARRIER: &str = "noc_async_write_barrier();";
const COMPLETION_SIGNAL: &str = "noc_semaphore_set(completion_semaphore, completed_count);";
const CORRECTION_SCALAR_CAPACITY: &str = "kCorrectionScalarTileCapacity, core_ranges";
const REDUCE_UNINIT: &str = "reduce_uninit(input_cb);";
const BINARY_ADD_INIT: &str = "add_binary_tile_init();";
const REDUCE_FUNCTION_START: &str = "void reduce_row_vector(";
const REDUCE_FUNCTION_END: &str = "\n}\n\nvoid center_readout";
const MAXIMUM_BATCH_PROBE: &str =
    "kFusedWkvProbeBatchSize =\n    rwkv7::fused_wkv::kMaximumBatchSize;";
const FUSED_LAYER_SMOKE_POLICY: &str =
    "if (execution_policy == WkvExecutionPolicy::RequireFusedCandidate) {";
const FUSED_LAYER_SMOKE_INVOCATION: &str = "run_simulator_smoke(parse_fused_wkv_layer_smoke_output(arguments),\n                          WkvExecutionPolicy::RequireFusedCandidate);";
const FUSED_LAYER_SMOKE_OUTCOME: &str = "simulated_single_card_fused_wkv_layer_state_carry";
const STATE_CARRY_GUARD: &str = "require(state_changed_after_carry,";
const RESET_CONTROL_GUARD: &str = "require(carried_state_differs_from_reset_control,";
const STATE_RANK_VALIDATION: &str = "bool has_state_shape(const Tensor &tensor, std::uint32_t batch_size) {\n  const auto &shape = tensor.logical_shape();\n  return shape.rank() == kTensorRank &&";
const STATE_DTYPE_VALIDATION: &str = "inputs.state.dtype() != tt::tt_metal::DataType::FLOAT32";
const VECTOR_DTYPE_VALIDATION: &str = "vector->dtype() != tt::tt_metal::DataType::BFLOAT16";
const VECTOR_LAYOUT_VALIDATION: &str = "vector->layout() != ttnn::Layout::TILE";

const REQUIRED_RECURRENCE: [(&str, &str); 5] = [
    (
        "state_dot_a",
        "matmul_tiles(kStateCb, kACb, state_tile, column_tile, row_tile);",
    ),
    (
        "decay",
        "mul_tiles_bcast_rows(\n        kStateCb, kWCb, output_tile, column_tile, output_tile);",
    ),
    (
        "rank_update",
        "matmul_tiles(\n        kStateDotACb, kBCb, row_tile, column_tile, output_tile);",
    ),
    (
        "outer_update",
        "matmul_tiles(kVCb, kKCb, row_tile, column_tile, output_tile);",
    ),
    (
        "readout",
        "matmul_tiles(\n          kNextStateCb, kRCb, state_tile, column_tile, row_tile);",
    ),
];

const REQUIRED_CORRECTION: [(&str, &str); 6] = [
    (
        "per-head group normalization",
        "reduce_row_vector(kReadoutRowCb, kMeanCb);",
    ),
    ("group-normalization affine", "apply_group_norm_affine();"),
    (
        "receptance-key-rk correction",
        "multiply_rows(kCorrectionScalarCb, kRawRkCb, kCorrectionVectorCb);",
    ),
    (
        "correction reduction",
        "reduce_row_vector(kCorrectionVectorCb, kCorrectionScalarCb);",
    ),
    (
        "value-scaled correction",
        "mul_tiles_bcast_scalar(\n        kRawVCb, kCorrectionScalarCb, tile, kFirstTile, tile);",
    ),
    ("gate before output", "apply_gate_and_emit_output();"),
];

const BALANCED_COMPUTE_CBS: [(&str, &str, &str); 6] = [
    (
        "raw_vectors",
        "cb_wait_front(kRawVectorCbs[vector]",
        "cb_pop_front(kRawVectorCbs[vector]",
    ),
    (
        "oriented_vectors",
        "cb_wait_front(kOrientedVectorCbs[vector]",
        "cb_pop_front(kOrientedVectorCbs[vector]",
    ),
    ("state", "cb_wait_front(kStateCb", "cb_pop_front(kStateCb"),
    (
        "state_dot_a",
        "cb_wait_front(kStateDotACb",
        "cb_pop_front(kStateDotACb",
    ),
    (
        "next_state",
        "cb_wait_front(kNextStateCb",
        "cb_pop_front(kNextStateCb",
    ),
    (
        "readout",
        "cb_wait_front(kReadoutCb",
        "cb_pop_front(kReadoutCb",
    ),
];

fn count(source: &str, pattern: &str) -> usize {
    source.match_indices(pattern).count()
}

fn require_once(source: &str, name: &str, pattern: &str) -> Result<(), KernelContractError> {
    let actual = count(source, pattern);
    if actual != 1 {
        return Err(KernelContractError::new(
            "source_pattern_count",
            format!("{name} occurs {actual} times, expected exactly one"),
        ));
    }
    Ok(())
}

fn require_count(
    source: &str,
    name: &str,
    pattern: &str,
    expected: usize,
) -> Result<(), KernelContractError> {
    let actual = count(source, pattern);
    if actual != expected {
        return Err(KernelContractError::new(
            "source_pattern_count",
            format!("{name} occurs {actual} times, expected {expected}"),
        ));
    }
    Ok(())
}

fn validate_blackhole_only(source: &str, kernel: &str) -> Result<(), KernelContractError> {
    require_once(
        source,
        &format!("{kernel} Blackhole guard"),
        BLACKHOLE_GUARD,
    )?;
    if source.contains(WORMHOLE_BRANCH) {
        return Err(KernelContractError::new(
            "unsupported_architecture_branch",
            format!("the fused {kernel} kernel must not contain a Wormhole branch"),
        ));
    }
    Ok(())
}

/// Validate the fused compute-kernel source structure.
pub fn validate_fused_compute_source(source: &str) -> Result<(), KernelContractError> {
    require_once(source, "64-wide head geometry", HEAD_GEOMETRY)?;
    require_once(source, "four-tile state geometry", STATE_GEOMETRY)?;
    validate_blackhole_only(source, "compute")?;

    for (name, pattern) in REQUIRED_RECURRENCE {
        require_once(source, name, pattern)?;
    }
    for (name, pattern) in REQUIRED_CORRECTION {
        require_once(source, name, pattern)?;
    }
    for (name, pattern) in [
        (
            "vector transpose table",
            "kTransposeVector[kRecurrenceVectorCount]",
        ),
        ("vector orientation", "orient_vectors();"),
    ] {
        require_once(source, name, pattern)?;
    }

    require_once(source, "BF16 output CB", OUTPUT_CB)?;
    require_once(source, "FP32 next-state output CB", NEXT_STATE_OUTPUT_CB)?;
    require_once(source, "BF16 pack boundary", BF16_PACK_BOUNDARY)?;
    require_once(source, "FP32 pack boundary", FP32_PACK_BOUNDARY)?;
    require_once(source, "reduce uninitialization", REDUCE_UNINIT)?;

    let reduce_function_start = source.find(REDUCE_FUNCTION_START).ok_or_else(|| {
        KernelContractError::new(
            "missing_reduce_function",
            "the row reduction function is absent",
        )
    })?;
    let reduce_function_tail = &source[reduce_function_start..];
    let reduce_function_end = reduce_function_tail
        .find(REDUCE_FUNCTION_END)
        .ok_or_else(|| {
            KernelContractError::new(
                "missing_reduce_function_end",
                "the row reduction function end is absent",
            )
        })?;
    let reduce_function = &reduce_function_tail[..reduce_function_end];
    let reduce_uninit = reduce_function.find(REDUCE_UNINIT).ok_or_else(|| {
        KernelContractError::new(
            "missing_reduce_uninit",
            "the reduction state is not uninitialized",
        )
    })?;
    let binary_add_init = reduce_function.find(BINARY_ADD_INIT).ok_or_else(|| {
        KernelContractError::new(
            "missing_binary_add_init",
            "the post-reduce binary operation is not initialized",
        )
    })?;
    if reduce_uninit >= binary_add_init {
        return Err(KernelContractError::new(
            "reduce_reconfiguration_order",
            "reduce_uninit must run before the next binary operation is initialized",
        ));
    }

    let bf16_boundary = source.find(BF16_PACK_BOUNDARY).ok_or_else(|| {
        KernelContractError::new("missing_boundary", "BF16 output boundary is absent")
    })?;
    let fp32_boundary = source.find(FP32_PACK_BOUNDARY).ok_or_else(|| {
        KernelContractError::new("missing_boundary", "FP32 state boundary is absent")
    })?;
    if bf16_boundary >= fp32_boundary {
        return Err(KernelContractError::new(
            "boundary_order",
            "the BF16 output pack must occur before the FP32 state pack",
        ));
    }

    for (name, wait_pattern, pop_pattern) in BALANCED_COMPUTE_CBS {
        let waits = count(source, wait_pattern);
        let pops = count(source, pop_pattern);
        if waits != 1 || pops != 1 {
            return Err(KernelContractError::new(
                "unbalanced_circular_buffer",
                format!("{name} has {waits} waits and {pops} pops, expected one of each"),
            ));
        }
    }

    Ok(())
}

/// Validate the BRISC reader source structure.
pub fn validate_fused_reader_source(source: &str) -> Result<(), KernelContractError> {
    require_once(source, "reader head geometry", HEAD_GEOMETRY)?;
    require_once(source, "reader state geometry", STATE_GEOMETRY)?;
    validate_blackhole_only(source, "reader")?;
    require_count(source, "reader page reads", "noc_async_read_page(", 2)?;
    require_count(source, "reader barriers", READ_BARRIER, 2)?;
    require_count(source, "reader CB reservations", "cb_reserve_back(", 2)?;
    require_count(source, "reader CB credits", "cb_push_back(", 2)?;
    require_once(source, "reader vector padding clear", PADDING_CLEAR)?;
    require_once(
        source,
        "reader work-list argument",
        "kWorkListStartArg + work_index",
    )?;

    let vector_barrier = source.find(READ_BARRIER).ok_or_else(|| {
        KernelContractError::new("missing_read_barrier", "reader barrier is absent")
    })?;
    let padding_clear = source.find(PADDING_CLEAR).ok_or_else(|| {
        KernelContractError::new("missing_padding_clear", "vector padding clear is absent")
    })?;
    let vector_credit = source.find(VECTOR_CREDIT).ok_or_else(|| {
        KernelContractError::new("missing_vector_credit", "vector CB credit is absent")
    })?;
    if !(vector_barrier < padding_clear && padding_clear < vector_credit) {
        return Err(KernelContractError::new(
            "padding_clear_order",
            "the vector padding clear must follow the NoC barrier and precede CB publication",
        ));
    }
    if source.contains("noc_async_read(") {
        return Err(KernelContractError::new(
            "unbounded_reader_transfer",
            "the fused reader must use page reads for vector and state tiles",
        ));
    }
    Ok(())
}

/// Validate the NCRISC writer source structure.
pub fn validate_fused_writer_source(source: &str) -> Result<(), KernelContractError> {
    require_once(source, "writer head geometry", HEAD_GEOMETRY)?;
    require_once(source, "writer state geometry", STATE_GEOMETRY)?;
    validate_blackhole_only(source, "writer")?;
    require_once(source, "writer BF16 output CB", OUTPUT_CB)?;
    require_once(source, "writer FP32 state CB", NEXT_STATE_OUTPUT_CB)?;
    require_once(source, "writer page write", "noc_async_write_page(")?;
    require_once(source, "writer completion barrier", WRITE_BARRIER)?;
    require_once(source, "writer completion count", "++completed_count;")?;
    require_once(source, "writer completion signal", COMPLETION_SIGNAL)?;
    require_once(
        source,
        "writer work-list argument",
        "kWorkListStartArg + work_index",
    )?;
    require_once(source, "writer CB wait", "cb_wait_front(cb, tile_count);")?;
    require_once(source, "writer CB pop", "cb_pop_front(cb, tile_count);")?;

    let final_barrier = source.rfind(WRITE_BARRIER).ok_or_else(|| {
        KernelContractError::new("missing_write_barrier", "writer barrier is absent")
    })?;
    let completion_signal = source.find(COMPLETION_SIGNAL).ok_or_else(|| {
        KernelContractError::new("missing_completion", "completion signal is absent")
    })?;
    if final_barrier >= completion_signal {
        return Err(KernelContractError::new(
            "data_before_signal_order",
            "all NoC writes must complete before the completion signal",
        ));
    }
    Ok(())
}

/// Validate host program construction and fail-closed runtime selection.
pub fn validate_fused_program_source(
    program_source: &str,
    runtime_source: &str,
) -> Result<(), KernelContractError> {
    for (name, pattern) in [
        (
            "round-robin owner",
            "const std::size_t owner = work_item % active_core_count;",
        ),
        (
            "single-writer assertion",
            "fused WKV ownership contains more than one writer",
        ),
        ("reader kernel", "fused_wkv_reader.cpp"),
        ("writer kernel", "fused_wkv_writer.cpp"),
        ("compute kernel", "fused_wkv_compute.cpp"),
        (
            "completion semaphore runtime argument",
            "args.push_back(kCompletionSemaphoreId);",
        ),
        (
            "completion semaphore descriptor",
            "semaphore.id = kCompletionSemaphoreId;",
        ),
        (
            "separate next-state output",
            "output.buffer(), next_state.buffer()",
        ),
        ("state-rank validation", STATE_RANK_VALIDATION),
        ("FP32 state validation", STATE_DTYPE_VALIDATION),
        ("BF16 vector validation", VECTOR_DTYPE_VALIDATION),
        ("tile vector validation", VECTOR_LAYOUT_VALIDATION),
        (
            "generic TT-NN device operation",
            "ttnn::generic_op(io_tensors, mesh_descriptor);",
        ),
        (
            "three-tile correction scalar CB cycle",
            CORRECTION_SCALAR_CAPACITY,
        ),
        ("r_k fused input", "&inputs.r_k,"),
        (
            "group-normalization fused input",
            "&inputs.group_norm_weight,",
        ),
        ("gate fused input", "&inputs.gate,"),
    ] {
        require_once(program_source, name, pattern)?;
    }

    for (name, pattern) in [
        (
            "closed physical admission",
            "constexpr bool kFusedWkvPhysicallyAdmitted = false;",
        ),
        ("capability selection", "rwkv7::fused_wkv::select_dispatch("),
        ("fused dispatch branch", "if (decision.use_fused_path)"),
        ("generic fallback", "return generic_wkv_step("),
        ("completed fused correction", "k_rows, v_rows, true"),
        ("maximum-batch cycle probe", MAXIMUM_BATCH_PROBE),
        (
            "explicit fused layer smoke policy",
            FUSED_LAYER_SMOKE_POLICY,
        ),
        (
            "explicit fused layer smoke invocation",
            FUSED_LAYER_SMOKE_INVOCATION,
        ),
        ("fused layer smoke outcome", FUSED_LAYER_SMOKE_OUTCOME),
        ("recurrent state carry guard", STATE_CARRY_GUARD),
        ("reset-matrix control guard", RESET_CONTROL_GUARD),
    ] {
        require_once(runtime_source, name, pattern)?;
    }
    if runtime_source.contains("kFusedWkvPhysicallyAdmitted = true") {
        return Err(KernelContractError::new(
            "unadmitted_fused_dispatch",
            "the production runtime must stay on the generic path before physical admission",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPUTE_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_compute.cpp");
    const READER_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_reader.cpp");
    const WRITER_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_writer.cpp");
    const PROGRAM_SOURCE: &str = include_str!("../../rwkv7-p150x2/src/fused_wkv_program.cpp");
    const RUNTIME_SOURCE: &str = include_str!("../../rwkv7-p150x2/src/main.cpp");

    #[test]
    fn accepts_the_reviewed_compute_source() {
        validate_fused_compute_source(COMPUTE_SOURCE).expect("reviewed compute source");
    }

    #[test]
    fn rejects_a_missing_decay_term() {
        let invalid = COMPUTE_SOURCE.replace(
            "mul_tiles_bcast_rows(\n        kStateCb, kWCb, output_tile, column_tile, output_tile);",
            "// invalid fixture removed the decay term",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("missing decay");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("decay"));
    }

    #[test]
    fn rejects_a_missing_attention_correction() {
        let invalid = COMPUTE_SOURCE.replace(
            "apply_group_norm_affine();",
            "// invalid fixture removed group-normalization affinity",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("missing correction");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("group-normalization affine"));
    }

    #[test]
    fn rejects_wrong_head_geometry() {
        let invalid =
            COMPUTE_SOURCE.replace(HEAD_GEOMETRY, "constexpr std::uint32_t kHeadTileCount = 1;");
        let error = validate_fused_compute_source(&invalid).expect_err("wrong geometry");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("64-wide head geometry"));
    }

    #[test]
    fn rejects_an_unbalanced_state_buffer() {
        let invalid = COMPUTE_SOURCE.replace(
            "cb_pop_front(kStateCb, kStateTileCount);",
            "// invalid fixture removed the state pop",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("unbalanced state");
        assert_eq!(error.code, "unbalanced_circular_buffer");
        assert!(error.detail.contains("state"));
    }

    #[test]
    fn rejects_binary_initialization_before_reduce_uninit() {
        let valid_order = format!("{REDUCE_UNINIT}\n  {BINARY_ADD_INIT}");
        let invalid_order = format!("{BINARY_ADD_INIT}\n  {REDUCE_UNINIT}");
        let invalid = COMPUTE_SOURCE.replace(&valid_order, &invalid_order);
        let error = validate_fused_compute_source(&invalid)
            .expect_err("binary initialization before reduce uninit");
        assert_eq!(error.code, "reduce_reconfiguration_order");
    }

    #[test]
    fn rejects_a_wormhole_branch() {
        let invalid = format!("{COMPUTE_SOURCE}\n#if defined(ARCH_WORMHOLE)\n#endif\n");
        let error = validate_fused_compute_source(&invalid).expect_err("Wormhole branch");
        assert_eq!(error.code, "unsupported_architecture_branch");
    }

    #[test]
    fn accepts_the_reviewed_reader_source() {
        validate_fused_reader_source(READER_SOURCE).expect("reviewed reader source");
    }

    #[test]
    fn rejects_a_reader_without_data_before_credit_barriers() {
        let invalid =
            READER_SOURCE.replace(READ_BARRIER, "// invalid fixture removed a read barrier");
        let error = validate_fused_reader_source(&invalid).expect_err("missing barriers");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reader barriers"));
    }

    #[test]
    fn rejects_a_reader_without_vector_padding_sanitation() {
        let invalid = READER_SOURCE.replace(
            PADDING_CLEAR,
            "// invalid fixture removed vector padding sanitation",
        );
        let error = validate_fused_reader_source(&invalid).expect_err("missing padding clear");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reader vector padding clear"));
    }

    #[test]
    fn accepts_the_reviewed_writer_source() {
        validate_fused_writer_source(WRITER_SOURCE).expect("reviewed writer source");
    }

    #[test]
    fn rejects_a_writer_without_a_data_completion_barrier() {
        let invalid = WRITER_SOURCE.replace(
            WRITE_BARRIER,
            "// invalid fixture removed the write barrier",
        );
        let error = validate_fused_writer_source(&invalid).expect_err("missing write barrier");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("writer completion barrier"));
    }

    #[test]
    fn rejects_a_writer_without_a_completion_signal() {
        let invalid = WRITER_SOURCE.replace(
            COMPLETION_SIGNAL,
            "// invalid fixture removed the completion signal",
        );
        let error = validate_fused_writer_source(&invalid).expect_err("missing completion");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("completion signal"));
    }

    #[test]
    fn accepts_the_reviewed_host_program_and_runtime() {
        validate_fused_program_source(PROGRAM_SOURCE, RUNTIME_SOURCE)
            .expect("reviewed host sources");
    }

    #[test]
    fn rejects_host_ownership_without_round_robin_mapping() {
        let invalid = PROGRAM_SOURCE.replace(
            "const std::size_t owner = work_item % active_core_count;",
            "const std::size_t owner = 0U;",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing ownership mapping");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("round-robin owner"));
    }

    #[test]
    fn rejects_host_program_without_completion_semaphore_wiring() {
        let invalid = PROGRAM_SOURCE.replace(
            "semaphore.id = kCompletionSemaphoreId;",
            "// invalid fixture removed the completion semaphore",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing completion wiring");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("completion semaphore descriptor"));
    }

    #[test]
    fn rejects_host_program_without_state_rank_validation() {
        let invalid = PROGRAM_SOURCE.replace(
            STATE_RANK_VALIDATION,
            "bool has_state_shape(const Tensor &, std::uint32_t) {\n  return true &&",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing state-rank validation");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("state-rank validation"));
    }

    #[test]
    fn rejects_host_program_without_vector_dtype_validation() {
        let invalid = PROGRAM_SOURCE.replace(
            VECTOR_DTYPE_VALIDATION,
            "false /* invalid fixture accepts every vector dtype */",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing vector-dtype validation");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("BF16 vector validation"));
    }

    #[test]
    fn rejects_host_program_without_vector_layout_validation() {
        let invalid = PROGRAM_SOURCE.replace(
            VECTOR_LAYOUT_VALIDATION,
            "false /* invalid fixture accepts every vector layout */",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing vector-layout validation");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("tile vector validation"));
    }

    #[test]
    fn rejects_an_aliased_next_state_output() {
        let invalid = PROGRAM_SOURCE.replace(
            "output.buffer(), next_state.buffer()",
            "output.buffer(), state.buffer()",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("aliased next-state output");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("separate next-state output"));
    }

    #[test]
    fn rejects_a_two_tile_correction_scalar_buffer() {
        let invalid =
            PROGRAM_SOURCE.replace(CORRECTION_SCALAR_CAPACITY, "kHeadTileCount, core_ranges");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("two-tile correction scalar CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("three-tile correction scalar CB cycle")
        );
    }

    #[test]
    fn rejects_a_probe_that_does_not_cover_the_maximum_batch() {
        let invalid = RUNTIME_SOURCE.replace(MAXIMUM_BATCH_PROBE, "kFusedWkvProbeBatchSize = 2U;");
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("probe below the maximum batch");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("maximum-batch cycle probe"));
    }

    #[test]
    fn rejects_a_layer_smoke_that_does_not_require_the_fused_candidate() {
        let invalid = RUNTIME_SOURCE.replace(
            FUSED_LAYER_SMOKE_INVOCATION,
            "run_simulator_smoke(parse_fused_wkv_layer_smoke_output(arguments),\n                          WkvExecutionPolicy::Production);",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("generic layer smoke cannot prove the fused candidate");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("explicit fused layer smoke invocation")
        );
    }

    #[test]
    fn rejects_a_layer_smoke_without_a_state_carry_guard() {
        let invalid = RUNTIME_SOURCE.replace(STATE_CARRY_GUARD, "if (state_changed_after_carry) {");
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("layer smoke without a state carry guard");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("recurrent state carry guard"));
    }

    #[test]
    fn rejects_a_layer_smoke_without_a_reset_matrix_control() {
        let invalid = RUNTIME_SOURCE.replace(
            RESET_CONTROL_GUARD,
            "if (carried_state_differs_from_reset_control) {",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("layer smoke without a reset-matrix control");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reset-matrix control guard"));
    }

    #[test]
    fn rejects_production_runtime_that_admits_fused_dispatch() {
        let invalid = RUNTIME_SOURCE.replace(
            "constexpr bool kFusedWkvPhysicallyAdmitted = false;",
            "constexpr bool kFusedWkvPhysicallyAdmitted = true;",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("unadmitted runtime");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("closed physical admission"));
    }

    #[test]
    fn rejects_production_runtime_without_generic_fallback() {
        let invalid = RUNTIME_SOURCE.replace(
            "return generic_wkv_step(",
            "return removed_generic_wkv_step(",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("missing generic fallback");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("generic fallback"));
    }
}
