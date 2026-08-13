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
const FP32_READOUT_CB: &str = "make_cb(kReadoutCb, fp32_format, kHeadTileCount, core_ranges)";
const BF16_READOUT_ROW_CB: &str =
    "make_cb(kReadoutRowCb, bf16_format, kHeadTileCount, core_ranges)";
const FP32_READOUT_ROW_CB: &str =
    "make_cb(kReadoutRowCb, fp32_format, kHeadTileCount, core_ranges)";
const DIAGNOSTIC_READOUT_GUARD: &str = "#if RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY";
const DIAGNOSTIC_RECURRENCE_GUARD: &str = "#if RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS";
const STATE_DOT_A_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kStateDotAOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const DECAY_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kDecayOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const RANK_UPDATE_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kRankUpdateOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const OUTER_UPDATE_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kOuterUpdateOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const FIRST_ADD_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kFirstAddOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const BF16_READOUT_DIAGNOSTIC_OUTPUT_CB: &str =
    "constexpr std::uint32_t kBf16ReadoutOutputCb = static_cast<std::uint32_t>(tt::CBIndex::c_16);";
const BF16_READOUT_REVERSE_SOURCE_ORDER: &str =
    "compute_kernel_hw_startup<SrcOrder::Reverse>(kStateCb, kRawRCb,";
const BF16_READOUT_TRANSPOSED_BLOCK_INIT: &str =
    "matmul_block_init(kStateCb, kRawRCb, kTransposeR, kOutputBlockWidth,";
const BF16_READOUT_BLOCK_MATMUL: &str = "matmul_block(";
const BF16_READOUT_OLD_TRANSPOSE_STAGE: &str = "transpose_wh_tile(";
const BF16_READOUT_HIFI2: &str =
    "compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi2;";
const BF16_READOUT_BF16_DESTINATION: &str = "compute_config.fp32_dest_acc_en = false;";
const BF16_READOUT_FP32_STATE_CB: &str =
    "make_cb(kBf16ReadoutStateCb, fp32_format, kStateTileCount,";
const BF16_READOUT_BF16_R_CB: &str = "make_cb(kBf16ReadoutRCb, bf16_format, kHeadTileCount,";
const BLACKHOLE_CIRCULAR_BUFFER_COUNT: &str = "constexpr std::size_t kCircularBufferCount = 64U;";
const DECAY_EXACT_FP32_UNPACK: &str = "compute_config.unpack_to_dest_mode[kRawWCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;";
const SPLIT_READOUT_MODE: &str = "constexpr bool kSplitReadout =\n    RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY != 0 && RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS != 0;";
const SPLIT_READOUT_DEVICE_DEFINE: &str =
    "{std::string(kSplitReadoutDefine), kSplitReadout ? \"1\" : \"0\"}";
const SPLIT_EXACT_FP32_UNPACK_CBS: &str = "const std::array<std::uint8_t, 6U> exact_fp32_cbs = {\n          kExactStateCb, kWCb, kDecayCb, kRankUpdateCb, kOuterUpdateCb, kFirstAddCb};";
const SPLIT_EXACT_FP32_UNPACK_ASSIGNMENT: &str =
    "compute_config.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;";
const SPLIT_READOUT_LAUNCH: &str =
    "run_bf16_readout_diagnostic(mesh, inputs, returned_state, kernel_root);";
const SPLIT_RAW_OUTPUT_OWNERSHIP: &str =
    "return Outputs{std::move(returned_state), std::move(raw_readout), false};";
const MONOLITHIC_CORRECTED_OUTPUT_OWNERSHIP: &str =
    "return Outputs{std::move(returned_state), std::move(output), true};";
const RUNTIME_CORRECTION_OWNERSHIP: &str = "fused.correction_complete,";
const SPLIT_READER_STATE_DUPLICATION: &str = "read_state(kExactStateCb, state_address, work_item,";
const SPLIT_WRITER_STATE_OWNERSHIP: &str = "split_readout ? kSplitNextStateCb : kNextStateOutputCb";
const SPLIT_WRITER_OUTPUT_SKIP: &str = "if (!split_readout) {";
const SHARED_FULL_KERNEL_STATE_DOT_A_TAP: &str = "kStateDotAStageCb";
const SINGLE_PATH_PATTERN_COUNT: usize = 1;
const DIAGNOSTIC_RECURRENCE_GUARD_COUNT: usize = 1;
const GENERIC_FP32_RECURRENCE_VECTOR_POLICY: &str =
    "kGenericFp32RecurrenceVector = {true, true, true, true, false, true};";
const GENERIC_FP32_RECURRENCE_FORMAT: &str =
    "const tt::DataFormat format = kGenericFp32RecurrenceVector[vector]";
const PRODUCTION_BF16_RECURRENCE_FORMAT: &str = "const tt::DataFormat format = bf16_format;";
const RECURRENCE_DEVICE_DEFINE_FACTORY: &str =
    "KernelDescriptor::Defines recurrence_kernel_defines()";
const RECURRENCE_DEVICE_DEFINE_VALUE: &str =
    "RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS ? \"1\" : \"0\"";
const READER_RECURRENCE_RUNTIME_ARG: &str =
    "args.push_back(RWKV7_FUSED_WKV_FP32_RECURRENCE_VECTORS ? 1U : 0U);";
const COMPUTE_RECURRENCE_DEVICE_DEFINE: &str = "compute.defines = recurrence_kernel_defines();";
const BF16_READOUT_PACK_BOUNDARY: &str = "pack_reconfig_data_format(kReadoutRowCb);";
const FP32_PACK_BOUNDARY: &str = "pack_reconfig_data_format(kOutputCb, kNextStateOutputCb);";
const READ_BARRIER: &str = "noc_async_read_barrier();";
const PADDING_CLEAR: &str = "clear_vector_padding(tile_address);";
const SOFTWARE_ROW_BROADCAST: &str =
    "void materialize_bf16_row_broadcast(std::uint32_t tile_address)";
const SOFTWARE_ROW_BROADCAST_CALL: &str = "materialize_bf16_row_broadcast(tile_address);";
const DIAGNOSTIC_ROW_BROADCAST_POLICY: &str = "const bool materialize_w_row_broadcast = get_arg_val<std::uint32_t>(kMaterializeWRowBroadcastArg) == 1U;";
const DIAGNOSTIC_DECAY_MULTIPLY: &str =
    "mul_binary_tile(output_tile, decay_operand_tile, output_tile);";
const PRODUCTION_DECAY_BROADCAST: &str =
    "mul_tiles_bcast_rows(kStateCb, kWCb, output_tile, column_tile, output_tile);";
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
const PHYSICAL_WORKER_POLICY_PARSE: &str =
    "parse_wkv_execution_policy(arguments[kWkvPolicyValueIndex]);";
const INITIALIZATION_READY_SIGNAL: &str =
    "signal_fast_barrier(arguments.initialization_ready_barrier_path,";
const SYNCHRONIZED_WARMUP_START: &str =
    "wait_for_fast_barrier(arguments.warmup_start_barrier_path,";
const TIMED_READY_SIGNAL: &str =
    "signal_fast_barrier(arguments.timed_ready_barrier_path, \"timed ready\");";
const SYNCHRONIZED_WORKER_START: &str =
    "wait_for_fast_barrier(arguments.timed_start_barrier_path, \"timed start\");";
const DISABLED_WORKER_INSPECTOR: &str =
    "std::string_view(inspector_value) == kDisabledEnvironmentValue";
const STALL_PROBE_STEP_COUNT: &str = "require(step_wall_seconds.size() == batch_step_count,";
const STALL_PROBE_RECEIPT: &str = "receipt[\"stall_probe_step_wall_seconds\"] = step_wall_seconds;";
const STALL_PROBE_BOUNDARY_COUNT: &str =
    "require(step_boundary_timings.size() == batch_step_count,";
const STALL_PROBE_BOUNDARY_RECEIPT: &str = "receipt[\"stall_probe_step_boundary_timings\"] =";
const FETCH_QUEUE_DIAGNOSTICS_ENABLE: &str =
    "std::getenv(kFetchQueueDiagnosticsEnvironment.data());";
const FETCH_QUEUE_DIAGNOSTICS_RECEIPT: &str = "receipt[\"fetch_queue_diagnostics\"] = {";
const HARDWARE_COUNTER_SESSION_START: &str = "hardware_counter_session->start();";
const HARDWARE_COUNTER_SESSION_STOP: &str =
    "hardware_counter_diagnostics = hardware_counter_session->stop();";
const HARDWARE_COUNTER_RECEIPT: &str = "receipt[\"hardware_counter_diagnostics\"] =";
const PRIMARY_THREAD_AFFINITY: &str = "sched_setaffinity(0, sizeof(cpu_set), &cpu_set)";
const PRIMARY_THREAD_AFFINITY_RECEIPT: &str = "receipt[\"primary_thread_cpu_affinity\"] =";
const PRIMARY_THREAD_FIFO_SCHEDULING: &str = "sched_setscheduler(0, SCHED_FIFO, &parameters)";
const PRIMARY_THREAD_FIFO_RECEIPT: &str = "receipt[\"primary_thread_fifo_priority\"] =";
const PROCESS_CPU_TIMING_RECEIPT: &str = "{\"model_submission_process_cpu_seconds\",";
const STALL_PROBE_BOUNDED_WARMUP: &str = "stall_probe_enabled ? kStallProbeWarmupTokenCount";
const WALL_CLOCK_CORRELATION_RECEIPT: &str =
    "receipt[\"timed_start_unix_ns\"] = timed_start_unix_ns;";
const STATE_RANK_VALIDATION: &str = "bool has_state_shape(const Tensor &tensor, std::uint32_t batch_size) {\n  const auto &shape = tensor.logical_shape();\n  return shape.rank() == kTensorRank &&";
const STATE_DTYPE_VALIDATION: &str = "inputs.state.dtype() != tt::tt_metal::DataType::FLOAT32";
const VECTOR_DTYPE_VALIDATION: &str = "vector->dtype() != tt::tt_metal::DataType::BFLOAT16";
const VECTOR_LAYOUT_VALIDATION: &str = "vector->layout() != ttnn::Layout::TILE";
const ASYMMETRIC_PROBE_START: &str = "void run_fused_wkv_asymmetric_probe(";
const ASYMMETRIC_PROBE_END: &str = "\n}\n\nvoid run_fused_wkv_probe(";
const ASYMMETRIC_HOST_REFERENCE: &str = "rwkv7::fused_wkv_reference::evaluate(ReferenceInputs{";
const ASYMMETRIC_BUILD_BINDING: &str = "rwkv7::fused_wkv_reference::validate_diagnostic_build(";
const ASYMMETRIC_FUSED_LAUNCH: &str = "rwkv7::fused_wkv::run_state_dot_a_diagnostic(";
const ASYMMETRIC_STAGE_MARKER: &str = "fused-wkv-asymmetric-stage: fused-launch-start";
const ASYMMETRIC_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: state-dot-a-readback-finished";
const ASYMMETRIC_DECAY_LAUNCH: &str = "rwkv7::fused_wkv::run_decay_diagnostic(";
const ASYMMETRIC_DECAY_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: decay-readback-finished";
const ASYMMETRIC_RANK_UPDATE_LAUNCH: &str = "rwkv7::fused_wkv::run_rank_update_diagnostic(";
const ASYMMETRIC_RANK_UPDATE_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: rank-update-readback-finished";
const ASYMMETRIC_OUTER_UPDATE_LAUNCH: &str = "rwkv7::fused_wkv::run_outer_update_diagnostic(";
const ASYMMETRIC_OUTER_UPDATE_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: outer-update-readback-finished";
const ASYMMETRIC_FIRST_ADD_LAUNCH: &str = "rwkv7::fused_wkv::run_first_add_diagnostic(";
const ASYMMETRIC_FIRST_ADD_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: first-add-readback-finished";
const ASYMMETRIC_FINAL_STATE_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: final-state-readback-finished";
const ASYMMETRIC_BF16_READOUT_LAUNCH: &str = "rwkv7::fused_wkv::run_bf16_readout_diagnostic(";
const ASYMMETRIC_BF16_READOUT_READBACK_MARKER: &str =
    "fused-wkv-asymmetric-stage: bf16-readout-readback-finished";
const ASYMMETRIC_FULL_SPLIT_READOUT_ROW_RESHAPE: &str =
    "const Tensor full_split_readout_rows = op_reshape(context, full_split_readout, vector_shape);";
const ASYMMETRIC_GENERIC_READOUT_ROW_RESHAPE: &str = "const Tensor generic_readout_rows = op_reshape(context, generic.output_for_hidden, vector_shape);";
const ASYMMETRIC_GENERIC_CONTROL: &str = "generic_wkv_step(context, state, a, b, w, v, k, r);";
const ASYMMETRIC_RUNTIME_RECEIPT: &str = "{\"runtime_name\", kRuntimeName}";
const ASYMMETRIC_FP32_RECURRENCE_RECEIPT: &str =
    "{\"fp32_recurrence_vectors\", kFusedWkvFp32RecurrenceVectors}";
const ASYMMETRIC_PREWORK_COUNT: &str = "{\"generic_device_operations_before_fused_launch\", 0U}";
const ASYMMETRIC_GENERIC_RECURRENCE: &str = "generic_wkv_step(";
const ASYMMETRIC_GENERIC_CORRECTION: &str = "corrected_attention_rows(";

const REQUIRED_RECURRENCE: [(&str, &str, usize); 5] = [
    (
        "state_dot_a",
        "matmul_tiles(kStateCb, kACb, state_tile, column_tile, row_tile);",
        SINGLE_PATH_PATTERN_COUNT,
    ),
    (
        "decay",
        "mul_tiles_bcast_rows(kStateCb, kWCb, output_tile, column_tile, output_tile);",
        SINGLE_PATH_PATTERN_COUNT,
    ),
    (
        "rank_update",
        "matmul_tiles(kStateDotACb, kBCb, row_tile, column_tile, output_tile);",
        SINGLE_PATH_PATTERN_COUNT,
    ),
    (
        "outer_update",
        "matmul_tiles(kVCb, kKCb, row_tile, column_tile, output_tile);",
        SINGLE_PATH_PATTERN_COUNT,
    ),
    (
        "readout",
        "matmul_tiles(\n          kNextStateCb, kRCb, state_tile, column_tile, row_tile);",
        SINGLE_PATH_PATTERN_COUNT,
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

    for (name, pattern, expected) in REQUIRED_RECURRENCE {
        require_count(source, name, pattern, expected)?;
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
    require_once(
        source,
        "diagnostic software decay multiply",
        DIAGNOSTIC_DECAY_MULTIPLY,
    )?;
    require_once(
        source,
        "production decay row broadcast",
        PRODUCTION_DECAY_BROADCAST,
    )?;
    require_once(
        source,
        "BF16 readout pack boundary",
        BF16_READOUT_PACK_BOUNDARY,
    )?;
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
        "reader software row-broadcast function",
        SOFTWARE_ROW_BROADCAST,
    )?;
    require_once(
        source,
        "reader software row-broadcast call",
        SOFTWARE_ROW_BROADCAST_CALL,
    )?;
    require_once(
        source,
        "diagnostic w row-broadcast policy",
        DIAGNOSTIC_ROW_BROADCAST_POLICY,
    )?;
    require_once(
        source,
        "reader work-list argument",
        "kWorkListStartArg + work_index",
    )?;
    require_once(
        source,
        "split reader state duplication",
        SPLIT_READER_STATE_DUPLICATION,
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
    if source.contains(SHARED_FULL_KERNEL_STATE_DOT_A_TAP) {
        return Err(KernelContractError::new(
            "shared_diagnostic_circular_buffer",
            "the full fused writer must not tap a compute-internal recurrence CB",
        ));
    }
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
    require_once(
        source,
        "split writer state ownership",
        SPLIT_WRITER_STATE_OWNERSHIP,
    )?;
    require_once(source, "split writer output skip", SPLIT_WRITER_OUTPUT_SKIP)?;

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

/// Validate the dedicated state_dot_a diagnostic ownership boundary.
pub fn validate_state_dot_a_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "state_dot_a reader")?;
    validate_blackhole_only(compute_source, "state_dot_a compute")?;
    validate_blackhole_only(writer_source, "state_dot_a writer")?;
    if reader_source.contains(STATE_DOT_A_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the state_dot_a output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated state_dot_a compute output CB",
        STATE_DOT_A_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated state_dot_a writer output CB",
        STATE_DOT_A_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated state_dot_a output publication",
        "cb_push_back(kStateDotAOutputCb, kHeadTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kStateDotAOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned diagnostic output",
        ));
    }
    require_once(
        writer_source,
        "dedicated state_dot_a output wait",
        "cb_wait_front(kStateDotAOutputCb, kHeadTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated state_dot_a output acknowledgement",
        "cb_pop_front(kStateDotAOutputCb, kHeadTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated state_dot_a data completion barrier",
        WRITE_BARRIER,
    )?;
    Ok(())
}

/// Validate the dedicated decay diagnostic ownership boundary.
pub fn validate_decay_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "decay reader")?;
    validate_blackhole_only(compute_source, "decay compute")?;
    validate_blackhole_only(writer_source, "decay writer")?;
    if reader_source.contains(DECAY_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the decay output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated decay compute output CB",
        DECAY_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated decay writer output CB",
        DECAY_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated decay output publication",
        "cb_push_back(kDecayOutputCb, kStateTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kDecayOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned decay output",
        ));
    }
    require_once(
        writer_source,
        "dedicated decay output wait",
        "cb_wait_front(kDecayOutputCb, kStateTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated decay output acknowledgement",
        "cb_pop_front(kDecayOutputCb, kStateTileCount);",
    )?;
    require_once(
        reader_source,
        "dedicated decay software row broadcast",
        "materialize_bf16_row_broadcast(tile_address);",
    )?;
    require_once(
        compute_source,
        "dedicated decay multiply",
        "mul_binary_tile(state_dst, decay_dst, state_dst);",
    )?;
    require_once(
        writer_source,
        "dedicated decay data completion barrier",
        WRITE_BARRIER,
    )?;
    Ok(())
}

/// Validate the dedicated rank_update diagnostic ownership boundary.
pub fn validate_rank_update_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "rank_update reader")?;
    validate_blackhole_only(compute_source, "rank_update compute")?;
    validate_blackhole_only(writer_source, "rank_update writer")?;
    if reader_source.contains(RANK_UPDATE_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the rank_update output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated rank_update compute output CB",
        RANK_UPDATE_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated rank_update writer output CB",
        RANK_UPDATE_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated rank_update output publication",
        "cb_push_back(kRankUpdateOutputCb, kStateTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kRankUpdateOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned rank_update output",
        ));
    }
    require_once(
        writer_source,
        "dedicated rank_update output wait",
        "cb_wait_front(kRankUpdateOutputCb, kStateTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated rank_update output acknowledgement",
        "cb_pop_front(kRankUpdateOutputCb, kStateTileCount);",
    )?;
    require_once(
        compute_source,
        "dedicated rank_update matmul",
        "matmul_tiles(kStateDotACb, kOrientedBCb, row_tile, column_tile, output_tile);",
    )?;
    require_once(
        writer_source,
        "dedicated rank_update data completion barrier",
        WRITE_BARRIER,
    )?;
    Ok(())
}

/// Validate the dedicated outer_update diagnostic ownership boundary.
pub fn validate_outer_update_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "outer_update reader")?;
    validate_blackhole_only(compute_source, "outer_update compute")?;
    validate_blackhole_only(writer_source, "outer_update writer")?;
    if reader_source.contains(OUTER_UPDATE_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the outer_update output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated outer_update compute output CB",
        OUTER_UPDATE_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated outer_update writer output CB",
        OUTER_UPDATE_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated outer_update output publication",
        "cb_push_back(kOuterUpdateOutputCb, kStateTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kOuterUpdateOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned outer_update output",
        ));
    }
    require_once(
        writer_source,
        "dedicated outer_update output wait",
        "cb_wait_front(kOuterUpdateOutputCb, kStateTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated outer_update output acknowledgement",
        "cb_pop_front(kOuterUpdateOutputCb, kStateTileCount);",
    )?;
    require_once(
        compute_source,
        "dedicated outer_update matmul",
        "matmul_tiles(kOrientedVCb, kOrientedKCb, row_tile, column_tile, output_tile);",
    )?;
    require_once(
        writer_source,
        "dedicated outer_update data completion barrier",
        WRITE_BARRIER,
    )?;
    Ok(())
}

/// Validate the dedicated first_add diagnostic ownership boundary.
pub fn validate_first_add_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "first_add reader")?;
    validate_blackhole_only(compute_source, "first_add compute")?;
    validate_blackhole_only(writer_source, "first_add writer")?;
    if reader_source.contains(FIRST_ADD_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the first_add output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated first_add compute output CB",
        FIRST_ADD_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated first_add writer output CB",
        FIRST_ADD_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated first_add output publication",
        "cb_push_back(kFirstAddOutputCb, kStateTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kFirstAddOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned first_add output",
        ));
    }
    require_once(
        writer_source,
        "dedicated first_add output wait",
        "cb_wait_front(kFirstAddOutputCb, kStateTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated first_add output acknowledgement",
        "cb_pop_front(kFirstAddOutputCb, kStateTileCount);",
    )?;
    require_once(
        compute_source,
        "dedicated first_add SFPU operation",
        "add_binary_tile(lhs_dst, rhs_dst, lhs_dst);",
    )?;
    require_once(
        writer_source,
        "dedicated first_add data completion barrier",
        WRITE_BARRIER,
    )?;
    Ok(())
}

/// Validate the dedicated BF16 readout diagnostic ownership boundary.
pub fn validate_bf16_readout_diagnostic_sources(
    reader_source: &str,
    compute_source: &str,
    writer_source: &str,
    program_source: &str,
) -> Result<(), KernelContractError> {
    validate_blackhole_only(reader_source, "BF16 readout reader")?;
    validate_blackhole_only(compute_source, "BF16 readout compute")?;
    validate_blackhole_only(writer_source, "BF16 readout writer")?;
    if reader_source.contains(BF16_READOUT_DIAGNOSTIC_OUTPUT_CB) {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "the dedicated reader must not own the BF16 readout output CB",
        ));
    }
    require_once(
        compute_source,
        "dedicated BF16 readout compute output CB",
        BF16_READOUT_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        writer_source,
        "dedicated BF16 readout writer output CB",
        BF16_READOUT_DIAGNOSTIC_OUTPUT_CB,
    )?;
    require_once(
        compute_source,
        "dedicated BF16 readout output publication",
        "cb_push_back(kBf16ReadoutOutputCb, kHeadTileCount);",
    )?;
    if compute_source.contains("cb_pop_front(kBf16ReadoutOutputCb") {
        return Err(KernelContractError::new(
            "diagnostic_output_ownership",
            "compute must not acknowledge the writer-owned BF16 readout output",
        ));
    }
    require_once(
        writer_source,
        "dedicated BF16 readout output wait",
        "cb_wait_front(kBf16ReadoutOutputCb, kHeadTileCount);",
    )?;
    require_once(
        writer_source,
        "dedicated BF16 readout output acknowledgement",
        "cb_pop_front(kBf16ReadoutOutputCb, kHeadTileCount);",
    )?;
    for (name, pattern) in [
        (
            "dedicated BF16 readout reverse source order",
            BF16_READOUT_REVERSE_SOURCE_ORDER,
        ),
        (
            "dedicated BF16 readout transposed block initialization",
            BF16_READOUT_TRANSPOSED_BLOCK_INIT,
        ),
        (
            "dedicated BF16 readout block matmul",
            BF16_READOUT_BLOCK_MATMUL,
        ),
    ] {
        require_once(compute_source, name, pattern)?;
    }
    if compute_source.contains(BF16_READOUT_OLD_TRANSPOSE_STAGE) {
        return Err(KernelContractError::new(
            "bf16_readout_matmul_contract",
            "the dedicated BF16 readout must transpose R inside matmul",
        ));
    }
    for (name, pattern) in [
        ("dedicated BF16 readout HiFi2", BF16_READOUT_HIFI2),
        (
            "dedicated BF16 readout BF16 destination accumulator",
            BF16_READOUT_BF16_DESTINATION,
        ),
        (
            "dedicated BF16 readout FP32 state CB",
            BF16_READOUT_FP32_STATE_CB,
        ),
        ("dedicated BF16 readout BF16 R CB", BF16_READOUT_BF16_R_CB),
    ] {
        require_once(program_source, name, pattern)?;
    }
    require_once(
        writer_source,
        "dedicated BF16 readout data completion barrier",
        WRITE_BARRIER,
    )?;
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
            "separate next-state output",
            "output.buffer(), next_state.buffer()",
        ),
        ("state-rank validation", STATE_RANK_VALIDATION),
        ("FP32 state validation", STATE_DTYPE_VALIDATION),
        ("BF16 vector validation", VECTOR_DTYPE_VALIDATION),
        ("tile vector validation", VECTOR_LAYOUT_VALIDATION),
        (
            "three-tile correction scalar CB cycle",
            CORRECTION_SCALAR_CAPACITY,
        ),
        ("diagnostic readout guard", DIAGNOSTIC_READOUT_GUARD),
        (
            "generic FP32 recurrence-vector policy",
            GENERIC_FP32_RECURRENCE_VECTOR_POLICY,
        ),
        (
            "generic FP32 recurrence-vector format",
            GENERIC_FP32_RECURRENCE_FORMAT,
        ),
        (
            "production BF16 recurrence-vector format",
            PRODUCTION_BF16_RECURRENCE_FORMAT,
        ),
        (
            "recurrence device-define factory",
            RECURRENCE_DEVICE_DEFINE_FACTORY,
        ),
        (
            "recurrence device-define value",
            RECURRENCE_DEVICE_DEFINE_VALUE,
        ),
        (
            "compute recurrence device define",
            COMPUTE_RECURRENCE_DEVICE_DEFINE,
        ),
        ("FP32 recurrent readout CB", FP32_READOUT_CB),
        ("diagnostic BF16 readout row CB", BF16_READOUT_ROW_CB),
        ("production FP32 readout row CB", FP32_READOUT_ROW_CB),
        ("r_k fused input", "&inputs.r_k,"),
        (
            "group-normalization fused input",
            "&inputs.group_norm_weight,",
        ),
        ("gate fused input", "&inputs.gate,"),
        ("Blackhole CB mode count", BLACKHOLE_CIRCULAR_BUFFER_COUNT),
        ("exact FP32 decay unpack", DECAY_EXACT_FP32_UNPACK),
        ("split readout mode binding", SPLIT_READOUT_MODE),
        ("split readout device define", SPLIT_READOUT_DEVICE_DEFINE),
        (
            "split exact FP32 unpack CB set",
            SPLIT_EXACT_FP32_UNPACK_CBS,
        ),
        (
            "split exact FP32 unpack assignment",
            SPLIT_EXACT_FP32_UNPACK_ASSIGNMENT,
        ),
        ("split HiFi2 readout launch", SPLIT_READOUT_LAUNCH),
        ("split raw output ownership", SPLIT_RAW_OUTPUT_OWNERSHIP),
        (
            "monolithic corrected output ownership",
            MONOLITHIC_CORRECTED_OUTPUT_OWNERSHIP,
        ),
    ] {
        require_once(program_source, name, pattern)?;
    }
    require_count(
        program_source,
        "completion semaphore runtime arguments",
        "args.push_back(kCompletionSemaphoreId);",
        7,
    )?;
    require_count(
        program_source,
        "completion semaphore descriptors",
        "semaphore.id = kCompletionSemaphoreId;",
        7,
    )?;
    require_count(
        program_source,
        "generic TT-NN device operations",
        "ttnn::generic_op(io_tensors, mesh_descriptor);",
        7,
    )?;
    require_count(
        program_source,
        "reader recurrence runtime argument",
        READER_RECURRENCE_RUNTIME_ARG,
        SINGLE_PATH_PATTERN_COUNT,
    )?;
    for (name, pattern) in [
        (
            "dedicated state_dot_a reader",
            "fused_wkv_state_dot_a_reader.cpp",
        ),
        (
            "dedicated state_dot_a writer",
            "fused_wkv_state_dot_a_writer.cpp",
        ),
        (
            "dedicated state_dot_a compute",
            "fused_wkv_state_dot_a_compute.cpp",
        ),
        (
            "dedicated state_dot_a host launch",
            "Tensor run_state_dot_a_diagnostic(",
        ),
        ("dedicated decay reader", "fused_wkv_decay_reader.cpp"),
        ("dedicated decay writer", "fused_wkv_decay_writer.cpp"),
        ("dedicated decay compute", "fused_wkv_decay_compute.cpp"),
        (
            "dedicated decay host launch",
            "Tensor run_decay_diagnostic(",
        ),
        (
            "dedicated rank_update reader",
            "fused_wkv_rank_update_reader.cpp",
        ),
        (
            "dedicated rank_update writer",
            "fused_wkv_rank_update_writer.cpp",
        ),
        (
            "dedicated rank_update compute",
            "fused_wkv_rank_update_compute.cpp",
        ),
        (
            "dedicated rank_update host launch",
            "Tensor run_rank_update_diagnostic(",
        ),
        (
            "dedicated outer_update reader",
            "fused_wkv_outer_update_reader.cpp",
        ),
        (
            "dedicated outer_update writer",
            "fused_wkv_outer_update_writer.cpp",
        ),
        (
            "dedicated outer_update compute",
            "fused_wkv_outer_update_compute.cpp",
        ),
        (
            "dedicated outer_update host launch",
            "Tensor run_outer_update_diagnostic(",
        ),
        (
            "dedicated first_add reader",
            "fused_wkv_first_add_reader.cpp",
        ),
        (
            "dedicated first_add writer",
            "fused_wkv_first_add_writer.cpp",
        ),
        (
            "dedicated first_add compute",
            "fused_wkv_first_add_compute.cpp",
        ),
        (
            "dedicated first_add host launch",
            "Tensor run_first_add_diagnostic(",
        ),
        (
            "dedicated BF16 readout reader",
            "fused_wkv_bf16_readout_reader.cpp",
        ),
        (
            "dedicated BF16 readout writer",
            "fused_wkv_bf16_readout_writer.cpp",
        ),
        (
            "dedicated BF16 readout compute",
            "fused_wkv_bf16_readout_compute.cpp",
        ),
        (
            "dedicated BF16 readout host launch",
            "Tensor run_bf16_readout_diagnostic(",
        ),
    ] {
        require_once(program_source, name, pattern)?;
    }
    require_count(
        program_source,
        "diagnostic recurrence-vector guard",
        DIAGNOSTIC_RECURRENCE_GUARD,
        DIAGNOSTIC_RECURRENCE_GUARD_COUNT,
    )?;

    let asymmetric_probe_start = runtime_source.find(ASYMMETRIC_PROBE_START).ok_or_else(|| {
        KernelContractError::new(
            "missing_asymmetric_probe",
            "the fused-only asymmetric probe is absent",
        )
    })?;
    let asymmetric_probe_tail = &runtime_source[asymmetric_probe_start..];
    let asymmetric_probe_end = asymmetric_probe_tail
        .find(ASYMMETRIC_PROBE_END)
        .ok_or_else(|| {
            KernelContractError::new(
                "missing_asymmetric_probe_end",
                "the fused-only asymmetric probe boundary is absent",
            )
        })?;
    let asymmetric_probe = &asymmetric_probe_tail[..asymmetric_probe_end];
    for (name, pattern) in [
        ("pure host asymmetric reference", ASYMMETRIC_HOST_REFERENCE),
        ("diagnostic build binding", ASYMMETRIC_BUILD_BINDING),
        ("fused-only asymmetric launch", ASYMMETRIC_FUSED_LAUNCH),
        ("bounded asymmetric stage marker", ASYMMETRIC_STAGE_MARKER),
        ("completed readback marker", ASYMMETRIC_READBACK_MARKER),
        ("dedicated decay launch", ASYMMETRIC_DECAY_LAUNCH),
        (
            "completed decay readback marker",
            ASYMMETRIC_DECAY_READBACK_MARKER,
        ),
        (
            "dedicated rank_update launch",
            ASYMMETRIC_RANK_UPDATE_LAUNCH,
        ),
        (
            "completed rank_update readback marker",
            ASYMMETRIC_RANK_UPDATE_READBACK_MARKER,
        ),
        (
            "dedicated outer_update launch",
            ASYMMETRIC_OUTER_UPDATE_LAUNCH,
        ),
        (
            "completed outer_update readback marker",
            ASYMMETRIC_OUTER_UPDATE_READBACK_MARKER,
        ),
        (
            "completed first_add readback marker",
            ASYMMETRIC_FIRST_ADD_READBACK_MARKER,
        ),
        (
            "completed final_state readback marker",
            ASYMMETRIC_FINAL_STATE_READBACK_MARKER,
        ),
        (
            "dedicated BF16 readout launch",
            ASYMMETRIC_BF16_READOUT_LAUNCH,
        ),
        (
            "completed BF16 readout marker",
            ASYMMETRIC_BF16_READOUT_READBACK_MARKER,
        ),
        (
            "full-split readout row reshape",
            ASYMMETRIC_FULL_SPLIT_READOUT_ROW_RESHAPE,
        ),
        (
            "generic readout row reshape",
            ASYMMETRIC_GENERIC_READOUT_ROW_RESHAPE,
        ),
        ("post-readback generic control", ASYMMETRIC_GENERIC_CONTROL),
        (
            "runtime-bound asymmetric receipt",
            ASYMMETRIC_RUNTIME_RECEIPT,
        ),
        (
            "FP32 recurrence-bound asymmetric receipt",
            ASYMMETRIC_FP32_RECURRENCE_RECEIPT,
        ),
        (
            "zero generic device pre-work receipt",
            ASYMMETRIC_PREWORK_COUNT,
        ),
    ] {
        require_once(asymmetric_probe, name, pattern)?;
    }
    require_count(
        asymmetric_probe,
        "dedicated first_add and final_state launches",
        ASYMMETRIC_FIRST_ADD_LAUNCH,
        2,
    )?;
    let readback_marker = asymmetric_probe
        .find(ASYMMETRIC_READBACK_MARKER)
        .expect("required readback marker was checked above");
    let generic_control = asymmetric_probe
        .find(ASYMMETRIC_GENERIC_CONTROL)
        .expect("required generic control was checked above");
    if generic_control <= readback_marker {
        return Err(KernelContractError::new(
            "forbidden_asymmetric_prework",
            "the generic control must run only after fused readback",
        ));
    }
    if asymmetric_probe.contains(ASYMMETRIC_GENERIC_CORRECTION) {
        return Err(KernelContractError::new(
            "forbidden_asymmetric_prework",
            "the fused-only asymmetric probe contains generic correction pre-work",
        ));
    }
    if count(asymmetric_probe, ASYMMETRIC_GENERIC_RECURRENCE) != 1 {
        return Err(KernelContractError::new(
            "unbounded_generic_control",
            "the asymmetric probe must contain exactly one generic recurrence control",
        ));
    }

    for (name, pattern) in [
        (
            "measured physical admission",
            "constexpr bool kFusedWkvPhysicallyAdmitted = true;",
        ),
        (
            "FIFO-bound production admission",
            "*primary_thread_fifo_priority != kAdmittedPrimaryThreadFifoPriority",
        ),
        (
            "card-bound production admission",
            "constexpr std::uint32_t kAdmittedCardZeroPrimaryThreadCpu = 2U;",
        ),
        (
            "explicit generic reference policy",
            "kGenericReferenceWkvPolicy = \"generic-reference\"",
        ),
        ("capability selection", "rwkv7::fused_wkv::select_dispatch("),
        ("fused dispatch branch", "if (decision.use_fused_path)"),
        ("generic fallback", "return generic_wkv_step("),
        ("runtime correction ownership", RUNTIME_CORRECTION_OWNERSHIP),
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
        (
            "explicit physical worker WKV policy",
            PHYSICAL_WORKER_POLICY_PARSE,
        ),
        (
            "physical worker initialization-ready signal",
            INITIALIZATION_READY_SIGNAL,
        ),
        (
            "synchronized physical worker warmup start",
            SYNCHRONIZED_WARMUP_START,
        ),
        ("physical worker timed-ready signal", TIMED_READY_SIGNAL),
        (
            "synchronized physical worker start",
            SYNCHRONIZED_WORKER_START,
        ),
        (
            "disabled physical worker Inspector",
            DISABLED_WORKER_INSPECTOR,
        ),
        (
            "bounded physical worker stall step count",
            STALL_PROBE_STEP_COUNT,
        ),
        ("physical worker stall step receipt", STALL_PROBE_RECEIPT),
        (
            "bounded physical worker stall boundary count",
            STALL_PROBE_BOUNDARY_COUNT,
        ),
        (
            "physical worker stall boundary receipt",
            STALL_PROBE_BOUNDARY_RECEIPT,
        ),
        (
            "diagnostic-only fetch queue counter enable",
            FETCH_QUEUE_DIAGNOSTICS_ENABLE,
        ),
        (
            "physical worker fetch queue diagnostics receipt",
            FETCH_QUEUE_DIAGNOSTICS_RECEIPT,
        ),
        (
            "diagnostic timed hardware counter start",
            HARDWARE_COUNTER_SESSION_START,
        ),
        (
            "diagnostic timed hardware counter stop",
            HARDWARE_COUNTER_SESSION_STOP,
        ),
        (
            "physical worker hardware counter receipt",
            HARDWARE_COUNTER_RECEIPT,
        ),
        ("primary-thread-only CPU affinity", PRIMARY_THREAD_AFFINITY),
        (
            "primary thread CPU affinity receipt",
            PRIMARY_THREAD_AFFINITY_RECEIPT,
        ),
        (
            "primary thread FIFO scheduling",
            PRIMARY_THREAD_FIFO_SCHEDULING,
        ),
        ("primary thread FIFO receipt", PRIMARY_THREAD_FIFO_RECEIPT),
        (
            "physical worker process CPU timing receipt",
            PROCESS_CPU_TIMING_RECEIPT,
        ),
        (
            "bounded physical worker stall warmup",
            STALL_PROBE_BOUNDED_WARMUP,
        ),
        (
            "physical worker wall-clock correlation receipt",
            WALL_CLOCK_CORRELATION_RECEIPT,
        ),
    ] {
        require_once(runtime_source, name, pattern)?;
    }
    Ok(())
}

pub fn validate_hardware_counter_diagnostics_source(
    source: &str,
) -> Result<(), KernelContractError> {
    for (name, pattern) in [
        ("Linux perf event syscall", "SYS_perf_event_open"),
        (
            "simultaneous task counter enable",
            "PR_TASK_PERF_EVENTS_ENABLE",
        ),
        (
            "simultaneous task counter disable",
            "if (::prctl(PR_TASK_PERF_EVENTS_DISABLE) != 0)",
        ),
        ("CPU cycle counter", "PERF_COUNT_HW_CPU_CYCLES"),
        ("instruction counter", "PERF_COUNT_HW_INSTRUCTIONS"),
        ("cache reference counter", "PERF_COUNT_HW_CACHE_REFERENCES"),
        ("cache miss counter", "PERF_COUNT_HW_CACHE_MISSES"),
        ("task clock counter", "PERF_COUNT_SW_TASK_CLOCK"),
        ("context switch counter", "PERF_COUNT_SW_CONTEXT_SWITCHES"),
        ("CPU migration counter", "PERF_COUNT_SW_CPU_MIGRATIONS"),
        (
            "APERF event",
            "constexpr std::uint64_t kAperfEventConfig = 0x01U;",
        ),
        (
            "MPERF event",
            "constexpr std::uint64_t kMperfEventConfig = 0x02U;",
        ),
        (
            "counter running-time bound",
            "constexpr double kMinimumRunningFraction = 0.99;",
        ),
        (
            "counter multiplex validation",
            "counter was multiplexed below the bound",
        ),
    ] {
        require_once(source, name, pattern)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPUTE_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_compute.cpp");
    const READER_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_reader.cpp");
    const WRITER_SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_writer.cpp");
    const STATE_DOT_A_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_state_dot_a_reader.cpp");
    const STATE_DOT_A_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_state_dot_a_compute.cpp");
    const STATE_DOT_A_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_state_dot_a_writer.cpp");
    const DECAY_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_decay_reader.cpp");
    const DECAY_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_decay_compute.cpp");
    const DECAY_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_decay_writer.cpp");
    const RANK_UPDATE_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_rank_update_reader.cpp");
    const RANK_UPDATE_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_rank_update_compute.cpp");
    const RANK_UPDATE_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_rank_update_writer.cpp");
    const OUTER_UPDATE_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_outer_update_reader.cpp");
    const OUTER_UPDATE_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_outer_update_compute.cpp");
    const OUTER_UPDATE_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_outer_update_writer.cpp");
    const FIRST_ADD_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_first_add_reader.cpp");
    const FIRST_ADD_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_first_add_compute.cpp");
    const FIRST_ADD_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_first_add_writer.cpp");
    const BF16_READOUT_READER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_bf16_readout_reader.cpp");
    const BF16_READOUT_COMPUTE_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_bf16_readout_compute.cpp");
    const BF16_READOUT_WRITER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/kernel/fused_wkv_bf16_readout_writer.cpp");
    const PROGRAM_SOURCE: &str = include_str!("../../rwkv7-p150x2/src/fused_wkv_program.cpp");
    const RUNTIME_SOURCE: &str = include_str!("../../rwkv7-p150x2/src/main.cpp");
    const HARDWARE_COUNTER_SOURCE: &str =
        include_str!("../../rwkv7-p150x2/src/hardware_counter_diagnostics.cpp");

    #[test]
    fn accepts_the_reviewed_compute_source() {
        validate_fused_compute_source(COMPUTE_SOURCE).expect("reviewed compute source");
    }

    #[test]
    fn accepts_the_reviewed_hardware_counter_source() {
        validate_hardware_counter_diagnostics_source(HARDWARE_COUNTER_SOURCE)
            .expect("reviewed hardware counter source");
    }

    #[test]
    fn rejects_a_hardware_counter_source_without_instruction_counts() {
        let invalid = HARDWARE_COUNTER_SOURCE
            .replace("PERF_COUNT_HW_INSTRUCTIONS", "REMOVED_INSTRUCTION_COUNTER");
        let error = validate_hardware_counter_diagnostics_source(&invalid)
            .expect_err("hardware counters without instructions");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("instruction counter"));
    }

    #[test]
    fn rejects_a_missing_decay_term() {
        let invalid = COMPUTE_SOURCE.replace(
            PRODUCTION_DECAY_BROADCAST,
            "// invalid fixture removed the decay term",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("missing decay");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("decay"));
    }

    #[test]
    fn rejects_a_missing_diagnostic_software_decay_multiply() {
        let invalid = COMPUTE_SOURCE.replace(
            DIAGNOSTIC_DECAY_MULTIPLY,
            "// invalid fixture restored the faulty FP32 row-broadcast path",
        );
        let error = validate_fused_compute_source(&invalid)
            .expect_err("missing diagnostic software decay multiply");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("diagnostic software decay multiply"));
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
    fn rejects_a_reader_without_split_state_duplication() {
        let invalid = READER_SOURCE.replace(
            SPLIT_READER_STATE_DUPLICATION,
            "read_state(kStateCb, state_address, work_item,",
        );
        let error = validate_fused_reader_source(&invalid)
            .expect_err("split reader must duplicate state for exact SFPU unpack");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split reader state duplication"));
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
    fn rejects_a_reader_without_diagnostic_software_row_broadcast() {
        let invalid = READER_SOURCE.replace(
            SOFTWARE_ROW_BROADCAST_CALL,
            "// invalid fixture removed software row broadcast",
        );
        let error =
            validate_fused_reader_source(&invalid).expect_err("missing software row broadcast");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reader software row-broadcast call"));
    }

    #[test]
    fn rejects_a_reader_without_production_row_broadcast_isolation() {
        let invalid = PROGRAM_SOURCE.replace(READER_RECURRENCE_RUNTIME_ARG, "args.push_back(1U);");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("missing production row-broadcast isolation");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reader recurrence runtime argument"));
    }

    #[test]
    fn accepts_the_reviewed_writer_source() {
        validate_fused_writer_source(WRITER_SOURCE).expect("reviewed writer source");
    }

    #[test]
    fn rejects_a_writer_without_split_state_ownership() {
        let invalid = WRITER_SOURCE.replace(SPLIT_WRITER_STATE_OWNERSHIP, "kNextStateOutputCb");
        let error = validate_fused_writer_source(&invalid)
            .expect_err("split writer must consume the recurrence output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split writer state ownership"));
    }

    #[test]
    fn rejects_a_writer_without_split_output_skip() {
        let invalid = WRITER_SOURCE.replace(SPLIT_WRITER_OUTPUT_SKIP, "if (true) {");
        let error = validate_fused_writer_source(&invalid)
            .expect_err("split writer must leave raw readout to the HiFi2 program");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split writer output skip"));
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
    fn rejects_full_writer_tapping_a_compute_internal_recurrence_cb() {
        let invalid = format!("{WRITER_SOURCE}\nconstexpr std::uint32_t kStateDotAStageCb = 8U;\n");
        let error = validate_fused_writer_source(&invalid)
            .expect_err("full fused writer must not tap compute-internal recurrence data");
        assert_eq!(error.code, "shared_diagnostic_circular_buffer");
    }

    #[test]
    fn accepts_the_dedicated_state_dot_a_kernel_ownership() {
        validate_state_dot_a_diagnostic_sources(
            STATE_DOT_A_READER_SOURCE,
            STATE_DOT_A_COMPUTE_SOURCE,
            STATE_DOT_A_WRITER_SOURCE,
        )
        .expect("dedicated state_dot_a ownership");
    }

    #[test]
    fn rejects_a_dedicated_state_dot_a_writer_without_output_acknowledgement() {
        let invalid = STATE_DOT_A_WRITER_SOURCE.replace(
            "cb_pop_front(kStateDotAOutputCb, kHeadTileCount);",
            "// invalid fixture removed dedicated output acknowledgement",
        );
        let error = validate_state_dot_a_diagnostic_sources(
            STATE_DOT_A_READER_SOURCE,
            STATE_DOT_A_COMPUTE_SOURCE,
            &invalid,
        )
        .expect_err("dedicated writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn accepts_the_dedicated_decay_kernel_ownership() {
        validate_decay_diagnostic_sources(
            DECAY_READER_SOURCE,
            DECAY_COMPUTE_SOURCE,
            DECAY_WRITER_SOURCE,
        )
        .expect("dedicated decay ownership");
    }

    #[test]
    fn rejects_a_dedicated_decay_writer_without_output_acknowledgement() {
        let invalid = DECAY_WRITER_SOURCE.replace(
            "cb_pop_front(kDecayOutputCb, kStateTileCount);",
            "// invalid fixture removed dedicated decay output acknowledgement",
        );
        let error =
            validate_decay_diagnostic_sources(DECAY_READER_SOURCE, DECAY_COMPUTE_SOURCE, &invalid)
                .expect_err("dedicated decay writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn accepts_the_dedicated_rank_update_kernel_ownership() {
        validate_rank_update_diagnostic_sources(
            RANK_UPDATE_READER_SOURCE,
            RANK_UPDATE_COMPUTE_SOURCE,
            RANK_UPDATE_WRITER_SOURCE,
        )
        .expect("dedicated rank_update ownership");
    }

    #[test]
    fn rejects_a_dedicated_rank_update_writer_without_output_acknowledgement() {
        let invalid = RANK_UPDATE_WRITER_SOURCE.replace(
            "cb_pop_front(kRankUpdateOutputCb, kStateTileCount);",
            "// invalid fixture removed dedicated rank_update output acknowledgement",
        );
        let error = validate_rank_update_diagnostic_sources(
            RANK_UPDATE_READER_SOURCE,
            RANK_UPDATE_COMPUTE_SOURCE,
            &invalid,
        )
        .expect_err("dedicated rank_update writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn accepts_the_dedicated_outer_update_kernel_ownership() {
        validate_outer_update_diagnostic_sources(
            OUTER_UPDATE_READER_SOURCE,
            OUTER_UPDATE_COMPUTE_SOURCE,
            OUTER_UPDATE_WRITER_SOURCE,
        )
        .expect("dedicated outer_update ownership");
    }

    #[test]
    fn rejects_a_dedicated_outer_update_writer_without_output_acknowledgement() {
        let invalid = OUTER_UPDATE_WRITER_SOURCE.replace(
            "cb_pop_front(kOuterUpdateOutputCb, kStateTileCount);",
            "// invalid fixture removed dedicated outer_update output acknowledgement",
        );
        let error = validate_outer_update_diagnostic_sources(
            OUTER_UPDATE_READER_SOURCE,
            OUTER_UPDATE_COMPUTE_SOURCE,
            &invalid,
        )
        .expect_err("dedicated outer_update writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn accepts_the_dedicated_first_add_kernel_ownership() {
        validate_first_add_diagnostic_sources(
            FIRST_ADD_READER_SOURCE,
            FIRST_ADD_COMPUTE_SOURCE,
            FIRST_ADD_WRITER_SOURCE,
        )
        .expect("dedicated first_add ownership");
    }

    #[test]
    fn rejects_a_dedicated_first_add_writer_without_output_acknowledgement() {
        let invalid = FIRST_ADD_WRITER_SOURCE.replace(
            "cb_pop_front(kFirstAddOutputCb, kStateTileCount);",
            "// invalid fixture removed dedicated first_add output acknowledgement",
        );
        let error = validate_first_add_diagnostic_sources(
            FIRST_ADD_READER_SOURCE,
            FIRST_ADD_COMPUTE_SOURCE,
            &invalid,
        )
        .expect_err("dedicated first_add writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn accepts_the_dedicated_bf16_readout_kernel_ownership() {
        validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            BF16_READOUT_COMPUTE_SOURCE,
            BF16_READOUT_WRITER_SOURCE,
            PROGRAM_SOURCE,
        )
        .expect("dedicated BF16 readout ownership");
    }

    #[test]
    fn rejects_a_dedicated_bf16_readout_writer_without_output_acknowledgement() {
        let invalid = BF16_READOUT_WRITER_SOURCE.replace(
            "cb_pop_front(kBf16ReadoutOutputCb, kHeadTileCount);",
            "// invalid fixture removed dedicated BF16 readout output acknowledgement",
        );
        let error = validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            BF16_READOUT_COMPUTE_SOURCE,
            &invalid,
            PROGRAM_SOURCE,
        )
        .expect_err("dedicated BF16 readout writer must acknowledge its output CB");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("output acknowledgement"));
    }

    #[test]
    fn rejects_a_dedicated_bf16_readout_without_reverse_source_order() {
        let invalid = BF16_READOUT_COMPUTE_SOURCE.replace(
            BF16_READOUT_REVERSE_SOURCE_ORDER,
            "compute_kernel_hw_startup(kStateCb, kRawRCb,",
        );
        let error = validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            &invalid,
            BF16_READOUT_WRITER_SOURCE,
            PROGRAM_SOURCE,
        )
        .expect_err("dedicated BF16 readout must use reverse source order");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reverse source order"));
    }

    #[test]
    fn rejects_a_dedicated_bf16_readout_with_a_separate_transpose_stage() {
        let invalid = format!("{BF16_READOUT_COMPUTE_SOURCE}\ntranspose_wh_tile();");
        let error = validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            &invalid,
            BF16_READOUT_WRITER_SOURCE,
            PROGRAM_SOURCE,
        )
        .expect_err("dedicated BF16 readout must transpose inside matmul");
        assert_eq!(error.code, "bf16_readout_matmul_contract");
        assert!(error.detail.contains("transpose R inside matmul"));
    }

    #[test]
    fn rejects_a_dedicated_bf16_readout_without_hifi2() {
        let invalid = PROGRAM_SOURCE.replace(
            BF16_READOUT_HIFI2,
            "compute_config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;",
        );
        let error = validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            BF16_READOUT_COMPUTE_SOURCE,
            BF16_READOUT_WRITER_SOURCE,
            &invalid,
        )
        .expect_err("dedicated BF16 readout must use generic HiFi2");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("HiFi2"));
    }

    #[test]
    fn rejects_a_dedicated_bf16_readout_with_fp32_destination_accumulation() {
        let invalid = PROGRAM_SOURCE.replace(
            BF16_READOUT_BF16_DESTINATION,
            "compute_config.fp32_dest_acc_en = true;",
        );
        let error = validate_bf16_readout_diagnostic_sources(
            BF16_READOUT_READER_SOURCE,
            BF16_READOUT_COMPUTE_SOURCE,
            BF16_READOUT_WRITER_SOURCE,
            &invalid,
        )
        .expect_err("dedicated BF16 readout must use a BF16 destination accumulator");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("BF16 destination accumulator"));
    }

    #[test]
    fn rejects_a_short_blackhole_unpack_mode_vector() {
        let invalid = PROGRAM_SOURCE.replace(
            BLACKHOLE_CIRCULAR_BUFFER_COUNT,
            "constexpr std::size_t kCircularBufferCount = 32U;",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("short Blackhole unpack mode vector");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("Blackhole CB mode count"));
    }

    #[test]
    fn rejects_decay_without_exact_fp32_unpack() {
        let invalid = PROGRAM_SOURCE.replace(
            DECAY_EXACT_FP32_UNPACK,
            "// invalid fixture removed exact FP32 decay unpack",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("decay without exact FP32 unpack");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("exact FP32 decay unpack"));
    }

    #[test]
    fn rejects_split_readout_without_both_precision_flags() {
        let invalid = PROGRAM_SOURCE.replace(
            SPLIT_READOUT_MODE,
            "constexpr bool kSplitReadout =\n    RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY != 0;",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("split readout must require both diagnostic precision flags");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split readout mode binding"));
    }

    #[test]
    fn rejects_split_readout_without_exact_fp32_stage_unpack() {
        let invalid = PROGRAM_SOURCE.replace(
            SPLIT_EXACT_FP32_UNPACK_CBS,
            "const std::array<std::uint8_t, 1U> exact_fp32_cbs = {kWCb};",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("split recurrence stages require exact FP32 unpack");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split exact FP32 unpack CB set"));
    }

    #[test]
    fn rejects_split_readout_without_the_hifi2_program() {
        let invalid = PROGRAM_SOURCE.replace(SPLIT_READOUT_LAUNCH, "raw_readout = output;");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("split path must launch the proven HiFi2 readout");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split HiFi2 readout launch"));
    }

    #[test]
    fn rejects_split_readout_that_claims_correction_complete() {
        let invalid = PROGRAM_SOURCE.replace(
            SPLIT_RAW_OUTPUT_OWNERSHIP,
            "return Outputs{std::move(returned_state), std::move(raw_readout), true};",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("raw split output must retain caller correction ownership");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("split raw output ownership"));
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
    fn rejects_asymmetric_probe_with_generic_device_prework() {
        let without_post_readback_control = RUNTIME_SOURCE.replace(
            ASYMMETRIC_GENERIC_CONTROL,
            "invalid_generic_control_removed();",
        );
        let invalid = without_post_readback_control.replace(
            ASYMMETRIC_FUSED_LAUNCH,
            "generic_wkv_step(context, state, a, b, w, v, k, r);\n  rwkv7::fused_wkv::run_state_dot_a_diagnostic(",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("generic recurrence before fused launch");
        assert_eq!(error.code, "forbidden_asymmetric_prework");
        assert!(
            error
                .detail
                .contains("generic control must run only after fused readback")
        );
    }

    #[test]
    fn rejects_asymmetric_probe_without_full_split_readout_row_reshape() {
        let invalid = RUNTIME_SOURCE.replace(
            ASYMMETRIC_FULL_SPLIT_READOUT_ROW_RESHAPE,
            "const Tensor full_split_readout_rows = full_split_readout;",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("full-split correction requires row-shaped readout");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("full-split readout row reshape"));
    }

    #[test]
    fn rejects_asymmetric_probe_without_generic_readout_row_reshape() {
        let invalid = RUNTIME_SOURCE.replace(
            ASYMMETRIC_GENERIC_READOUT_ROW_RESHAPE,
            "const Tensor generic_readout_rows = generic.output_for_hidden;",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("generic correction requires row-shaped readout");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("generic readout row reshape"));
    }

    #[test]
    fn rejects_asymmetric_probe_without_runtime_binding() {
        let invalid = RUNTIME_SOURCE.replace(
            ASYMMETRIC_BUILD_BINDING,
            "rwkv7::fused_wkv_reference::invalid_diagnostic_build(",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("missing diagnostic build binding");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("diagnostic build binding"));
    }

    #[test]
    fn rejects_asymmetric_probe_without_fused_stage_marker() {
        let invalid =
            RUNTIME_SOURCE.replace(ASYMMETRIC_STAGE_MARKER, "invalid-asymmetric-stage-marker");
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("missing fused launch marker");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("bounded asymmetric stage marker"));
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
    fn rejects_a_missing_diagnostic_fp32_recurrence_policy() {
        let invalid = PROGRAM_SOURCE.replace(
            GENERIC_FP32_RECURRENCE_VECTOR_POLICY,
            "kGenericFp32RecurrenceVector = {false, false, false, false, false, false};",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("diagnostic recurrence must match generic FP32 casts");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("generic FP32 recurrence-vector policy")
        );
    }

    #[test]
    fn rejects_a_missing_production_bfloat16_recurrence_branch() {
        let invalid = PROGRAM_SOURCE.replace(
            PRODUCTION_BF16_RECURRENCE_FORMAT,
            "const tt::DataFormat format = fp32_format;",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("production recurrence vectors must remain BF16");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("production BF16 recurrence-vector format")
        );
    }

    #[test]
    fn rejects_a_missing_reader_recurrence_runtime_argument() {
        let invalid =
            PROGRAM_SOURCE.replace(READER_RECURRENCE_RUNTIME_ARG, "static_cast<void>(args);");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("reader must receive the recurrence diagnostic mode");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("reader recurrence runtime argument"));
    }

    #[test]
    fn rejects_a_missing_compute_recurrence_device_define() {
        let invalid =
            PROGRAM_SOURCE.replace(COMPUTE_RECURRENCE_DEVICE_DEFINE, "compute.defines.clear();");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("compute must receive the recurrence diagnostic define");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("compute recurrence device define"));
    }

    #[test]
    fn rejects_a_hard_coded_recurrence_device_define() {
        let invalid =
            PROGRAM_SOURCE.replace(RECURRENCE_DEVICE_DEFINE_VALUE, "true ? \"1\" : \"0\"");
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("device define must match the configured runtime mode");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("recurrence device-define value"));
    }

    #[test]
    fn rejects_an_asymmetric_receipt_without_fp32_recurrence_binding() {
        let invalid = RUNTIME_SOURCE.replace(
            ASYMMETRIC_FP32_RECURRENCE_RECEIPT,
            "{\"fp32_recurrence_vectors_removed\", false}",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("physical receipt must bind FP32 recurrence vectors");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("FP32 recurrence-bound asymmetric receipt")
        );
    }

    #[test]
    fn rejects_a_direct_bfloat16_recurrent_readout_buffer() {
        let invalid = PROGRAM_SOURCE.replace(
            FP32_READOUT_CB,
            "make_cb(kReadoutCb, bf16_format, kHeadTileCount, core_ranges)",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("the recurrent readout matmul must retain its FP32 output buffer");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("FP32 recurrent readout CB"));
    }

    #[test]
    fn rejects_a_missing_diagnostic_bfloat16_readout_row_buffer() {
        let invalid = PROGRAM_SOURCE.replace(
            BF16_READOUT_ROW_CB,
            "make_cb(kReadoutRowCb, fp32_format, kHeadTileCount, core_ranges)",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("diagnostic correction needs a BF16 readout row buffer");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("diagnostic BF16 readout row CB"));
    }

    #[test]
    fn rejects_a_missing_production_fp32_readout_row_buffer() {
        let invalid = PROGRAM_SOURCE.replace(
            FP32_READOUT_ROW_CB,
            "make_cb(kReadoutRowCb, fp32_format, kStateTileCount, core_ranges)",
        );
        let error = validate_fused_program_source(&invalid, RUNTIME_SOURCE)
            .expect_err("production must retain its FP32 readout row");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("production FP32 readout row CB"));
    }

    #[test]
    fn rejects_a_missing_bfloat16_readout_pack() {
        let invalid = COMPUTE_SOURCE.replace(
            BF16_READOUT_PACK_BOUNDARY,
            "// invalid fixture removed the BF16 readout pack",
        );
        let error = validate_fused_compute_source(&invalid)
            .expect_err("missing BF16 readout pack boundary");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("BF16 readout pack boundary"));
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
    fn rejects_a_physical_worker_without_an_explicit_wkv_policy() {
        let invalid = RUNTIME_SOURCE.replace(
            PHYSICAL_WORKER_POLICY_PARSE,
            "WkvExecutionPolicy::Production;",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("physical worker without an explicit WKV policy");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("explicit physical worker WKV policy"));
    }

    #[test]
    fn rejects_a_physical_worker_without_a_synchronized_start() {
        let invalid = RUNTIME_SOURCE.replace(
            SYNCHRONIZED_WORKER_START,
            "static_cast<void>(arguments.timed_start_barrier_path);",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("physical worker without a synchronized start");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("synchronized physical worker start"));
    }

    #[test]
    fn rejects_a_physical_worker_with_inspector_enabled() {
        let invalid = RUNTIME_SOURCE.replace(
            DISABLED_WORKER_INSPECTOR,
            "std::string_view(inspector_value) != kDisabledEnvironmentValue",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("physical worker with Inspector enabled");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("disabled physical worker Inspector"));
    }

    #[test]
    fn rejects_a_stall_probe_without_complete_step_timing() {
        let invalid = RUNTIME_SOURCE.replace(
            STALL_PROBE_STEP_COUNT,
            "require(!step_wall_seconds.empty(),",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without complete step timing");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("bounded physical worker stall step count")
        );
    }

    #[test]
    fn rejects_a_stall_probe_without_boundary_timing() {
        let invalid = RUNTIME_SOURCE.replace(
            STALL_PROBE_BOUNDARY_COUNT,
            "require(!step_boundary_timings.empty(),",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without complete boundary timing");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("bounded physical worker stall boundary count")
        );
    }

    #[test]
    fn rejects_a_stall_probe_without_primary_thread_affinity_receipt() {
        let invalid = RUNTIME_SOURCE.replace(
            PRIMARY_THREAD_AFFINITY_RECEIPT,
            "receipt[\"primary_thread_cpu_affinity_removed\"] =",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without primary thread affinity receipt");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("primary thread CPU affinity receipt"));
    }

    #[test]
    fn rejects_a_stall_probe_without_primary_thread_fifo_receipt() {
        let invalid = RUNTIME_SOURCE.replace(
            PRIMARY_THREAD_FIFO_RECEIPT,
            "receipt[\"primary_thread_fifo_priority_removed\"] =",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without primary thread FIFO receipt");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("primary thread FIFO receipt"));
    }

    #[test]
    fn rejects_a_stall_probe_without_fetch_queue_diagnostics() {
        let invalid = RUNTIME_SOURCE.replace(
            FETCH_QUEUE_DIAGNOSTICS_RECEIPT,
            "receipt[\"fetch_queue_diagnostics_removed\"] = {",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without fetch queue diagnostics");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("physical worker fetch queue diagnostics receipt")
        );
    }

    #[test]
    fn rejects_a_stall_probe_without_process_cpu_timing() {
        let invalid = RUNTIME_SOURCE.replace(
            PROCESS_CPU_TIMING_RECEIPT,
            "{\"model_submission_process_cpu_seconds_removed\",",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without process CPU timing");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("physical worker process CPU timing receipt")
        );
    }

    #[test]
    fn rejects_a_stall_probe_without_bounded_warmup() {
        let invalid = RUNTIME_SOURCE.replace(
            STALL_PROBE_BOUNDED_WARMUP,
            "false ? kStallProbeWarmupTokenCount",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without bounded warmup");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("bounded physical worker stall warmup")
        );
    }

    #[test]
    fn rejects_a_stall_probe_without_wall_clock_correlation() {
        let invalid = RUNTIME_SOURCE.replace(
            WALL_CLOCK_CORRELATION_RECEIPT,
            "receipt[\"timed_start_unix_ns\"] = 0U;",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("stall probe without wall-clock correlation");
        assert_eq!(error.code, "source_pattern_count");
        assert!(
            error
                .detail
                .contains("physical worker wall-clock correlation receipt")
        );
    }

    #[test]
    fn rejects_production_runtime_without_fifo_bound_admission() {
        let invalid = RUNTIME_SOURCE.replace(
            "*primary_thread_fifo_priority != kAdmittedPrimaryThreadFifoPriority",
            "*primary_thread_fifo_priority == kAdmittedPrimaryThreadFifoPriority",
        );
        let error = validate_fused_program_source(PROGRAM_SOURCE, &invalid)
            .expect_err("production runtime without FIFO-bound admission");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("FIFO-bound production admission"));
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
