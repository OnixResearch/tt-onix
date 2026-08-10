//! Static contract guard for the fused RWKV-7 WKV compute kernel.
//!
//! This guard does not compile or run the device kernel. It rejects source
//! drift that removes one recurrence term, changes the fixed head geometry,
//! aliases the FP32 state output with the BF16 readout, or breaks CB balance.

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

const BALANCED_CBS: [(&str, &str, &str); 10] = [
    ("a", "cb_wait_front(kACb", "cb_pop_front(kACb"),
    ("w", "cb_wait_front(kWCb", "cb_pop_front(kWCb"),
    ("k", "cb_wait_front(kKCb", "cb_pop_front(kKCb"),
    ("v", "cb_wait_front(kVCb", "cb_pop_front(kVCb"),
    ("r", "cb_wait_front(kRCb", "cb_pop_front(kRCb"),
    ("b", "cb_wait_front(kBCb", "cb_pop_front(kBCb"),
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

/// Validate the source structure of the first fused compute-kernel slice.
///
/// The deterministic kernel cross-compile and LLK audits remain separate
/// gates. This function only guards the source contract that those gates use.
pub fn validate_fused_compute_source(source: &str) -> Result<(), KernelContractError> {
    require_once(source, "64-wide head geometry", HEAD_GEOMETRY)?;
    require_once(source, "four-tile state geometry", STATE_GEOMETRY)?;
    require_once(source, "Blackhole guard", BLACKHOLE_GUARD)?;
    if source.contains(WORMHOLE_BRANCH) {
        return Err(KernelContractError::new(
            "unsupported_architecture_branch",
            "the fused compute kernel must not contain a Wormhole branch",
        ));
    }

    for (name, pattern) in REQUIRED_RECURRENCE {
        require_once(source, name, pattern)?;
    }

    require_once(source, "BF16 output CB", OUTPUT_CB)?;
    require_once(source, "FP32 next-state output CB", NEXT_STATE_OUTPUT_CB)?;
    require_once(source, "BF16 pack boundary", BF16_PACK_BOUNDARY)?;
    require_once(source, "FP32 pack boundary", FP32_PACK_BOUNDARY)?;

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

    for (name, wait_pattern, pop_pattern) in BALANCED_CBS {
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

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = include_str!("../../rwkv7-p150x2/kernel/fused_wkv_compute.cpp");

    #[test]
    fn accepts_the_reviewed_compute_source() {
        validate_fused_compute_source(SOURCE).expect("reviewed source");
    }

    #[test]
    fn rejects_a_missing_decay_term() {
        let invalid = SOURCE.replace(
            "mul_tiles_bcast_rows(\n        kStateCb, kWCb, output_tile, column_tile, output_tile);",
            "// invalid fixture removed the decay term",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("missing decay");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("decay"));
    }

    #[test]
    fn rejects_wrong_head_geometry() {
        let invalid = SOURCE.replace(HEAD_GEOMETRY, "constexpr std::uint32_t kHeadTileCount = 1;");
        let error = validate_fused_compute_source(&invalid).expect_err("wrong geometry");
        assert_eq!(error.code, "source_pattern_count");
        assert!(error.detail.contains("64-wide head geometry"));
    }

    #[test]
    fn rejects_an_unbalanced_state_buffer() {
        let invalid = SOURCE.replace(
            "cb_pop_front(kStateCb, kStateTileCount);",
            "// invalid fixture removed the state pop",
        );
        let error = validate_fused_compute_source(&invalid).expect_err("unbalanced state");
        assert_eq!(error.code, "unbalanced_circular_buffer");
        assert!(error.detail.contains("state"));
    }

    #[test]
    fn rejects_a_wormhole_branch() {
        let invalid = format!("{SOURCE}\n#if defined(ARCH_WORMHOLE)\n#endif\n");
        let error = validate_fused_compute_source(&invalid).expect_err("Wormhole branch");
        assert_eq!(error.code, "unsupported_architecture_branch");
    }
}
