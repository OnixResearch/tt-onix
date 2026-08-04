use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const REQUIRED_PROCESS_ATTEMPT_COUNT: u32 = 1;
const CANDIDATE_FACT_NAMES: [&str; 11] = [
    "upstream_identity",
    "revision",
    "license",
    "artifacts",
    "tokenizer",
    "state_kind",
    "required_operators",
    "memory_bounds",
    "reference_authority",
    "evidence_adapter",
    "topology_compatibility",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TargetIdentity {
    pub target_name: String,
    pub model_id: String,
    pub revision: String,
    pub architecture_family: String,
    pub state_kind: String,
    pub precision: String,
    pub architecture_adapter: String,
    pub evidence_adapter: String,
    pub artifact_blake3: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TopologyIdentity {
    pub id: String,
    pub mesh_shape: Vec<u32>,
    pub physical_device_ids: Vec<u32>,
    pub replication: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CommonEvidenceEnvelope {
    pub target: TargetIdentity,
    pub topology: TopologyIdentity,
    pub process_attempt_count: u32,
    pub profiler_attributed: bool,
    pub host_model_arithmetic: bool,
    pub cpu_model_fallback: bool,
    pub generated_text_published: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArchitectureEvidence {
    pub adapter_id: String,
    pub architecture_family: String,
    pub state_kind: String,
    pub observed_operations: BTreeSet<String>,
    pub payload_complete: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CandidateFacts {
    pub candidate_name: String,
    pub upstream_identity: Option<String>,
    pub revision: Option<String>,
    pub license: Option<String>,
    pub artifacts: Option<String>,
    pub tokenizer: Option<String>,
    pub state_kind: Option<String>,
    pub required_operators: Option<String>,
    pub memory_bounds: Option<String>,
    pub reference_authority: Option<String>,
    pub evidence_adapter: Option<String>,
    pub topology_compatibility: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Eligible,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CandidateReport {
    pub candidate_name: String,
    pub status: CandidateStatus,
    pub simulator_change_eligible: bool,
    pub runnable_output_exposed: bool,
    pub blockers: Vec<String>,
}

// r[impl tenstorrent.model_targets.evidence_contract]
// r[impl tenstorrent.model_targets.single_authority]
pub fn validate_common_evidence(
    expected_target: &TargetIdentity,
    expected_topology: &TopologyIdentity,
    evidence: &CommonEvidenceEnvelope,
) -> Result<(), String> {
    validate_target_identity(expected_target, &evidence.target)?;
    validate_topology_identity(expected_topology, &evidence.topology)?;
    require(
        evidence.process_attempt_count == REQUIRED_PROCESS_ATTEMPT_COUNT,
        "common evidence process policy mismatch",
    )?;
    require(
        evidence.profiler_attributed,
        "common evidence lacks profiler attribution",
    )?;
    require(
        !evidence.host_model_arithmetic,
        "common evidence reports host model arithmetic",
    )?;
    require(
        !evidence.cpu_model_fallback,
        "common evidence reports CPU model fallback",
    )?;
    require(
        !evidence.generated_text_published,
        "common evidence reports text publication before classification",
    )
}

// r[impl tenstorrent.model_targets.architecture_adapter]
// r[impl tenstorrent.model_targets.evidence_contract]
pub fn validate_architecture_evidence(
    expected_target: &TargetIdentity,
    required_operations: &BTreeSet<String>,
    evidence: &ArchitectureEvidence,
) -> Result<(), String> {
    require(
        evidence.adapter_id == expected_target.architecture_adapter,
        "architecture evidence adapter identity mismatch",
    )?;
    require(
        evidence.architecture_family == expected_target.architecture_family,
        "architecture evidence family mismatch",
    )?;
    require(
        evidence.state_kind == expected_target.state_kind,
        "architecture evidence state kind mismatch",
    )?;
    require(
        evidence.payload_complete,
        "architecture evidence payload is incomplete",
    )?;
    let missing = required_operations
        .difference(&evidence.observed_operations)
        .cloned()
        .collect::<Vec<_>>();
    require(
        missing.is_empty(),
        &format!("architecture evidence lacks operations: {}", missing.join(", ")),
    )
}

// r[impl tenstorrent.model_targets.candidate_report]
// r[impl tenstorrent.model_targets.no_unverified_candidate_claim]
pub fn evaluate_candidate(facts: &CandidateFacts) -> CandidateReport {
    let blockers = candidate_blockers(facts);
    let simulator_change_eligible = blockers.is_empty();
    let status = if simulator_change_eligible {
        CandidateStatus::Eligible
    } else {
        CandidateStatus::Unsupported
    };
    CandidateReport {
        candidate_name: facts.candidate_name.clone(),
        status,
        simulator_change_eligible,
        runnable_output_exposed: false,
        blockers,
    }
}

fn validate_target_identity(
    expected: &TargetIdentity,
    actual: &TargetIdentity,
) -> Result<(), String> {
    require_field("target name", &expected.target_name, &actual.target_name)?;
    require_field("model ID", &expected.model_id, &actual.model_id)?;
    require_field("revision", &expected.revision, &actual.revision)?;
    require_field(
        "architecture family",
        &expected.architecture_family,
        &actual.architecture_family,
    )?;
    require_field("state kind", &expected.state_kind, &actual.state_kind)?;
    require_field("precision", &expected.precision, &actual.precision)?;
    require_field(
        "architecture adapter",
        &expected.architecture_adapter,
        &actual.architecture_adapter,
    )?;
    require_field(
        "evidence adapter",
        &expected.evidence_adapter,
        &actual.evidence_adapter,
    )?;
    require(
        expected.artifact_blake3 == actual.artifact_blake3,
        "target artifact identity mismatch",
    )
}

fn validate_topology_identity(
    expected: &TopologyIdentity,
    actual: &TopologyIdentity,
) -> Result<(), String> {
    require_field("topology ID", &expected.id, &actual.id)?;
    require(
        expected.mesh_shape == actual.mesh_shape,
        "topology mesh shape mismatch",
    )?;
    require(
        expected.physical_device_ids == actual.physical_device_ids,
        "topology physical device identity mismatch",
    )?;
    require_field("topology replication", &expected.replication, &actual.replication)
}

fn candidate_blockers(facts: &CandidateFacts) -> Vec<String> {
    let values = [
        facts.upstream_identity.as_deref(),
        facts.revision.as_deref(),
        facts.license.as_deref(),
        facts.artifacts.as_deref(),
        facts.tokenizer.as_deref(),
        facts.state_kind.as_deref(),
        facts.required_operators.as_deref(),
        facts.memory_bounds.as_deref(),
        facts.reference_authority.as_deref(),
        facts.evidence_adapter.as_deref(),
        facts.topology_compatibility.as_deref(),
    ];
    CANDIDATE_FACT_NAMES
        .iter()
        .zip(values)
        .filter_map(|(name, value)| {
            if value.is_some_and(|text| !text.trim().is_empty()) {
                None
            } else {
                Some(format!("missing reviewed candidate fact: {name}"))
            }
        })
        .collect()
}

fn require_field(name: &str, expected: &str, actual: &str) -> Result<(), String> {
    require(
        expected == actual,
        &format!("target {name} mismatch: expected '{expected}', got '{actual}'"),
    )
}

fn require(condition: bool, diagnostic: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(diagnostic.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArchitectureEvidence, CandidateFacts, CandidateStatus, CommonEvidenceEnvelope,
        TargetIdentity, TopologyIdentity, evaluate_candidate, validate_architecture_evidence,
        validate_common_evidence,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const EXPECTED_ATTEMPT_COUNT: u32 = 1;

    fn target() -> TargetIdentity {
        TargetIdentity {
            target_name: "rwkv-target".to_owned(),
            model_id: "RWKV/model".to_owned(),
            revision: "revision".to_owned(),
            architecture_family: "rwkv7".to_owned(),
            state_kind: "recurrent_matrix".to_owned(),
            precision: "bf16".to_owned(),
            architecture_adapter: "rwkv-adapter".to_owned(),
            evidence_adapter: "rwkv-evidence".to_owned(),
            artifact_blake3: BTreeMap::from([("checkpoint".to_owned(), "hash".to_owned())]),
        }
    }

    fn topology() -> TopologyIdentity {
        TopologyIdentity {
            id: "p150x2".to_owned(),
            mesh_shape: vec![1, 2],
            physical_device_ids: vec![0, 1],
            replication: "full-lockstep".to_owned(),
        }
    }

    fn common_evidence() -> CommonEvidenceEnvelope {
        CommonEvidenceEnvelope {
            target: target(),
            topology: topology(),
            process_attempt_count: EXPECTED_ATTEMPT_COUNT,
            profiler_attributed: true,
            host_model_arithmetic: false,
            cpu_model_fallback: false,
            generated_text_published: false,
        }
    }

    fn candidate(value: Option<&str>) -> CandidateFacts {
        CandidateFacts {
            candidate_name: "candidate".to_owned(),
            upstream_identity: value.map(str::to_owned),
            revision: value.map(str::to_owned),
            license: value.map(str::to_owned),
            artifacts: value.map(str::to_owned),
            tokenizer: value.map(str::to_owned),
            state_kind: value.map(str::to_owned),
            required_operators: value.map(str::to_owned),
            memory_bounds: value.map(str::to_owned),
            reference_authority: value.map(str::to_owned),
            evidence_adapter: value.map(str::to_owned),
            topology_compatibility: value.map(str::to_owned),
        }
    }

    // r[verify tenstorrent.model_targets.evidence_contract.complete]
    #[test]
    fn common_and_architecture_evidence_accept_complete_inputs() {
        validate_common_evidence(&target(), &topology(), &common_evidence())
            .expect("complete common evidence must pass");
        let required = BTreeSet::from(["matmul".to_owned(), "wkv".to_owned()]);
        let architecture = ArchitectureEvidence {
            adapter_id: "rwkv-adapter".to_owned(),
            architecture_family: "rwkv7".to_owned(),
            state_kind: "recurrent_matrix".to_owned(),
            observed_operations: required.clone(),
            payload_complete: true,
        };
        validate_architecture_evidence(&target(), &required, &architecture)
            .expect("complete architecture evidence must pass");
    }

    // r[verify tenstorrent.model_targets.evidence_contract.incomplete]
    // r[verify tenstorrent.model_targets.evidence_contract.wrong_adapter]
    #[test]
    fn evidence_rejects_early_publication_and_wrong_adapter() {
        let mut common = common_evidence();
        common.generated_text_published = true;
        assert!(
            validate_common_evidence(&target(), &topology(), &common)
                .expect_err("early publication must fail")
                .contains("publication")
        );
        let required = BTreeSet::from(["wkv".to_owned()]);
        let architecture = ArchitectureEvidence {
            adapter_id: "transformer-adapter".to_owned(),
            architecture_family: "rwkv7".to_owned(),
            state_kind: "recurrent_matrix".to_owned(),
            observed_operations: required.clone(),
            payload_complete: true,
        };
        assert!(
            validate_architecture_evidence(&target(), &required, &architecture)
                .expect_err("wrong adapter must fail")
                .contains("adapter identity")
        );
    }

    // r[verify tenstorrent.model_targets.candidate_report.complete]
    #[test]
    fn complete_candidate_is_eligible_only_for_a_simulator_change() {
        let report = evaluate_candidate(&candidate(Some("reviewed")));
        assert_eq!(report.status, CandidateStatus::Eligible);
        assert!(report.simulator_change_eligible);
        assert!(!report.runnable_output_exposed);
        assert!(report.blockers.is_empty());
    }

    // r[verify tenstorrent.model_targets.candidate_report.incomplete]
    // r[verify tenstorrent.model_targets.no_unverified_candidate_claim.unreviewed]
    #[test]
    fn incomplete_candidate_is_unsupported_and_exposes_no_runtime() {
        let report = evaluate_candidate(&candidate(None));
        assert_eq!(report.status, CandidateStatus::Unsupported);
        assert!(!report.simulator_change_eligible);
        assert!(!report.runnable_output_exposed);
        assert_eq!(report.blockers.len(), super::CANDIDATE_FACT_NAMES.len());
    }
}
