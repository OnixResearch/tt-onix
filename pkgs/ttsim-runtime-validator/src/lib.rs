// r[impl tenstorrent.ttsim.evidence]
// r[verify tenstorrent.ttsim.single_chip_smoke.invalid]
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 3_600;
const MAXIMUM_RECEIPT_BYTES: usize = 1_048_576;
const BLAKE3_HEX_LENGTH: usize = 64;
pub const SUPPORTED_SIMULATOR_VERSION: &str = "1.10.6";
const SUPPORTED_HOST_COUNT: usize = 2;
const SUPPORTED_HOSTS: [&str; SUPPORTED_HOST_COUNT] = ["aarch64-linux", "x86_64-linux"];
const REQUIRED_NON_CLAIM_COUNT: usize = 6;
const REQUIRED_NON_CLAIMS: [&str; REQUIRED_NON_CLAIM_COUNT] = [
    "physical placement",
    "hardware profiling",
    "performance",
    "thermals",
    "device health",
    "silicon success",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SmokeStatus {
    Passed,
    Blocked,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SmokeClassification {
    Passed,
    Timeout,
    UnsupportedOperation,
    NonzeroExit,
    IncompleteDeviceOpen,
    IncompleteFirmwareCompilation,
    IncompleteRepresentativeOperations,
    IncompleteClose,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessObservation<'a> {
    pub exit_code: i32,
    pub timed_out: bool,
    pub stdout: &'a str,
    pub stderr: &'a str,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SmokeObservations {
    pub device_opened: bool,
    pub firmware_compiled: bool,
    pub representative_operations_completed: bool,
    pub clean_close: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SmokeOutcome {
    pub status: SmokeStatus,
    pub classification: SmokeClassification,
    pub blocker: Option<String>,
    pub observations: SmokeObservations,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BoundIdentity {
    pub value: String,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReceiptIdentities {
    pub library: BoundIdentity,
    pub descriptor: BoundIdentity,
    pub tt_metal: BoundIdentity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SimulatorReceipt {
    pub schema_version: u32,
    pub simulated: bool,
    pub simulator_version: String,
    pub host_system: String,
    pub topology: String,
    pub dispatch_mode: String,
    pub workload: String,
    pub timeout_seconds: u64,
    pub status: SmokeStatus,
    pub classification: SmokeClassification,
    pub blocker: Option<String>,
    pub observations: SmokeObservations,
    pub identities: ReceiptIdentities,
    pub claims: Vec<String>,
    pub non_claims: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptExpectations<'a> {
    pub host_system: &'a str,
    pub library_blake3: &'a str,
    pub descriptor_blake3: &'a str,
    pub tt_metal_blake3: &'a str,
}

pub fn classify_smoke(observation: &ProcessObservation<'_>) -> SmokeOutcome {
    let observations = parse_observations(observation.stdout);
    let unsupported = observation.stderr.contains("UnsupportedFunctionality")
        || observation.stderr.contains("Unsupported operation")
        || observation.stderr.contains("unsupported operation");

    let (status, classification, blocker) = if observation.timed_out {
        blocked(
            SmokeClassification::Timeout,
            "simulator smoke reached its named timeout",
        )
    } else if unsupported {
        blocked(
            SmokeClassification::UnsupportedOperation,
            "simulator reported an unsupported operation",
        )
    } else if observation.exit_code != 0 {
        blocked(
            SmokeClassification::NonzeroExit,
            &format!(
                "simulator smoke exited with status {}",
                observation.exit_code
            ),
        )
    } else if !observations.device_opened {
        blocked(
            SmokeClassification::IncompleteDeviceOpen,
            "simulator smoke did not record one device open",
        )
    } else if !observations.firmware_compiled {
        blocked(
            SmokeClassification::IncompleteFirmwareCompilation,
            "simulator smoke did not record firmware compilation",
        )
    } else if !observations.representative_operations_completed {
        blocked(
            SmokeClassification::IncompleteRepresentativeOperations,
            "simulator smoke did not complete representative operations",
        )
    } else if !observations.clean_close {
        blocked(
            SmokeClassification::IncompleteClose,
            "simulator smoke did not close cleanly",
        )
    } else {
        (SmokeStatus::Passed, SmokeClassification::Passed, None)
    };

    SmokeOutcome {
        status,
        classification,
        blocker,
        observations,
    }
}

fn parse_observations(stdout: &str) -> SmokeObservations {
    SmokeObservations {
        device_opened: stdout.contains("\"event\": \"device_opened\""),
        firmware_compiled: stdout.contains("\"event\": \"firmware_compiled\""),
        representative_operations_completed: stdout
            .contains("\"event\": \"representative_operations_completed\""),
        clean_close: stdout.contains("\"event\": \"clean_close\""),
    }
}

fn blocked(
    classification: SmokeClassification,
    message: &str,
) -> (SmokeStatus, SmokeClassification, Option<String>) {
    (
        SmokeStatus::Blocked,
        classification,
        Some(message.to_owned()),
    )
}

pub fn validate_receipt(
    receipt_bytes: &[u8],
    expectations: Option<&ReceiptExpectations<'_>>,
) -> Result<SimulatorReceipt, String> {
    if receipt_bytes.is_empty() {
        return Err("simulator receipt is empty".to_owned());
    }
    if receipt_bytes.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(format!(
            "simulator receipt exceeds {MAXIMUM_RECEIPT_BYTES} bytes"
        ));
    }

    let receipt: SimulatorReceipt = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("simulator receipt JSON is invalid: {error}"))?;
    validate_receipt_fields(&receipt)?;
    if let Some(expected) = expectations {
        validate_expected_identity(&receipt, expected)?;
    }
    Ok(receipt)
}

fn validate_receipt_fields(receipt: &SimulatorReceipt) -> Result<(), String> {
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION {
        return Err("simulator receipt schema version is unsupported".to_owned());
    }
    if !receipt.simulated {
        return Err("simulator receipt must declare simulated execution".to_owned());
    }
    if receipt.simulator_version != SUPPORTED_SIMULATOR_VERSION {
        return Err(format!(
            "simulator receipt version is unsupported: {}",
            receipt.simulator_version
        ));
    }
    if !SUPPORTED_HOSTS.contains(&receipt.host_system.as_str()) {
        return Err(format!(
            "simulator receipt host system is unsupported: {}",
            receipt.host_system
        ));
    }
    if receipt.topology != "blackhole-x1" {
        return Err(format!(
            "simulator receipt topology is unsupported: {}",
            receipt.topology
        ));
    }
    if receipt.dispatch_mode != "slow" {
        return Err("simulator receipt must bind slow dispatch".to_owned());
    }
    if receipt.workload.is_empty() {
        return Err("simulator receipt workload is empty".to_owned());
    }
    if receipt.timeout_seconds == 0 || receipt.timeout_seconds > DEFAULT_TIMEOUT_SECONDS {
        return Err("simulator receipt timeout is outside the admitted bound".to_owned());
    }
    if !receipt.claims.is_empty() {
        return Err("simulator receipt contains a hardware claim".to_owned());
    }

    let actual_non_claims: BTreeSet<&str> = receipt.non_claims.iter().map(String::as_str).collect();
    let required_non_claims: BTreeSet<&str> = REQUIRED_NON_CLAIMS.into_iter().collect();
    if actual_non_claims != required_non_claims {
        return Err("simulator receipt non-claims are incomplete".to_owned());
    }

    validate_identity("library", &receipt.identities.library)?;
    validate_identity("descriptor", &receipt.identities.descriptor)?;
    validate_identity("tt_metal", &receipt.identities.tt_metal)?;
    validate_outcome_consistency(receipt)
}

fn validate_identity(name: &str, identity: &BoundIdentity) -> Result<(), String> {
    if identity.value.is_empty() {
        return Err(format!("simulator receipt {name} identity is empty"));
    }
    let hash_is_valid = identity.blake3.len() == BLAKE3_HEX_LENGTH
        && identity
            .blake3
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !hash_is_valid {
        return Err(format!(
            "simulator receipt {name} identity is not lowercase BLAKE3 hex"
        ));
    }
    Ok(())
}

fn validate_outcome_consistency(receipt: &SimulatorReceipt) -> Result<(), String> {
    match receipt.status {
        SmokeStatus::Passed => {
            if receipt.classification != SmokeClassification::Passed
                || receipt.blocker.is_some()
                || !receipt.observations.device_opened
                || !receipt.observations.firmware_compiled
                || !receipt.observations.representative_operations_completed
                || !receipt.observations.clean_close
            {
                return Err("passed simulator receipt has incomplete observations".to_owned());
            }
        }
        SmokeStatus::Blocked => {
            if receipt.classification == SmokeClassification::Passed
                || receipt.blocker.as_deref().unwrap_or_default().is_empty()
            {
                return Err("blocked simulator receipt has no exact blocker".to_owned());
            }
        }
    }
    Ok(())
}

fn validate_expected_identity(
    receipt: &SimulatorReceipt,
    expected: &ReceiptExpectations<'_>,
) -> Result<(), String> {
    if receipt.host_system != expected.host_system
        || receipt.identities.library.blake3 != expected.library_blake3
        || receipt.identities.descriptor.blake3 != expected.descriptor_blake3
        || receipt.identities.tt_metal.blake3 != expected.tt_metal_blake3
    {
        return Err("simulator receipt identity is stale".to_owned());
    }
    Ok(())
}

pub fn required_non_claims() -> Vec<String> {
    REQUIRED_NON_CLAIMS.into_iter().map(str::to_owned).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SUCCESS_LOG: &str = concat!(
        "TTSIM_OBSERVATION {\"event\": \"device_opened\"}\n",
        "TTSIM_OBSERVATION {\"event\": \"firmware_compiled\"}\n",
        "TTSIM_OBSERVATION {\"event\": \"representative_operations_completed\"}\n",
        "TTSIM_OBSERVATION {\"event\": \"clean_close\"}\n",
    );

    fn valid_receipt() -> SimulatorReceipt {
        SimulatorReceipt {
            schema_version: RECEIPT_SCHEMA_VERSION,
            simulated: true,
            simulator_version: SUPPORTED_SIMULATOR_VERSION.to_owned(),
            host_system: "x86_64-linux".to_owned(),
            topology: "blackhole-x1".to_owned(),
            dispatch_mode: "slow".to_owned(),
            workload: "ttnn-ones-add".to_owned(),
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            status: SmokeStatus::Passed,
            classification: SmokeClassification::Passed,
            blocker: None,
            observations: SmokeObservations {
                device_opened: true,
                firmware_compiled: true,
                representative_operations_completed: true,
                clean_close: true,
            },
            identities: ReceiptIdentities {
                library: identity("library"),
                descriptor: identity("descriptor"),
                tt_metal: identity("tt-metal-0.74.0"),
            },
            claims: Vec::new(),
            non_claims: required_non_claims(),
        }
    }

    fn identity(value: &str) -> BoundIdentity {
        BoundIdentity {
            value: value.to_owned(),
            blake3: HASH_A.to_owned(),
        }
    }

    fn encode(receipt: &SimulatorReceipt) -> Vec<u8> {
        serde_json::to_vec(receipt).expect("test receipt must serialize")
    }

    #[test]
    fn admits_complete_simulated_receipt() {
        let receipt = valid_receipt();
        let expected = ReceiptExpectations {
            host_system: "x86_64-linux",
            library_blake3: HASH_A,
            descriptor_blake3: HASH_A,
            tt_metal_blake3: HASH_A,
        };
        assert!(validate_receipt(&encode(&receipt), Some(&expected)).is_ok());
    }

    #[test]
    fn rejects_previous_simulator_version() {
        const PREVIOUS_SIMULATOR_VERSIONS: &[&str] = &["1.9.2", "1.10.1", "1.10.2"];
        for &version in PREVIOUS_SIMULATOR_VERSIONS {
            let mut receipt = valid_receipt();
            receipt.simulator_version = version.to_owned();
            let error = validate_receipt(&encode(&receipt), None).expect_err("version must fail");
            assert!(error.contains("version is unsupported"));
        }
    }

    #[test]
    fn classifies_complete_smoke_as_passed() {
        let outcome = classify_smoke(&ProcessObservation {
            exit_code: 0,
            timed_out: false,
            stdout: SUCCESS_LOG,
            stderr: "",
        });
        assert_eq!(outcome.status, SmokeStatus::Passed);
    }

    #[test]
    fn classifies_timeout_before_other_failures() {
        let outcome = classify_smoke(&ProcessObservation {
            exit_code: 1,
            timed_out: true,
            stdout: "",
            stderr: "UnsupportedFunctionality",
        });
        assert_eq!(outcome.classification, SmokeClassification::Timeout);
    }

    #[test]
    fn classifies_unsupported_operation() {
        let outcome = classify_smoke(&ProcessObservation {
            exit_code: 1,
            timed_out: false,
            stdout: "",
            stderr: "UnsupportedFunctionality: instruction",
        });
        assert_eq!(
            outcome.classification,
            SmokeClassification::UnsupportedOperation
        );
    }

    #[test]
    fn classifies_nonzero_exit() {
        let outcome = classify_smoke(&ProcessObservation {
            exit_code: 1,
            timed_out: false,
            stdout: "",
            stderr: "failure",
        });
        assert_eq!(outcome.classification, SmokeClassification::NonzeroExit);
    }

    #[test]
    fn rejects_incomplete_close() {
        let log = SUCCESS_LOG.replace("TTSIM_OBSERVATION {\"event\": \"clean_close\"}\n", "");
        let outcome = classify_smoke(&ProcessObservation {
            exit_code: 0,
            timed_out: false,
            stdout: &log,
            stderr: "",
        });
        assert_eq!(outcome.classification, SmokeClassification::IncompleteClose);
    }

    #[test]
    fn rejects_hardware_claim() {
        let mut receipt = valid_receipt();
        receipt.claims.push("silicon passed".to_owned());
        let error = validate_receipt(&encode(&receipt), None).expect_err("claim must fail");
        assert!(error.contains("hardware claim"));
    }

    #[test]
    fn rejects_stale_identity() {
        let receipt = valid_receipt();
        let expected = ReceiptExpectations {
            host_system: "x86_64-linux",
            library_blake3: HASH_B,
            descriptor_blake3: HASH_A,
            tt_metal_blake3: HASH_A,
        };
        let error = validate_receipt(&encode(&receipt), Some(&expected))
            .expect_err("stale identity must fail");
        assert!(error.contains("identity is stale"));
    }

    #[test]
    fn rejects_missing_non_claim() {
        let mut receipt = valid_receipt();
        receipt.non_claims.pop();
        let error = validate_receipt(&encode(&receipt), None).expect_err("non-claim must fail");
        assert!(error.contains("non-claims are incomplete"));
    }

    #[test]
    fn rejects_unsupported_topology() {
        let mut receipt = valid_receipt();
        receipt.topology = "blackhole-x2".to_owned();
        let error = validate_receipt(&encode(&receipt), None).expect_err("topology must fail");
        assert!(error.contains("topology is unsupported"));
    }

    #[test]
    fn rejects_non_blake3_identity() {
        let mut receipt = valid_receipt();
        receipt.identities.library.blake3 = "not-a-hash".to_owned();
        let error = validate_receipt(&encode(&receipt), None).expect_err("hash must fail");
        assert!(error.contains("not lowercase BLAKE3 hex"));
    }
}
