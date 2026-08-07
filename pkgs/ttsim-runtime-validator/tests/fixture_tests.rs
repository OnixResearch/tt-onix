use serde::Deserialize;
use ttsim_runtime_validator::{
    ProcessObservation, ReceiptExpectations, SmokeClassification, SmokeStatus, classify_smoke,
    validate_receipt,
};

const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[derive(Debug, Deserialize)]
struct ProcessFixture {
    exit_code: i32,
    timed_out: bool,
    stdout: String,
    stderr: String,
}

fn classify_fixture(source: &str) -> (SmokeStatus, SmokeClassification) {
    let fixture: ProcessFixture = serde_json::from_str(source).expect("process fixture must parse");
    let outcome = classify_smoke(&ProcessObservation {
        exit_code: fixture.exit_code,
        timed_out: fixture.timed_out,
        stdout: &fixture.stdout,
        stderr: &fixture.stderr,
    });
    (outcome.status, outcome.classification)
}

#[test]
fn classifies_saved_process_fixtures() {
    let cases = [
        (
            include_str!("fixtures/process/passed.json"),
            SmokeStatus::Passed,
            SmokeClassification::Passed,
        ),
        (
            include_str!("fixtures/process/timeout.json"),
            SmokeStatus::Blocked,
            SmokeClassification::Timeout,
        ),
        (
            include_str!("fixtures/process/unsupported-operation.json"),
            SmokeStatus::Blocked,
            SmokeClassification::UnsupportedOperation,
        ),
        (
            include_str!("fixtures/process/nonzero-exit.json"),
            SmokeStatus::Blocked,
            SmokeClassification::NonzeroExit,
        ),
        (
            include_str!("fixtures/process/incomplete-close.json"),
            SmokeStatus::Blocked,
            SmokeClassification::IncompleteClose,
        ),
    ];

    for (fixture, expected_status, expected_classification) in cases {
        assert_eq!(
            classify_fixture(fixture),
            (expected_status, expected_classification)
        );
    }
}

#[test]
fn rejects_saved_hardware_claim_fixture() {
    let error = validate_receipt(
        include_bytes!("fixtures/receipts/hardware-claim.json"),
        None,
    )
    .expect_err("hardware claim fixture must fail");
    assert!(error.contains("hardware claim"));
}

#[test]
fn rejects_saved_stale_identity_fixture() {
    let expectations = ReceiptExpectations {
        host_system: "x86_64-linux",
        library_blake3: HASH_B,
        descriptor_blake3: HASH_A,
        tt_metal_blake3: HASH_A,
    };
    let error = validate_receipt(
        include_bytes!("fixtures/receipts/stale-identity.json"),
        Some(&expectations),
    )
    .expect_err("stale identity fixture must fail");
    assert!(error.contains("identity is stale"));
}

#[test]
fn rejects_saved_missing_non_claim_fixture() {
    let error = validate_receipt(
        include_bytes!("fixtures/receipts/missing-non-claim.json"),
        None,
    )
    .expect_err("missing non-claim fixture must fail");
    assert!(error.contains("non-claims are incomplete"));
}

#[test]
fn rejects_saved_unsupported_topology_fixture() {
    let error = validate_receipt(
        include_bytes!("fixtures/receipts/unsupported-topology.json"),
        None,
    )
    .expect_err("unsupported topology fixture must fail");
    assert!(error.contains("topology is unsupported"));
}
