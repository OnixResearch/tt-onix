use serde::{Deserialize, Deserializer, de::IgnoredAny};

const RECEIPT_SCHEMA_VERSION: u32 = 1;
const SUCCESS_EXIT_STATUS: i32 = 0;
const HOST_RECEIPT_TARGET: &str = "rwkv_ttwkv7_persistent_physical_dispatch";
const CORE_RECEIPT_TARGET: &str = "rwkv_ttwkv7_persistent_metalium_core";
const CLOSED_TERMINAL_STATE: &str = "closed";
const MINIMUM_CALL_COUNT: u32 = 1;
const KIBIBYTE_BYTES: u64 = 1024;
const MAXIMUM_RECEIPT_KIBIBYTES: u64 = 2048;
pub const MAXIMUM_PERSISTENT_RECEIPT_BYTES: u64 =
    MAXIMUM_RECEIPT_KIBIBYTES * KIBIBYTE_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentReceiptExpectations {
    pub call_count: u32,
    pub continuity_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentReceiptSummary {
    pub call_count: u32,
    pub continuity_count: u32,
}

#[derive(Debug, Default)]
struct FieldPresence(bool);

impl<'de> Deserialize<'de> for FieldPresence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let _ignored = IgnoredAny::deserialize(deserializer)?;
        Ok(Self(true))
    }
}

#[derive(Debug, Deserialize)]
struct PersistentHostReceipt {
    schema_version: u32,
    target: String,
    physical_wkv_call_count: u32,
    same_layer_state_continuity_count: u32,
    child_exit_status: i32,
    core: PersistentCoreReceipt,
    terminal_state: String,
    #[serde(default)]
    session_call_count: FieldPresence,
    #[serde(default)]
    session_continuity_check_count: FieldPresence,
}

#[derive(Debug, Deserialize)]
struct PersistentCoreReceipt {
    schema_version: u32,
    target: String,
    physical_wkv_call_count: u32,
    session: PersistentSessionReceipt,
    terminal_state: String,
}

#[derive(Debug, Deserialize)]
struct PersistentSessionReceipt {
    call_count: u32,
    same_layer_state_continuity_count: u32,
    terminal_state: String,
}

pub fn validate_persistent_receipt(
    receipt_bytes: &[u8],
    expectations: PersistentReceiptExpectations,
) -> Result<PersistentReceiptSummary, String> {
    let receipt_byte_count = u64::try_from(receipt_bytes.len())
        .map_err(|error| format!("persistent receipt byte count is invalid: {error}"))?;
    if receipt_byte_count == 0 {
        return Err("persistent receipt is empty".to_owned());
    }
    if receipt_byte_count > MAXIMUM_PERSISTENT_RECEIPT_BYTES {
        return Err(format!(
            "persistent receipt exceeds {MAXIMUM_PERSISTENT_RECEIPT_BYTES} bytes"
        ));
    }
    validate_expectations(expectations)?;
    let receipt: PersistentHostReceipt = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("persistent receipt JSON is invalid: {error}"))?;
    validate_host_receipt(&receipt, expectations)?;
    validate_core_receipt(&receipt.core, expectations)?;
    validate_session_receipt(&receipt.core.session, expectations)?;

    let summary = PersistentReceiptSummary {
        call_count: receipt.core.session.call_count,
        continuity_count: receipt.core.session.same_layer_state_continuity_count,
    };
    debug_assert_eq!(summary.call_count, expectations.call_count);
    debug_assert_eq!(summary.continuity_count, expectations.continuity_count);
    Ok(summary)
}

fn validate_expectations(expectations: PersistentReceiptExpectations) -> Result<(), String> {
    if expectations.call_count < MINIMUM_CALL_COUNT {
        return Err("expected call count must be positive".to_owned());
    }
    if expectations.continuity_count > expectations.call_count {
        return Err("expected continuity count exceeds the expected call count".to_owned());
    }
    Ok(())
}

fn validate_host_receipt(
    receipt: &PersistentHostReceipt,
    expectations: PersistentReceiptExpectations,
) -> Result<(), String> {
    require_u32("host schema_version", receipt.schema_version, RECEIPT_SCHEMA_VERSION)?;
    require_text("host target", &receipt.target, HOST_RECEIPT_TARGET)?;
    require_u32(
        "host physical_wkv_call_count",
        receipt.physical_wkv_call_count,
        expectations.call_count,
    )?;
    require_u32(
        "host same_layer_state_continuity_count",
        receipt.same_layer_state_continuity_count,
        expectations.continuity_count,
    )?;
    require_i32(
        "host child_exit_status",
        receipt.child_exit_status,
        SUCCESS_EXIT_STATUS,
    )?;
    require_text(
        "host terminal_state",
        &receipt.terminal_state,
        CLOSED_TERMINAL_STATE,
    )?;
    if receipt.session_call_count.0 {
        return Err("host receipt contains obsolete session_call_count".to_owned());
    }
    if receipt.session_continuity_check_count.0 {
        return Err("host receipt contains obsolete session_continuity_check_count".to_owned());
    }
    Ok(())
}

fn validate_core_receipt(
    receipt: &PersistentCoreReceipt,
    expectations: PersistentReceiptExpectations,
) -> Result<(), String> {
    require_u32("core schema_version", receipt.schema_version, RECEIPT_SCHEMA_VERSION)?;
    require_text("core target", &receipt.target, CORE_RECEIPT_TARGET)?;
    require_u32(
        "core physical_wkv_call_count",
        receipt.physical_wkv_call_count,
        expectations.call_count,
    )?;
    require_text(
        "core terminal_state",
        &receipt.terminal_state,
        CLOSED_TERMINAL_STATE,
    )
}

fn validate_session_receipt(
    receipt: &PersistentSessionReceipt,
    expectations: PersistentReceiptExpectations,
) -> Result<(), String> {
    require_u32("core.session.call_count", receipt.call_count, expectations.call_count)?;
    require_u32(
        "core.session.same_layer_state_continuity_count",
        receipt.same_layer_state_continuity_count,
        expectations.continuity_count,
    )?;
    require_text(
        "core.session.terminal_state",
        &receipt.terminal_state,
        CLOSED_TERMINAL_STATE,
    )
}

fn require_u32(field: &str, actual: u32, expected: u32) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{field} mismatch: expected {expected}, found {actual}"
        ));
    }
    Ok(())
}

fn require_i32(field: &str, actual: i32, expected: i32) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{field} mismatch: expected {expected}, found {actual}"
        ));
    }
    Ok(())
}

fn require_text(field: &str, actual: &str, expected: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{field} mismatch: expected {expected:?}, found {actual:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const EXPECTED_CALL_COUNT: u32 = 24;
    const EXPECTED_CONTINUITY_COUNT: u32 = 12;
    const WRONG_CALL_COUNT: u32 = EXPECTED_CALL_COUNT - 1;
    const WRONG_CONTINUITY_COUNT: u32 = EXPECTED_CONTINUITY_COUNT - 1;
    const WRONG_SCHEMA_VERSION: u32 = RECEIPT_SCHEMA_VERSION + 1;
    const FAILURE_EXIT_STATUS: i32 = 1;
    const OVERSIZED_RECEIPT_BYTES: u64 = MAXIMUM_PERSISTENT_RECEIPT_BYTES + 1;

    fn expectations() -> PersistentReceiptExpectations {
        PersistentReceiptExpectations {
            call_count: EXPECTED_CALL_COUNT,
            continuity_count: EXPECTED_CONTINUITY_COUNT,
        }
    }

    fn valid_receipt() -> Value {
        json!({
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "target": HOST_RECEIPT_TARGET,
            "physical_wkv_call_count": EXPECTED_CALL_COUNT,
            "same_layer_state_continuity_count": EXPECTED_CONTINUITY_COUNT,
            "child_exit_status": SUCCESS_EXIT_STATUS,
            "core": {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "target": CORE_RECEIPT_TARGET,
                "physical_wkv_call_count": EXPECTED_CALL_COUNT,
                "session": {
                    "call_count": EXPECTED_CALL_COUNT,
                    "same_layer_state_continuity_count": EXPECTED_CONTINUITY_COUNT,
                    "terminal_state": CLOSED_TERMINAL_STATE
                },
                "terminal_state": CLOSED_TERMINAL_STATE
            },
            "terminal_state": CLOSED_TERMINAL_STATE
        })
    }

    fn validate_value(value: &Value) -> Result<PersistentReceiptSummary, String> {
        let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
        validate_persistent_receipt(&bytes, expectations())
    }

    #[test]
    fn accepts_nested_persistent_receipt() {
        let summary = validate_value(&valid_receipt()).expect("valid receipt must pass");
        assert_eq!(summary.call_count, EXPECTED_CALL_COUNT);
        assert_eq!(summary.continuity_count, EXPECTED_CONTINUITY_COUNT);
    }

    #[test]
    fn accepts_zero_continuity_when_expected() {
        let mut receipt = valid_receipt();
        receipt["same_layer_state_continuity_count"] = json!(0);
        receipt["core"]["session"]["same_layer_state_continuity_count"] = json!(0);
        let bytes = serde_json::to_vec(&receipt).expect("receipt must serialize");
        let summary = validate_persistent_receipt(
            &bytes,
            PersistentReceiptExpectations {
                call_count: EXPECTED_CALL_COUNT,
                continuity_count: 0,
            },
        )
        .expect("zero expected continuity must pass");
        assert_eq!(summary.call_count, EXPECTED_CALL_COUNT);
        assert_eq!(summary.continuity_count, 0);
    }

    #[test]
    fn rejects_stale_flat_session_fields() {
        let mut receipt = valid_receipt();
        receipt["session_call_count"] = json!(EXPECTED_CALL_COUNT);
        receipt["session_continuity_check_count"] = json!(EXPECTED_CONTINUITY_COUNT);
        let error = validate_value(&receipt).expect_err("obsolete fields must fail");
        assert!(error.contains("obsolete session_call_count"));

        let mut null_receipt = valid_receipt();
        null_receipt["session_call_count"] = Value::Null;
        let null_error = validate_value(&null_receipt).expect_err("obsolete null field must fail");
        assert!(null_error.contains("obsolete session_call_count"));
    }

    #[test]
    fn rejects_missing_nested_session() {
        let mut receipt = valid_receipt();
        receipt["core"]
            .as_object_mut()
            .expect("core must be an object")
            .remove("session");
        let error = validate_value(&receipt).expect_err("missing session must fail");
        assert!(error.contains("missing field `session`"));
    }

    #[test]
    fn rejects_wrong_host_and_nested_counts() {
        let mut host_receipt = valid_receipt();
        host_receipt["physical_wkv_call_count"] = json!(WRONG_CALL_COUNT);
        let host_error = validate_value(&host_receipt).expect_err("wrong host count must fail");
        assert!(host_error.contains("host physical_wkv_call_count mismatch"));

        let mut session_receipt = valid_receipt();
        session_receipt["core"]["session"]["call_count"] = json!(WRONG_CALL_COUNT);
        let session_error =
            validate_value(&session_receipt).expect_err("wrong session count must fail");
        assert!(session_error.contains("core.session.call_count mismatch"));
    }

    #[test]
    fn rejects_wrong_continuity_and_terminal_state() {
        let mut continuity_receipt = valid_receipt();
        continuity_receipt["core"]["session"]["same_layer_state_continuity_count"] =
            json!(WRONG_CONTINUITY_COUNT);
        let continuity_error =
            validate_value(&continuity_receipt).expect_err("wrong continuity count must fail");
        assert!(continuity_error.contains("core.session.same_layer_state_continuity_count mismatch"));

        let mut terminal_receipt = valid_receipt();
        terminal_receipt["terminal_state"] = json!("open");
        let terminal_error =
            validate_value(&terminal_receipt).expect_err("open terminal state must fail");
        assert!(terminal_error.contains("host terminal_state mismatch"));
    }

    #[test]
    fn rejects_wrong_authority_and_child_status() {
        let mut schema_receipt = valid_receipt();
        schema_receipt["schema_version"] = json!(WRONG_SCHEMA_VERSION);
        let schema_error = validate_value(&schema_receipt).expect_err("wrong schema must fail");
        assert!(schema_error.contains("host schema_version mismatch"));

        let mut target_receipt = valid_receipt();
        target_receipt["core"]["target"] = json!("wrong_target");
        let target_error = validate_value(&target_receipt).expect_err("wrong target must fail");
        assert!(target_error.contains("core target mismatch"));

        let mut status_receipt = valid_receipt();
        status_receipt["child_exit_status"] = json!(FAILURE_EXIT_STATUS);
        let status_error = validate_value(&status_receipt).expect_err("child failure must fail");
        assert!(status_error.contains("host child_exit_status mismatch"));
    }

    #[test]
    fn rejects_empty_oversized_malformed_and_duplicate_fields() {
        let empty_error =
            validate_persistent_receipt(b"", expectations()).expect_err("empty receipt must fail");
        assert!(empty_error.contains("persistent receipt is empty"));

        let oversized_length =
            usize::try_from(OVERSIZED_RECEIPT_BYTES).expect("test size must fit usize");
        let oversized_receipt = vec![b' '; oversized_length];
        let oversized_error = validate_persistent_receipt(&oversized_receipt, expectations())
            .expect_err("oversized receipt must fail");
        assert!(oversized_error.contains("persistent receipt exceeds"));

        let malformed_error = validate_persistent_receipt(b"{", expectations())
            .expect_err("malformed JSON must fail");
        assert!(malformed_error.contains("persistent receipt JSON is invalid"));

        let duplicate = format!(
            r#"{{"schema_version":{RECEIPT_SCHEMA_VERSION},"target":"{HOST_RECEIPT_TARGET}","physical_wkv_call_count":{EXPECTED_CALL_COUNT},"same_layer_state_continuity_count":{EXPECTED_CONTINUITY_COUNT},"child_exit_status":{SUCCESS_EXIT_STATUS},"core":{{"schema_version":{RECEIPT_SCHEMA_VERSION},"target":"{CORE_RECEIPT_TARGET}","physical_wkv_call_count":{EXPECTED_CALL_COUNT},"session":{{"call_count":{EXPECTED_CALL_COUNT},"call_count":{EXPECTED_CALL_COUNT},"same_layer_state_continuity_count":{EXPECTED_CONTINUITY_COUNT},"terminal_state":"{CLOSED_TERMINAL_STATE}"}},"terminal_state":"{CLOSED_TERMINAL_STATE}"}},"terminal_state":"{CLOSED_TERMINAL_STATE}"}}"#
        );
        let duplicate_error = validate_persistent_receipt(duplicate.as_bytes(), expectations())
            .expect_err("duplicate fields must fail");
        assert!(duplicate_error.contains("duplicate field `call_count`"));
    }

    #[test]
    fn rejects_invalid_expectations() {
        let zero_call_error = validate_persistent_receipt(
            b"{}",
            PersistentReceiptExpectations {
                call_count: 0,
                continuity_count: EXPECTED_CONTINUITY_COUNT,
            },
        )
        .expect_err("zero calls must fail");
        assert!(zero_call_error.contains("call count must be positive"));

        let excessive_continuity_error = validate_persistent_receipt(
            b"{}",
            PersistentReceiptExpectations {
                call_count: EXPECTED_CONTINUITY_COUNT,
                continuity_count: EXPECTED_CALL_COUNT,
            },
        )
        .expect_err("excessive continuity must fail");
        assert!(excessive_continuity_error.contains("exceeds the expected call count"));
    }
}
