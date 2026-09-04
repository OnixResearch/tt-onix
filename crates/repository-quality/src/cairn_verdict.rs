//! Admission for the pinned native `cairn validate` protocol, not lifecycle gates.
use crate::{Policy, json_input::parse_json};
use serde_json::{Map, Value};

const ISSUE_FIELDS: &[&str] = &[
    "issues",
    "findings",
    "change_issues",
    "cross_repo_evidence_issues",
    "spec_issues",
    "spec_findings",
    "substance_issues",
    "substance_findings",
];
const INVENTORY_FIELDS: &[&str] = &["cross_repo_dependencies", "spec_substance", "substance"];
const OTHER_FIELDS: &[&str] = &[
    "valid",
    "resolution",
    "changes",
    "specs_validated",
    "policy",
    "layout",
    "profile_issues",
    "workflow_profiles",
];

#[derive(Debug, Eq, PartialEq)]
pub struct ValidationOutcome {
    pub valid: bool,
    pub resolution_valid: bool,
    pub issue_counts: Vec<(String, usize)>,
}

// r[impl tenstorrent.quality.aggregate_gate]
pub fn admit_validation(outcome: &ValidationOutcome) -> Result<(), String> {
    if !outcome.valid {
        return Err("Cairn top-level valid must be true".to_owned());
    }
    if !outcome.resolution_valid {
        return Err("Cairn resolution.valid must be true".to_owned());
    }
    for (field, count) in &outcome.issue_counts {
        if *count != 0 {
            return Err(format!("Cairn {field} contains {count} blocking entries"));
        }
    }
    Ok(())
}

/// Decode all verdict-bearing fields. Inventories remain metadata, not proof.
pub fn parse_validation(policy: &Policy, bytes: &[u8]) -> Result<ValidationOutcome, String> {
    let limit = policy
        .parser_limits
        .iter()
        .find(|limit| limit.family == "receipt")
        .ok_or("quality policy lacks the receipt limit")?;
    let value = parse_json(bytes, limit)?;
    let object = value.as_object().ok_or("Cairn receipt must be an object")?;
    for key in object.keys() {
        if !ISSUE_FIELDS.contains(&key.as_str())
            && !INVENTORY_FIELDS.contains(&key.as_str())
            && !OTHER_FIELDS.contains(&key.as_str())
        {
            return Err(format!("unsupported Cairn receipt field: {key}"));
        }
    }
    let valid = boolean(object, "valid")?;
    let resolution = object
        .get("resolution")
        .and_then(Value::as_object)
        .ok_or("Cairn resolution must be an object")?;
    let resolution_valid = boolean(resolution, "valid")?;
    let mut issue_counts = vec![(
        "resolution.diagnostics".to_owned(),
        array_len(resolution, "diagnostics")?,
    )];
    for field in ISSUE_FIELDS {
        issue_counts.push(((*field).to_owned(), array_len(object, field)?));
    }
    for field in INVENTORY_FIELDS {
        array_len(object, field)?;
    }
    if object.contains_key("profile_issues") || object.contains_key("workflow_profiles") {
        issue_counts.push((
            "profile_issues".to_owned(),
            array_len(object, "profile_issues")?,
        ));
        array_len(object, "workflow_profiles")?;
    }
    for field in ["changes", "specs_validated"] {
        object
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("Cairn {field} must be a nonnegative integer"))?;
    }
    if object.get("layout").and_then(Value::as_str) != Some("cairn") {
        return Err("Cairn layout must be cairn".to_owned());
    }
    if object
        .get("policy")
        .and_then(Value::as_str)
        .is_none_or(|policy| policy.is_empty())
    {
        return Err("Cairn policy must be a nonempty string".to_owned());
    }
    Ok(ValidationOutcome {
        valid,
        resolution_valid,
        issue_counts,
    })
}

fn boolean(object: &Map<String, Value>, field: &str) -> Result<bool, String> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("Cairn {field} must be a boolean"))
}

fn array_len(object: &Map<String, Value>, field: &str) -> Result<usize, String> {
    object
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| format!("Cairn {field} must be an array"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_policy;
    use serde_json::json;

    const VALID: &[u8] = include_bytes!("../../../quality/fixtures/cairn/valid.json");

    fn admit_bytes(bytes: &[u8]) -> Result<(), String> {
        admit_validation(&parse_validation(&load_policy()?, bytes)?)
    }

    fn admit_value(value: &Value) -> Result<(), String> {
        admit_bytes(&serde_json::to_vec(value).unwrap())
    }

    #[test]
    fn admits_native_validation_with_empty_and_nonempty_inventories() {
        let mut value: Value = serde_json::from_slice(VALID).unwrap();
        assert!(admit_value(&value).is_ok());
        value["substance"] = json!([{"path":".cairn/specs/example/spec.md"}]);
        assert!(admit_value(&value).is_ok());
    }

    #[test]
    fn rejects_nested_success_and_contradictory_findings() {
        for seed in [
            include_bytes!("../../../quality/fixtures/cairn/nested-success.json").as_slice(),
            include_bytes!("../../../quality/fixtures/cairn/duplicate-verdict.json").as_slice(),
        ] {
            assert!(admit_bytes(seed).is_err());
        }
        for field in ISSUE_FIELDS {
            let mut value: Value = serde_json::from_slice(VALID).unwrap();
            value[*field] = json!(["blocking fixture"]);
            assert!(admit_value(&value).unwrap_err().contains(field));
        }
        let mut value: Value = serde_json::from_slice(VALID).unwrap();
        value["resolution"]["valid"] = json!(false);
        assert!(
            admit_value(&value)
                .unwrap_err()
                .contains("resolution.valid")
        );
        value["resolution"]["valid"] = json!(true);
        value["resolution"]["diagnostics"] = json!(["invalid root"]);
        assert!(admit_value(&value).is_err());
    }

    #[test]
    fn rejects_incomplete_malformed_and_unknown_verdicts() {
        let valid: Value = serde_json::from_slice(VALID).unwrap();
        for field in valid.as_object().unwrap().keys() {
            let mut value = valid.clone();
            value.as_object_mut().unwrap().remove(field);
            assert!(admit_value(&value).is_err(), "accepted missing {field}");
            value[field] = Value::Null;
            assert!(admit_value(&value).is_err(), "accepted null {field}");
        }
        for wrong_type in [json!("true"), json!(1)] {
            let mut value = valid.clone();
            value["valid"] = wrong_type;
            assert!(admit_value(&value).is_err());
        }
        let mut value = valid;
        value["blocked"] = json!(true);
        assert!(admit_value(&value).is_err());
        for bytes in [b"".as_slice(), b"[]", b"true", br#"{"valid":true}"#, b"{"] {
            assert!(admit_bytes(bytes).is_err());
        }
    }
}
