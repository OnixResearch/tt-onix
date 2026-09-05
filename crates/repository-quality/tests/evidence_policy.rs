use repository_quality::{
    Policy, classify_artifact, contains_private_content, load_policy, validate_policy,
};
use serde_json::{Value, json};

const PORTABLE_EVIDENCE: &str = "portable_evidence";
const SOURCE: &str = "source";
const NON_UTF8_BYTE: u8 = 0xff;

#[test]
fn research_registry_covers_nested_paths_but_not_similar_names() {
    let policy = load_policy().unwrap();
    for path in [
        "docs/research/receipt.json",
        "docs/research/run/receipt.json",
        "./docs/research/run/receipt.json",
        "docs/research/fixtures/receipt.bin",
    ] {
        assert_eq!(
            classify_artifact(&policy, path, 0).unwrap(),
            PORTABLE_EVIDENCE
        );
    }
    for path in [
        "docs/research-more/receipt.json",
        "docs/research.md",
        "docs/receipt.json",
    ] {
        assert_eq!(classify_artifact(&policy, path, 0).unwrap(), SOURCE);
    }
    let source_limit = policy
        .artifact_rules
        .iter()
        .find(|rule| rule.class == SOURCE)
        .unwrap()
        .max_bytes;
    assert_eq!(
        classify_artifact(
            &policy,
            "docs/research/fixtures/receipt.bin",
            source_limit + 1
        )
        .unwrap(),
        PORTABLE_EVIDENCE
    );
}

#[test]
fn registry_rejects_missing_wrong_type_empty_unowned_and_ambiguous_roots() {
    let original = serde_json::to_value(load_policy().unwrap()).unwrap();
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("evidence_roots");
    assert!(serde_json::from_value::<Policy>(missing).is_err());
    for roots in [
        json!(false),
        json!("docs/research"),
        json!([]),
        json!([""]),
        json!(["."]),
        json!(["/docs/research"]),
        json!(["../docs/research"]),
        json!(["docs//research"]),
        json!(["docs/research/"]),
        json!(["docs/./research"]),
        json!(["docs/research/../escape"]),
        json!(["docs\\research"]),
        json!(["docs/research\u{0001}"]),
        json!(["unowned/reports"]),
        json!(["docs/research", "docs/research"]),
        json!(["docs/research", "docs/research/nested"]),
    ] {
        let mut candidate = original.clone();
        candidate["evidence_roots"] = roots.clone();
        let result = serde_json::from_value::<Policy>(candidate)
            .map_err(|error| error.to_string())
            .and_then(|policy| validate_policy(&policy));
        assert!(result.is_err(), "accepted registry {roots}");
    }
}

#[test]
fn registry_cannot_disable_identity_or_permit_private_content() {
    let original = serde_json::to_value(load_policy().unwrap()).unwrap();
    for (field, value) in [
        ("blake3_required", false),
        ("private_content_allowed", true),
    ] {
        let mut candidate = original.clone();
        let rule = candidate["artifact_rules"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|rule| rule["class"] == PORTABLE_EVIDENCE)
            .unwrap();
        rule[field] = Value::Bool(value);
        let policy: Policy = serde_json::from_value(candidate).unwrap();
        assert!(
            validate_policy(&policy).is_err(),
            "accepted unsafe evidence rule {field}"
        );
    }
}

#[test]
fn registry_cannot_overlap_reserved_artifact_classes() {
    for root in [
        ".pi",
        "generated",
        "result",
        "result-cache",
        "quality",
        "quality/generated",
        "model-targets/generated",
    ] {
        let mut policy = load_policy().unwrap();
        policy.evidence_roots = vec![root.to_owned()];
        policy
            .artifact_rules
            .iter_mut()
            .find(|rule| rule.class == PORTABLE_EVIDENCE)
            .unwrap()
            .allowed_roots
            .push(root.to_owned());
        let error = validate_policy(&policy).unwrap_err();
        assert!(error.contains("reserved class"), "{error}");
    }
}

#[test]
fn private_marker_detection_does_not_depend_on_utf8_decoding() {
    let mut bytes = b"credential=fixture-only".to_vec();
    bytes.push(NON_UTF8_BYTE);
    assert!(contains_private_content(&bytes));
    assert!(!contains_private_content(&[NON_UTF8_BYTE]));
    assert!(!contains_private_content(b"portable receipt"));
}
