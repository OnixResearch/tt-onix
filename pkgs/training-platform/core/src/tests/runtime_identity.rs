//! Synthetic identity fixtures protect receipt admission, not physical readiness.

use super::{valid_authority, valid_receipt};
use crate::{
    HASH_HEX_LEN, RuntimeAuthority, admit_runtime_authority, build_authorized_launch_plan,
    validate_terminal_receipt,
};
use serde_json::{Value, json};

const STALE_DIGEST_DIGIT: &str = "0";
const REPLACEMENT_DIGEST_DIGIT: &str = "c";
const RELOCATED_DESCRIPTOR_PATH: &str = "/nix/store/relocated-descriptor.textproto";
const RELOCATED_WHEELHOUSE_PATH: &str = "/nix/store/relocated-wheelhouse";
const RELOCATED_REQUIREMENTS_PATH: &str = "/nix/store/relocated-wheelhouse/requirements.lock";

fn admit_fixture(authority: &Value) -> RuntimeAuthority {
    let descriptor_hash = authority["descriptor"]["blake3"]
        .as_str()
        .expect("the fixture contains a descriptor hash");
    let requirements_hash = authority["wheelhouse"]["requirements_blake3"]
        .as_str()
        .expect("the fixture contains a requirements hash");
    admit_runtime_authority(authority, descriptor_hash, requirements_hash)
        .expect("the synthetic authority admits with matching supplied hashes")
}

fn assert_changed_authority_rejects_original_receipt(original: &Value, changed: &Value) {
    let original = admit_fixture(original);
    let changed = admit_fixture(changed);
    assert_eq!(
        original.bundle.identity_blake3,
        changed.bundle.identity_blake3
    );
    assert_eq!(
        original.config.identity_blake3,
        changed.config.identity_blake3
    );
    assert_ne!(original.identity_blake3, changed.identity_blake3);

    let original_plan = build_authorized_launch_plan(&original).expect("original plan builds");
    let changed_plan = build_authorized_launch_plan(&changed).expect("changed plan builds");
    assert_ne!(original_plan.identity_blake3, changed_plan.identity_blake3);
    let receipt = valid_receipt(&original.bundle, &original.config, &original_plan);
    assert!(
        validate_terminal_receipt(
            &receipt,
            &original.bundle.identity_blake3,
            &original.config.identity_blake3,
            &original_plan.identity_blake3,
        )
        .expect("the synthetic receipt matches its original authority")
    );
    let rejection = validate_terminal_receipt(
        &receipt,
        &changed.bundle.identity_blake3,
        &changed.config.identity_blake3,
        &changed_plan.identity_blake3,
    )
    .expect_err("an old receipt cannot admit a different launch plan");
    assert_eq!(rejection.code, "stale_identity");
    assert_eq!(
        rejection.detail,
        "receipt launch_plan_identity does not match the current admitted identity"
    );
}

// r[verify tenstorrent.training_platform.runtime_bundle.valid]
// r[verify tenstorrent.training_platform.evidence.valid]
#[test]
fn matching_authority_accepts_a_synthetic_receipt() {
    let authority_json = valid_authority();
    let reformatted = serde_json::to_string_pretty(&authority_json).expect("fixture serializes");
    let parsed: Value = serde_json::from_str(&reformatted).expect("fixture parses again");
    let authority = admit_fixture(&authority_json);
    let repeated = admit_fixture(&parsed);
    assert_eq!(authority.identity_blake3, repeated.identity_blake3);
    let plan = build_authorized_launch_plan(&authority).expect("plan builds");
    let repeated_plan = build_authorized_launch_plan(&repeated).expect("plan rebuilds");
    assert_eq!(plan.identity_blake3, repeated_plan.identity_blake3);
    assert_eq!(plan.plan, repeated_plan.plan);
    let receipt = valid_receipt(&authority.bundle, &authority.config, &plan);
    assert!(
        validate_terminal_receipt(
            &receipt,
            &authority.bundle.identity_blake3,
            &authority.config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect("the synthetic receipt matches the admitted plan")
    );
}

// r[verify tenstorrent.training_platform.evidence.invalid]
#[test]
fn every_receipt_identity_rejects_a_stale_digest() {
    let authority = admit_fixture(&valid_authority());
    let plan = build_authorized_launch_plan(&authority).expect("plan builds");
    for field in ["bundle_identity", "config_identity", "launch_plan_identity"] {
        let mut receipt = valid_receipt(&authority.bundle, &authority.config, &plan);
        receipt[field] = json!(STALE_DIGEST_DIGIT.repeat(HASH_HEX_LEN));
        let rejection = validate_terminal_receipt(
            &receipt,
            &authority.bundle.identity_blake3,
            &authority.config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("each stale identity must fail separately");
        assert_eq!(rejection.code, "stale_identity", "{field}");
        assert_eq!(
            rejection.detail,
            format!("receipt {field} does not match the current admitted identity")
        );
    }
}

// r[verify tenstorrent.training_platform.evidence.invalid]
#[test]
fn path_changes_reject_old_receipts_with_unchanged_content_hashes() {
    let original = valid_authority();
    for (pointer, replacement) in [
        ("/descriptor/path", RELOCATED_DESCRIPTOR_PATH),
        ("/wheelhouse/path", RELOCATED_WHEELHOUSE_PATH),
        ("/wheelhouse/requirements_path", RELOCATED_REQUIREMENTS_PATH),
    ] {
        let mut changed = original.clone();
        *changed
            .pointer_mut(pointer)
            .expect("the fixture path exists") = json!(replacement);
        assert_changed_authority_rejects_original_receipt(&original, &changed);
    }
}

// r[verify tenstorrent.training_platform.evidence.invalid]
#[test]
fn content_hash_changes_reject_old_receipts_with_unchanged_paths() {
    let original = valid_authority();
    for pointer in ["/descriptor/blake3", "/wheelhouse/requirements_blake3"] {
        let mut changed = original.clone();
        *changed
            .pointer_mut(pointer)
            .expect("the fixture hash exists") =
            json!(REPLACEMENT_DIGEST_DIGIT.repeat(HASH_HEX_LEN));
        assert_changed_authority_rejects_original_receipt(&original, &changed);
    }
}
