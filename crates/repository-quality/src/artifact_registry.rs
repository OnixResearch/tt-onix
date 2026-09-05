//! Repository-owned evidence scope policy, not native filesystem path admission.

use crate::{PORTABLE_EVIDENCE_CLASS, Policy, artifact_rule, path_has_allowed_root, root_contains};

const RESERVED_ROOTS: &[&str] = &[
    ".pi",
    "generated",
    "result",
    "model-targets/generated",
    "quality/generated",
];

pub(crate) fn validate(policy: &Policy) -> Result<(), String> {
    let rule = artifact_rule(policy, PORTABLE_EVIDENCE_CLASS)?;
    if !rule.blake3_required || rule.private_content_allowed {
        return Err("registered evidence requires BLAKE3 and forbids private content".to_owned());
    }
    if policy.evidence_roots.is_empty() {
        return Err("evidence registry must not be empty".to_owned());
    }
    for (index, root) in policy.evidence_roots.iter().enumerate() {
        if !canonical_root(root) {
            return Err(format!("evidence registry root is not canonical: {root:?}"));
        }
        if RESERVED_ROOTS
            .iter()
            .any(|reserved| root_contains(reserved, root) || root_contains(root, reserved))
            || root.starts_with("result-")
        {
            return Err(format!(
                "evidence registry root overlaps a reserved class: {root}"
            ));
        }
        if !path_has_allowed_root(root, &rule.allowed_roots) {
            return Err(format!(
                "evidence registry root is not owned by portable_evidence: {root}"
            ));
        }
        for previous in &policy.evidence_roots[..index] {
            if root_contains(previous, root) || root_contains(root, previous) {
                return Err(format!(
                    "evidence registry roots overlap: {previous} and {root}"
                ));
            }
        }
    }
    Ok(())
}

fn canonical_root(root: &str) -> bool {
    !root.contains('\\')
        && !root.contains(':')
        && !root.chars().any(char::is_control)
        && root.split('/').all(|part| !matches!(part, "" | "." | ".."))
}
