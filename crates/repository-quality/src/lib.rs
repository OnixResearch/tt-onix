mod artifact_registry;
pub mod cairn_verdict;
pub mod json_input;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_JSON: &str = include_str!("../../../quality/generated/repository-quality.json");
const POLICY_SCHEMA_VERSION: u32 = 1;
const BLAKE3_HEX_LENGTH: usize = 64;
const SOURCE_CLASS: &str = "source";
const GENERATED_SOURCE_CLASS: &str = "generated_source";
const PORTABLE_EVIDENCE_CLASS: &str = "portable_evidence";
const PRIVATE_LOG_CLASS: &str = "private_operational_log";
const LARGE_FIXTURE_CLASS: &str = "large_fixture";
const EXTERNAL_OBJECT_CLASS: &str = "external_large_object";
const FIXTURE_SEGMENT: &str = "/fixtures/";
const EVIDENCE_SEGMENT: &str = "/evidence/";
const TOP_LEVEL_GENERATED_ROOT: &str = "generated/";
const PRIVATE_ROOT: &str = ".pi/";
const RESULT_ROOT: &str = "result";
const PRIVATE_CONTENT_MARKERS: [&str; 3] = ["BEGIN PRIVATE KEY", "private_token=", "credential="];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Policy {
    pub schema_version: u32,
    pub parser_limits: Vec<ParserLimit>,
    pub artifact_rules: Vec<ArtifactRule>,
    pub evidence_roots: Vec<String>,
    pub forbidden_device_paths: Vec<String>,
    pub forbidden_device_commands: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParserLimit {
    pub family: String,
    pub max_bytes: usize,
    pub max_items: usize,
    pub max_depth: usize,
    pub max_fields: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactRule {
    pub class: String,
    pub max_bytes: u64,
    pub allowed_roots: Vec<String>,
    pub git_allowed: bool,
    pub blake3_required: bool,
    pub private_content_allowed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParserObservation<'a> {
    pub family: &'a str,
    pub byte_count: usize,
    pub item_count: usize,
    pub depth: usize,
    pub field_count: usize,
    pub malformed: bool,
    pub truncated: bool,
    pub duplicate: bool,
    pub non_finite: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub entries: Vec<ArtifactManifestEntry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactManifestEntry {
    pub path: String,
    pub class: String,
    pub byte_count: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactObservation<'a> {
    pub path: &'a str,
    pub class: &'a str,
    pub byte_count: u64,
    pub blake3: Option<&'a str>,
    pub contains_private_content: bool,
    pub tracked: bool,
    pub fresh: bool,
}

pub fn load_policy() -> Result<Policy, String> {
    let policy: Policy = serde_json::from_str(POLICY_JSON)
        .map_err(|error| format!("repository quality policy is invalid JSON: {error}"))?;
    validate_policy(&policy)?;
    Ok(policy)
}

pub fn validate_policy(policy: &Policy) -> Result<(), String> {
    if policy.schema_version != POLICY_SCHEMA_VERSION {
        return Err(format!(
            "repository quality policy schema must be {POLICY_SCHEMA_VERSION}, found {}",
            policy.schema_version
        ));
    }
    let parser_families = policy
        .parser_limits
        .iter()
        .map(|limit| limit.family.as_str())
        .collect::<BTreeSet<_>>();
    if parser_families.len() != policy.parser_limits.len() {
        return Err("repository quality policy has duplicate parser families".to_owned());
    }
    let artifact_classes = policy
        .artifact_rules
        .iter()
        .map(|rule| rule.class.as_str())
        .collect::<BTreeSet<_>>();
    if artifact_classes.len() != policy.artifact_rules.len() {
        return Err("repository quality policy has duplicate artifact classes".to_owned());
    }
    for required in [
        SOURCE_CLASS,
        GENERATED_SOURCE_CLASS,
        PORTABLE_EVIDENCE_CLASS,
        PRIVATE_LOG_CLASS,
        LARGE_FIXTURE_CLASS,
        EXTERNAL_OBJECT_CLASS,
    ] {
        if !artifact_classes.contains(required) {
            return Err(format!("repository quality policy lacks class {required}"));
        }
    }
    artifact_registry::validate(policy)
}

pub fn validate_parser_observation(
    policy: &Policy,
    observation: &ParserObservation<'_>,
) -> Result<(), String> {
    let limit = policy
        .parser_limits
        .iter()
        .find(|candidate| candidate.family == observation.family)
        .ok_or_else(|| format!("unknown parser family: {}", observation.family))?;
    if observation.byte_count > limit.max_bytes {
        return Err(format!(
            "{} input byte count {} exceeds {}",
            observation.family, observation.byte_count, limit.max_bytes
        ));
    }
    if observation.item_count > limit.max_items {
        return Err(format!(
            "{} item count {} exceeds {}",
            observation.family, observation.item_count, limit.max_items
        ));
    }
    if observation.depth > limit.max_depth {
        return Err(format!(
            "{} depth {} exceeds {}",
            observation.family, observation.depth, limit.max_depth
        ));
    }
    if observation.field_count > limit.max_fields {
        return Err(format!(
            "{} field count {} exceeds {}",
            observation.family, observation.field_count, limit.max_fields
        ));
    }
    for (invalid, diagnostic) in [
        (observation.malformed, "malformed"),
        (observation.truncated, "truncated"),
        (observation.duplicate, "duplicate"),
        (observation.non_finite, "non-finite"),
    ] {
        if invalid {
            return Err(format!("{} input is {diagnostic}", observation.family));
        }
    }
    Ok(())
}

pub fn classify_artifact(
    policy: &Policy,
    path: &str,
    byte_count: u64,
) -> Result<&'static str, String> {
    let normalized = path.trim_start_matches("./");
    if normalized.starts_with(TOP_LEVEL_GENERATED_ROOT) || normalized.starts_with(PRIVATE_ROOT) {
        return Ok(PRIVATE_LOG_CLASS);
    }
    if normalized == RESULT_ROOT || normalized.starts_with("result-") {
        return Ok(EXTERNAL_OBJECT_CLASS);
    }
    if normalized.starts_with("model-targets/generated/")
        || normalized.starts_with("quality/generated/")
    {
        return Ok(GENERATED_SOURCE_CLASS);
    }
    if normalized.contains(EVIDENCE_SEGMENT)
        || path_has_allowed_root(normalized, &policy.evidence_roots)
    {
        return Ok(PORTABLE_EVIDENCE_CLASS);
    }
    if normalized.contains(FIXTURE_SEGMENT) {
        let source_limit = artifact_rule(policy, SOURCE_CLASS)?.max_bytes;
        if byte_count > source_limit {
            return Ok(LARGE_FIXTURE_CLASS);
        }
    }
    Ok(SOURCE_CLASS)
}

pub fn validate_artifact(
    policy: &Policy,
    observation: &ArtifactObservation<'_>,
) -> Result<(), String> {
    let rule = artifact_rule(policy, observation.class)?;
    if observation.byte_count > rule.max_bytes {
        return Err(format!(
            "{} exceeds {} byte limit for {}",
            observation.path, rule.max_bytes, rule.class
        ));
    }
    if observation.tracked && !rule.git_allowed {
        return Err(format!(
            "{} uses class {} which is forbidden in Git",
            observation.path, rule.class
        ));
    }
    if !path_has_allowed_root(observation.path, &rule.allowed_roots) {
        return Err(format!(
            "{} is outside allowed roots for {}",
            observation.path, rule.class
        ));
    }
    if rule.blake3_required && !observation.blake3.is_some_and(valid_blake3) {
        return Err(format!(
            "{} lacks a lowercase BLAKE3 identity",
            observation.path
        ));
    }
    if observation.contains_private_content && !rule.private_content_allowed {
        return Err(format!("{} contains private content", observation.path));
    }
    if !observation.fresh {
        return Err(format!(
            "{} has stale generated or manifest data",
            observation.path
        ));
    }
    Ok(())
}

pub fn validate_manifest(manifest: &ArtifactManifest) -> Result<(), String> {
    if manifest.schema_version != POLICY_SCHEMA_VERSION {
        return Err(format!(
            "artifact manifest schema must be {POLICY_SCHEMA_VERSION}, found {}",
            manifest.schema_version
        ));
    }
    let mut paths = BTreeSet::new();
    for entry in &manifest.entries {
        if !paths.insert(entry.path.as_str()) {
            return Err(format!("artifact manifest duplicates {}", entry.path));
        }
        if !valid_blake3(&entry.blake3) {
            return Err(format!(
                "artifact manifest has invalid BLAKE3 for {}",
                entry.path
            ));
        }
    }
    Ok(())
}

pub fn manifest_map(
    manifest: &ArtifactManifest,
) -> Result<BTreeMap<&str, &ArtifactManifestEntry>, String> {
    validate_manifest(manifest)?;
    Ok(manifest
        .entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry))
        .collect())
}

pub fn validate_device_free_actions(policy: &Policy, actions: &[&str]) -> Result<(), String> {
    for action in actions {
        for forbidden in policy
            .forbidden_device_paths
            .iter()
            .chain(policy.forbidden_device_commands.iter())
        {
            if action.contains(forbidden) {
                return Err(format!(
                    "device-free quality action contains forbidden token {forbidden}: {action}"
                ));
            }
        }
    }
    Ok(())
}

pub fn contains_private_content(bytes: &[u8]) -> bool {
    PRIVATE_CONTENT_MARKERS.iter().any(|marker| {
        bytes
            .windows(marker.len())
            .any(|window| window == marker.as_bytes())
    })
}

pub fn valid_blake3(value: &str) -> bool {
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn artifact_rule<'a>(policy: &'a Policy, class: &str) -> Result<&'a ArtifactRule, String> {
    policy
        .artifact_rules
        .iter()
        .find(|candidate| candidate.class == class)
        .ok_or_else(|| format!("unknown artifact class: {class}"))
}

fn path_has_allowed_root(path: &str, roots: &[String]) -> bool {
    let normalized = path.trim_start_matches("./");
    roots
        .iter()
        .any(|root| root == "." || root_contains(root, normalized))
}

fn root_contains(root: &str, path: &str) -> bool {
    path == root
        || path
            .strip_prefix(root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE_BYTES: u64 = 128;
    const OVERSIZED_BYTES: usize = usize::MAX;

    fn valid_parser_observation(family: &str) -> ParserObservation<'_> {
        ParserObservation {
            family,
            byte_count: 2,
            item_count: 1,
            depth: 1,
            field_count: 1,
            malformed: false,
            truncated: false,
            duplicate: false,
            non_finite: false,
        }
    }

    #[test]
    fn policy_and_all_parser_families_accept_bounded_input() {
        let policy = load_policy().expect("checked policy must load");
        for family in [
            "json",
            "yaml",
            "csv",
            "binary_frame",
            "vocabulary",
            "receipt",
        ] {
            validate_parser_observation(&policy, &valid_parser_observation(family))
                .expect("bounded parser observation must pass");
        }
    }

    #[test]
    fn parser_limits_accept_the_boundary_and_reject_each_excess() {
        let policy = load_policy().expect("checked policy must load");
        for limit in &policy.parser_limits {
            let boundary = ParserObservation {
                family: &limit.family,
                byte_count: limit.max_bytes,
                item_count: limit.max_items,
                depth: limit.max_depth,
                field_count: limit.max_fields,
                malformed: false,
                truncated: false,
                duplicate: false,
                non_finite: false,
            };
            validate_parser_observation(&policy, &boundary)
                .expect("an observation at every named limit must pass");

            for (name, error) in [
                (
                    "bytes",
                    format!(
                        "{} input byte count {} exceeds {}",
                        limit.family,
                        limit.max_bytes + 1,
                        limit.max_bytes
                    ),
                ),
                (
                    "items",
                    format!(
                        "{} item count {} exceeds {}",
                        limit.family,
                        limit.max_items + 1,
                        limit.max_items
                    ),
                ),
                (
                    "depth",
                    format!(
                        "{} depth {} exceeds {}",
                        limit.family,
                        limit.max_depth + 1,
                        limit.max_depth
                    ),
                ),
                (
                    "fields",
                    format!(
                        "{} field count {} exceeds {}",
                        limit.family,
                        limit.max_fields + 1,
                        limit.max_fields
                    ),
                ),
            ] {
                let mut excessive = boundary.clone();
                match name {
                    "bytes" => excessive.byte_count += 1,
                    "items" => excessive.item_count += 1,
                    "depth" => excessive.depth += 1,
                    "fields" => excessive.field_count += 1,
                    _ => unreachable!("parser limit dimensions are closed"),
                }
                assert_eq!(
                    validate_parser_observation(&policy, &excessive)
                        .expect_err("an observation above a named limit must fail"),
                    error
                );
            }
        }
    }

    #[test]
    fn parser_families_reject_every_unsafe_shape_with_stable_diagnostics() {
        let policy = load_policy().expect("checked policy must load");
        let mut oversized = valid_parser_observation("json");
        oversized.byte_count = OVERSIZED_BYTES;
        assert_eq!(
            validate_parser_observation(&policy, &oversized).expect_err("oversized JSON must fail"),
            format!(
                "json input byte count {OVERSIZED_BYTES} exceeds {}",
                policy
                    .parser_limits
                    .iter()
                    .find(|limit| limit.family == "json")
                    .expect("JSON limit must exist")
                    .max_bytes
            )
        );
        for mutation in ["malformed", "truncated", "duplicate", "non-finite"] {
            let mut observation = valid_parser_observation("receipt");
            match mutation {
                "malformed" => observation.malformed = true,
                "truncated" => observation.truncated = true,
                "duplicate" => observation.duplicate = true,
                "non-finite" => observation.non_finite = true,
                _ => unreachable!("fixture mutation is closed"),
            }
            assert_eq!(
                validate_parser_observation(&policy, &observation)
                    .expect_err("unsafe parser observation must fail"),
                format!("receipt input is {mutation}")
            );
        }
    }

    #[test]
    fn every_artifact_class_has_an_approved_case() {
        let policy = load_policy().expect("checked policy must load");
        let source_limit = artifact_rule(&policy, SOURCE_CLASS)
            .expect("source rule must exist")
            .max_bytes;
        let fixtures = [
            ("README.md", SOURCE_BYTES, SOURCE_CLASS, None, true, false),
            (
                "quality/generated/policy.json",
                SOURCE_BYTES,
                GENERATED_SOURCE_CLASS,
                Some(VALID_HASH),
                true,
                false,
            ),
            (
                "pkgs/demo/evidence/receipt.json",
                SOURCE_BYTES,
                PORTABLE_EVIDENCE_CLASS,
                Some(VALID_HASH),
                true,
                false,
            ),
            (
                "generated/inspector.log",
                SOURCE_BYTES,
                PRIVATE_LOG_CLASS,
                None,
                false,
                true,
            ),
            (
                "pkgs/demo/fixtures/model.bin",
                source_limit + 1,
                LARGE_FIXTURE_CLASS,
                Some(VALID_HASH),
                true,
                false,
            ),
            (
                "result",
                SOURCE_BYTES,
                EXTERNAL_OBJECT_CLASS,
                Some(VALID_HASH),
                false,
                false,
            ),
        ];
        for (path, byte_count, expected_class, blake3, tracked, private) in fixtures {
            let class = classify_artifact(&policy, path, byte_count)
                .expect("approved artifact must classify");
            assert_eq!(class, expected_class);
            validate_artifact(
                &policy,
                &ArtifactObservation {
                    path,
                    class,
                    byte_count,
                    blake3,
                    contains_private_content: private,
                    tracked,
                    fresh: true,
                },
            )
            .expect("approved artifact must pass");
        }
    }

    #[test]
    fn portable_evidence_accepts_native_cairn_and_rejects_unowned_roots() {
        let policy = load_policy().expect("checked policy must load");
        let cairn = ArtifactObservation {
            path: ".cairn/changes/demo/evidence/receipt.json",
            class: PORTABLE_EVIDENCE_CLASS,
            byte_count: SOURCE_BYTES,
            blake3: Some(VALID_HASH),
            contains_private_content: false,
            tracked: true,
            fresh: true,
        };
        validate_artifact(&policy, &cairn).expect("Cairn evidence must pass");

        for path in [
            "docs/evidence/receipt.json",
            "cairn/changes/demo/evidence/receipt.json",
        ] {
            let unowned = ArtifactObservation {
                path,
                ..cairn.clone()
            };
            assert_eq!(
                validate_artifact(&policy, &unowned).expect_err("unowned evidence must fail"),
                format!("{path} is outside allowed roots for portable_evidence")
            );
        }
    }

    #[test]
    fn artifact_validation_rejects_private_stale_and_unidentified_content() {
        let policy = load_policy().expect("checked policy must load");
        let private = ArtifactObservation {
            path: "pkgs/demo/evidence/receipt.json",
            class: PORTABLE_EVIDENCE_CLASS,
            byte_count: SOURCE_BYTES,
            blake3: Some(VALID_HASH),
            contains_private_content: true,
            tracked: true,
            fresh: true,
        };
        assert_eq!(
            validate_artifact(&policy, &private).expect_err("private evidence must fail"),
            "pkgs/demo/evidence/receipt.json contains private content"
        );
        let stale = ArtifactObservation {
            contains_private_content: false,
            fresh: false,
            ..private.clone()
        };
        assert_eq!(
            validate_artifact(&policy, &stale).expect_err("stale evidence must fail"),
            "pkgs/demo/evidence/receipt.json has stale generated or manifest data"
        );
        let unidentified = ArtifactObservation {
            blake3: None,
            fresh: true,
            ..stale
        };
        assert_eq!(
            validate_artifact(&policy, &unidentified).expect_err("unidentified evidence must fail"),
            "pkgs/demo/evidence/receipt.json lacks a lowercase BLAKE3 identity"
        );
        let tracked_private_log = ArtifactObservation {
            path: "generated/inspector.log",
            class: PRIVATE_LOG_CLASS,
            byte_count: SOURCE_BYTES,
            blake3: None,
            contains_private_content: true,
            tracked: true,
            fresh: true,
        };
        assert_eq!(
            validate_artifact(&policy, &tracked_private_log)
                .expect_err("tracked private logs must fail"),
            "generated/inspector.log uses class private_operational_log which is forbidden in Git"
        );
    }

    #[test]
    fn manifest_rejects_duplicate_paths_and_uppercase_hashes() {
        let duplicate = ArtifactManifest {
            schema_version: POLICY_SCHEMA_VERSION,
            entries: vec![
                ArtifactManifestEntry {
                    path: "quality/generated/policy.json".to_owned(),
                    class: GENERATED_SOURCE_CLASS.to_owned(),
                    byte_count: SOURCE_BYTES,
                    blake3: VALID_HASH.to_owned(),
                },
                ArtifactManifestEntry {
                    path: "quality/generated/policy.json".to_owned(),
                    class: GENERATED_SOURCE_CLASS.to_owned(),
                    byte_count: SOURCE_BYTES,
                    blake3: VALID_HASH.to_owned(),
                },
            ],
        };
        assert!(validate_manifest(&duplicate).is_err());
        let uppercase = ArtifactManifest {
            schema_version: POLICY_SCHEMA_VERSION,
            entries: vec![ArtifactManifestEntry {
                path: "quality/generated/policy.json".to_owned(),
                class: GENERATED_SOURCE_CLASS.to_owned(),
                byte_count: SOURCE_BYTES,
                blake3: VALID_HASH.to_uppercase(),
            }],
        };
        assert!(validate_manifest(&uppercase).is_err());
    }

    #[test]
    fn device_free_actions_accept_lints_and_reject_hardware_access() {
        let policy = load_policy().expect("checked policy must load");
        validate_device_free_actions(&policy, &["cargo test", "ruff check ."])
            .expect("device-free actions must pass");
        assert_eq!(
            validate_device_free_actions(&policy, &["open /dev/tenstorrent/0"])
                .expect_err("device access must fail"),
            "device-free quality action contains forbidden token /dev/tenstorrent: open /dev/tenstorrent/0"
        );
        assert_eq!(
            validate_device_free_actions(&policy, &["tt-flash --force"])
                .expect_err("firmware mutation must fail"),
            "device-free quality action contains forbidden token tt-flash: tt-flash --force"
        );
    }

    #[test]
    fn private_marker_detection_is_bounded_and_exact() {
        assert!(!contains_private_content(b"portable aggregate receipt"));
        assert!(contains_private_content(b"credential=secret"));
    }
}
