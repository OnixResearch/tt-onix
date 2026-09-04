use repository_quality::{
    ArtifactManifest, ArtifactManifestEntry, ArtifactObservation,
    cairn_verdict::{admit_validation, parse_validation},
    classify_artifact, contains_private_content, load_policy, manifest_map, validate_artifact,
    validate_manifest,
};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_ARGUMENT_COUNT: usize = 1;
const GENERATE_ARGUMENT_COUNT: usize = 2;
const CHECK_ARGUMENT_COUNT: usize = 3;
const GENERATE_COMMAND: &str = "manifest";
const CHECK_COMMAND: &str = "check";
const CAIRN_COMMAND: &str = "cairn-verdict";
const ARTIFACT_MANIFEST_PATH: &str = "quality/generated/artifact-manifest.json";
const PORTABLE_EVIDENCE_CLASS: &str = "portable_evidence";
const IGNORED_DIRECTORY_NAMES: [&str; 8] = [
    ".git",
    ".jj",
    ".pi",
    ".direnv",
    "target",
    "result",
    "result-1",
    "result-runtime",
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("repository-quality: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args().skip(PROGRAM_ARGUMENT_COUNT).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command, receipt] if command == CAIRN_COMMAND => {
            check_cairn_receipt(Path::new(receipt))?;
            println!("repository quality Cairn verdict: PASS");
            Ok(())
        }
        [command, root] if command == GENERATE_COMMAND => {
            let manifest = build_manifest(Path::new(root))?;
            write_json(&manifest)
        }
        [command, root, manifest] if command == CHECK_COMMAND => {
            check_tree(Path::new(root), Path::new(manifest))?;
            println!("repository quality artifact check: PASS");
            Ok(())
        }
        _ => Err(format!(
            "usage: repository-quality {GENERATE_COMMAND} ROOT | {CHECK_COMMAND} ROOT MANIFEST | {CAIRN_COMMAND} RECEIPT; expected {GENERATE_ARGUMENT_COUNT} or {CHECK_ARGUMENT_COUNT} arguments"
        )),
    }
}

fn check_cairn_receipt(path: &Path) -> Result<(), String> {
    let policy = load_policy()?;
    let limit = policy
        .parser_limits
        .iter()
        .find(|limit| limit.family == "receipt")
        .ok_or("quality policy lacks the receipt limit")?;
    let max_bytes = u64::try_from(limit.max_bytes).map_err(|error| error.to_string())?;
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or("receipt byte limit overflow")?;
    let metadata =
        fs::metadata(path).map_err(|error| format!("failed to inspect receipt: {error}"))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        return Err(
            "Cairn receipt must be a regular file within the receipt byte limit".to_owned(),
        );
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|error| format!("failed to open receipt: {error}"))?
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read receipt: {error}"))?;
    admit_validation(&parse_validation(&policy, &bytes)?)
}

fn check_tree(root: &Path, manifest_path: &Path) -> Result<(), String> {
    let policy = load_policy()?;
    let expected_bytes = fs::read(manifest_path)
        .map_err(|error| format!("failed to read {}: {error}", manifest_path.display()))?;
    let expected: ArtifactManifest = serde_json::from_slice(&expected_bytes)
        .map_err(|error| format!("artifact manifest is invalid JSON: {error}"))?;
    validate_manifest(&expected)?;
    let expected_map = manifest_map(&expected)?;
    let actual = build_manifest(root)?;
    let actual_map = manifest_map(&actual)?;
    if expected_map.len() != actual_map.len() {
        return Err(format!(
            "artifact manifest entry count drift: expected {}, found {}",
            expected_map.len(),
            actual_map.len()
        ));
    }
    for (path, expected_entry) in expected_map {
        let actual_entry = actual_map
            .get(path)
            .ok_or_else(|| format!("artifact manifest lacks {path}"))?;
        if *actual_entry != expected_entry {
            return Err(format!("artifact manifest entry is stale: {path}"));
        }
    }
    for path in collect_files(root)? {
        let bytes = fs::read(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let relative = relative_path(root, &path)?;
        if relative == ARTIFACT_MANIFEST_PATH {
            continue;
        }
        let class = classify_artifact(&policy, &relative, bytes.len() as u64)?;
        let manifest_entry = actual_map.get(relative.as_str());
        validate_artifact(
            &policy,
            &ArtifactObservation {
                path: &relative,
                class,
                byte_count: bytes.len() as u64,
                blake3: manifest_entry.map(|entry| entry.blake3.as_str()),
                contains_private_content: class == PORTABLE_EVIDENCE_CLASS
                    && contains_private_content(&bytes),
                tracked: true,
                fresh: true,
            },
        )?;
    }
    Ok(())
}

fn build_manifest(root: &Path) -> Result<ArtifactManifest, String> {
    let policy = load_policy()?;
    let mut entries = Vec::new();
    for path in collect_files(root)? {
        let bytes = fs::read(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let relative = relative_path(root, &path)?;
        if relative == ARTIFACT_MANIFEST_PATH {
            continue;
        }
        let class = classify_artifact(&policy, &relative, bytes.len() as u64)?;
        let rule = policy
            .artifact_rules
            .iter()
            .find(|rule| rule.class == class)
            .ok_or_else(|| format!("unknown artifact class: {class}"))?;
        if rule.blake3_required {
            entries.push(ArtifactManifestEntry {
                path: relative,
                class: class.to_owned(),
                byte_count: bytes.len() as u64,
                blake3: blake3::hash(&bytes).to_hex().to_string(),
            });
        }
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    let manifest = ArtifactManifest {
        schema_version: policy.schema_version,
        entries,
    };
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let metadata = fs::metadata(root)
        .map_err(|error| format!("failed to inspect {}: {error}", root.display()))?;
    if !metadata.is_dir() {
        return Err(format!("root is not a directory: {}", root.display()));
    }
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("failed to read {}: {error}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "failed to read an entry in {}: {error}",
                    directory.display()
                )
            })?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                if !is_ignored_directory(&path) {
                    pending.push(path);
                }
            } else if file_type.is_file() && path.file_name().is_none_or(|name| name != ".git") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_ignored_directory(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            IGNORED_DIRECTORY_NAMES.contains(&name)
                || name.starts_with("result-")
                || name == "node_modules"
        })
}

fn relative_path(root: &Path, path: &Path) -> Result<String, String> {
    path.strip_prefix(root)
        .map_err(|error| format!("{} is outside {}: {error}", path.display(), root.display()))?
        .to_str()
        .map(|path| path.replace('\\', "/"))
        .ok_or_else(|| format!("path is not UTF-8: {}", path.display()))
}

fn write_json(manifest: &ArtifactManifest) -> Result<(), String> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, manifest)
        .map_err(|error| format!("failed to serialize artifact manifest: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("failed to write artifact manifest: {error}"))
}
