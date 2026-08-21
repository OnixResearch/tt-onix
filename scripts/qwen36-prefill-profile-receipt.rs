#!/usr/bin/env -S RUSTC_BOOTSTRAP=1 nix shell "github:nix-community/fenix?rev=a0b1aef63d7e9c5568958e0d90f6fe48f0f4a7a2#stable.toolchain" nixpkgs#gcc -c cargo -q -Zscript

---
[package]
edition = "2024"

[dependencies]
blake3 = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
---

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_NAME: &str = "qwen36-prefill-profile-receipt.rs";
const EAGER_ROOT_FLAG: &str = "--eager-response-root";
const CANDIDATE_ROOT_FLAG: &str = "--candidate-response-root";
const EXECUTION_ORDER_FLAG: &str = "--execution-order";
const OUTPUT_FLAG: &str = "--output";
const SELF_TEST_FLAG: &str = "--self-test";
const MEASURED_MARKER: &str = "-measured-";
const JSON_EXTENSION: &str = "json";
const DIAGNOSTIC_MEASUREMENT_MODE: &str = "diagnostic-prefill-profile";
const RECEIPT_SCHEMA: &str = "qwen36.prefill-profile-attribution.v1";
const PROFILE_SCHEMA_VERSION: usize = 1;
const EXPECTED_ROOT_COUNT: usize = 2;
const EXPECTED_DEVICE_COUNT: usize = 2;
const EXPECTED_DEVICE_IDS: [i64; EXPECTED_DEVICE_COUNT] = [0, 1];
const FLAG_VALUE_ARGUMENT_WIDTH: usize = 2;
const MEDIAN_DIVISOR: usize = 2;
const MEDIAN_DIVISOR_FLOAT: f64 = 2.0;
const SUCCESS_EXIT_CODE: u8 = 0;
const FAILURE_EXIT_CODE: u8 = 1;
const SELF_TEST_EAGER_DURATION_NS: f64 = 100.0;
const SELF_TEST_CANDIDATE_DURATION_NS: f64 = 120.0;
const SELF_TEST_EXPECTED_DELTA_NS: f64 =
    SELF_TEST_CANDIDATE_DURATION_NS - SELF_TEST_EAGER_DURATION_NS;

#[derive(Debug)]
struct Config {
    eager_roots: Vec<PathBuf>,
    candidate_roots: Vec<PathBuf>,
    execution_order: Vec<String>,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    id: String,
    x_onix_decode_evidence: DecodeEvidence,
}

#[derive(Clone, Debug, Deserialize)]
struct DecodeEvidence {
    admission_eligible: bool,
    measurement_mode: String,
    model_identity: String,
    package_identity: String,
    physical_pcie_device_ids: Vec<i64>,
    prefill_page_table_blocks: usize,
    prefill_program_profile: ProgramProfile,
    profile_id: String,
    runtime_identity: String,
    simulated: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ProgramProfile {
    schema_version: usize,
    record_count: usize,
    runtime_ids: Vec<u64>,
    by_chip: Vec<ChipTotal>,
    signatures: Vec<SignatureTotal>,
}

#[derive(Clone, Debug, Deserialize)]
struct ChipTotal {
    chip_id: i64,
    record_count: usize,
    total_duration_ns: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct SignatureTotal {
    kernel_sources: Vec<String>,
    maximum_chip_total_duration_ns: f64,
    record_count: usize,
    by_chip: Vec<SignatureChipTotal>,
}

#[derive(Clone, Debug, Deserialize)]
struct SignatureChipTotal {
    chip_id: i64,
    maximum_duration_ns: f64,
    record_count: usize,
    total_duration_ns: f64,
}

#[derive(Debug)]
struct Sample {
    name: String,
    response_blake3: String,
    response: CompletionResponse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    model: String,
    package: String,
    profile: String,
    runtime: String,
    physical_pcie_device_ids: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SignatureKey {
    kernel_sources: Vec<String>,
    chip_id: i64,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::from(SUCCESS_EXIT_CODE),
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(FAILURE_EXIT_CODE)
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments == [SELF_TEST_FLAG] {
        self_test()?;
        println!("{PROGRAM_NAME} self-test: PASS");
        return Ok(());
    }
    let config = parse_config(&arguments)?;
    let receipt = build_receipt(&config)?;
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize attribution receipt: {error}"))?;
    fs::write(&config.output, bytes)
        .map_err(|error| format!("write {}: {error}", config.output.display()))?;
    println!("{}", config.output.display());
    Ok(())
}

fn parse_config(arguments: &[String]) -> Result<Config, String> {
    let mut eager_roots = Vec::new();
    let mut candidate_roots = Vec::new();
    let mut execution_order = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            EAGER_ROOT_FLAG => eager_roots.push(PathBuf::from(value)),
            CANDIDATE_ROOT_FLAG => candidate_roots.push(PathBuf::from(value)),
            EXECUTION_ORDER_FLAG => {
                execution_order = Some(value.split(',').map(str::to_owned).collect())
            }
            OUTPUT_FLAG => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
        index += FLAG_VALUE_ARGUMENT_WIDTH;
    }
    Ok(Config {
        eager_roots,
        candidate_roots,
        execution_order: execution_order
            .ok_or_else(|| format!("{EXECUTION_ORDER_FLAG} is required"))?,
        output: output.ok_or_else(|| format!("{OUTPUT_FLAG} is required"))?,
    })
}

fn build_receipt(config: &Config) -> Result<Value, String> {
    validate_config(config)?;
    let eager_roots = load_roots(&config.eager_roots)?;
    let candidate_roots = load_roots(&config.candidate_roots)?;
    let sample_names = validate_root_samples(&eager_roots, "eager")?;
    let candidate_names = validate_root_samples(&candidate_roots, "candidate")?;
    if sample_names != candidate_names {
        return Err("eager and candidate roots contain different measured samples".to_owned());
    }

    let mut canonical_identity = None;
    let mut input_hashes = Vec::new();
    for (path_name, roots) in [("eager", &eager_roots), ("candidate", &candidate_roots)] {
        for (root_index, root) in roots.iter().enumerate() {
            for sample in root {
                validate_sample(sample)?;
                let identity = identity(&sample.response.x_onix_decode_evidence);
                if let Some(expected) = &canonical_identity {
                    if identity != *expected {
                        return Err(format!("{} has identity drift", sample.name));
                    }
                } else {
                    canonical_identity = Some(identity);
                }
                input_hashes.push(json!({
                    "path": path_name,
                    "root_index": root_index,
                    "sample": sample.name,
                    "blake3": sample.response_blake3,
                }));
            }
        }
    }
    validate_duplicate_coverage(&eager_roots, "eager")?;
    validate_duplicate_coverage(&candidate_roots, "candidate")?;

    let eager_values = aggregate_signature_values(&eager_roots)?;
    let candidate_values = aggregate_signature_values(&candidate_roots)?;
    let expected_value_count = sample_names.len() * EXPECTED_ROOT_COUNT;
    let complete_eager = complete_keys(&eager_values, expected_value_count);
    let complete_candidate = complete_keys(&candidate_values, expected_value_count);
    let common_keys = complete_eager
        .intersection(&complete_candidate)
        .cloned()
        .collect::<BTreeSet<_>>();
    if common_keys.is_empty() {
        return Err("profile roots have no complete common program signatures".to_owned());
    }

    let mut differences = common_keys
        .iter()
        .map(|key| {
            let eager_median = median(
                eager_values
                    .get(key)
                    .ok_or_else(|| "missing eager signature values".to_owned())?,
            )?;
            let candidate_median = median(
                candidate_values
                    .get(key)
                    .ok_or_else(|| "missing candidate signature values".to_owned())?,
            )?;
            let delta = candidate_median - eager_median;
            let ratio = candidate_median / eager_median;
            Ok((key.clone(), eager_median, candidate_median, delta, ratio))
        })
        .collect::<Result<Vec<_>, String>>()?;
    differences.sort_by(|left, right| {
        right
            .3
            .total_cmp(&left.3)
            .then_with(|| left.0.cmp(&right.0))
    });
    let strongest = differences
        .first()
        .ok_or_else(|| "no program difference survived validation".to_owned())?;
    let identity = canonical_identity.ok_or_else(|| "profile roots are empty".to_owned())?;

    Ok(json!({
        "schema": RECEIPT_SCHEMA,
        "admission_eligible": false,
        "execution": {
            "execution_order": config.execution_order,
            "measured_samples_per_root": sample_names.len(),
            "response_roots_per_path": EXPECTED_ROOT_COUNT,
            "simulated": false,
        },
        "identity": {
            "model_identity": identity.model,
            "package_identity": identity.package,
            "physical_pcie_device_ids": identity.physical_pcie_device_ids,
            "profile_id": identity.profile,
            "runtime_identity": identity.runtime,
        },
        "coverage": {
            "candidate_complete_signature_chip_count": complete_candidate.len(),
            "candidate_only_signature_chip_count": complete_candidate.difference(&complete_eager).count(),
            "common_signature_chip_count": common_keys.len(),
            "eager_complete_signature_chip_count": complete_eager.len(),
            "eager_only_signature_chip_count": complete_eager.difference(&complete_candidate).count(),
        },
        "strongest_checked_difference": {
            "candidate_median_total_duration_ns": strongest.2,
            "candidate_to_eager_ratio": strongest.4,
            "chip_id": strongest.0.chip_id,
            "delta_duration_ns": strongest.3,
            "eager_median_total_duration_ns": strongest.1,
            "kernel_sources": strongest.0.kernel_sources,
        },
        "input_hashes": input_hashes,
        "non_claims": [
            "profiled responses are not TTFT admission evidence",
            "program-duration sums are not wall-clock TTFT",
            "the strongest signature difference is not an admitted optimization",
        ],
    }))
}

fn validate_config(config: &Config) -> Result<(), String> {
    if config.eager_roots.len() != EXPECTED_ROOT_COUNT
        || config.candidate_roots.len() != EXPECTED_ROOT_COUNT
    {
        return Err(format!(
            "exactly {EXPECTED_ROOT_COUNT} eager and candidate roots are required"
        ));
    }
    let expected_order = ["eager", "candidate", "candidate", "eager"];
    if config.execution_order != expected_order {
        return Err("execution order must be eager,candidate,candidate,eager".to_owned());
    }
    Ok(())
}

fn load_roots(paths: &[PathBuf]) -> Result<Vec<Vec<Sample>>, String> {
    paths.iter().map(|path| load_root(path)).collect()
}

fn load_root(root: &Path) -> Result<Vec<Sample>, String> {
    let entries = fs::read_dir(root)
        .map_err(|error| format!("read response root {}: {error}", root.display()))?;
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().and_then(|value| value.to_str()) == Some(JSON_EXTENSION)
                && path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.contains(MEASURED_MARKER))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "response root {} has no measured JSON files",
            root.display()
        ));
    }
    paths
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read response {}: {error}", path.display()))?;
            let response = serde_json::from_slice(&bytes)
                .map_err(|error| format!("parse response {}: {error}", path.display()))?;
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| format!("response name is not UTF-8: {}", path.display()))?
                .to_owned();
            Ok(Sample {
                name,
                response_blake3: blake3::hash(&bytes).to_hex().to_string(),
                response,
            })
        })
        .collect()
}

fn validate_root_samples(roots: &[Vec<Sample>], label: &str) -> Result<Vec<String>, String> {
    let first = roots
        .first()
        .ok_or_else(|| format!("{label} roots are empty"))?;
    let expected = first
        .iter()
        .map(|sample| sample.name.clone())
        .collect::<Vec<_>>();
    for root in roots.iter().skip(1) {
        let observed = root
            .iter()
            .map(|sample| sample.name.clone())
            .collect::<Vec<_>>();
        if observed != expected {
            return Err(format!("{label} roots contain different measured samples"));
        }
    }
    Ok(expected)
}

fn validate_sample(sample: &Sample) -> Result<(), String> {
    if sample.response.id.is_empty() {
        return Err(format!("{} has an empty response identity", sample.name));
    }
    let evidence = &sample.response.x_onix_decode_evidence;
    if evidence.admission_eligible || evidence.measurement_mode != DIAGNOSTIC_MEASUREMENT_MODE {
        return Err(format!("{} is not diagnostic profile evidence", sample.name));
    }
    if evidence.simulated || evidence.physical_pcie_device_ids != EXPECTED_DEVICE_IDS {
        return Err(format!("{} is not physical P150x2 evidence", sample.name));
    }
    if evidence.model_identity.is_empty()
        || evidence.package_identity.is_empty()
        || evidence.profile_id.is_empty()
        || evidence.runtime_identity.is_empty()
        || evidence.prefill_page_table_blocks == 0
    {
        return Err(format!("{} has incomplete identity or shape evidence", sample.name));
    }
    validate_program_profile(&evidence.prefill_program_profile, &sample.name)
}

fn validate_program_profile(profile: &ProgramProfile, sample_name: &str) -> Result<(), String> {
    if profile.schema_version != PROFILE_SCHEMA_VERSION
        || profile.record_count == 0
        || profile.runtime_ids.is_empty()
        || profile.signatures.is_empty()
    {
        return Err(format!("{sample_name} has an empty or unsupported program profile"));
    }
    if profile.runtime_ids.iter().any(|runtime_id| *runtime_id == 0) {
        return Err(format!("{sample_name} has an invalid runtime ID"));
    }
    validate_chip_rows(&profile.by_chip, profile.record_count, sample_name)?;
    let mut previous_total = f64::INFINITY;
    let mut previous_sources = Vec::new();
    let mut signature_record_count = 0;
    for signature in &profile.signatures {
        if signature.kernel_sources.is_empty()
            || signature
                .kernel_sources
                .iter()
                .any(|source| source.is_empty())
        {
            return Err(format!("{sample_name} has an empty kernel-source signature"));
        }
        let sources = signature
            .kernel_sources
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if sources.len() != signature.kernel_sources.len()
            || signature.kernel_sources != sources.into_iter().collect::<Vec<_>>()
        {
            return Err(format!("{sample_name} has a non-canonical kernel signature"));
        }
        if signature.record_count == 0
            || !finite_positive(signature.maximum_chip_total_duration_ns)
        {
            return Err(format!("{sample_name} has an invalid signature total"));
        }
        validate_signature_chip_rows(signature, sample_name)?;
        if signature.maximum_chip_total_duration_ns > previous_total
            || (signature.maximum_chip_total_duration_ns == previous_total
                && signature.kernel_sources < previous_sources)
        {
            return Err(format!("{sample_name} signatures are not deterministically sorted"));
        }
        previous_total = signature.maximum_chip_total_duration_ns;
        previous_sources = signature.kernel_sources.clone();
        signature_record_count += signature.record_count;
    }
    if signature_record_count != profile.record_count {
        return Err(format!("{sample_name} signature counts do not cover all records"));
    }
    Ok(())
}

fn validate_chip_rows(
    rows: &[ChipTotal],
    expected_record_count: usize,
    sample_name: &str,
) -> Result<(), String> {
    if rows.len() != EXPECTED_DEVICE_COUNT {
        return Err(format!("{sample_name} has the wrong chip summary count"));
    }
    let mut record_count = 0;
    for (index, row) in rows.iter().enumerate() {
        if row.chip_id != EXPECTED_DEVICE_IDS[index]
            || row.record_count == 0
            || !finite_positive(row.total_duration_ns)
        {
            return Err(format!("{sample_name} has an invalid chip summary"));
        }
        record_count += row.record_count;
    }
    if record_count != expected_record_count {
        return Err(format!("{sample_name} chip counts do not cover all records"));
    }
    Ok(())
}

fn validate_signature_chip_rows(
    signature: &SignatureTotal,
    sample_name: &str,
) -> Result<(), String> {
    if signature.by_chip.is_empty() || signature.by_chip.len() > EXPECTED_DEVICE_COUNT {
        return Err(format!("{sample_name} has invalid signature chip coverage"));
    }
    let mut previous_chip = None;
    let mut record_count = 0;
    let mut maximum_chip_total = 0.0_f64;
    for row in &signature.by_chip {
        if !EXPECTED_DEVICE_IDS.contains(&row.chip_id)
            || previous_chip.is_some_and(|chip| row.chip_id <= chip)
            || row.record_count == 0
            || !finite_positive(row.total_duration_ns)
            || !finite_positive(row.maximum_duration_ns)
            || row.maximum_duration_ns > row.total_duration_ns
        {
            return Err(format!("{sample_name} has an invalid signature chip row"));
        }
        previous_chip = Some(row.chip_id);
        record_count += row.record_count;
        maximum_chip_total = maximum_chip_total.max(row.total_duration_ns);
    }
    if record_count != signature.record_count
        || maximum_chip_total != signature.maximum_chip_total_duration_ns
    {
        return Err(format!("{sample_name} has inconsistent signature totals"));
    }
    Ok(())
}

fn validate_duplicate_coverage(roots: &[Vec<Sample>], label: &str) -> Result<(), String> {
    let first = roots
        .first()
        .ok_or_else(|| format!("{label} roots are empty"))?;
    for sample_index in 0..first.len() {
        let expected = signature_keys(&first[sample_index].response.x_onix_decode_evidence);
        for root in roots.iter().skip(1) {
            let observed = signature_keys(&root[sample_index].response.x_onix_decode_evidence);
            if observed != expected {
                return Err(format!(
                    "{label} duplicate roots have different signature coverage for {}",
                    first[sample_index].name
                ));
            }
        }
    }
    Ok(())
}

fn signature_keys(evidence: &DecodeEvidence) -> BTreeSet<SignatureKey> {
    evidence
        .prefill_program_profile
        .signatures
        .iter()
        .flat_map(|signature| {
            signature.by_chip.iter().map(|row| SignatureKey {
                kernel_sources: signature.kernel_sources.clone(),
                chip_id: row.chip_id,
            })
        })
        .collect()
}

fn aggregate_signature_values(
    roots: &[Vec<Sample>],
) -> Result<BTreeMap<SignatureKey, Vec<f64>>, String> {
    let mut values: BTreeMap<SignatureKey, Vec<f64>> = BTreeMap::new();
    for root in roots {
        for sample in root {
            for signature in &sample.response.x_onix_decode_evidence.prefill_program_profile.signatures {
                for row in &signature.by_chip {
                    values
                        .entry(SignatureKey {
                            kernel_sources: signature.kernel_sources.clone(),
                            chip_id: row.chip_id,
                        })
                        .or_default()
                        .push(row.total_duration_ns);
                }
            }
        }
    }
    if values.is_empty() {
        return Err("profile roots have no signature values".to_owned());
    }
    Ok(values)
}

fn complete_keys(
    values: &BTreeMap<SignatureKey, Vec<f64>>,
    expected_value_count: usize,
) -> BTreeSet<SignatureKey> {
    values
        .iter()
        .filter(|(_, durations)| durations.len() == expected_value_count)
        .map(|(key, _)| key.clone())
        .collect()
}

fn identity(evidence: &DecodeEvidence) -> Identity {
    Identity {
        model: evidence.model_identity.clone(),
        package: evidence.package_identity.clone(),
        profile: evidence.profile_id.clone(),
        runtime: evidence.runtime_identity.clone(),
        physical_pcie_device_ids: evidence.physical_pcie_device_ids.clone(),
    }
}

fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !finite_positive(*value)) {
        return Err("median requires finite positive values".to_owned());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let midpoint = ordered.len() / MEDIAN_DIVISOR;
    if ordered.len() % MEDIAN_DIVISOR == 0 {
        Ok((ordered[midpoint - 1] + ordered[midpoint]) / MEDIAN_DIVISOR_FLOAT)
    } else {
        Ok(ordered[midpoint])
    }
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn self_test() -> Result<(), String> {
    let root = env::temp_dir().join(format!(
        "qwen36-prefill-profile-receipt-{}",
        std::process::id()
    ));
    let eager_one = root.join("eager-one");
    let eager_two = root.join("eager-two");
    let candidate_one = root.join("candidate-one");
    let candidate_two = root.join("candidate-two");
    for path in [&eager_one, &eager_two, &candidate_one, &candidate_two] {
        fs::create_dir_all(path)
            .map_err(|error| format!("create self-test root {}: {error}", path.display()))?;
    }
    let sample_name = "p-measured-1.json";
    write_fixture(
        &eager_one.join(sample_name),
        SELF_TEST_EAGER_DURATION_NS,
        "kernel.cpp",
        false,
    )?;
    write_fixture(
        &eager_two.join(sample_name),
        SELF_TEST_EAGER_DURATION_NS,
        "kernel.cpp",
        false,
    )?;
    write_fixture(
        &candidate_one.join(sample_name),
        SELF_TEST_CANDIDATE_DURATION_NS,
        "kernel.cpp",
        false,
    )?;
    write_fixture(
        &candidate_two.join(sample_name),
        SELF_TEST_CANDIDATE_DURATION_NS,
        "kernel.cpp",
        false,
    )?;
    let config = Config {
        eager_roots: vec![eager_one.clone(), eager_two.clone()],
        candidate_roots: vec![candidate_one.clone(), candidate_two.clone()],
        execution_order: vec![
            "eager".to_owned(),
            "candidate".to_owned(),
            "candidate".to_owned(),
            "eager".to_owned(),
        ],
        output: root.join("receipt.json"),
    };
    let positive = build_receipt(&config)?;
    if positive["strongest_checked_difference"]["delta_duration_ns"]
        != json!(SELF_TEST_EXPECTED_DELTA_NS)
    {
        return Err("positive fixture computed the wrong program difference".to_owned());
    }

    write_fixture(
        &candidate_two.join(sample_name),
        SELF_TEST_CANDIDATE_DURATION_NS,
        "other-kernel.cpp",
        false,
    )?;
    let coverage_error = build_receipt(&config)
        .expect_err("mismatched duplicate coverage unexpectedly produced a receipt");
    if !coverage_error.contains("different signature coverage") {
        return Err(format!("coverage fixture had the wrong error: {coverage_error}"));
    }
    write_fixture(
        &candidate_two.join(sample_name),
        SELF_TEST_CANDIDATE_DURATION_NS,
        "kernel.cpp",
        true,
    )?;
    let admission_error = build_receipt(&config)
        .expect_err("admission-eligible profile unexpectedly produced a receipt");
    if !admission_error.contains("not diagnostic") {
        return Err(format!("admission fixture had the wrong error: {admission_error}"));
    }
    let wrong_order = Config {
        execution_order: vec![
            "eager".to_owned(),
            "eager".to_owned(),
            "candidate".to_owned(),
            "candidate".to_owned(),
        ],
        ..config
    };
    let order_error = build_receipt(&wrong_order)
        .expect_err("wrong execution order unexpectedly produced a receipt");
    if !order_error.contains("execution order") {
        return Err(format!("order fixture had the wrong error: {order_error}"));
    }
    fs::remove_dir_all(&root)
        .map_err(|error| format!("remove self-test fixture {}: {error}", root.display()))?;
    Ok(())
}

fn write_fixture(
    path: &Path,
    duration_ns: f64,
    kernel_source: &str,
    admission_eligible: bool,
) -> Result<(), String> {
    let response = json!({
        "id": "fixture-id",
        "x_onix_decode_evidence": {
            "admission_eligible": admission_eligible,
            "measurement_mode": DIAGNOSTIC_MEASUREMENT_MODE,
            "model_identity": "model",
            "package_identity": "package",
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "prefill_page_table_blocks": 1,
            "prefill_program_profile": {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "record_count": EXPECTED_DEVICE_COUNT,
                "runtime_ids": [1],
                "by_chip": [
                    {"chip_id": 0, "record_count": 1, "total_duration_ns": duration_ns},
                    {"chip_id": 1, "record_count": 1, "total_duration_ns": duration_ns}
                ],
                "signatures": [{
                    "kernel_sources": [kernel_source],
                    "maximum_chip_total_duration_ns": duration_ns,
                    "record_count": EXPECTED_DEVICE_COUNT,
                    "by_chip": [
                        {
                            "chip_id": 0,
                            "maximum_duration_ns": duration_ns,
                            "record_count": 1,
                            "total_duration_ns": duration_ns
                        },
                        {
                            "chip_id": 1,
                            "maximum_duration_ns": duration_ns,
                            "record_count": 1,
                            "total_duration_ns": duration_ns
                        }
                    ]
                }]
            },
            "profile_id": "profile",
            "runtime_identity": "runtime",
            "simulated": false
        }
    });
    fs::write(
        path,
        serde_json::to_vec(&response)
            .map_err(|error| format!("serialize fixture: {error}"))?,
    )
    .map_err(|error| format!("write fixture {}: {error}", path.display()))
}
