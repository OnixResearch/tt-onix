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
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_NAME: &str = "qwen36-decode-receipt.rs";
const PROFILE_FLAG: &str = "--profile";
const EAGER_ROOT_FLAG: &str = "--eager-response-root";
const CANDIDATE_ROOT_FLAG: &str = "--candidate-response-root";
const EXECUTION_ORDER_FLAG: &str = "--execution-order";
const OUTPUT_FLAG: &str = "--output";
const SELF_TEST_FLAG: &str = "--self-test";
const MEASURED_MARKER: &str = "-measured-";
const JSON_EXTENSION: &str = "json";
const PERCENTILE_NUMERATOR: usize = 99;
const PERCENTILE_DENOMINATOR: usize = 100;
const MINIMUM_ROOT_COUNT: usize = 1;
const EXPECTED_DEVICE_COUNT: usize = 2;
const EXPECTED_DEVICE_IDS: [i64; EXPECTED_DEVICE_COUNT] = [0, 1];
const FLAG_VALUE_ARGUMENT_WIDTH: usize = 2;
const MEDIAN_DIVISOR: usize = 2;
const MEDIAN_DIVISOR_FLOAT: f64 = 2.0;
const SELF_TEST_CANDIDATE_RATE: f64 = 2.0;
const SELF_TEST_ADMISSION_BOUND: f64 = 1.1;
const SELF_TEST_LATENCY_SECONDS: f64 = 0.1;
const SELF_TEST_MATCHING_TOKEN: i64 = 7;
const SELF_TEST_MISMATCH_TOKEN: i64 = 8;
const SUCCESS_EXIT_CODE: u8 = 0;
const FAILURE_EXIT_CODE: u8 = 1;

#[derive(Debug)]
struct Config {
    profile: PathBuf,
    eager_roots: Vec<PathBuf>,
    candidate_roots: Vec<PathBuf>,
    execution_order: Vec<String>,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Profile {
    profile_id: String,
    identities: ProfileIdentities,
    workload: Workload,
    admission: AdmissionBounds,
}

#[derive(Debug, Deserialize)]
struct ProfileIdentities {
    authority_blake3: String,
    baseline_receipt_blake3: String,
    checkpoint_revision: String,
    model_source_revision: String,
    tokenizer_revision: String,
}

#[derive(Debug, Deserialize)]
struct Workload {
    prompts: Vec<String>,
    warmup_runs: usize,
    measured_runs: usize,
    generated_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct AdmissionBounds {
    maximum_tail_latency_regression_ratio: f64,
    maximum_ttft_regression_ratio: f64,
    minimum_median_decode_speedup_ratio: f64,
    require_clean_reset: bool,
    require_exact_token_parity: bool,
    require_physical_evidence: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,
    usage: Usage,
    x_onix_decode_evidence: DecodeEvidence,
}

#[derive(Debug, Deserialize)]
struct Choice {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    completion_tokens: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DecodeEvidence {
    clean_reset: bool,
    cleanup_seconds: f64,
    decode_path: String,
    decode_tokens_per_second: f64,
    fallback_reason: Option<String>,
    generated_token_ids: Vec<i64>,
    model_identity: String,
    package_identity: String,
    physical_pcie_device_ids: Vec<i64>,
    profile_id: String,
    runtime_identity: String,
    simulated: bool,
    tail_token_latency_seconds: f64,
    token_latency_seconds: Vec<f64>,
    trace_capture_seconds: Option<f64>,
    ttft_seconds: f64,
}

#[derive(Debug)]
struct Sample {
    name: String,
    response_blake3: String,
    response: CompletionResponse,
}

#[derive(Default)]
struct Metrics {
    decode_rates: Vec<f64>,
    ttft_seconds: Vec<f64>,
    token_latency_seconds: Vec<f64>,
    cleanup_seconds: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    model: String,
    package: String,
    runtime: String,
    physical_pcie_device_ids: Vec<i64>,
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
    if arguments.as_slice() == [SELF_TEST_FLAG] {
        self_test()?;
        println!("{PROGRAM_NAME} self-test: PASS");
        return Ok(());
    }
    let config = parse_config(arguments)?;
    let receipt = build_receipt(&config)?;
    let parent = config
        .output
        .parent()
        .ok_or_else(|| "output path has no parent directory".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    let encoded = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("serialize receipt: {error}"))?;
    let file_bytes = format!("{encoded}\n");
    let receipt_blake3 = blake3::hash(file_bytes.as_bytes()).to_hex().to_string();
    fs::write(&config.output, file_bytes)
        .map_err(|error| format!("write {}: {error}", config.output.display()))?;
    let classification = receipt["classification"]
        .as_str()
        .ok_or_else(|| "receipt classification is not a string".to_owned())?;
    println!("qwen36 decode receipt: {classification}");
    println!("receipt blake3: {receipt_blake3}");
    Ok(())
}

fn parse_config(arguments: Vec<String>) -> Result<Config, String> {
    let mut profile = None;
    let mut eager_roots = Vec::new();
    let mut candidate_roots = Vec::new();
    let mut execution_order: Option<Vec<String>> = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            PROFILE_FLAG => profile = Some(PathBuf::from(value)),
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
    if eager_roots.len() < MINIMUM_ROOT_COUNT || eager_roots.len() != candidate_roots.len() {
        return Err("eager and candidate response roots must have the same non-zero count".to_owned());
    }
    let execution_order = execution_order.ok_or_else(|| format!("missing {EXECUTION_ORDER_FLAG}"))?;
    let expected_order_items = eager_roots.len() + candidate_roots.len();
    if execution_order.len() != expected_order_items {
        return Err(format!(
            "execution order has {} items, expected {expected_order_items}",
            execution_order.len()
        ));
    }
    if execution_order
        .iter()
        .any(|item| item != "eager" && item != "candidate")
    {
        return Err("execution order contains an unknown path".to_owned());
    }
    Ok(Config {
        profile: profile.ok_or_else(|| format!("missing {PROFILE_FLAG}"))?,
        eager_roots,
        candidate_roots,
        execution_order,
        output: output.ok_or_else(|| format!("missing {OUTPUT_FLAG}"))?,
    })
}

fn build_receipt(config: &Config) -> Result<Value, String> {
    let profile: Profile = read_json(&config.profile)?;
    validate_profile(&profile)?;
    let expected_samples_per_root = profile.workload.prompts.len() * profile.workload.measured_runs;
    let mut eager_metrics = Metrics::default();
    let mut candidate_metrics = Metrics::default();
    let mut token_parity = true;
    let mut text_parity = true;
    let mut clean_reset = true;
    let mut identities_match = true;
    let mut physical = true;
    let mut eager_hashes = Vec::new();
    let mut candidate_hashes = Vec::new();
    let mut canonical_identity: Option<Identity> = None;

    for (eager_root, candidate_root) in config
        .eager_roots
        .iter()
        .zip(&config.candidate_roots)
    {
        let eager_samples = read_measured_samples(eager_root)?;
        let candidate_samples = read_measured_samples(candidate_root)?;
        if eager_samples.len() != expected_samples_per_root {
            return Err(format!(
                "{} has {} measured samples, expected {expected_samples_per_root}",
                eager_root.display(),
                eager_samples.len()
            ));
        }
        if candidate_samples.len() != expected_samples_per_root {
            return Err(format!(
                "{} has {} measured samples, expected {expected_samples_per_root}",
                candidate_root.display(),
                candidate_samples.len()
            ));
        }
        for (eager, candidate) in eager_samples.iter().zip(&candidate_samples) {
            if eager.name != candidate.name {
                return Err(format!(
                    "response names differ: {} != {}",
                    eager.name, candidate.name
                ));
            }
            validate_response(eager, &profile, "eager")?;
            validate_response(candidate, &profile, "traced")?;
            token_parity &= eager.response.x_onix_decode_evidence.generated_token_ids
                == candidate.response.x_onix_decode_evidence.generated_token_ids;
            text_parity &= eager.response.choices[0].text == candidate.response.choices[0].text;
            clean_reset &= eager.response.x_onix_decode_evidence.clean_reset
                && candidate.response.x_onix_decode_evidence.clean_reset;
            physical &= !eager.response.x_onix_decode_evidence.simulated
                && !candidate.response.x_onix_decode_evidence.simulated
                && eager.response.x_onix_decode_evidence.physical_pcie_device_ids
                    == EXPECTED_DEVICE_IDS
                && candidate.response.x_onix_decode_evidence.physical_pcie_device_ids
                    == EXPECTED_DEVICE_IDS;
            let eager_identity = identity(&eager.response.x_onix_decode_evidence);
            let candidate_identity = identity(&candidate.response.x_onix_decode_evidence);
            identities_match &= eager_identity == candidate_identity;
            if let Some(expected) = &canonical_identity {
                identities_match &= eager_identity == *expected;
            } else {
                canonical_identity = Some(eager_identity.clone());
            }
            push_metrics(&mut eager_metrics, &eager.response.x_onix_decode_evidence)?;
            push_metrics(
                &mut candidate_metrics,
                &candidate.response.x_onix_decode_evidence,
            )?;
            eager_hashes.push(json!({"name": eager.name, "blake3": eager.response_blake3}));
            candidate_hashes.push(json!({
                "name": candidate.name,
                "blake3": candidate.response_blake3
            }));
        }
    }

    let eager_decode_median = median(&eager_metrics.decode_rates)?;
    let candidate_decode_median = median(&candidate_metrics.decode_rates)?;
    let eager_ttft_median = median(&eager_metrics.ttft_seconds)?;
    let candidate_ttft_median = median(&candidate_metrics.ttft_seconds)?;
    let eager_tail_p99 = percentile_99(&eager_metrics.token_latency_seconds)?;
    let candidate_tail_p99 = percentile_99(&candidate_metrics.token_latency_seconds)?;
    let speedup_ratio = candidate_decode_median / eager_decode_median;
    let ttft_ratio = candidate_ttft_median / eager_ttft_median;
    let tail_ratio = candidate_tail_p99 / eager_tail_p99;

    let mut reasons = Vec::new();
    if profile.admission.require_physical_evidence && !physical {
        reasons.push("physical-evidence-missing");
    }
    if !identities_match {
        reasons.push("identity-mismatch");
    }
    if profile.admission.require_exact_token_parity && !token_parity {
        reasons.push("token-parity-failed");
    }
    if !text_parity {
        reasons.push("text-parity-failed");
    }
    if profile.admission.require_clean_reset && !clean_reset {
        reasons.push("cleanup-failed");
    }
    if speedup_ratio < profile.admission.minimum_median_decode_speedup_ratio {
        reasons.push("decode-speedup-below-bound");
    }
    if ttft_ratio > profile.admission.maximum_ttft_regression_ratio {
        reasons.push("ttft-regression-above-bound");
    }
    if tail_ratio > profile.admission.maximum_tail_latency_regression_ratio {
        reasons.push("tail-latency-regression-above-bound");
    }
    let admitted = reasons.is_empty();
    let classification = if admitted { "passed" } else { "rejected" };
    let identity = canonical_identity.ok_or_else(|| "no measured identity".to_owned())?;

    Ok(json!({
        "schema_version": 1,
        "classification": classification,
        "date": "2026-08-20",
        "profile": {
            "profile_id": profile.profile_id,
            "authority_blake3": profile.identities.authority_blake3,
            "baseline_receipt_blake3": profile.identities.baseline_receipt_blake3,
            "checkpoint_revision": profile.identities.checkpoint_revision,
            "model_source_revision": profile.identities.model_source_revision,
            "tokenizer_revision": profile.identities.tokenizer_revision,
        },
        "execution": {
            "execution_order": config.execution_order,
            "generated_tokens": profile.workload.generated_tokens,
            "measured_runs_per_prompt": profile.workload.measured_runs,
            "prompt_count": profile.workload.prompts.len(),
            "response_root_pairs": config.eager_roots.len(),
            "simulated": false,
            "warmup_runs_per_prompt": profile.workload.warmup_runs,
        },
        "identity": {
            "model_identity": identity.model,
            "package_identity": identity.package,
            "runtime_identity": identity.runtime,
            "physical_pcie_device_ids": identity.physical_pcie_device_ids,
            "identities_match": identities_match,
        },
        "parity": {
            "exact_generated_token_ids": token_parity,
            "exact_generated_text": text_parity,
            "clean_reset": clean_reset,
        },
        "metrics": {
            "eager": {
                "median_decode_tokens_per_second": eager_decode_median,
                "median_ttft_seconds": eager_ttft_median,
                "p99_token_latency_seconds": eager_tail_p99,
                "median_cleanup_seconds": median(&eager_metrics.cleanup_seconds)?,
            },
            "candidate": {
                "median_decode_tokens_per_second": candidate_decode_median,
                "median_ttft_seconds": candidate_ttft_median,
                "p99_token_latency_seconds": candidate_tail_p99,
                "median_cleanup_seconds": median(&candidate_metrics.cleanup_seconds)?,
            },
            "decode_speedup_ratio": speedup_ratio,
            "ttft_regression_ratio": ttft_ratio,
            "tail_latency_regression_ratio": tail_ratio,
        },
        "admission": {
            "admitted": admitted,
            "reasons": reasons,
            "bounds": {
                "minimum_median_decode_speedup_ratio": profile.admission.minimum_median_decode_speedup_ratio,
                "maximum_ttft_regression_ratio": profile.admission.maximum_ttft_regression_ratio,
                "maximum_tail_latency_regression_ratio": profile.admission.maximum_tail_latency_regression_ratio,
            }
        },
        "response_hashes": {
            "eager": eager_hashes,
            "candidate": candidate_hashes,
        },
        "non_claims": [
            "No concurrent-request throughput claim",
            "No prefill speedup claim",
            "No mesh other than physical P150x2",
            "No generated length other than the profile workload",
            "A rejected candidate does not change the admitted eager service"
        ]
    }))
}

fn validate_profile(profile: &Profile) -> Result<(), String> {
    if profile.profile_id.is_empty() || profile.workload.prompts.is_empty() {
        return Err("profile identity and prompts must be non-empty".to_owned());
    }
    if profile.workload.measured_runs == 0 || profile.workload.generated_tokens == 0 {
        return Err("profile measured runs and generated tokens must be positive".to_owned());
    }
    let unique_prompts = profile.workload.prompts.iter().collect::<BTreeSet<_>>();
    if unique_prompts.len() != profile.workload.prompts.len() {
        return Err("profile prompts must be unique".to_owned());
    }
    Ok(())
}

fn read_measured_samples(root: &Path) -> Result<Vec<Sample>, String> {
    let mut paths = fs::read_dir(root)
        .map_err(|error| format!("read {}: {error}", root.display()))?
        .map(|entry| entry.map(|item| item.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read entry in {}: {error}", root.display()))?;
    paths.retain(|path| {
        path.extension().and_then(|value| value.to_str()) == Some(JSON_EXTENSION)
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.contains(MEASURED_MARKER))
    });
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            let response = serde_json::from_slice(&bytes)
                .map_err(|error| format!("parse {}: {error}", path.display()))?;
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| format!("response path is not UTF-8: {}", path.display()))?
                .to_owned();
            Ok(Sample {
                name,
                response_blake3: blake3::hash(&bytes).to_hex().to_string(),
                response,
            })
        })
        .collect()
}

fn validate_response(sample: &Sample, profile: &Profile, expected_path: &str) -> Result<(), String> {
    let response = &sample.response;
    if response.id.is_empty() || response.choices.len() != MINIMUM_ROOT_COUNT {
        return Err(format!("{} has an invalid response identity or choice count", sample.name));
    }
    if response.usage.completion_tokens != profile.workload.generated_tokens {
        return Err(format!("{} has the wrong completion token count", sample.name));
    }
    let evidence = &response.x_onix_decode_evidence;
    if evidence.generated_token_ids.len() != profile.workload.generated_tokens {
        return Err(format!("{} has the wrong generated token ID count", sample.name));
    }
    if evidence.decode_path != expected_path {
        return Err(format!("{} used {}, expected {expected_path}", sample.name, evidence.decode_path));
    }
    if evidence.profile_id != profile.profile_id {
        return Err(format!("{} has the wrong profile identity", sample.name));
    }
    if expected_path == "traced" && evidence.fallback_reason.is_some() {
        return Err(format!("{} traced request used a fallback", sample.name));
    }
    if expected_path == "traced" && evidence.trace_capture_seconds.is_none() {
        return Err(format!("{} lacks trace capture timing", sample.name));
    }
    Ok(())
}

fn identity(evidence: &DecodeEvidence) -> Identity {
    Identity {
        model: evidence.model_identity.clone(),
        package: evidence.package_identity.clone(),
        runtime: evidence.runtime_identity.clone(),
        physical_pcie_device_ids: evidence.physical_pcie_device_ids.clone(),
    }
}

fn push_metrics(metrics: &mut Metrics, evidence: &DecodeEvidence) -> Result<(), String> {
    let scalar_metrics = [
        evidence.decode_tokens_per_second,
        evidence.ttft_seconds,
        evidence.tail_token_latency_seconds,
        evidence.cleanup_seconds,
    ];
    if scalar_metrics.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err("response contains a non-positive or non-finite metric".to_owned());
    }
    if evidence
        .token_latency_seconds
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("response contains an invalid token latency".to_owned());
    }
    metrics.decode_rates.push(evidence.decode_tokens_per_second);
    metrics.ttft_seconds.push(evidence.ttft_seconds);
    metrics
        .token_latency_seconds
        .extend(&evidence.token_latency_seconds);
    metrics.cleanup_seconds.push(evidence.cleanup_seconds);
    Ok(())
}

fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("median requires at least one value".to_owned());
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

fn percentile_99(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("percentile requires at least one value".to_owned());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = (PERCENTILE_NUMERATOR * ordered.len()).div_ceil(PERCENTILE_DENOMINATOR);
    Ok(ordered[rank - 1])
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn self_test() -> Result<(), String> {
    let root = env::temp_dir().join(format!("qwen36-decode-receipt-{}", std::process::id()));
    let eager = root.join("eager");
    let candidate = root.join("candidate");
    fs::create_dir_all(&eager).map_err(|error| format!("create eager fixture: {error}"))?;
    fs::create_dir_all(&candidate).map_err(|error| format!("create candidate fixture: {error}"))?;
    let profile_path = root.join("profile.json");
    fs::write(
        &profile_path,
        serde_json::to_vec(&json!({
            "profile_id": "fixture",
            "identities": {
                "authority_blake3": "a",
                "baseline_receipt_blake3": "b",
                "checkpoint_revision": "c",
                "model_source_revision": "d",
                "tokenizer_revision": "e"
            },
            "workload": {"prompts": ["p"], "warmup_runs": 1, "measured_runs": 1, "generated_tokens": 1},
            "admission": {
                "maximum_tail_latency_regression_ratio": SELF_TEST_ADMISSION_BOUND,
                "maximum_ttft_regression_ratio": SELF_TEST_ADMISSION_BOUND,
                "minimum_median_decode_speedup_ratio": SELF_TEST_ADMISSION_BOUND,
                "require_clean_reset": true,
                "require_exact_token_parity": true,
                "require_physical_evidence": true
            }
        }))
        .map_err(|error| format!("serialize profile fixture: {error}"))?,
    )
    .map_err(|error| format!("write profile fixture: {error}"))?;
    write_fixture(
        &eager.join("p-measured-1.json"),
        "eager",
        1.0,
        SELF_TEST_MATCHING_TOKEN,
    )?;
    write_fixture(
        &candidate.join("p-measured-1.json"),
        "traced",
        SELF_TEST_CANDIDATE_RATE,
        SELF_TEST_MATCHING_TOKEN,
    )?;
    let config = Config {
        profile: profile_path,
        eager_roots: vec![eager.clone()],
        candidate_roots: vec![candidate.clone()],
        execution_order: vec!["eager".to_owned(), "candidate".to_owned()],
        output: root.join("receipt.json"),
    };
    let positive = build_receipt(&config)?;
    if positive["admission"]["admitted"] != true {
        return Err("positive fixture was not admitted".to_owned());
    }
    write_fixture(
        &candidate.join("p-measured-1.json"),
        "traced",
        SELF_TEST_CANDIDATE_RATE,
        SELF_TEST_MISMATCH_TOKEN,
    )?;
    let negative = build_receipt(&config)?;
    if negative["admission"]["admitted"] != false
        || negative["admission"]["reasons"]
            .as_array()
            .is_none_or(|reasons| !reasons.iter().any(|reason| reason == "token-parity-failed"))
    {
        return Err("negative token-parity fixture was not rejected".to_owned());
    }
    fs::remove_dir_all(&root).map_err(|error| format!("remove self-test fixture: {error}"))?;
    Ok(())
}

fn write_fixture(path: &Path, decode_path: &str, rate: f64, token: i64) -> Result<(), String> {
    let trace_capture = if decode_path == "traced" { Some(1.0) } else { None };
    let response = json!({
        "id": "fixture-id",
        "choices": [{"text": "x"}],
        "usage": {"completion_tokens": 1},
        "x_onix_decode_evidence": {
            "clean_reset": true,
            "cleanup_seconds": SELF_TEST_LATENCY_SECONDS,
            "decode_path": decode_path,
            "decode_tokens_per_second": rate,
            "fallback_reason": if decode_path == "traced" { Value::Null } else { json!("eager-selected") },
            "generated_token_ids": [token],
            "model_identity": "model",
            "package_identity": "package",
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "profile_id": "fixture",
            "runtime_identity": "runtime",
            "simulated": false,
            "tail_token_latency_seconds": SELF_TEST_LATENCY_SECONDS,
            "token_latency_seconds": [SELF_TEST_LATENCY_SECONDS],
            "trace_capture_seconds": trace_capture,
            "ttft_seconds": 1.0
        }
    });
    fs::write(
        path,
        serde_json::to_vec(&response).map_err(|error| format!("serialize fixture: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))
}
