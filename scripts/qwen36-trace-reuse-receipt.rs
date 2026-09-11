#!/usr/bin/env -S RUSTC_BOOTSTRAP=1 nix shell "github:nix-community/fenix?rev=a0b1aef63d7e9c5568958e0d90f6fe48f0f4a7a2#stable.toolchain" nixpkgs#gcc -c cargo -q -Zscript

---
[package]
edition = "2024"

[dependencies]
blake3 = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
---

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_NAME: &str = "qwen36-trace-reuse-receipt.rs";
const PROFILE_FLAG: &str = "--profile";
const RECAPTURE_ROOT_FLAG: &str = "--recapture-response-root";
const REUSE_ROOT_FLAG: &str = "--reuse-response-root";
const EXECUTION_ORDER_FLAG: &str = "--execution-order";
const OUTPUT_FLAG: &str = "--output";
const SELF_TEST_FLAG: &str = "--self-test";
const MEASURED_MARKER: &str = "-measured-";
const JSON_EXTENSION: &str = "json";
const RECAPTURE_MEASUREMENT_MODE: &str = "physical-batched-service";
const REUSE_MEASUREMENT_MODE: &str = "same-width-trace-reuse";
const DECODE_PATH: &str = "batched-traced";
const NOT_MEASURED: &str = "not-measured";
const PROFILE_SCHEMA_VERSION: usize = 1;
const BLAKE3_LENGTH: usize = 64;
const REVISION_LENGTH: usize = 40;
const MAXIMUM_RESIDENT_TRACE_COUNT: usize = 1;
const EXPECTED_ROOTS_PER_ROUTE: usize = 2;
const EXPECTED_CHOICE_COUNT: usize = 1;
const EXPECTED_DEVICE_COUNT: usize = 2;
const EXPECTED_EXECUTION_ORDER_COUNT: usize = 4;
const EXPECTED_DEVICE_IDS: [i64; EXPECTED_DEVICE_COUNT] = [0, 1];
const EXPECTED_EXECUTION_ORDER: [&str; EXPECTED_EXECUTION_ORDER_COUNT] =
    ["recapture", "reuse", "reuse", "recapture"];
const FLAG_ARGUMENT_WIDTH: usize = 2;
const MEDIAN_DIVISOR: usize = 2;
const MEDIAN_DIVISOR_FLOAT: f64 = 2.0;
const SUCCESS_EXIT_CODE: u8 = 0;
const FAILURE_EXIT_CODE: u8 = 1;
const ZERO_CAPTURE_COUNT: usize = 0;
const SELF_TEST_WIDTH: usize = 4;
const SELF_TEST_PROMPT_COUNT: usize = 4;
const SELF_TEST_MEASURED_RUNS: usize = 1;
const SELF_TEST_GENERATED_TOKENS: usize = 2;
const SELF_TEST_RECAPTURE_WALL_SECONDS: f64 = 6.0;
const SELF_TEST_REUSE_WALL_SECONDS: f64 = 5.0;
const SELF_TEST_DECODE_RATE: f64 = 14.0;
const SELF_TEST_TTFT_SECONDS: f64 = 2.0;
const SELF_TEST_TAIL_SECONDS: f64 = 0.1;
const SELF_TEST_RECAPTURE_PREPARE_SECONDS: f64 = 0.65;
const SELF_TEST_RECAPTURE_CAPTURE_SECONDS: f64 = 0.64;
const SELF_TEST_REUSE_PREPARE_SECONDS: f64 = 0.01;
const SELF_TEST_CLEANUP_SECONDS: f64 = 0.01;
const SELF_TEST_MAXIMUM_DUPLICATE_RATIO: f64 = 1.25;
const SELF_TEST_MAXIMUM_REUSE_PREPARE_SECONDS: f64 = 0.05;
const SELF_TEST_MAXIMUM_TAIL_RATIO: f64 = 1.10;
const SELF_TEST_MAXIMUM_TTFT_RATIO: f64 = 1.05;
const SELF_TEST_MINIMUM_DECODE_RETENTION: f64 = 0.95;
const SELF_TEST_MINIMUM_WALL_SPEEDUP: f64 = 1.05;
const SELF_TEST_TOKEN_BASE: i64 = 100;
const SELF_TEST_TOKEN_STRIDE: i64 = 100;
const SELF_TEST_MISMATCH_TOKEN: i64 = 999;
const SELF_TEST_FILE_INDEX_WIDTH: usize = 2;
const SELF_TEST_FIRST_RESPONSE_NAME: &str = "00-measured-0.json";

#[derive(Debug)]
struct Config {
    profile: PathBuf,
    recapture_roots: Vec<PathBuf>,
    reuse_roots: Vec<PathBuf>,
    execution_order: Vec<String>,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema_version: usize,
    profile_id: String,
    physically_admitted: bool,
    identities: ProfileIdentities,
    workload: Workload,
    admission: Admission,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProfileIdentities {
    authority_blake3: String,
    checkpoint_revision: String,
    model_source_revision: String,
    receipt_blake3: String,
    tokenizer_revision: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    width: usize,
    execution_order: Vec<String>,
    generated_tokens: usize,
    measured_runs: usize,
    prompts: Vec<String>,
    warmup_runs: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Admission {
    maximum_duplicate_root_ratio: f64,
    maximum_resident_trace_count: usize,
    maximum_reused_trace_prepare_seconds: f64,
    maximum_tail_latency_ratio: f64,
    maximum_ttft_ratio: f64,
    minimum_decode_rate_retention_ratio: f64,
    minimum_request_wall_speedup_ratio: f64,
    require_clean_reset: bool,
    require_exact_token_parity: bool,
    require_physical_evidence: bool,
    require_stable_binding: bool,
    require_zero_measured_recaptures: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Response {
    id: String,
    choices: Vec<Choice>,
    usage: Usage,
    x_onix_decode_evidence: Evidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Choice {
    text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Usage {
    completion_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Evidence {
    admission_eligible: bool,
    batch_width: usize,
    clean_reset: bool,
    cleanup_seconds: f64,
    decode_path: String,
    decode_tokens_per_second: f64,
    generated_token_ids: Vec<i64>,
    measurement_mode: String,
    model_identity: String,
    package_identity: String,
    physical_pcie_device_ids: Vec<i64>,
    profile_id: String,
    request_wall_seconds: f64,
    resident_trace_count: usize,
    runtime_identity: String,
    simulated: bool,
    state_slice: usize,
    tail_token_latency_seconds: f64,
    token_latency_seconds: Vec<f64>,
    trace_binding_stable: bool,
    trace_capture_count: usize,
    trace_capture_seconds: f64,
    trace_prepare_seconds: f64,
    trace_reuse_enabled: bool,
    trace_reused: bool,
    trace_transition_reason: Option<String>,
    ttft_seconds: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    model: String,
    package: String,
    runtime: String,
    devices: Vec<i64>,
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Recapture,
    Reuse,
}

#[derive(Debug)]
struct Root {
    path: PathBuf,
    samples: Vec<Sample>,
}

#[derive(Clone, Debug)]
struct Sample {
    path: PathBuf,
    response: Response,
    hash: String,
}

#[derive(Default)]
struct Metrics {
    wall: Vec<f64>,
    decode_rate: Vec<f64>,
    ttft: Vec<f64>,
    tail: Vec<f64>,
    trace_prepare: Vec<f64>,
    cleanup: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct MetricSummary {
    median_request_wall_seconds: f64,
    median_decode_tokens_per_second: f64,
    median_ttft_seconds: f64,
    median_tail_token_latency_seconds: f64,
    median_trace_prepare_seconds: f64,
    median_cleanup_seconds: f64,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::from(SUCCESS_EXIT_CODE),
        Err(error) => {
            eprintln!("{PROGRAM_NAME}: {error}");
            ExitCode::from(FAILURE_EXIT_CODE)
        }
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.as_slice() == [SELF_TEST_FLAG] {
        return self_test();
    }
    let config = parse_config(&arguments)?;
    let receipt = build_receipt(&config)?;
    let bytes = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    fs::write(&config.output, bytes)
        .map_err(|error| format!("write {}: {error}", config.output.display()))?;
    Ok(())
}

fn parse_config(arguments: &[String]) -> Result<Config, String> {
    let mut profile = None;
    let mut recapture_roots = Vec::new();
    let mut reuse_roots = Vec::new();
    let mut execution_order = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        if index + 1 >= arguments.len() {
            return Err(format!("missing value for {flag}"));
        }
        let value = &arguments[index + 1];
        match flag.as_str() {
            PROFILE_FLAG => profile = Some(PathBuf::from(value)),
            RECAPTURE_ROOT_FLAG => recapture_roots.push(PathBuf::from(value)),
            REUSE_ROOT_FLAG => reuse_roots.push(PathBuf::from(value)),
            EXECUTION_ORDER_FLAG => {
                execution_order = Some(value.split(',').map(str::to_owned).collect())
            }
            OUTPUT_FLAG => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
        index += FLAG_ARGUMENT_WIDTH;
    }
    if recapture_roots.len() != EXPECTED_ROOTS_PER_ROUTE
        || reuse_roots.len() != EXPECTED_ROOTS_PER_ROUTE
    {
        return Err("exactly two recapture and two reuse roots are required".to_owned());
    }
    let execution_order = execution_order.ok_or("missing --execution-order")?;
    if execution_order != EXPECTED_EXECUTION_ORDER {
        return Err("execution order must be recapture,reuse,reuse,recapture".to_owned());
    }
    Ok(Config {
        profile: profile.ok_or("missing --profile")?,
        recapture_roots,
        reuse_roots,
        execution_order,
        output: output.ok_or("missing --output")?,
    })
}

fn build_receipt(config: &Config) -> Result<Value, String> {
    let profile: Profile = read_json(&config.profile)?;
    validate_profile(&profile, config)?;
    let recapture_roots = read_roots(&config.recapture_roots)?;
    let reuse_roots = read_roots(&config.reuse_roots)?;
    let expected_samples = profile
        .workload
        .prompts
        .len()
        .checked_mul(profile.workload.measured_runs)
        .ok_or("sample count overflow")?;
    let mut identity = None;
    for root in &recapture_roots {
        validate_root(
            root,
            &profile,
            Route::Recapture,
            expected_samples,
            &mut identity,
        )?;
    }
    for root in &reuse_roots {
        validate_root(
            root,
            &profile,
            Route::Reuse,
            expected_samples,
            &mut identity,
        )?;
    }
    validate_state_slices(&recapture_roots, &profile)?;
    validate_state_slices(&reuse_roots, &profile)?;

    let canonical_outputs = outputs(&recapture_roots[0]);
    let token_parity = recapture_roots
        .iter()
        .chain(&reuse_roots)
        .all(|root| output_tokens(root) == output_tokens(&recapture_roots[0]));
    let text_parity = recapture_roots
        .iter()
        .chain(&reuse_roots)
        .all(|root| outputs(root) == canonical_outputs);

    let recapture_metrics = metrics(&flatten(&recapture_roots))?;
    let reuse_metrics = metrics(&flatten(&reuse_roots))?;
    let recapture_summary = summarize(&recapture_metrics)?;
    let reuse_summary = summarize(&reuse_metrics)?;
    let recapture_duplicate_ratio = duplicate_ratio(&recapture_roots)?;
    let reuse_duplicate_ratio = duplicate_ratio(&reuse_roots)?;
    let wall_speedup =
        recapture_summary.median_request_wall_seconds / reuse_summary.median_request_wall_seconds;
    let decode_retention = reuse_summary.median_decode_tokens_per_second
        / recapture_summary.median_decode_tokens_per_second;
    let ttft_ratio = reuse_summary.median_ttft_seconds / recapture_summary.median_ttft_seconds;
    let tail_ratio = reuse_summary.median_tail_token_latency_seconds
        / recapture_summary.median_tail_token_latency_seconds;

    let mut reasons = Vec::new();
    if profile.admission.require_exact_token_parity && !token_parity {
        reasons.push("token-parity-failed");
    }
    if !text_parity {
        reasons.push("text-parity-failed");
    }
    if recapture_duplicate_ratio > profile.admission.maximum_duplicate_root_ratio {
        reasons.push("recapture-duplicate-root-instability");
    }
    if reuse_duplicate_ratio > profile.admission.maximum_duplicate_root_ratio {
        reasons.push("reuse-duplicate-root-instability");
    }
    if wall_speedup < profile.admission.minimum_request_wall_speedup_ratio {
        reasons.push("request-wall-speedup-below-bound");
    }
    if decode_retention < profile.admission.minimum_decode_rate_retention_ratio {
        reasons.push("decode-rate-retention-below-bound");
    }
    if ttft_ratio > profile.admission.maximum_ttft_ratio {
        reasons.push("ttft-ratio-above-bound");
    }
    if tail_ratio > profile.admission.maximum_tail_latency_ratio {
        reasons.push("tail-latency-ratio-above-bound");
    }
    if reuse_summary.median_trace_prepare_seconds
        > profile.admission.maximum_reused_trace_prepare_seconds
    {
        reasons.push("reused-trace-prepare-above-bound");
    }
    let admitted = reasons.is_empty();
    let identity = identity.ok_or("response roots are empty")?;

    Ok(json!({
        "schema": "onix.qwen36.trace-reuse-receipt.v1",
        "profile_id": profile.profile_id,
        "identity": {
            "authority_blake3": profile.identities.authority_blake3,
            "checkpoint_revision": profile.identities.checkpoint_revision,
            "model_source_revision": profile.identities.model_source_revision,
            "tokenizer_revision": profile.identities.tokenizer_revision,
            "runtime_identity": identity.runtime,
            "package_identity": identity.package,
            "model_identity": identity.model,
            "physical_pcie_device_ids": identity.devices,
        },
        "execution": {
            "execution_order": config.execution_order,
            "width": profile.workload.width,
            "prompt_count": profile.workload.prompts.len(),
            "measured_runs": profile.workload.measured_runs,
            "generated_tokens": profile.workload.generated_tokens,
            "simulated": false,
        },
        "correctness": {
            "exact_token_parity": token_parity,
            "exact_text_parity": text_parity,
            "clean_reset": true,
            "stable_binding": true,
            "one_resident_trace": true,
            "zero_measured_recaptures": true,
        },
        "metrics": {
            "recapture": recapture_summary,
            "reuse": reuse_summary,
            "recapture_duplicate_root_ratio": recapture_duplicate_ratio,
            "reuse_duplicate_root_ratio": reuse_duplicate_ratio,
            "request_wall_speedup_ratio": wall_speedup,
            "decode_rate_retention_ratio": decode_retention,
            "ttft_ratio": ttft_ratio,
            "tail_latency_ratio": tail_ratio,
        },
        "admission": {
            "admitted": admitted,
            "reasons": reasons,
            "bounds": profile.admission,
        },
        "response_hashes": {
            "recapture": root_hashes("recapture", &recapture_roots),
            "reuse": root_hashes("reuse", &reuse_roots),
        },
        "non_claims": [
            "this receipt does not admit arbitrary continuous batching",
            "this receipt does not admit width changes without trace release",
            "this receipt does not admit slot recycling or live request insertion",
            "this receipt does not replace the B=1 versus B=4 throughput gate",
        ],
    }))
}

fn validate_profile(profile: &Profile, config: &Config) -> Result<(), String> {
    if profile.schema_version != PROFILE_SCHEMA_VERSION {
        return Err("profile schema version is unsupported".to_owned());
    }
    if profile.physically_admitted || profile.identities.receipt_blake3 != NOT_MEASURED {
        return Err("input profile must be unadmitted and unmeasured".to_owned());
    }
    if profile.workload.execution_order != EXPECTED_EXECUTION_ORDER
        || config.execution_order != profile.workload.execution_order
    {
        return Err("profile execution order differs from the command".to_owned());
    }
    if profile.workload.width != SELF_TEST_WIDTH
        || profile.workload.prompts.is_empty()
        || profile.workload.measured_runs == 0
        || profile.workload.generated_tokens == 0
        || profile.workload.warmup_runs == 0
    {
        return Err("profile workload is unsupported or empty".to_owned());
    }
    if profile.admission.maximum_resident_trace_count != MAXIMUM_RESIDENT_TRACE_COUNT
        || !profile.admission.require_clean_reset
        || !profile.admission.require_exact_token_parity
        || !profile.admission.require_physical_evidence
        || !profile.admission.require_stable_binding
        || !profile.admission.require_zero_measured_recaptures
    {
        return Err("profile weakens a required trace-reuse gate".to_owned());
    }
    Ok(())
}

fn read_roots(paths: &[PathBuf]) -> Result<Vec<Root>, String> {
    paths
        .iter()
        .map(|path| {
            let mut files = fs::read_dir(path)
                .map_err(|error| format!("read {}: {error}", path.display()))?
                .map(|entry| entry.map(|value| value.path()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            files.retain(|file| {
                file.extension().and_then(|value| value.to_str()) == Some(JSON_EXTENSION)
                    && file
                        .file_name()
                        .and_then(|value| value.to_str())
                        .is_some_and(|name| name.contains(MEASURED_MARKER))
            });
            files.sort();
            let samples = files
                .into_iter()
                .map(|file| {
                    let bytes = fs::read(&file)
                        .map_err(|error| format!("read {}: {error}", file.display()))?;
                    let response = serde_json::from_slice(&bytes)
                        .map_err(|error| format!("parse {}: {error}", file.display()))?;
                    Ok(Sample {
                        path: file,
                        response,
                        hash: blake3::hash(&bytes).to_hex().to_string(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Root {
                path: path.clone(),
                samples,
            })
        })
        .collect()
}

fn validate_root(
    root: &Root,
    profile: &Profile,
    route: Route,
    expected_samples: usize,
    canonical_identity: &mut Option<Identity>,
) -> Result<(), String> {
    if root.samples.len() != expected_samples {
        return Err(format!(
            "{} has {} measured samples, expected {expected_samples}",
            root.path.display(),
            root.samples.len()
        ));
    }
    for sample in &root.samples {
        validate_sample(sample, profile, route, canonical_identity)?;
    }
    Ok(())
}

fn validate_sample(
    sample: &Sample,
    profile: &Profile,
    route: Route,
    canonical_identity: &mut Option<Identity>,
) -> Result<(), String> {
    let response = &sample.response;
    let evidence = &response.x_onix_decode_evidence;
    if response.id.is_empty()
        || response.choices.len() != EXPECTED_CHOICE_COUNT
        || response.usage.completion_tokens != profile.workload.generated_tokens
        || evidence.generated_token_ids.len() != profile.workload.generated_tokens
    {
        return Err(format!(
            "{} has invalid response cardinality",
            sample.path.display()
        ));
    }
    if evidence.simulated || evidence.physical_pcie_device_ids != EXPECTED_DEVICE_IDS {
        return Err(format!(
            "{} is not physical P150x2 evidence",
            sample.path.display()
        ));
    }
    if evidence.batch_width != profile.workload.width || evidence.decode_path != DECODE_PATH {
        return Err(format!("{} has route drift", sample.path.display()));
    }
    match route {
        Route::Recapture => {
            if evidence.admission_eligible
                || evidence.measurement_mode != RECAPTURE_MEASUREMENT_MODE
                || evidence.trace_reuse_enabled
                || evidence.trace_reused
                || evidence.trace_capture_count == ZERO_CAPTURE_COUNT
                || evidence.trace_capture_seconds <= 0.0
            {
                return Err(format!(
                    "{} is not valid recapture evidence",
                    sample.path.display()
                ));
            }
        }
        Route::Reuse => {
            if !evidence.admission_eligible
                || evidence.measurement_mode != REUSE_MEASUREMENT_MODE
                || evidence.profile_id != profile.profile_id
                || !evidence.trace_reuse_enabled
                || !evidence.trace_reused
                || evidence.trace_capture_count != ZERO_CAPTURE_COUNT
                || evidence.trace_capture_seconds != 0.0
                || evidence.trace_transition_reason.is_some()
            {
                return Err(format!(
                    "{} is not valid reuse evidence",
                    sample.path.display()
                ));
            }
        }
    }
    if profile.admission.require_clean_reset && !evidence.clean_reset {
        return Err(format!("{} has incomplete cleanup", sample.path.display()));
    }
    if profile.admission.require_stable_binding && !evidence.trace_binding_stable {
        return Err(format!("{} has trace binding drift", sample.path.display()));
    }
    if evidence.resident_trace_count != profile.admission.maximum_resident_trace_count {
        return Err(format!("{} has trace-count drift", sample.path.display()));
    }
    let scalars = [
        evidence.cleanup_seconds,
        evidence.decode_tokens_per_second,
        evidence.request_wall_seconds,
        evidence.tail_token_latency_seconds,
        evidence.trace_capture_seconds,
        evidence.trace_prepare_seconds,
        evidence.ttft_seconds,
    ];
    if scalars
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || evidence.decode_tokens_per_second <= 0.0
        || evidence.request_wall_seconds <= 0.0
        || evidence.ttft_seconds <= 0.0
        || evidence.token_latency_seconds.is_empty()
        || evidence
            .token_latency_seconds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "{} has invalid timing evidence",
            sample.path.display()
        ));
    }
    let identity = Identity {
        model: evidence.model_identity.clone(),
        package: evidence.package_identity.clone(),
        runtime: evidence.runtime_identity.clone(),
        devices: evidence.physical_pcie_device_ids.clone(),
    };
    if identity.model.is_empty() || identity.package.is_empty() || identity.runtime.is_empty() {
        return Err(format!("{} has an empty identity", sample.path.display()));
    }
    match canonical_identity {
        Some(canonical) if canonical != &identity => {
            return Err(format!("{} has identity drift", sample.path.display()));
        }
        Some(_) => {}
        None => *canonical_identity = Some(identity),
    }
    Ok(())
}

fn validate_state_slices(roots: &[Root], profile: &Profile) -> Result<(), String> {
    for root in roots {
        for measured_run in 0..profile.workload.measured_runs {
            let mut observed = BTreeSet::new();
            for prompt in 0..profile.workload.prompts.len() {
                let index = prompt
                    .checked_mul(profile.workload.measured_runs)
                    .and_then(|base| base.checked_add(measured_run))
                    .ok_or("state-slice sample index overflow")?;
                let state_slice = root.samples[index]
                    .response
                    .x_onix_decode_evidence
                    .state_slice;
                if state_slice >= profile.workload.width || !observed.insert(state_slice) {
                    return Err(format!(
                        "{} has a state-slice collision or gap",
                        root.path.display()
                    ));
                }
            }
            let expected: BTreeSet<usize> = (0..profile.workload.width).collect();
            if observed != expected {
                return Err(format!(
                    "{} does not use one active prefix",
                    root.path.display()
                ));
            }
        }
    }
    Ok(())
}

fn flatten(roots: &[Root]) -> Vec<Sample> {
    roots
        .iter()
        .flat_map(|root| root.samples.iter().cloned())
        .collect()
}

fn output_tokens(root: &Root) -> Vec<Vec<i64>> {
    root.samples
        .iter()
        .map(|sample| {
            sample
                .response
                .x_onix_decode_evidence
                .generated_token_ids
                .clone()
        })
        .collect()
}

fn outputs(root: &Root) -> Vec<String> {
    root.samples
        .iter()
        .map(|sample| sample.response.choices[0].text.clone())
        .collect()
}

fn metrics(samples: &[Sample]) -> Result<Metrics, String> {
    let mut result = Metrics::default();
    for sample in samples {
        let evidence = &sample.response.x_onix_decode_evidence;
        result.wall.push(evidence.request_wall_seconds);
        result.decode_rate.push(evidence.decode_tokens_per_second);
        result.ttft.push(evidence.ttft_seconds);
        result.tail.push(evidence.tail_token_latency_seconds);
        result.trace_prepare.push(evidence.trace_prepare_seconds);
        result.cleanup.push(evidence.cleanup_seconds);
    }
    if result.wall.is_empty() {
        return Err("metrics are empty".to_owned());
    }
    Ok(result)
}

fn summarize(metrics: &Metrics) -> Result<MetricSummary, String> {
    Ok(MetricSummary {
        median_request_wall_seconds: median(&metrics.wall)?,
        median_decode_tokens_per_second: median(&metrics.decode_rate)?,
        median_ttft_seconds: median(&metrics.ttft)?,
        median_tail_token_latency_seconds: median(&metrics.tail)?,
        median_trace_prepare_seconds: median(&metrics.trace_prepare)?,
        median_cleanup_seconds: median(&metrics.cleanup)?,
    })
}

fn duplicate_ratio(roots: &[Root]) -> Result<f64, String> {
    if roots.len() != EXPECTED_ROOTS_PER_ROUTE {
        return Err("duplicate ratio needs exactly two roots".to_owned());
    }
    let first = summarize(&metrics(&roots[0].samples)?)?;
    let second = summarize(&metrics(&roots[1].samples)?)?;
    let ratios = [
        ratio(
            first.median_request_wall_seconds,
            second.median_request_wall_seconds,
        )?,
        ratio(
            first.median_decode_tokens_per_second,
            second.median_decode_tokens_per_second,
        )?,
        ratio(first.median_ttft_seconds, second.median_ttft_seconds)?,
        ratio(
            first.median_tail_token_latency_seconds,
            second.median_tail_token_latency_seconds,
        )?,
    ];
    ratios
        .into_iter()
        .max_by(f64::total_cmp)
        .ok_or("duplicate ratio is empty".to_owned())
}

fn ratio(first: f64, second: f64) -> Result<f64, String> {
    if !first.is_finite() || !second.is_finite() || first <= 0.0 || second <= 0.0 {
        return Err("ratio input is invalid".to_owned());
    }
    Ok(first.max(second) / first.min(second))
}

fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("median input is empty or non-finite".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / MEDIAN_DIVISOR;
    if sorted.len() % MEDIAN_DIVISOR == 0 {
        Ok((sorted[middle - 1] + sorted[middle]) / MEDIAN_DIVISOR_FLOAT)
    } else {
        Ok(sorted[middle])
    }
}

fn root_hashes(route: &str, roots: &[Root]) -> Vec<Value> {
    roots
        .iter()
        .enumerate()
        .map(|(index, root)| {
            json!({
                "root": format!("{route}-{}", index + 1),
                "samples": root.samples.iter().map(|sample| json!({
                    "path": sample.path.file_name(),
                    "blake3": sample.hash,
                })).collect::<Vec<_>>(),
            })
        })
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn self_test() -> Result<(), String> {
    let root = env::temp_dir().join(format!("qwen36-trace-reuse-receipt-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
    }
    let recapture_one = root.join("recapture-one");
    let recapture_two = root.join("recapture-two");
    let reuse_one = root.join("reuse-one");
    let reuse_two = root.join("reuse-two");
    for directory in [&recapture_one, &recapture_two, &reuse_one, &reuse_two] {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    }
    let profile_path = root.join("profile.json");
    let output_path = root.join("receipt.json");
    let profile = self_test_profile();
    fs::write(
        &profile_path,
        serde_json::to_vec_pretty(&profile).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_self_test_root(&recapture_one, Route::Recapture, false)?;
    write_self_test_root(&recapture_two, Route::Recapture, false)?;
    write_self_test_root(&reuse_one, Route::Reuse, false)?;
    write_self_test_root(&reuse_two, Route::Reuse, false)?;
    let config = Config {
        profile: profile_path,
        recapture_roots: vec![recapture_one.clone(), recapture_two.clone()],
        reuse_roots: vec![reuse_one.clone(), reuse_two.clone()],
        execution_order: EXPECTED_EXECUTION_ORDER
            .into_iter()
            .map(str::to_owned)
            .collect(),
        output: output_path,
    };
    let valid = build_receipt(&config)?;
    if valid["admission"]["admitted"] != json!(true) {
        return Err("positive trace-reuse fixture was rejected".to_owned());
    }

    write_self_test_root(&reuse_one, Route::Reuse, true)?;
    let trace_error =
        build_receipt(&config).expect_err("negative trace fixture unexpectedly produced a receipt");
    if !trace_error.contains("not valid reuse evidence") {
        return Err(format!("wrong negative trace diagnostic: {trace_error}"));
    }

    write_self_test_root(&reuse_one, Route::Reuse, false)?;
    let first_reuse_response = reuse_one.join(SELF_TEST_FIRST_RESPONSE_NAME);
    let mut mismatched: Value = read_json(&first_reuse_response)?;
    mismatched["x_onix_decode_evidence"]["generated_token_ids"][0] =
        json!(SELF_TEST_MISMATCH_TOKEN);
    fs::write(
        &first_reuse_response,
        serde_json::to_vec_pretty(&mismatched).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mismatch = build_receipt(&config)?;
    if mismatch["admission"]["reasons"]
        .as_array()
        .is_none_or(|reasons| !reasons.contains(&json!("token-parity-failed")))
    {
        return Err("token mismatch did not reject trace reuse".to_owned());
    }
    fs::remove_dir_all(root).map_err(|error| error.to_string())?;
    Ok(())
}

fn self_test_profile() -> Value {
    json!({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": "trace-reuse-self-test",
        "physically_admitted": false,
        "identities": {
            "authority_blake3": "a".repeat(BLAKE3_LENGTH),
            "checkpoint_revision": "b".repeat(REVISION_LENGTH),
            "model_source_revision": "c".repeat(REVISION_LENGTH),
            "receipt_blake3": NOT_MEASURED,
            "tokenizer_revision": "d".repeat(REVISION_LENGTH),
        },
        "workload": {
            "width": SELF_TEST_WIDTH,
            "execution_order": EXPECTED_EXECUTION_ORDER,
            "generated_tokens": SELF_TEST_GENERATED_TOKENS,
            "measured_runs": SELF_TEST_MEASURED_RUNS,
            "prompts": (0..SELF_TEST_PROMPT_COUNT).map(|index| format!("prompt-{index}")).collect::<Vec<_>>(),
            "warmup_runs": 1,
        },
        "admission": {
            "maximum_duplicate_root_ratio": SELF_TEST_MAXIMUM_DUPLICATE_RATIO,
            "maximum_resident_trace_count": MAXIMUM_RESIDENT_TRACE_COUNT,
            "maximum_reused_trace_prepare_seconds": SELF_TEST_MAXIMUM_REUSE_PREPARE_SECONDS,
            "maximum_tail_latency_ratio": SELF_TEST_MAXIMUM_TAIL_RATIO,
            "maximum_ttft_ratio": SELF_TEST_MAXIMUM_TTFT_RATIO,
            "minimum_decode_rate_retention_ratio": SELF_TEST_MINIMUM_DECODE_RETENTION,
            "minimum_request_wall_speedup_ratio": SELF_TEST_MINIMUM_WALL_SPEEDUP,
            "require_clean_reset": true,
            "require_exact_token_parity": true,
            "require_physical_evidence": true,
            "require_stable_binding": true,
            "require_zero_measured_recaptures": true,
        },
    })
}

fn write_self_test_root(path: &Path, route: Route, invalid_reuse: bool) -> Result<(), String> {
    for prompt in 0..SELF_TEST_PROMPT_COUNT {
        let response = self_test_response(prompt, route, invalid_reuse && prompt == 0);
        let name = format!(
            "{prompt:0width$}-measured-0.json",
            width = SELF_TEST_FILE_INDEX_WIDTH
        );
        fs::write(
            path.join(name),
            serde_json::to_vec_pretty(&response).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn self_test_response(prompt: usize, route: Route, invalid_reuse: bool) -> Value {
    let reuse = matches!(route, Route::Reuse);
    let token_base = SELF_TEST_TOKEN_BASE + i64::try_from(prompt).expect("prompt fits i64");
    json!({
        "id": format!("response-{prompt}"),
        "choices": [{"text": format!("text-{prompt}")}],
        "usage": {"completion_tokens": SELF_TEST_GENERATED_TOKENS},
        "x_onix_decode_evidence": {
            "admission_eligible": reuse,
            "batch_width": SELF_TEST_WIDTH,
            "clean_reset": true,
            "cleanup_seconds": SELF_TEST_CLEANUP_SECONDS,
            "decode_path": DECODE_PATH,
            "decode_tokens_per_second": SELF_TEST_DECODE_RATE,
            "generated_token_ids": [token_base, token_base + SELF_TEST_TOKEN_STRIDE],
            "measurement_mode": if reuse { REUSE_MEASUREMENT_MODE } else { RECAPTURE_MEASUREMENT_MODE },
            "model_identity": "model",
            "package_identity": "package",
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "profile_id": if reuse { "trace-reuse-self-test" } else { "decode-profile" },
            "request_wall_seconds": if reuse { SELF_TEST_REUSE_WALL_SECONDS } else { SELF_TEST_RECAPTURE_WALL_SECONDS },
            "resident_trace_count": MAXIMUM_RESIDENT_TRACE_COUNT,
            "runtime_identity": "runtime",
            "simulated": false,
            "state_slice": prompt,
            "tail_token_latency_seconds": SELF_TEST_TAIL_SECONDS,
            "token_latency_seconds": [SELF_TEST_TAIL_SECONDS],
            "trace_binding_stable": true,
            "trace_capture_count": if reuse { ZERO_CAPTURE_COUNT } else { 1 },
            "trace_capture_seconds": if reuse { 0.0 } else { SELF_TEST_RECAPTURE_CAPTURE_SECONDS },
            "trace_prepare_seconds": if reuse { SELF_TEST_REUSE_PREPARE_SECONDS } else { SELF_TEST_RECAPTURE_PREPARE_SECONDS },
            "trace_reuse_enabled": reuse,
            "trace_reused": reuse && !invalid_reuse,
            "trace_transition_reason": if reuse { Value::Null } else { json!("reuse-disabled") },
            "ttft_seconds": SELF_TEST_TTFT_SECONDS,
        },
    })
}
