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

const PROGRAM_NAME: &str = "qwen36-batch-receipt.rs";
const PROFILE_FLAG: &str = "--profile";
const SINGLE_ROOT_FLAG: &str = "--single-response-root";
const BATCH_ROOT_FLAG: &str = "--batch-response-root";
const EXECUTION_ORDER_FLAG: &str = "--execution-order";
const OUTPUT_FLAG: &str = "--output";
const SELF_TEST_FLAG: &str = "--self-test";
const MEASURED_MARKER: &str = "-measured-";
const JSON_EXTENSION: &str = "json";
const MEASUREMENT_MODE: &str = "batch-admission";
const DECODE_PATH: &str = "batched-traced";
const EXPECTED_ROOTS_PER_ROUTE: usize = 2;
const EXPECTED_CHOICE_COUNT: usize = 1;
const EXPECTED_DEVICE_COUNT: usize = 2;
const EXPECTED_EXECUTION_ORDER_COUNT: usize = 4;
const EXPECTED_DEVICE_IDS: [i64; EXPECTED_DEVICE_COUNT] = [0, 1];
const EXPECTED_EXECUTION_ORDER: [&str; EXPECTED_EXECUTION_ORDER_COUNT] =
    ["single", "batch", "batch", "single"];
const FLAG_ARGUMENT_WIDTH: usize = 2;
const MEDIAN_DIVISOR: usize = 2;
const MEDIAN_DIVISOR_FLOAT: f64 = 2.0;
const PHASE_TOLERANCE_SECONDS: f64 = 1.0e-6;
const SUCCESS_EXIT_CODE: u8 = 0;
const FAILURE_EXIT_CODE: u8 = 1;
const SELF_TEST_BASELINE_WIDTH: usize = 1;
const SELF_TEST_CANDIDATE_WIDTH: usize = 4;
const SELF_TEST_B8_CANDIDATE_WIDTH: usize = 8;
const ALLOWED_CANDIDATE_WIDTHS: [usize; 2] =
    [SELF_TEST_CANDIDATE_WIDTH, SELF_TEST_B8_CANDIDATE_WIDTH];
const SELF_TEST_PROMPT_COUNT: usize = 4;
const SELF_TEST_MEASURED_RUNS: usize = 1;
const SELF_TEST_GENERATED_TOKENS: usize = 2;
const SELF_TEST_BASELINE_RATE: f64 = 10.0;
const SELF_TEST_CANDIDATE_RATE: f64 = 8.0;
const SELF_TEST_CANDIDATE_AGGREGATE_RATE: f64 = 32.0;
const SELF_TEST_BASELINE_TTFT: f64 = 1.0;
const SELF_TEST_CANDIDATE_TTFT: f64 = 4.0;
const SELF_TEST_BASELINE_TAIL: f64 = 0.1;
const SELF_TEST_CANDIDATE_TAIL: f64 = 0.12;
const SELF_TEST_TRACE_PREPARE: f64 = 0.2;
const ROUTE_COUNT_DIAGNOSTIC: &str = "exactly two single and two batch roots are required";
const ARGUMENT_ORDER_DIAGNOSTIC: &str = "execution order must be single,batch,batch,single";
const PROFILE_ORDER_DIAGNOSTIC: &str = "profile execution order differs from the command";
const SAMPLE_COUNT_DIAGNOSTIC: &str = "measured samples, expected";
const SELF_TEST_STATE_INSTALL: f64 = 0.3;
const SELF_TEST_CLEANUP: f64 = 0.01;
const SELF_TEST_MAXIMUM_TTFT_RATIO: f64 = 4.5;
const SELF_TEST_MAXIMUM_TAIL_RATIO: f64 = 1.5;
const SELF_TEST_MINIMUM_AGGREGATE_SPEEDUP: f64 = 3.0;
const SELF_TEST_MINIMUM_RETENTION: f64 = 0.75;
const SELF_TEST_B8_PROMPT_COUNT: usize = 8;
const SELF_TEST_B8_AGGREGATE_RATE: f64 = 64.0;
const SELF_TEST_B8_MINIMUM_AGGREGATE_SPEEDUP: f64 = 6.0;
const SELF_TEST_B8_MAXIMUM_TTFT_RATIO: f64 = 9.0;
const SELF_TEST_TOKEN_BASE: i64 = 100;
const SELF_TEST_FILE_INDEX_WIDTH: usize = 2;
const SELF_TEST_FIRST_RESPONSE_NAME: &str = "00-measured-0.json";

#[derive(Debug)]
struct Config {
    profile: PathBuf,
    single_roots: Vec<PathBuf>,
    batch_roots: Vec<PathBuf>,
    execution_order: Vec<String>,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct Profile {
    profile_id: String,
    physically_admitted: bool,
    identities: ProfileIdentities,
    workload: Workload,
    admission: Admission,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProfileIdentities {
    authority_blake3: String,
    checkpoint_revision: String,
    model_source_revision: String,
    receipt_blake3: String,
    tokenizer_revision: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Workload {
    baseline_width: usize,
    candidate_width: usize,
    execution_order: Vec<String>,
    generated_tokens: usize,
    measured_runs: usize,
    prompts: Vec<String>,
    warmup_runs: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct Admission {
    maximum_tail_latency_ratio: f64,
    maximum_ttft_ratio: f64,
    minimum_aggregate_decode_speedup_ratio: f64,
    minimum_per_user_decode_retention_ratio: f64,
    require_active_prefix: bool,
    require_clean_reset: bool,
    require_distinct_state_slices: bool,
    require_exact_token_parity: bool,
    require_physical_evidence: bool,
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
    aggregate_decode_tokens_per_second: f64,
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
    prefill_first_token_seconds: f64,
    prefill_state_install_seconds: f64,
    profile_id: String,
    queue_seconds: f64,
    runtime_identity: String,
    simulated: bool,
    state_slice: usize,
    tail_token_latency_seconds: f64,
    token_latency_seconds: Vec<f64>,
    trace_prepare_seconds: f64,
    ttft_seconds: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    model: String,
    package: String,
    runtime: String,
    devices: Vec<i64>,
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
    aggregate_rates: Vec<f64>,
    per_user_rates: Vec<f64>,
    ttft: Vec<f64>,
    tail: Vec<f64>,
    trace_prepare: Vec<f64>,
    state_install: Vec<f64>,
    cleanup: Vec<f64>,
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
    let mut single_roots = Vec::new();
    let mut batch_roots = Vec::new();
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
            SINGLE_ROOT_FLAG => single_roots.push(PathBuf::from(value)),
            BATCH_ROOT_FLAG => batch_roots.push(PathBuf::from(value)),
            EXECUTION_ORDER_FLAG => {
                execution_order = Some(value.split(',').map(str::to_owned).collect())
            }
            OUTPUT_FLAG => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
        index += FLAG_ARGUMENT_WIDTH;
    }
    if single_roots.len() != EXPECTED_ROOTS_PER_ROUTE
        || batch_roots.len() != EXPECTED_ROOTS_PER_ROUTE
    {
        return Err("exactly two single and two batch roots are required".to_owned());
    }
    let execution_order = execution_order.ok_or("missing --execution-order")?;
    if execution_order != EXPECTED_EXECUTION_ORDER {
        return Err("execution order must be single,batch,batch,single".to_owned());
    }
    Ok(Config {
        profile: profile.ok_or("missing --profile")?,
        single_roots,
        batch_roots,
        execution_order,
        output: output.ok_or("missing --output")?,
    })
}

fn build_receipt(config: &Config) -> Result<Value, String> {
    let profile: Profile = read_json(&config.profile)?;
    validate_profile(&profile, config)?;
    let single_roots = read_roots(&config.single_roots)?;
    let batch_roots = read_roots(&config.batch_roots)?;
    let expected_samples = profile
        .workload
        .prompts
        .len()
        .checked_mul(profile.workload.measured_runs)
        .ok_or("sample count overflow")?;
    let mut identity = None;
    for root in &single_roots {
        validate_root(
            root,
            &profile,
            profile.workload.baseline_width,
            expected_samples,
            &mut identity,
        )?;
    }
    for root in &batch_roots {
        validate_root(
            root,
            &profile,
            profile.workload.candidate_width,
            expected_samples,
            &mut identity,
        )?;
    }
    validate_state_slices(&single_roots, &profile, false)?;
    validate_state_slices(&batch_roots, &profile, true)?;

    let single_samples = flatten(&single_roots);
    let batch_samples = flatten(&batch_roots);
    let token_parity = sample_tokens(&single_samples) == sample_tokens(&batch_samples);
    let text_parity = sample_text(&single_samples) == sample_text(&batch_samples);
    let single_metrics = metrics(&single_samples)?;
    let batch_metrics = metrics(&batch_samples)?;
    let aggregate_speedup =
        median(&batch_metrics.aggregate_rates)? / median(&single_metrics.aggregate_rates)?;
    let per_user_retention =
        median(&batch_metrics.per_user_rates)? / median(&single_metrics.per_user_rates)?;
    let ttft_ratio = median(&batch_metrics.ttft)? / median(&single_metrics.ttft)?;
    let tail_ratio = median(&batch_metrics.tail)? / median(&single_metrics.tail)?;

    let mut reasons = Vec::new();
    if profile.admission.require_exact_token_parity && !token_parity {
        reasons.push("token-parity-failed");
    }
    if !text_parity {
        reasons.push("text-parity-failed");
    }
    if aggregate_speedup < profile.admission.minimum_aggregate_decode_speedup_ratio {
        reasons.push("aggregate-decode-speedup-below-bound");
    }
    if per_user_retention < profile.admission.minimum_per_user_decode_retention_ratio {
        reasons.push("per-user-decode-retention-below-bound");
    }
    if ttft_ratio > profile.admission.maximum_ttft_ratio {
        reasons.push("ttft-ratio-above-bound");
    }
    if tail_ratio > profile.admission.maximum_tail_latency_ratio {
        reasons.push("tail-latency-ratio-above-bound");
    }
    let admitted = reasons.is_empty();
    let identity = identity.ok_or("response roots are empty")?;

    Ok(json!({
        "schema": "onix.qwen36.batch-admission-receipt.v1",
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
            "baseline_width": profile.workload.baseline_width,
            "candidate_width": profile.workload.candidate_width,
            "prompt_count": profile.workload.prompts.len(),
            "measured_runs": profile.workload.measured_runs,
            "generated_tokens": profile.workload.generated_tokens,
            "simulated": false,
        },
        "correctness": {
            "exact_token_parity": token_parity,
            "exact_text_parity": text_parity,
            "clean_reset": true,
            "active_prefix": true,
            "distinct_state_slices": true,
        },
        "metrics": {
            "single": metric_json(&single_metrics)?,
            "batch": metric_json(&batch_metrics)?,
            "aggregate_decode_speedup_ratio": aggregate_speedup,
            "per_user_decode_retention_ratio": per_user_retention,
            "ttft_ratio": ttft_ratio,
            "tail_latency_ratio": tail_ratio,
        },
        "admission": {
            "admitted": admitted,
            "reasons": reasons,
            "bounds": profile.admission,
            "single_stream_ttft_gate_unchanged": true,
        },
        "response_hashes": {
            "single": root_hashes("single", &single_roots),
            "batch": root_hashes("batch", &batch_roots),
        },
        "non_claims": [
            format!(
                "this receipt admits only candidate width {} on the recorded execution identity and does not admit any other width",
                profile.workload.candidate_width
            ),
            "this receipt does not admit arbitrary continuous batching".to_owned(),
            "this receipt does not replace the single-stream eager-versus-traced gate".to_owned(),
        ],
    }))
}

fn validate_profile(profile: &Profile, config: &Config) -> Result<(), String> {
    if profile.physically_admitted || profile.identities.receipt_blake3 != "not-measured" {
        return Err("input profile must be unadmitted and unmeasured".to_owned());
    }
    if profile.workload.execution_order != EXPECTED_EXECUTION_ORDER
        || config.execution_order != profile.workload.execution_order
    {
        return Err("profile execution order differs from the command".to_owned());
    }
    if profile.workload.baseline_width != SELF_TEST_BASELINE_WIDTH
        || !ALLOWED_CANDIDATE_WIDTHS.contains(&profile.workload.candidate_width)
    {
        return Err("profile widths are unsupported".to_owned());
    }
    if profile.workload.prompts.is_empty()
        || profile.workload.measured_runs == 0
        || profile.workload.generated_tokens == 0
        || profile.workload.warmup_runs == 0
    {
        return Err("profile workload is empty".to_owned());
    }
    if !profile.admission.require_active_prefix
        || !profile.admission.require_clean_reset
        || !profile.admission.require_distinct_state_slices
        || !profile.admission.require_exact_token_parity
        || !profile.admission.require_physical_evidence
    {
        return Err("profile weakens a required admission gate".to_owned());
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
    expected_width: usize,
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
        validate_sample(sample, profile, expected_width, canonical_identity)?;
    }
    Ok(())
}

fn validate_sample(
    sample: &Sample,
    profile: &Profile,
    expected_width: usize,
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
    if !evidence.admission_eligible || evidence.measurement_mode != MEASUREMENT_MODE {
        return Err(format!(
            "{} is not batch admission evidence",
            sample.path.display()
        ));
    }
    if evidence.simulated || evidence.physical_pcie_device_ids != EXPECTED_DEVICE_IDS {
        return Err(format!(
            "{} is not physical P150x2 evidence",
            sample.path.display()
        ));
    }
    if evidence.batch_width != expected_width
        || evidence.decode_path != DECODE_PATH
        || evidence.profile_id != profile.profile_id
    {
        return Err(format!(
            "{} has route or profile drift",
            sample.path.display()
        ));
    }
    if profile.admission.require_clean_reset && !evidence.clean_reset {
        return Err(format!("{} has incomplete cleanup", sample.path.display()));
    }
    let scalars = [
        evidence.aggregate_decode_tokens_per_second,
        evidence.cleanup_seconds,
        evidence.decode_tokens_per_second,
        evidence.prefill_first_token_seconds,
        evidence.prefill_state_install_seconds,
        evidence.queue_seconds,
        evidence.tail_token_latency_seconds,
        evidence.trace_prepare_seconds,
        evidence.ttft_seconds,
    ];
    if scalars
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || evidence.aggregate_decode_tokens_per_second <= 0.0
        || evidence.decode_tokens_per_second <= 0.0
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
    if (evidence.ttft_seconds - evidence.prefill_first_token_seconds).abs()
        > PHASE_TOLERANCE_SECONDS
    {
        return Err(format!(
            "{} has inconsistent TTFT phases",
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

fn validate_state_slices(roots: &[Root], profile: &Profile, candidate: bool) -> Result<(), String> {
    let width = if candidate {
        profile.workload.candidate_width
    } else {
        profile.workload.baseline_width
    };
    for root in roots {
        if candidate {
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
                    if state_slice >= width || !observed.insert(state_slice) {
                        return Err(format!(
                            "{} has a state-slice collision or gap",
                            root.path.display()
                        ));
                    }
                }
                let expected: BTreeSet<usize> = (0..width).collect();
                if observed != expected {
                    return Err(format!(
                        "{} does not use one active prefix",
                        root.path.display()
                    ));
                }
            }
        } else if root
            .samples
            .iter()
            .any(|sample| sample.response.x_onix_decode_evidence.state_slice != 0)
        {
            return Err(format!(
                "{} changed the single-stream state slice",
                root.path.display()
            ));
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

fn sample_tokens(samples: &[Sample]) -> Vec<Vec<i64>> {
    samples
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

fn sample_text(samples: &[Sample]) -> Vec<String> {
    samples
        .iter()
        .map(|sample| sample.response.choices[0].text.clone())
        .collect()
}

fn metrics(samples: &[Sample]) -> Result<Metrics, String> {
    let mut result = Metrics::default();
    for sample in samples {
        let evidence = &sample.response.x_onix_decode_evidence;
        result
            .aggregate_rates
            .push(evidence.aggregate_decode_tokens_per_second);
        result
            .per_user_rates
            .push(evidence.decode_tokens_per_second);
        result.ttft.push(evidence.ttft_seconds);
        result.tail.push(evidence.tail_token_latency_seconds);
        result.trace_prepare.push(evidence.trace_prepare_seconds);
        result
            .state_install
            .push(evidence.prefill_state_install_seconds);
        result.cleanup.push(evidence.cleanup_seconds);
    }
    if result.aggregate_rates.is_empty() {
        return Err("metrics are empty".to_owned());
    }
    Ok(result)
}

fn metric_json(metrics: &Metrics) -> Result<Value, String> {
    Ok(json!({
        "median_aggregate_decode_tokens_per_second": median(&metrics.aggregate_rates)?,
        "median_per_user_decode_tokens_per_second": median(&metrics.per_user_rates)?,
        "median_ttft_seconds": median(&metrics.ttft)?,
        "median_tail_token_latency_seconds": median(&metrics.tail)?,
        "median_trace_prepare_seconds": median(&metrics.trace_prepare)?,
        "median_prefill_state_install_seconds": median(&metrics.state_install)?,
        "median_cleanup_seconds": median(&metrics.cleanup)?,
    }))
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

struct SelfTestCase {
    width: usize,
    prompt_count: usize,
    candidate_per_user_rate: f64,
    candidate_aggregate_rate: f64,
    minimum_aggregate_speedup: f64,
    minimum_retention: f64,
    maximum_ttft_ratio: f64,
    maximum_tail_ratio: f64,
}

fn self_test() -> Result<(), String> {
    let cases = [
        SelfTestCase {
            width: SELF_TEST_CANDIDATE_WIDTH,
            prompt_count: SELF_TEST_PROMPT_COUNT,
            candidate_per_user_rate: SELF_TEST_CANDIDATE_RATE,
            candidate_aggregate_rate: SELF_TEST_CANDIDATE_AGGREGATE_RATE,
            minimum_aggregate_speedup: SELF_TEST_MINIMUM_AGGREGATE_SPEEDUP,
            minimum_retention: SELF_TEST_MINIMUM_RETENTION,
            maximum_ttft_ratio: SELF_TEST_MAXIMUM_TTFT_RATIO,
            maximum_tail_ratio: SELF_TEST_MAXIMUM_TAIL_RATIO,
        },
        SelfTestCase {
            width: SELF_TEST_B8_CANDIDATE_WIDTH,
            prompt_count: SELF_TEST_B8_PROMPT_COUNT,
            candidate_per_user_rate: SELF_TEST_CANDIDATE_RATE,
            candidate_aggregate_rate: SELF_TEST_B8_AGGREGATE_RATE,
            minimum_aggregate_speedup: SELF_TEST_B8_MINIMUM_AGGREGATE_SPEEDUP,
            minimum_retention: SELF_TEST_MINIMUM_RETENTION,
            maximum_ttft_ratio: SELF_TEST_B8_MAXIMUM_TTFT_RATIO,
            maximum_tail_ratio: SELF_TEST_MAXIMUM_TAIL_RATIO,
        },
    ];
    for case in &cases {
        run_self_test_case(case)?;
    }
    Ok(())
}

fn run_self_test_case(case: &SelfTestCase) -> Result<(), String> {
    let root = env::temp_dir().join(format!(
        "qwen36-batch-receipt-{}-{}",
        std::process::id(),
        case.width
    ));
    if root.exists() {
        fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
    }
    let single_one = root.join("single-one");
    let single_two = root.join("single-two");
    let batch_one = root.join("batch-one");
    let batch_two = root.join("batch-two");
    for directory in [&single_one, &single_two, &batch_one, &batch_two] {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    }
    let profile_path = root.join("profile.json");
    write_profile(&profile_path, case)?;
    for prompt in 0..case.prompt_count {
        let token =
            SELF_TEST_TOKEN_BASE + i64::try_from(prompt).map_err(|error| error.to_string())?;
        for directory in [&single_one, &single_two] {
            write_response(
                directory,
                case,
                prompt,
                SELF_TEST_BASELINE_WIDTH,
                0,
                token,
                false,
            )?;
        }
        for directory in [&batch_one, &batch_two] {
            write_response(directory, case, prompt, case.width, prompt, token, false)?;
        }
    }
    let config = Config {
        profile: profile_path.clone(),
        single_roots: vec![single_one.clone(), single_two.clone()],
        batch_roots: vec![batch_one.clone(), batch_two.clone()],
        execution_order: EXPECTED_EXECUTION_ORDER
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        output: root.join("receipt.json"),
    };
    let receipt = build_receipt(&config)?;
    if receipt["admission"]["admitted"] != json!(true) {
        return Err(format!(
            "positive width-{} fixture was not admitted",
            case.width
        ));
    }

    let mismatch_path = batch_one.join(SELF_TEST_FIRST_RESPONSE_NAME);
    let mut mismatch: Response = read_json(&mismatch_path)?;
    mismatch.x_onix_decode_evidence.generated_token_ids[0] += 1;
    fs::write(
        &mismatch_path,
        serde_json::to_vec(&mismatch).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let rejected = build_receipt(&config)?;
    if rejected["admission"]["admitted"] != json!(false)
        || !rejected["admission"]["reasons"]
            .as_array()
            .is_some_and(|reasons| reasons.contains(&json!("token-parity-failed")))
    {
        return Err(format!(
            "width-{} token mismatch did not reject admission",
            case.width
        ));
    }
    write_response(
        &batch_one,
        case,
        0,
        case.width,
        0,
        SELF_TEST_TOKEN_BASE,
        true,
    )?;
    let error = build_receipt(&config).expect_err("ineligible evidence was accepted");
    if !error.contains("not batch admission evidence") {
        return Err(format!("wrong ineligible diagnostic: {error}"));
    }

    // Negative: a reduced root count is rejected before any measurement.
    let mut reduced_arguments = vec![
        PROFILE_FLAG.to_owned(),
        profile_path.display().to_string(),
        SINGLE_ROOT_FLAG.to_owned(),
        single_one.display().to_string(),
        SINGLE_ROOT_FLAG.to_owned(),
        single_two.display().to_string(),
        BATCH_ROOT_FLAG.to_owned(),
        batch_one.display().to_string(),
        EXECUTION_ORDER_FLAG.to_owned(),
        EXPECTED_EXECUTION_ORDER.join(","),
        OUTPUT_FLAG.to_owned(),
        root.join("unused.json").display().to_string(),
    ];
    let error =
        parse_config(&reduced_arguments).expect_err("a reduced root count was accepted");
    require_diagnostic(&error, ROUTE_COUNT_DIAGNOSTIC, "reduced root count")?;

    // Negative: a changed order is rejected at the argument layer.
    let order_index = reduced_arguments
        .iter()
        .position(|argument| argument == EXECUTION_ORDER_FLAG)
        .ok_or("the self-test order flag is missing")?;
    reduced_arguments[order_index + 1] = ["single", "batch", "single", "batch"].join(",");
    reduced_arguments.push(BATCH_ROOT_FLAG.to_owned());
    reduced_arguments.push(batch_two.display().to_string());
    let error =
        parse_config(&reduced_arguments).expect_err("a changed order was accepted");
    require_diagnostic(&error, ARGUMENT_ORDER_DIAGNOSTIC, "changed order argument")?;

    // Negative: a changed order inside the profile is rejected before measurement.
    let mut reordered: Value = read_json(&profile_path)?;
    reordered["workload"]["execution_order"] =
        json!(["single", "batch", "single", "batch"]);
    let reordered_path = root.join("profile-reordered.json");
    fs::write(
        &reordered_path,
        serde_json::to_vec(&reordered).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let reordered_config = Config {
        profile: reordered_path,
        single_roots: config.single_roots.clone(),
        batch_roots: config.batch_roots.clone(),
        execution_order: config.execution_order.clone(),
        output: config.output.clone(),
    };
    let error = build_receipt(&reordered_config).expect_err("a reordered profile was accepted");
    require_diagnostic(&error, PROFILE_ORDER_DIAGNOSTIC, "changed order profile")?;

    // Negative: a shortened corpus cannot satisfy the frozen sample count.
    let mut shortened: Value = read_json(&profile_path)?;
    let prompts = shortened["workload"]["prompts"]
        .as_array_mut()
        .ok_or("the self-test profile carries no prompts")?;
    prompts.pop();
    let shortened_path = root.join("profile-short.json");
    fs::write(
        &shortened_path,
        serde_json::to_vec(&shortened).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let shortened_config = Config {
        profile: shortened_path,
        single_roots: config.single_roots.clone(),
        batch_roots: config.batch_roots.clone(),
        execution_order: config.execution_order.clone(),
        output: config.output.clone(),
    };
    let error = build_receipt(&shortened_config).expect_err("a shortened corpus was accepted");
    require_diagnostic(&error, SAMPLE_COUNT_DIAGNOSTIC, "shortened corpus")?;

    fs::remove_dir_all(root).map_err(|error| error.to_string())?;
    Ok(())
}

fn require_diagnostic(error: &str, needle: &str, label: &str) -> Result<(), String> {
    if error.contains(needle) {
        Ok(())
    } else {
        Err(format!("wrong {label} diagnostic: {error}"))
    }
}

fn write_profile(path: &Path, case: &SelfTestCase) -> Result<(), String> {
    let prompts: Vec<String> = (0..case.prompt_count)
        .map(|index| format!("prompt-{index}"))
        .collect();
    let profile = json!({
        "profile_id": "batch-profile-test",
        "physically_admitted": false,
        "identities": {
            "authority_blake3": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "checkpoint_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "model_source_revision": "cccccccccccccccccccccccccccccccccccccccc",
            "receipt_blake3": "not-measured",
            "tokenizer_revision": "dddddddddddddddddddddddddddddddddddddddd",
        },
        "workload": {
            "baseline_width": SELF_TEST_BASELINE_WIDTH,
            "candidate_width": case.width,
            "execution_order": EXPECTED_EXECUTION_ORDER,
            "generated_tokens": SELF_TEST_GENERATED_TOKENS,
            "measured_runs": SELF_TEST_MEASURED_RUNS,
            "prompts": prompts,
            "warmup_runs": 1,
        },
        "admission": {
            "maximum_tail_latency_ratio": case.maximum_tail_ratio,
            "maximum_ttft_ratio": case.maximum_ttft_ratio,
            "minimum_aggregate_decode_speedup_ratio": case.minimum_aggregate_speedup,
            "minimum_per_user_decode_retention_ratio": case.minimum_retention,
            "require_active_prefix": true,
            "require_clean_reset": true,
            "require_distinct_state_slices": true,
            "require_exact_token_parity": true,
            "require_physical_evidence": true,
        },
    });
    fs::write(
        path,
        serde_json::to_vec(&profile).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn write_response(
    root: &Path,
    case: &SelfTestCase,
    prompt: usize,
    width: usize,
    state_slice: usize,
    token: i64,
    ineligible: bool,
) -> Result<(), String> {
    let candidate = width == case.width;
    let per_user_rate = if candidate {
        case.candidate_per_user_rate
    } else {
        SELF_TEST_BASELINE_RATE
    };
    let aggregate_rate = if candidate {
        case.candidate_aggregate_rate
    } else {
        SELF_TEST_BASELINE_RATE
    };
    let ttft = if candidate {
        SELF_TEST_CANDIDATE_TTFT
    } else {
        SELF_TEST_BASELINE_TTFT
    };
    let tail = if candidate {
        SELF_TEST_CANDIDATE_TAIL
    } else {
        SELF_TEST_BASELINE_TAIL
    };
    let response = json!({
        "id": format!("response-{prompt}-{width}"),
        "choices": [{"text": format!("token-{token}")}],
        "usage": {"completion_tokens": SELF_TEST_GENERATED_TOKENS},
        "x_onix_decode_evidence": {
            "admission_eligible": !ineligible,
            "aggregate_decode_tokens_per_second": aggregate_rate,
            "batch_width": width,
            "clean_reset": true,
            "cleanup_seconds": SELF_TEST_CLEANUP,
            "decode_path": DECODE_PATH,
            "decode_tokens_per_second": per_user_rate,
            "generated_token_ids": [token, token + 1],
            "measurement_mode": if ineligible { "physical-batched-service" } else { MEASUREMENT_MODE },
            "model_identity": "model",
            "package_identity": "package",
            "physical_pcie_device_ids": EXPECTED_DEVICE_IDS,
            "prefill_first_token_seconds": ttft,
            "prefill_state_install_seconds": SELF_TEST_STATE_INSTALL,
            "profile_id": "batch-profile-test",
            "queue_seconds": 0.0,
            "runtime_identity": "runtime",
            "simulated": false,
            "state_slice": state_slice,
            "tail_token_latency_seconds": tail,
            "token_latency_seconds": [tail],
            "trace_prepare_seconds": SELF_TEST_TRACE_PREPARE,
            "ttft_seconds": ttft,
        },
    });
    let path = root.join(format!(
        "{prompt:0width$}-measured-0.json",
        width = SELF_TEST_FILE_INDEX_WIDTH
    ));
    fs::write(
        path,
        serde_json::to_vec(&response).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}
