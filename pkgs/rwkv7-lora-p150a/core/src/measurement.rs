//! Pure measurement-profile, statistics, and receipt admission.
//!
//! r[impl tenstorrent.harness.measurement_profile]
//! r[impl tenstorrent.harness.stage_breakdown]
//! r[impl tenstorrent.harness.speedup_ratio]
//! r[impl tenstorrent.harness.machine_facts]
//! r[impl tenstorrent.harness.repeat_stats]

use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::{Rejection, blake3_identity, is_hash_hex};

const MEASUREMENT_SCHEMA_VERSION: u64 = 1;
const MEASUREMENT_RECEIPT_KIND: &str = "rwkv7_lora_measurement";
const MINIMUM_REPEAT_COUNT: u64 = 3;
const PERCENTILE_MEDIAN: f64 = 0.50;
const PERCENTILE_P99: f64 = 0.99;
const STAGE_SUM_TOLERANCE_SECONDS: f64 = 0.000_001;

const MEASUREMENT_PROFILE_FIELDS: [&str; 5] = [
    "profile_id",
    "parity_profile_id",
    "repeat_count",
    "metric_set",
    "machine_fact_set",
];
const REQUIRED_METRICS: [&str; 2] = ["stage_time_seconds", "loss"];
const REQUIRED_MACHINE_FACTS: [&str; 5] = [
    "board_identity",
    "firmware_version",
    "mesh_descriptor_identity",
    "pcie_link",
    "power_state",
];
const REQUIRED_STAGES: [&str; 5] = ["launch", "compile", "step", "checkpoint", "teardown"];
const RAW_MEASUREMENT_FIELDS: [&str; 4] = [
    "stage_times_seconds",
    "cpu_baseline",
    "machine_facts",
    "runs",
];
const RECEIPT_FIELDS: [&str; 10] = [
    "schema_version",
    "kind",
    "profile_identity",
    "measurement_profile_id",
    "stage_times_seconds",
    "cpu_baseline",
    "machine_facts",
    "runs",
    "statistics",
    "speedup_ratio",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementProfile {
    pub profile_id: String,
    pub repeat_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub standard_deviation: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementAdmission {
    pub identity_blake3: String,
    pub measurement_profile_id: String,
    pub repeats: u64,
    pub speedup_ratio: f64,
    pub time_seconds: Distribution,
    pub loss: Distribution,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, Rejection> {
    value.as_object().ok_or_else(|| {
        Rejection::new(
            "malformed_measurement",
            format!("field '{name}' is not an object"),
        )
    })
}

fn exact_fields(value: &Value, name: &str, required_fields: &[&str]) -> Result<(), Rejection> {
    let fields = object(value, name)?;
    let actual = fields.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = required_fields.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(Rejection::new(
            "unexpected_measurement_field",
            format!("field '{name}' has fields {actual:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

fn required_field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Rejection> {
    value.get(key).ok_or_else(|| {
        Rejection::new(
            "missing_measurement_field",
            format!("missing measurement field '{key}'"),
        )
    })
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    required_field(value, key)?.as_str().ok_or_else(|| {
        Rejection::new(
            "malformed_measurement",
            format!("measurement field '{key}' is not a string"),
        )
    })
}

fn required_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    required_field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "malformed_measurement",
            format!("measurement field '{key}' is not an unsigned integer"),
        )
    })
}

fn finite_number(value: &Value, key: &str, is_positive: bool) -> Result<f64, Rejection> {
    let number = required_field(value, key)?.as_f64().ok_or_else(|| {
        Rejection::new(
            "invalid_metric",
            format!("metric '{key}' is not a finite number"),
        )
    })?;
    if !number.is_finite() {
        return Err(Rejection::new(
            "invalid_metric",
            format!("metric '{key}' is not finite"),
        ));
    }
    if is_positive && number <= 0.0 {
        return Err(Rejection::new(
            "invalid_metric",
            format!("metric '{key}' must be positive, got {number}"),
        ));
    }
    if !is_positive && number < 0.0 {
        return Err(Rejection::new(
            "invalid_metric",
            format!("metric '{key}' must be non-negative, got {number}"),
        ));
    }
    Ok(number)
}

fn exact_string_array(value: &Value, key: &str, expected: &[&str]) -> Result<(), Rejection> {
    let entries = required_field(value, key)?.as_array().ok_or_else(|| {
        Rejection::new(
            "malformed_measurement_profile",
            format!("measurement profile field '{key}' is not an array"),
        )
    })?;
    let actual = entries
        .iter()
        .map(|entry| entry.as_str())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Rejection::new(
                "malformed_measurement_profile",
                format!("measurement profile field '{key}' contains a non-string value"),
            )
        })?;
    if actual != expected {
        return Err(Rejection::new(
            "bad_measurement_profile",
            format!("measurement profile field '{key}' is {actual:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

pub fn admit_measurement_profile(profile: &Value) -> Result<MeasurementProfile, Rejection> {
    let parity_profile = required_field(profile, "parity_profile")?;
    let parity_profile_id = required_string(parity_profile, "profile_id")?;
    let mut projected_parity = object(profile, "measurement_document")?.clone();
    projected_parity.remove("measurement_profile");
    projected_parity.remove("parity_profile");
    if Value::Object(projected_parity) != *parity_profile {
        return Err(Rejection::new(
            "parity_profile_mismatch",
            "measurement profile changes one or more parity fields",
        ));
    }
    let measurement = required_field(profile, "measurement_profile")?;
    exact_fields(
        measurement,
        "measurement_profile",
        &MEASUREMENT_PROFILE_FIELDS,
    )?;

    let measurement_profile_id = required_string(measurement, "profile_id")?;
    if measurement_profile_id.is_empty() || measurement_profile_id == parity_profile_id {
        return Err(Rejection::new(
            "bad_measurement_profile",
            "measurement profile identity must be non-empty and distinct from parity",
        ));
    }
    let bound_parity_profile_id = required_string(measurement, "parity_profile_id")?;
    if bound_parity_profile_id != parity_profile_id {
        return Err(Rejection::new(
            "parity_profile_mismatch",
            format!(
                "measurement profile binds parity profile '{bound_parity_profile_id}', expected '{parity_profile_id}'"
            ),
        ));
    }

    let repeat_count = required_u64(measurement, "repeat_count")?;
    if repeat_count < MINIMUM_REPEAT_COUNT {
        return Err(Rejection::new(
            "repeat_count_below_bound",
            format!(
                "measurement repeat_count is {repeat_count}, minimum is {MINIMUM_REPEAT_COUNT}"
            ),
        ));
    }
    exact_string_array(measurement, "metric_set", &REQUIRED_METRICS)?;
    exact_string_array(measurement, "machine_fact_set", &REQUIRED_MACHINE_FACTS)?;

    Ok(MeasurementProfile {
        profile_id: measurement_profile_id.to_string(),
        repeat_count,
    })
}

fn stage_times(raw: &Value) -> Result<Value, Rejection> {
    let input = required_field(raw, "stage_times_seconds")?;
    let mut output = Map::new();
    let mut end_to_end_seconds = 0.0;
    for stage in REQUIRED_STAGES {
        let seconds = finite_number(input, stage, false)?;
        end_to_end_seconds += seconds;
        output.insert(stage.to_string(), json!(seconds));
    }
    let reported_end_to_end = finite_number(input, "end_to_end", false)?;
    let sum_deviation = (reported_end_to_end - end_to_end_seconds).abs();
    if sum_deviation > STAGE_SUM_TOLERANCE_SECONDS {
        return Err(Rejection::new(
            "stage_time_sum_mismatch",
            format!("stage sum is {end_to_end_seconds}, end_to_end is {reported_end_to_end}"),
        ));
    }
    output.insert("end_to_end".to_string(), json!(reported_end_to_end));
    Ok(Value::Object(output))
}

pub fn bind_machine_facts(values: &Value) -> Result<Value, Rejection> {
    exact_fields(values, "machine_fact_values", &REQUIRED_MACHINE_FACTS)?;
    let mut output = Map::new();
    for fact_name in REQUIRED_MACHINE_FACTS {
        let fact_value = required_string(values, fact_name)?;
        if fact_value.is_empty() {
            return Err(Rejection::new(
                "missing_machine_fact",
                format!("machine fact '{fact_name}' is empty"),
            ));
        }
        output.insert(
            fact_name.to_string(),
            json!({
                "value": fact_value,
                "identity_blake3": blake3_identity(&json!({
                    "name": fact_name,
                    "value": fact_value,
                })),
            }),
        );
    }
    Ok(Value::Object(output))
}

fn machine_facts(raw: &Value) -> Result<Value, Rejection> {
    let input = required_field(raw, "machine_facts")?;
    exact_fields(input, "machine_facts", &REQUIRED_MACHINE_FACTS)?;
    let mut values = Map::new();
    for fact_name in REQUIRED_MACHINE_FACTS {
        let fact = required_field(input, fact_name)?;
        exact_fields(fact, fact_name, &["value", "identity_blake3"])?;
        let fact_value = required_string(fact, "value")?;
        let identity = required_string(fact, "identity_blake3")?;
        if !is_hash_hex(identity) {
            return Err(Rejection::new(
                "bad_machine_fact_identity",
                format!("machine fact '{fact_name}' has an invalid BLAKE3 identity"),
            ));
        }
        values.insert(fact_name.to_string(), json!(fact_value));
    }
    let expected = bind_machine_facts(&Value::Object(values))?;
    if input != &expected {
        return Err(Rejection::new(
            "machine_fact_identity_mismatch",
            "one or more machine fact identities do not bind their values",
        ));
    }
    Ok(input.clone())
}

fn cpu_baseline(raw: &Value) -> Result<(Value, f64), Rejection> {
    let baseline = required_field(raw, "cpu_baseline")?;
    exact_fields(
        baseline,
        "cpu_baseline",
        &["identity_blake3", "step_seconds"],
    )?;
    let identity = required_string(baseline, "identity_blake3")?;
    if !is_hash_hex(identity) {
        return Err(Rejection::new(
            "missing_cpu_baseline",
            "CPU baseline identity is not a BLAKE3 hash",
        ));
    }
    let step_seconds = finite_number(baseline, "step_seconds", true)?;
    Ok((baseline.clone(), step_seconds))
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    assert!(!sorted.is_empty());
    assert!(percentile > 0.0);
    assert!(percentile <= 1.0);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len().saturating_sub(1));
    sorted[index]
}

fn distribution(values: &[f64]) -> Distribution {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    let sample_count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / sample_count;
    let variance = values
        .iter()
        .map(|value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>()
        / sample_count;
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let result = Distribution {
        mean,
        p50: percentile(&sorted, PERCENTILE_MEDIAN),
        p99: percentile(&sorted, PERCENTILE_P99),
        standard_deviation: variance.sqrt(),
    };
    assert!(result.mean.is_finite());
    assert!(result.standard_deviation.is_finite());
    result
}

fn distribution_json(distribution: &Distribution) -> Value {
    json!({
        "mean": distribution.mean,
        "p50": distribution.p50,
        "p99": distribution.p99,
        "standard_deviation": distribution.standard_deviation,
    })
}

fn runs(raw: &Value, repeat_count: u64) -> Result<(Value, Distribution, Distribution), Rejection> {
    let entries = required_field(raw, "runs")?.as_array().ok_or_else(|| {
        Rejection::new("malformed_measurement", "measurement runs is not an array")
    })?;
    let actual_repeat_count = u64::try_from(entries.len()).map_err(|_| {
        Rejection::new(
            "repeat_count_overflow",
            "measurement run count does not fit in u64",
        )
    })?;
    if actual_repeat_count != repeat_count {
        return Err(Rejection::new(
            "repeat_count_mismatch",
            format!("measurement has {actual_repeat_count} runs, expected {repeat_count}"),
        ));
    }

    let mut time_values = Vec::with_capacity(entries.len());
    let mut loss_values = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        exact_fields(entry, "run", &["time_seconds", "loss"])?;
        let time_seconds = finite_number(entry, "time_seconds", true).map_err(|error| {
            Rejection::new(error.code, format!("run {index}: {}", error.detail))
        })?;
        let loss = finite_number(entry, "loss", false).map_err(|error| {
            Rejection::new(error.code, format!("run {index}: {}", error.detail))
        })?;
        time_values.push(time_seconds);
        loss_values.push(loss);
    }
    Ok((
        Value::Array(entries.clone()),
        distribution(&time_values),
        distribution(&loss_values),
    ))
}

pub fn summarize_measurement(raw: &Value, profile: &Value) -> Result<Value, Rejection> {
    exact_fields(raw, "raw_measurement", &RAW_MEASUREMENT_FIELDS)?;
    let measurement_profile = admit_measurement_profile(profile)?;
    let stage_times_seconds = stage_times(raw)?;
    let (cpu_baseline, cpu_step_seconds) = cpu_baseline(raw)?;
    let machine_facts = machine_facts(raw)?;
    let (runs, time_seconds, loss) = runs(raw, measurement_profile.repeat_count)?;
    let speedup_ratio = cpu_step_seconds / time_seconds.mean;
    if !speedup_ratio.is_finite() || speedup_ratio <= 0.0 {
        return Err(Rejection::new(
            "invalid_speedup_ratio",
            format!("computed speedup ratio is {speedup_ratio}"),
        ));
    }

    Ok(json!({
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "kind": MEASUREMENT_RECEIPT_KIND,
        "profile_identity": blake3_identity(profile),
        "measurement_profile_id": measurement_profile.profile_id,
        "stage_times_seconds": stage_times_seconds,
        "cpu_baseline": cpu_baseline,
        "machine_facts": machine_facts,
        "runs": runs,
        "statistics": {
            "time_seconds": distribution_json(&time_seconds),
            "loss": distribution_json(&loss),
        },
        "speedup_ratio": speedup_ratio,
    }))
}

fn distribution_from_receipt(
    receipt: &Value,
    metric_name: &str,
) -> Result<Distribution, Rejection> {
    let statistics = required_field(receipt, "statistics")?;
    let metric = required_field(statistics, metric_name)?;
    exact_fields(
        metric,
        metric_name,
        &["mean", "p50", "p99", "standard_deviation"],
    )?;
    Ok(Distribution {
        mean: finite_number(metric, "mean", false)?,
        p50: finite_number(metric, "p50", false)?,
        p99: finite_number(metric, "p99", false)?,
        standard_deviation: finite_number(metric, "standard_deviation", false)?,
    })
}

pub fn admit_measurement_receipt(
    receipt: &Value,
    profile: &Value,
) -> Result<MeasurementAdmission, Rejection> {
    exact_fields(receipt, "measurement_receipt", &RECEIPT_FIELDS)?;
    let raw = json!({
        "stage_times_seconds": required_field(receipt, "stage_times_seconds")?,
        "cpu_baseline": required_field(receipt, "cpu_baseline")?,
        "machine_facts": required_field(receipt, "machine_facts")?,
        "runs": required_field(receipt, "runs")?,
    });
    let expected = summarize_measurement(&raw, profile)?;
    if receipt != &expected {
        return Err(Rejection::new(
            "measurement_receipt_mismatch",
            "measurement receipt differs from the deterministic summary",
        ));
    }
    let repeats = u64::try_from(
        required_field(receipt, "runs")?
            .as_array()
            .expect("summarize_measurement requires an array")
            .len(),
    )
    .map_err(|_| Rejection::new("repeat_count_overflow", "run count does not fit in u64"))?;
    Ok(MeasurementAdmission {
        identity_blake3: blake3_identity(receipt),
        measurement_profile_id: required_string(receipt, "measurement_profile_id")?.to_string(),
        repeats,
        speedup_ratio: finite_number(receipt, "speedup_ratio", true)?,
        time_seconds: distribution_from_receipt(receipt, "time_seconds")?,
        loss: distribution_from_receipt(receipt, "loss")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE_ID: &str = "parity-profile";
    const MEASUREMENT_PROFILE_ID: &str = "parity-profile-measurement";
    const REPEAT_COUNT: u64 = 3;

    fn profile() -> Value {
        json!({
            "profile_id": PROFILE_ID,
            "parity_profile": {
                "profile_id": PROFILE_ID,
            },
            "measurement_profile": {
                "profile_id": MEASUREMENT_PROFILE_ID,
                "parity_profile_id": PROFILE_ID,
                "repeat_count": REPEAT_COUNT,
                "metric_set": REQUIRED_METRICS,
                "machine_fact_set": REQUIRED_MACHINE_FACTS,
            },
        })
    }

    fn fact(name: &str, value: &str) -> Value {
        json!({
            "value": value,
            "identity_blake3": blake3_identity(&json!({"name": name, "value": value})),
        })
    }

    fn raw_measurement() -> Value {
        json!({
            "stage_times_seconds": {
                "launch": 1.0,
                "compile": 2.0,
                "step": 3.0,
                "checkpoint": 0.0,
                "teardown": 4.0,
                "end_to_end": 10.0,
            },
            "cpu_baseline": {
                "identity_blake3": "a".repeat(crate::HEX_HASH_LEN),
                "step_seconds": 12.0,
            },
            "machine_facts": {
                "board_identity": fact("board_identity", "p150a"),
                "firmware_version": fact("firmware_version", "firmware-1"),
                "mesh_descriptor_identity": fact("mesh_descriptor_identity", "mesh-hash"),
                "pcie_link": fact("pcie_link", "gen4-x16"),
                "power_state": fact("power_state", "performance"),
            },
            "runs": [
                {"time_seconds": 4.0, "loss": 3.0},
                {"time_seconds": 5.0, "loss": 2.0},
                {"time_seconds": 6.0, "loss": 1.0},
            ],
        })
    }

    #[test]
    fn profile_and_receipt_admit_deterministically() {
        let profile = profile();
        let first = summarize_measurement(&raw_measurement(), &profile).unwrap();
        let second = summarize_measurement(&raw_measurement(), &profile).unwrap();
        assert_eq!(first, second);
        let admission = admit_measurement_receipt(&first, &profile).unwrap();
        assert_eq!(admission.repeats, REPEAT_COUNT);
        assert_eq!(admission.measurement_profile_id, MEASUREMENT_PROFILE_ID);
        assert_eq!(admission.time_seconds.mean, 5.0);
        assert_eq!(admission.speedup_ratio, 2.4);
    }

    #[test]
    fn parity_relaxation_field_is_rejected() {
        let mut profile = profile();
        profile["measurement_profile"]["tolerances"] = json!({"loss": 99.0});
        let rejection = admit_measurement_profile(&profile).unwrap_err();
        assert_eq!(rejection.code, "unexpected_measurement_field");
    }

    #[test]
    fn changed_parity_profile_is_rejected() {
        let mut profile = profile();
        profile["tolerances"] = json!({"loss": 99.0});
        let rejection = admit_measurement_profile(&profile).unwrap_err();
        assert_eq!(rejection.code, "parity_profile_mismatch");
    }

    #[test]
    fn low_repeat_count_is_rejected() {
        let mut profile = profile();
        profile["measurement_profile"]["repeat_count"] = json!(MINIMUM_REPEAT_COUNT - 1);
        let rejection = admit_measurement_profile(&profile).unwrap_err();
        assert_eq!(rejection.code, "repeat_count_below_bound");
    }

    #[test]
    fn negative_stage_time_is_rejected() {
        let mut raw = raw_measurement();
        raw["stage_times_seconds"]["compile"] = json!(-1.0);
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "invalid_metric");
    }

    #[test]
    fn stage_sum_mismatch_is_rejected() {
        let mut raw = raw_measurement();
        raw["stage_times_seconds"]["end_to_end"] = json!(11.0);
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "stage_time_sum_mismatch");
    }

    #[test]
    fn missing_cpu_baseline_is_rejected() {
        let mut raw = raw_measurement();
        raw["cpu_baseline"]["identity_blake3"] = json!("missing");
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "missing_cpu_baseline");
    }

    #[test]
    fn missing_machine_fact_is_rejected() {
        let mut raw = raw_measurement();
        raw["machine_facts"]
            .as_object_mut()
            .unwrap()
            .remove("pcie_link");
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "unexpected_measurement_field");
    }

    #[test]
    fn machine_fact_identity_mismatch_is_rejected() {
        let mut raw = raw_measurement();
        raw["machine_facts"]["power_state"]["value"] = json!("idle");
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "machine_fact_identity_mismatch");
    }

    #[test]
    fn repeat_count_mismatch_is_rejected() {
        let mut raw = raw_measurement();
        raw["runs"].as_array_mut().unwrap().pop();
        let rejection = summarize_measurement(&raw, &profile()).unwrap_err();
        assert_eq!(rejection.code, "repeat_count_mismatch");
    }

    #[test]
    fn modified_statistics_are_rejected() {
        let profile = profile();
        let mut receipt = summarize_measurement(&raw_measurement(), &profile).unwrap();
        receipt["statistics"]["time_seconds"]["p99"] = json!(99.0);
        let rejection = admit_measurement_receipt(&receipt, &profile).unwrap_err();
        assert_eq!(rejection.code, "measurement_receipt_mismatch");
    }
}
