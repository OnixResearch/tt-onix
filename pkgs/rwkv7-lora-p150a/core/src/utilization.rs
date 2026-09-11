//! Pure device-utilization, memory-pressure, operation-latency, and scaling admission.
//!
//! The imperative shell converts a package-matched TT-Metal profiler report into
//! the closed telemetry input accepted here. This core does not read profiler files.
//!
//! r[impl tenstorrent.harness.utilization]
//! r[impl tenstorrent.harness.memory_pressure]
//! r[impl tenstorrent.harness.op_latency]
//! r[impl tenstorrent.harness.scaling_efficiency]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::{Rejection, admit_measurement_receipt, blake3_identity};

const UTILIZATION_SCHEMA_VERSION: u64 = 1;
const UTILIZATION_RECEIPT_KIND: &str = "rwkv7_lora_device_utilization";
const SINGLE_DEVICE_COUNT: u64 = 1;
const DUAL_DEVICE_COUNT: u64 = 2;
const IDEAL_DATA_PARALLEL_SPEEDUP: f64 = 2.0;
const PERCENTILE_P99: f64 = 0.99;

const RAW_FIELDS: [&str; 3] = ["measurement_receipt", "telemetry", "scaling_baseline"];
const TELEMETRY_FIELDS: [&str; 4] = [
    "utilization",
    "memory",
    "executed_operations",
    "operation_samples",
];
const UTILIZATION_FIELDS: [&str; 4] = [
    "math_busy_fraction",
    "math_busy_source",
    "noc_idle_fraction",
    "noc_idle_source",
];
const MEMORY_FIELDS: [&str; 5] = [
    "device_ordinal",
    "l1_peak_bytes",
    "l1_capacity_bytes",
    "dram_peak_bytes",
    "dram_capacity_bytes",
];
const OPERATION_SAMPLE_FIELDS: [&str; 3] = ["operation", "source", "latency_seconds"];
const SCALING_BASELINE_FIELDS: [&str; 2] = ["profile", "receipt"];
const RECEIPT_FIELDS: [&str; 10] = [
    "schema_version",
    "kind",
    "profile_identity",
    "measurement_identity",
    "telemetry_identity",
    "utilization",
    "memory",
    "executed_operations",
    "operation_latency",
    "scaling_efficiency",
];

#[derive(Clone, Debug, PartialEq)]
pub struct UtilizationAdmission {
    pub identity_blake3: String,
    pub device_count: u64,
    pub operation_count: u64,
    pub math_busy_fraction: f64,
    pub noc_idle_fraction: f64,
    pub scaling_efficiency: Option<f64>,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, Rejection> {
    value.as_object().ok_or_else(|| {
        Rejection::new(
            "malformed_utilization",
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
            "unexpected_utilization_field",
            format!("field '{name}' has fields {actual:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

fn required_field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Rejection> {
    value.get(key).ok_or_else(|| {
        Rejection::new(
            "missing_utilization_field",
            format!("missing utilization field '{key}'"),
        )
    })
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    let text = required_field(value, key)?.as_str().ok_or_else(|| {
        Rejection::new(
            "malformed_utilization",
            format!("utilization field '{key}' is not a string"),
        )
    })?;
    if text.is_empty() {
        return Err(Rejection::new(
            "malformed_utilization",
            format!("utilization field '{key}' is empty"),
        ));
    }
    Ok(text)
}

fn required_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    required_field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "malformed_utilization",
            format!("utilization field '{key}' is not an unsigned integer"),
        )
    })
}

fn finite_number(value: &Value, key: &str, positive: bool) -> Result<f64, Rejection> {
    let number = required_field(value, key)?.as_f64().ok_or_else(|| {
        Rejection::new(
            "invalid_utilization_metric",
            format!("metric '{key}' is not a finite number"),
        )
    })?;
    if !number.is_finite() {
        return Err(Rejection::new(
            "invalid_utilization_metric",
            format!("metric '{key}' is not finite"),
        ));
    }
    if positive && number <= 0.0 {
        return Err(Rejection::new(
            "invalid_utilization_metric",
            format!("metric '{key}' must be positive, got {number}"),
        ));
    }
    if !positive && number < 0.0 {
        return Err(Rejection::new(
            "invalid_utilization_metric",
            format!("metric '{key}' must be non-negative, got {number}"),
        ));
    }
    Ok(number)
}

fn fraction(value: &Value, key: &str) -> Result<f64, Rejection> {
    let number = finite_number(value, key, false)?;
    if number > 1.0 {
        return Err(Rejection::new(
            "utilization_fraction_out_of_range",
            format!("fraction '{key}' is {number}; expected the closed interval zero to one"),
        ));
    }
    Ok(number)
}

fn profile_device_count(profile: &Value) -> Result<u64, Rejection> {
    let device = required_field(profile, "device")?;
    let mesh = required_field(device, "logical_mesh")?;
    let rows = required_u64(mesh, "rows")?;
    let columns = required_u64(mesh, "columns")?;
    let count = rows.checked_mul(columns).ok_or_else(|| {
        Rejection::new(
            "device_count_overflow",
            "logical mesh device count does not fit in u64",
        )
    })?;
    if count != SINGLE_DEVICE_COUNT && count != DUAL_DEVICE_COUNT {
        return Err(Rejection::new(
            "unsupported_device_count",
            format!("logical mesh has {count} devices; expected one or two"),
        ));
    }
    Ok(count)
}

fn utilization(telemetry: &Value) -> Result<Value, Rejection> {
    let input = required_field(telemetry, "utilization")?;
    exact_fields(input, "utilization", &UTILIZATION_FIELDS)?;
    let math_busy_fraction = fraction(input, "math_busy_fraction")?;
    let noc_idle_fraction = fraction(input, "noc_idle_fraction")?;
    let math_busy_source = required_string(input, "math_busy_source")?;
    let noc_idle_source = required_string(input, "noc_idle_source")?;
    Ok(json!({
        "math_busy_fraction": math_busy_fraction,
        "math_busy_source": math_busy_source,
        "noc_idle_fraction": noc_idle_fraction,
        "noc_idle_source": noc_idle_source,
    }))
}

fn memory(telemetry: &Value, device_count: u64) -> Result<Value, Rejection> {
    let entries = required_field(telemetry, "memory")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_utilization", "memory is not an array"))?;
    let actual_count = u64::try_from(entries.len()).map_err(|_| {
        Rejection::new(
            "device_count_overflow",
            "memory entry count does not fit in u64",
        )
    })?;
    if actual_count != device_count {
        return Err(Rejection::new(
            "memory_device_count_mismatch",
            format!("memory has {actual_count} devices, expected {device_count}"),
        ));
    }

    let mut output = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        exact_fields(entry, "memory_entry", &MEMORY_FIELDS)?;
        let expected_ordinal = u64::try_from(index).map_err(|_| {
            Rejection::new("device_count_overflow", "memory index does not fit in u64")
        })?;
        let ordinal = required_u64(entry, "device_ordinal")?;
        if ordinal != expected_ordinal {
            return Err(Rejection::new(
                "memory_device_ordinal_mismatch",
                format!("memory device ordinal is {ordinal}, expected {expected_ordinal}"),
            ));
        }
        let l1_peak = finite_number(entry, "l1_peak_bytes", false)?;
        let l1_capacity = finite_number(entry, "l1_capacity_bytes", true)?;
        let dram_peak = finite_number(entry, "dram_peak_bytes", false)?;
        let dram_capacity = finite_number(entry, "dram_capacity_bytes", true)?;
        if l1_peak > l1_capacity {
            return Err(Rejection::new(
                "memory_peak_exceeds_capacity",
                format!("device {ordinal} L1 peak {l1_peak} exceeds capacity {l1_capacity}"),
            ));
        }
        if dram_peak > dram_capacity {
            return Err(Rejection::new(
                "memory_peak_exceeds_capacity",
                format!("device {ordinal} DRAM peak {dram_peak} exceeds capacity {dram_capacity}"),
            ));
        }
        output.push(json!({
            "device_ordinal": ordinal,
            "l1_peak_bytes": l1_peak,
            "l1_capacity_bytes": l1_capacity,
            "dram_peak_bytes": dram_peak,
            "dram_capacity_bytes": dram_capacity,
        }));
    }
    Ok(Value::Array(output))
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    assert!(!sorted.is_empty());
    assert!(percentile > 0.0);
    assert!(percentile <= 1.0);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len().saturating_sub(1));
    sorted[index]
}

fn operation_latency(telemetry: &Value) -> Result<(Value, Value), Rejection> {
    let executed_entries = required_field(telemetry, "executed_operations")?
        .as_array()
        .ok_or_else(|| {
            Rejection::new(
                "malformed_utilization",
                "executed_operations is not an array",
            )
        })?;
    let mut executed = BTreeSet::new();
    for entry in executed_entries {
        let operation = entry.as_str().ok_or_else(|| {
            Rejection::new(
                "malformed_utilization",
                "executed_operations contains a non-string value",
            )
        })?;
        if operation.is_empty() || !executed.insert(operation.to_string()) {
            return Err(Rejection::new(
                "invalid_executed_operation",
                format!("executed operation '{operation}' is empty or duplicated"),
            ));
        }
    }
    if executed.is_empty() {
        return Err(Rejection::new(
            "missing_operation_latency",
            "executed operation set is empty",
        ));
    }

    let samples = required_field(telemetry, "operation_samples")?
        .as_array()
        .ok_or_else(|| {
            Rejection::new("malformed_utilization", "operation_samples is not an array")
        })?;
    let mut summaries = BTreeMap::new();
    for sample in samples {
        exact_fields(sample, "operation_sample", &OPERATION_SAMPLE_FIELDS)?;
        let operation = required_string(sample, "operation")?;
        let source = required_string(sample, "source")?;
        let values = required_field(sample, "latency_seconds")?
            .as_array()
            .ok_or_else(|| {
                Rejection::new(
                    "malformed_utilization",
                    format!("operation '{operation}' latency_seconds is not an array"),
                )
            })?;
        if values.is_empty() {
            return Err(Rejection::new(
                "missing_operation_latency",
                format!("operation '{operation}' has no latency samples"),
            ));
        }
        let mut latencies = Vec::with_capacity(values.len());
        for value in values {
            let seconds = value.as_f64().ok_or_else(|| {
                Rejection::new(
                    "invalid_operation_latency",
                    format!("operation '{operation}' has a non-finite latency"),
                )
            })?;
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(Rejection::new(
                    "invalid_operation_latency",
                    format!("operation '{operation}' latency must be positive, got {seconds}"),
                ));
            }
            latencies.push(seconds);
        }
        latencies.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p99 = percentile(&latencies, PERCENTILE_P99);
        let summary = json!({
            "operation": operation,
            "source": source,
            "mean_seconds": mean,
            "p99_seconds": p99,
            "sample_count": latencies.len(),
        });
        if summaries.insert(operation.to_string(), summary).is_some() {
            return Err(Rejection::new(
                "duplicate_operation_latency",
                format!("operation '{operation}' appears more than once"),
            ));
        }
    }

    let observed = summaries.keys().cloned().collect::<BTreeSet<_>>();
    if observed != executed {
        let missing = executed.difference(&observed).cloned().collect::<Vec<_>>();
        let extra = observed.difference(&executed).cloned().collect::<Vec<_>>();
        return Err(Rejection::new(
            "incomplete_operation_latency",
            format!(
                "operation latency coverage is incomplete; missing {missing:?}, extra {extra:?}"
            ),
        ));
    }
    let executed_output = executed.into_iter().map(Value::String).collect::<Vec<_>>();
    let latency_output = summaries.into_values().collect::<Vec<_>>();
    Ok((Value::Array(executed_output), Value::Array(latency_output)))
}

fn compatible_measurement_profiles(current: &Value, baseline: &Value) -> Result<(), Rejection> {
    for key in ["repeat_count", "metric_set", "machine_fact_set"] {
        let current_value = required_field(required_field(current, "measurement_profile")?, key)?;
        let baseline_value = required_field(required_field(baseline, "measurement_profile")?, key)?;
        if current_value != baseline_value {
            return Err(Rejection::new(
                "scaling_profile_mismatch",
                format!("measurement profile field '{key}' differs across meshes"),
            ));
        }
    }
    for key in ["micro_batch", "sequence_length"] {
        let current_value = required_field(required_field(current, "static_shape")?, key)?;
        let baseline_value = required_field(required_field(baseline, "static_shape")?, key)?;
        if current_value != baseline_value {
            return Err(Rejection::new(
                "scaling_profile_mismatch",
                format!("static shape field '{key}' differs across meshes"),
            ));
        }
    }
    Ok(())
}

fn mean_step_seconds(receipt: &Value) -> Result<f64, Rejection> {
    let statistics = required_field(receipt, "statistics")?;
    let time = required_field(statistics, "time_seconds")?;
    finite_number(time, "mean", true)
}

fn scaling_efficiency(
    raw: &Value,
    profile: &Value,
    current_receipt: &Value,
    device_count: u64,
) -> Result<Value, Rejection> {
    let baseline = required_field(raw, "scaling_baseline")?;
    if device_count == SINGLE_DEVICE_COUNT {
        if !baseline.is_null() {
            return Err(Rejection::new(
                "scaling_claim_without_two_device_run",
                "single-device receipt cannot report scaling efficiency",
            ));
        }
        return Ok(Value::Null);
    }
    if baseline.is_null() {
        return Err(Rejection::new(
            "missing_scaling_baseline",
            "two-device receipt requires a one-device measurement baseline",
        ));
    }
    exact_fields(baseline, "scaling_baseline", &SCALING_BASELINE_FIELDS)?;
    let baseline_profile = required_field(baseline, "profile")?;
    let baseline_receipt = required_field(baseline, "receipt")?;
    if profile_device_count(baseline_profile)? != SINGLE_DEVICE_COUNT {
        return Err(Rejection::new(
            "scaling_claim_without_two_device_run",
            "scaling baseline is not a one-device profile",
        ));
    }
    compatible_measurement_profiles(profile, baseline_profile)?;
    admit_measurement_receipt(baseline_receipt, baseline_profile)?;

    let one_device_seconds = mean_step_seconds(baseline_receipt)?;
    let two_device_seconds = mean_step_seconds(current_receipt)?;
    let efficiency = one_device_seconds / two_device_seconds;
    if !efficiency.is_finite() || efficiency <= 0.0 {
        return Err(Rejection::new(
            "invalid_scaling_efficiency",
            format!("computed scaling efficiency is {efficiency}"),
        ));
    }
    Ok(json!({
        "one_device_measurement_identity": blake3_identity(baseline_receipt),
        "two_device_measurement_identity": blake3_identity(current_receipt),
        "one_device_time_per_step_seconds": one_device_seconds,
        "two_device_time_per_step_seconds": two_device_seconds,
        "ideal_data_parallel_speedup": IDEAL_DATA_PARALLEL_SPEEDUP,
        "efficiency_ratio": efficiency,
    }))
}

pub fn summarize_utilization(raw: &Value, profile: &Value) -> Result<Value, Rejection> {
    exact_fields(raw, "raw_utilization", &RAW_FIELDS)?;
    let measurement_receipt = required_field(raw, "measurement_receipt")?;
    admit_measurement_receipt(measurement_receipt, profile)?;
    let telemetry = required_field(raw, "telemetry")?;
    exact_fields(telemetry, "telemetry", &TELEMETRY_FIELDS)?;
    let device_count = profile_device_count(profile)?;
    let utilization = utilization(telemetry)?;
    let memory = memory(telemetry, device_count)?;
    let (executed_operations, operation_latency) = operation_latency(telemetry)?;
    let scaling_efficiency = scaling_efficiency(raw, profile, measurement_receipt, device_count)?;

    Ok(json!({
        "schema_version": UTILIZATION_SCHEMA_VERSION,
        "kind": UTILIZATION_RECEIPT_KIND,
        "profile_identity": blake3_identity(profile),
        "measurement_identity": blake3_identity(measurement_receipt),
        "telemetry_identity": blake3_identity(telemetry),
        "utilization": utilization,
        "memory": memory,
        "executed_operations": executed_operations,
        "operation_latency": operation_latency,
        "scaling_efficiency": scaling_efficiency,
    }))
}

pub fn admit_utilization_receipt(
    receipt: &Value,
    raw: &Value,
    profile: &Value,
) -> Result<UtilizationAdmission, Rejection> {
    exact_fields(receipt, "utilization_receipt", &RECEIPT_FIELDS)?;
    let expected = summarize_utilization(raw, profile)?;
    if receipt != &expected {
        return Err(Rejection::new(
            "utilization_receipt_mismatch",
            "utilization receipt differs from the deterministic summary",
        ));
    }
    let utilization = required_field(receipt, "utilization")?;
    let operation_count = u64::try_from(
        required_field(receipt, "executed_operations")?
            .as_array()
            .expect("summarize_utilization emits an array")
            .len(),
    )
    .map_err(|_| {
        Rejection::new(
            "operation_count_overflow",
            "operation count does not fit in u64",
        )
    })?;
    let scaling = required_field(receipt, "scaling_efficiency")?;
    let scaling_efficiency = if scaling.is_null() {
        None
    } else {
        Some(finite_number(scaling, "efficiency_ratio", true)?)
    };
    Ok(UtilizationAdmission {
        identity_blake3: blake3_identity(receipt),
        device_count: profile_device_count(profile)?,
        operation_count,
        math_busy_fraction: fraction(utilization, "math_busy_fraction")?,
        noc_idle_fraction: fraction(utilization, "noc_idle_fraction")?,
        scaling_efficiency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HEX_HASH_LEN, bind_machine_facts, summarize_measurement};

    const REPEAT_COUNT: u64 = 3;

    fn profile(device_count: u64) -> Value {
        json!({
            "profile_id": format!("parity-{device_count}"),
            "device": {
                "logical_mesh": {"rows": 1, "columns": device_count},
            },
            "static_shape": {"micro_batch": 1, "sequence_length": 64},
            "parity_profile": {
                "profile_id": format!("parity-{device_count}"),
                "device": {
                    "logical_mesh": {"rows": 1, "columns": device_count},
                },
                "static_shape": {"micro_batch": 1, "sequence_length": 64},
            },
            "measurement_profile": {
                "profile_id": format!("measurement-{device_count}"),
                "parity_profile_id": format!("parity-{device_count}"),
                "repeat_count": REPEAT_COUNT,
                "metric_set": ["stage_time_seconds", "loss"],
                "machine_fact_set": [
                    "board_identity",
                    "firmware_version",
                    "mesh_descriptor_identity",
                    "pcie_link",
                    "power_state",
                ],
            },
        })
    }

    fn measurement_receipt(device_count: u64, step_seconds: f64) -> Value {
        let facts = bind_machine_facts(&json!({
            "board_identity": "p150a",
            "firmware_version": "firmware-1",
            "mesh_descriptor_identity": format!("mesh-{device_count}"),
            "pcie_link": "gen4-x16",
            "power_state": "performance",
        }))
        .unwrap();
        let raw = json!({
            "stage_times_seconds": {
                "launch": 1.0,
                "compile": 2.0,
                "step": step_seconds,
                "checkpoint": 0.0,
                "teardown": 1.0,
                "end_to_end": step_seconds + 4.0,
            },
            "cpu_baseline": {
                "identity_blake3": "a".repeat(HEX_HASH_LEN),
                "step_seconds": 12.0,
            },
            "machine_facts": facts,
            "runs": [
                {"time_seconds": step_seconds, "loss": 3.0},
                {"time_seconds": step_seconds, "loss": 2.0},
                {"time_seconds": step_seconds, "loss": 1.0},
            ],
        });
        summarize_measurement(&raw, &profile(device_count)).unwrap()
    }

    fn telemetry(device_count: u64) -> Value {
        let memory = (0..device_count)
            .map(|ordinal| {
                json!({
                    "device_ordinal": ordinal,
                    "l1_peak_bytes": 10.0,
                    "l1_capacity_bytes": 20.0,
                    "dram_peak_bytes": 30.0,
                    "dram_capacity_bytes": 40.0,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "utilization": {
                "math_busy_fraction": 0.75,
                "math_busy_source": "tt-metal-op-report:Avg Math util on full grid (%)",
                "noc_idle_fraction": 0.25,
                "noc_idle_source": "tt-metal-op-report:100-NOC UTIL (%)",
            },
            "memory": memory,
            "executed_operations": ["ttnn.linear", "ttnn.rms_norm"],
            "operation_samples": [
                {
                    "operation": "ttnn.linear",
                    "source": "tt-metal-op-report:DEVICE KERNEL DURATION [ns]",
                    "latency_seconds": [0.1, 0.2, 0.3],
                },
                {
                    "operation": "ttnn.rms_norm",
                    "source": "tt-metal-op-report:DEVICE KERNEL DURATION [ns]",
                    "latency_seconds": [0.01, 0.02, 0.03],
                },
            ],
        })
    }

    fn raw(device_count: u64) -> Value {
        let scaling_baseline = if device_count == DUAL_DEVICE_COUNT {
            json!({
                "profile": profile(SINGLE_DEVICE_COUNT),
                "receipt": measurement_receipt(SINGLE_DEVICE_COUNT, 2.0),
            })
        } else {
            Value::Null
        };
        json!({
            "measurement_receipt": measurement_receipt(device_count, 1.0),
            "telemetry": telemetry(device_count),
            "scaling_baseline": scaling_baseline,
        })
    }

    #[test]
    fn utilization_receipt_is_deterministic_and_complete() {
        let raw = raw(SINGLE_DEVICE_COUNT);
        let profile = profile(SINGLE_DEVICE_COUNT);
        let first = summarize_utilization(&raw, &profile).unwrap();
        let second = summarize_utilization(&raw, &profile).unwrap();
        assert_eq!(first, second);
        let admission = admit_utilization_receipt(&first, &raw, &profile).unwrap();
        assert_eq!(admission.device_count, SINGLE_DEVICE_COUNT);
        assert_eq!(admission.operation_count, DUAL_DEVICE_COUNT);
        assert_eq!(admission.math_busy_fraction, 0.75);
        assert_eq!(admission.scaling_efficiency, None);
    }

    #[test]
    fn dual_receipt_binds_scaling_efficiency() {
        let raw = raw(DUAL_DEVICE_COUNT);
        let profile = profile(DUAL_DEVICE_COUNT);
        let receipt = summarize_utilization(&raw, &profile).unwrap();
        let admission = admit_utilization_receipt(&receipt, &raw, &profile).unwrap();
        assert_eq!(admission.scaling_efficiency, Some(2.0));
    }

    #[test]
    fn out_of_range_utilization_is_rejected() {
        let mut raw = raw(SINGLE_DEVICE_COUNT);
        raw["telemetry"]["utilization"]["math_busy_fraction"] = json!(1.1);
        let rejection = summarize_utilization(&raw, &profile(SINGLE_DEVICE_COUNT)).unwrap_err();
        assert_eq!(rejection.code, "utilization_fraction_out_of_range");
        assert!(rejection.detail.contains("math_busy_fraction"));
    }

    #[test]
    fn invalid_memory_peak_is_rejected() {
        let mut raw = raw(SINGLE_DEVICE_COUNT);
        raw["telemetry"]["memory"][0]["l1_peak_bytes"] = json!(21.0);
        let rejection = summarize_utilization(&raw, &profile(SINGLE_DEVICE_COUNT)).unwrap_err();
        assert_eq!(rejection.code, "memory_peak_exceeds_capacity");
        assert!(rejection.detail.contains("L1"));
    }

    #[test]
    fn missing_operation_is_rejected() {
        let mut raw = raw(SINGLE_DEVICE_COUNT);
        raw["telemetry"]["operation_samples"]
            .as_array_mut()
            .unwrap()
            .pop();
        let rejection = summarize_utilization(&raw, &profile(SINGLE_DEVICE_COUNT)).unwrap_err();
        assert_eq!(rejection.code, "incomplete_operation_latency");
        assert!(rejection.detail.contains("ttnn.rms_norm"));
    }

    #[test]
    fn scaling_claim_without_two_device_run_is_rejected() {
        let mut raw = raw(SINGLE_DEVICE_COUNT);
        raw["scaling_baseline"] = json!({
            "profile": profile(SINGLE_DEVICE_COUNT),
            "receipt": measurement_receipt(SINGLE_DEVICE_COUNT, 2.0),
        });
        let rejection = summarize_utilization(&raw, &profile(SINGLE_DEVICE_COUNT)).unwrap_err();
        assert_eq!(rejection.code, "scaling_claim_without_two_device_run");
    }

    #[test]
    fn dual_receipt_without_single_baseline_is_rejected() {
        let mut raw = raw(DUAL_DEVICE_COUNT);
        raw["scaling_baseline"] = Value::Null;
        let rejection = summarize_utilization(&raw, &profile(DUAL_DEVICE_COUNT)).unwrap_err();
        assert_eq!(rejection.code, "missing_scaling_baseline");
    }

    #[test]
    fn changed_utilization_receipt_is_rejected() {
        let raw = raw(SINGLE_DEVICE_COUNT);
        let profile = profile(SINGLE_DEVICE_COUNT);
        let mut receipt = summarize_utilization(&raw, &profile).unwrap();
        receipt["operation_latency"][0]["p99_seconds"] = json!(99.0);
        let rejection = admit_utilization_receipt(&receipt, &raw, &profile).unwrap_err();
        assert_eq!(rejection.code, "utilization_receipt_mismatch");
    }
}
