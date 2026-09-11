//! Pure long-run, thermal, and held-out-eval admission logic.
//!
//! r[impl tenstorrent.harness.long_run]
//! r[impl tenstorrent.harness.thermal]
//! r[impl tenstorrent.harness.heldout_eval]

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::{Rejection, bind_machine_facts, blake3_identity};

const STABILITY_SCHEMA_VERSION: u64 = 1;
const STABILITY_KIND: &str = "rwkv7_lora_stability_training_value";
const MINIMUM_LONG_RUN_STEPS: u64 = 64;
const BLAKE3_HEX_LENGTH: usize = 64;
const POLICY_FIELDS: [&str; 14] = [
    "schema_version",
    "profile_id",
    "parity_profile_id",
    "step_budget",
    "trajectory_cadence_steps",
    "thermal_cadence_steps",
    "maximum_loss",
    "maximum_gradient_norm",
    "maximum_temperature_celsius",
    "maximum_power_watts",
    "maximum_eval_loss_ratio",
    "thermal_source",
    "generalization_non_claim",
    "hash_algorithm",
];
const RAW_FIELDS: [&str; 8] = [
    "status",
    "completed_steps",
    "checked_steps",
    "parity_profile_identity",
    "machine_facts",
    "trajectory",
    "thermal",
    "heldout_eval",
];
const TRAJECTORY_FIELDS: [&str; 3] = ["step", "loss", "gradient_norm"];
const THERMAL_FIELDS: [&str; 2] = ["source", "samples"];
const THERMAL_SAMPLE_FIELDS: [&str; 2] = ["step", "devices"];
const THERMAL_DEVICE_FIELDS: [&str; 3] = ["device_id", "temperature_celsius", "power_watts"];
const MACHINE_FACT_FIELDS: [&str; 5] = [
    "board_identity",
    "firmware_version",
    "mesh_descriptor_identity",
    "pcie_link",
    "power_state",
];
const HELDOUT_FIELDS: [&str; 7] = [
    "split_identity",
    "reference_batch_identity",
    "split_role",
    "metric",
    "before_loss",
    "after_loss",
    "generalization_non_claim",
];
const RECEIPT_FIELDS: [&str; 11] = [
    "schema_version",
    "kind",
    "profile_id",
    "parity_profile_identity",
    "policy_identity",
    "machine_facts_identity",
    "completed_steps",
    "trajectory",
    "thermal",
    "heldout_eval",
    "verdict",
];

#[derive(Clone, Debug, PartialEq)]
pub struct StabilityAdmission {
    pub identity_blake3: String,
    pub completed_steps: u64,
    pub trajectory_samples: u64,
    pub thermal_samples: u64,
    pub eval_loss_ratio: f64,
}

#[derive(Clone, Debug)]
struct StabilityPolicy {
    profile_id: String,
    step_budget: u64,
    trajectory_cadence_steps: u64,
    thermal_cadence_steps: u64,
    maximum_loss: f64,
    maximum_gradient_norm: f64,
    maximum_temperature_celsius: f64,
    maximum_power_watts: f64,
    maximum_eval_loss_ratio: f64,
    thermal_source: String,
    generalization_non_claim: String,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, Rejection> {
    value.as_object().ok_or_else(|| {
        Rejection::new(
            "malformed_stability",
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
            "unexpected_stability_field",
            format!("field '{name}' has fields {actual:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Rejection> {
    value.get(key).ok_or_else(|| {
        Rejection::new(
            "missing_stability_field",
            format!("missing stability field '{key}'"),
        )
    })
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    let text = field(value, key)?.as_str().ok_or_else(|| {
        Rejection::new(
            "malformed_stability",
            format!("stability field '{key}' is not a string"),
        )
    })?;
    if text.is_empty() {
        return Err(Rejection::new(
            "malformed_stability",
            format!("stability field '{key}' is empty"),
        ));
    }
    Ok(text)
}

fn unsigned(value: &Value, key: &str) -> Result<u64, Rejection> {
    field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "malformed_stability",
            format!("stability field '{key}' is not an unsigned integer"),
        )
    })
}

fn finite(value: &Value, key: &str, positive: bool) -> Result<f64, Rejection> {
    let number = field(value, key)?.as_f64().ok_or_else(|| {
        Rejection::new(
            "invalid_stability_metric",
            format!("metric '{key}' is not a finite number"),
        )
    })?;
    if !number.is_finite() || (positive && number <= 0.0) || (!positive && number < 0.0) {
        let expectation = if positive { "positive" } else { "non-negative" };
        return Err(Rejection::new(
            "invalid_stability_metric",
            format!("metric '{key}' must be finite and {expectation}, got {number}"),
        ));
    }
    Ok(number)
}

fn hash<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    let identity = string(value, key)?;
    let valid = identity.len() == BLAKE3_HEX_LENGTH
        && identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(Rejection::new(
            "invalid_stability_identity",
            format!("field '{key}' is not a lowercase BLAKE3 identity"),
        ));
    }
    Ok(identity)
}

fn parity_profile_id(profile: &Value) -> Result<&str, Rejection> {
    string(profile, "profile_id")
}

fn snapshot_metric(telemetry: &Value, key: &str, device_id: u64) -> Result<f64, Rejection> {
    let raw = field(telemetry, key)?;
    let number = match raw {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        Rejection::new(
            "invalid_thermal_snapshot",
            format!("device {device_id}: tt-smi telemetry field '{key}' is not a number"),
        )
    })?;
    if !number.is_finite() || number < 0.0 {
        return Err(Rejection::new(
            "invalid_thermal_snapshot",
            format!(
                "device {device_id}: tt-smi telemetry field '{key}' must be finite and non-negative, got {number}"
            ),
        ));
    }
    Ok(number)
}

fn device_ids(profile: &Value) -> Result<Vec<u64>, Rejection> {
    let device = field(profile, "device")?;
    let device_fields = object(device, "device")?;
    let ids = if let Some(ids) = device_fields.get("physical_device_ids") {
        ids.as_array()
            .ok_or_else(|| {
                Rejection::new(
                    "malformed_stability",
                    "profile physical_device_ids is not an array",
                )
            })?
            .iter()
            .map(|id| {
                id.as_u64().ok_or_else(|| {
                    Rejection::new(
                        "malformed_stability",
                        "profile physical device ID is not an unsigned integer",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![unsigned(device, "selection_ordinal")?]
    };
    if ids.is_empty() {
        return Err(Rejection::new(
            "malformed_stability",
            "profile has no physical device IDs",
        ));
    }
    let mut unique = BTreeSet::new();
    for id in &ids {
        if !unique.insert(*id) {
            return Err(Rejection::new(
                "duplicate_stability_device",
                format!("physical device ID {id} appears more than once"),
            ));
        }
    }
    Ok(ids)
}

/// Convert one package-matched tt-smi snapshot into the thermal-device shape.
///
/// The tt-smi `device_info` array uses the UMD logical device order. The
/// profile selects those same logical ordinals. This function is pure so the
/// host shell only owns snapshot collection and file orchestration.
pub fn normalize_tt_smi_snapshot(snapshot: &Value, profile: &Value) -> Result<Value, Rejection> {
    let device_info = field(snapshot, "device_info")?.as_array().ok_or_else(|| {
        Rejection::new(
            "invalid_thermal_snapshot",
            "tt-smi field 'device_info' is not an array",
        )
    })?;
    let mut devices = Vec::new();
    for device_id in device_ids(profile)? {
        let device_index = usize::try_from(device_id).map_err(|_| {
            Rejection::new(
                "invalid_thermal_snapshot",
                format!("device ID {device_id} does not fit a host array index"),
            )
        })?;
        let device = device_info.get(device_index).ok_or_else(|| {
            Rejection::new(
                "missing_thermal_device",
                format!("tt-smi snapshot has no device_info entry for device {device_id}"),
            )
        })?;
        let telemetry = field(device, "telemetry")?;
        object(telemetry, "telemetry")?;
        let temperature_celsius = snapshot_metric(telemetry, "asic_temperature", device_id)?;
        let power_watts = snapshot_metric(telemetry, "power", device_id)?;
        devices.push(json!({
            "device_id": device_id,
            "temperature_celsius": temperature_celsius,
            "power_watts": power_watts,
        }));
    }
    Ok(json!({"devices": devices}))
}

fn validate_policy(policy: &Value, profile: &Value) -> Result<StabilityPolicy, Rejection> {
    exact_fields(policy, "stability_policy", &POLICY_FIELDS)?;
    if unsigned(policy, "schema_version")? != STABILITY_SCHEMA_VERSION {
        return Err(Rejection::new(
            "stability_policy_schema_mismatch",
            "stability policy schema version is unsupported",
        ));
    }
    let parity_id = parity_profile_id(profile)?;
    if string(policy, "parity_profile_id")? != parity_id {
        return Err(Rejection::new(
            "stability_profile_mismatch",
            format!("stability policy does not bind parity profile '{parity_id}'"),
        ));
    }
    let expected_policy_id = format!("{parity_id}-stability");
    if string(policy, "profile_id")? != expected_policy_id {
        return Err(Rejection::new(
            "stability_profile_mismatch",
            format!("stability policy ID must be '{expected_policy_id}'"),
        ));
    }
    if string(policy, "hash_algorithm")? != "blake3" {
        return Err(Rejection::new(
            "stability_hash_mismatch",
            "stability policy must use BLAKE3",
        ));
    }
    let step_budget = unsigned(policy, "step_budget")?;
    let trajectory_cadence_steps = unsigned(policy, "trajectory_cadence_steps")?;
    let thermal_cadence_steps = unsigned(policy, "thermal_cadence_steps")?;
    if step_budget < MINIMUM_LONG_RUN_STEPS
        || trajectory_cadence_steps == 0
        || thermal_cadence_steps == 0
        || step_budget % trajectory_cadence_steps != 0
        || step_budget % thermal_cadence_steps != 0
    {
        return Err(Rejection::new(
            "invalid_stability_cadence",
            format!(
                "step budget {step_budget} must be at least {MINIMUM_LONG_RUN_STEPS} and divisible by both non-zero cadences"
            ),
        ));
    }
    let maximum_eval_loss_ratio = finite(policy, "maximum_eval_loss_ratio", true)?;
    if maximum_eval_loss_ratio > 1.0 {
        return Err(Rejection::new(
            "invalid_stability_metric",
            "maximum_eval_loss_ratio must not exceed one",
        ));
    }
    Ok(StabilityPolicy {
        profile_id: expected_policy_id,
        step_budget,
        trajectory_cadence_steps,
        thermal_cadence_steps,
        maximum_loss: finite(policy, "maximum_loss", true)?,
        maximum_gradient_norm: finite(policy, "maximum_gradient_norm", true)?,
        maximum_temperature_celsius: finite(policy, "maximum_temperature_celsius", true)?,
        maximum_power_watts: finite(policy, "maximum_power_watts", true)?,
        maximum_eval_loss_ratio,
        thermal_source: string(policy, "thermal_source")?.to_string(),
        generalization_non_claim: string(policy, "generalization_non_claim")?.to_string(),
    })
}

fn validate_machine_facts(raw: &Value) -> Result<String, Rejection> {
    let machine_facts = field(raw, "machine_facts")?;
    exact_fields(machine_facts, "machine_facts", &MACHINE_FACT_FIELDS)?;
    let mut values = Map::new();
    for fact_name in MACHINE_FACT_FIELDS {
        let fact = field(machine_facts, fact_name)?;
        exact_fields(fact, fact_name, &["value", "identity_blake3"])?;
        values.insert(fact_name.to_string(), json!(string(fact, "value")?));
    }
    let expected = bind_machine_facts(&Value::Object(values))?;
    if machine_facts != &expected {
        return Err(Rejection::new(
            "stability_machine_facts_mismatch",
            "machine facts identity does not bind its values",
        ));
    }
    Ok(blake3_identity(machine_facts))
}

fn expected_sample_count(step_budget: u64, cadence: u64) -> Result<usize, Rejection> {
    usize::try_from(step_budget / cadence).map_err(|_| {
        Rejection::new(
            "stability_sample_count_overflow",
            "stability sample count does not fit in usize",
        )
    })
}

fn validate_trajectory(raw: &Value, policy: &StabilityPolicy) -> Result<Value, Rejection> {
    let trajectory = field(raw, "trajectory")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_stability", "trajectory is not an array"))?;
    let expected_count =
        expected_sample_count(policy.step_budget, policy.trajectory_cadence_steps)?;
    if trajectory.len() != expected_count {
        return Err(Rejection::new(
            "stability_trajectory_incomplete",
            format!(
                "trajectory has {} samples, expected {expected_count}",
                trajectory.len()
            ),
        ));
    }
    for (index, sample) in trajectory.iter().enumerate() {
        exact_fields(sample, "trajectory_sample", &TRAJECTORY_FIELDS)?;
        let sample_ordinal = u64::try_from(index + 1).map_err(|_| {
            Rejection::new(
                "stability_step_overflow",
                "trajectory sample ordinal does not fit in u64",
            )
        })?;
        let expected_step = sample_ordinal * policy.trajectory_cadence_steps;
        let actual_step = unsigned(sample, "step")?;
        if actual_step != expected_step {
            return Err(Rejection::new(
                "stability_trajectory_incomplete",
                format!("trajectory step is {actual_step}, expected {expected_step}"),
            ));
        }
        let loss = finite(sample, "loss", false).map_err(|error| {
            Rejection::new(error.code, format!("step {actual_step}: {}", error.detail))
        })?;
        let gradient_norm = finite(sample, "gradient_norm", false).map_err(|error| {
            Rejection::new(error.code, format!("step {actual_step}: {}", error.detail))
        })?;
        if loss > policy.maximum_loss {
            return Err(Rejection::new(
                "long_run_divergence",
                format!(
                    "step {actual_step}: loss {loss} exceeds maximum {}",
                    policy.maximum_loss
                ),
            ));
        }
        if gradient_norm > policy.maximum_gradient_norm {
            return Err(Rejection::new(
                "long_run_divergence",
                format!(
                    "step {actual_step}: gradient norm {gradient_norm} exceeds maximum {}",
                    policy.maximum_gradient_norm
                ),
            ));
        }
    }
    Ok(Value::Array(trajectory.clone()))
}

fn validate_thermal(
    raw: &Value,
    profile: &Value,
    policy: &StabilityPolicy,
) -> Result<Value, Rejection> {
    let thermal = field(raw, "thermal")?;
    exact_fields(thermal, "thermal", &THERMAL_FIELDS)?;
    if string(thermal, "source")? != policy.thermal_source {
        return Err(Rejection::new(
            "thermal_source_mismatch",
            format!("thermal source does not match '{}'", policy.thermal_source),
        ));
    }
    let samples = field(thermal, "samples")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_stability", "thermal samples is not an array"))?;
    let expected_count = expected_sample_count(policy.step_budget, policy.thermal_cadence_steps)?;
    if samples.len() != expected_count {
        return Err(Rejection::new(
            "thermal_series_incomplete",
            format!(
                "thermal series has {} samples, expected {expected_count}",
                samples.len()
            ),
        ));
    }
    let expected_devices = device_ids(profile)?;
    for (index, sample) in samples.iter().enumerate() {
        exact_fields(sample, "thermal_sample", &THERMAL_SAMPLE_FIELDS)?;
        let sample_ordinal = u64::try_from(index + 1).map_err(|_| {
            Rejection::new(
                "stability_step_overflow",
                "thermal sample ordinal does not fit in u64",
            )
        })?;
        let expected_step = sample_ordinal * policy.thermal_cadence_steps;
        let actual_step = unsigned(sample, "step")?;
        if actual_step != expected_step {
            return Err(Rejection::new(
                "thermal_series_incomplete",
                format!("thermal step is {actual_step}, expected {expected_step}"),
            ));
        }
        let devices = field(sample, "devices")?.as_array().ok_or_else(|| {
            Rejection::new("malformed_stability", "thermal devices is not an array")
        })?;
        if devices.len() != expected_devices.len() {
            return Err(Rejection::new(
                "thermal_device_mismatch",
                format!("step {actual_step}: thermal device count does not match the profile"),
            ));
        }
        for (device, expected_device_id) in devices.iter().zip(&expected_devices) {
            exact_fields(device, "thermal_device", &THERMAL_DEVICE_FIELDS)?;
            let actual_device_id = unsigned(device, "device_id")?;
            if actual_device_id != *expected_device_id {
                return Err(Rejection::new(
                    "thermal_device_mismatch",
                    format!(
                        "step {actual_step}: device ID is {actual_device_id}, expected {expected_device_id}"
                    ),
                ));
            }
            let temperature = finite(device, "temperature_celsius", false).map_err(|error| {
                Rejection::new(
                    "invalid_thermal_sample",
                    format!(
                        "step {actual_step}, device {actual_device_id}: {}",
                        error.detail
                    ),
                )
            })?;
            let power = finite(device, "power_watts", false).map_err(|error| {
                Rejection::new(
                    "invalid_thermal_sample",
                    format!(
                        "step {actual_step}, device {actual_device_id}: {}",
                        error.detail
                    ),
                )
            })?;
            if temperature > policy.maximum_temperature_celsius {
                return Err(Rejection::new(
                    "thermal_bound_exceeded",
                    format!(
                        "step {actual_step}, device {actual_device_id}: temperature {temperature} exceeds maximum {}",
                        policy.maximum_temperature_celsius
                    ),
                ));
            }
            if power > policy.maximum_power_watts {
                return Err(Rejection::new(
                    "thermal_bound_exceeded",
                    format!(
                        "step {actual_step}, device {actual_device_id}: power {power} exceeds maximum {}",
                        policy.maximum_power_watts
                    ),
                ));
            }
        }
    }
    Ok(thermal.clone())
}

fn validate_heldout(raw: &Value, policy: &StabilityPolicy) -> Result<(Value, f64), Rejection> {
    let heldout = field(raw, "heldout_eval")?;
    exact_fields(heldout, "heldout_eval", &HELDOUT_FIELDS)?;
    let split_identity = hash(heldout, "split_identity")?;
    let reference_identity = hash(heldout, "reference_batch_identity")?;
    if split_identity == reference_identity || string(heldout, "split_role")? != "held_out" {
        return Err(Rejection::new(
            "eval_not_held_out",
            "eval split is not distinct from the periodic reference batch",
        ));
    }
    if string(heldout, "metric")? != "loss" {
        return Err(Rejection::new(
            "heldout_metric_mismatch",
            "held-out eval metric must be loss",
        ));
    }
    if string(heldout, "generalization_non_claim")? != policy.generalization_non_claim {
        return Err(Rejection::new(
            "heldout_non_claim_missing",
            "held-out eval result is missing the required generalization non-claim",
        ));
    }
    let before_loss = finite(heldout, "before_loss", true)?;
    let after_loss = finite(heldout, "after_loss", false)?;
    let ratio = after_loss / before_loss;
    if ratio > policy.maximum_eval_loss_ratio {
        return Err(Rejection::new(
            "heldout_eval_no_improvement",
            format!(
                "held-out eval loss ratio {ratio} exceeds maximum {}",
                policy.maximum_eval_loss_ratio
            ),
        ));
    }
    Ok((heldout.clone(), ratio))
}

pub fn summarize_stability(
    raw: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    let policy_values = validate_policy(policy, profile)?;
    exact_fields(raw, "stability_raw", &RAW_FIELDS)?;
    if string(raw, "status")? != "completed" {
        return Err(Rejection::new(
            "long_run_incomplete",
            "long run status is not completed",
        ));
    }
    let completed_steps = unsigned(raw, "completed_steps")?;
    let checked_steps = unsigned(raw, "checked_steps")?;
    if completed_steps != policy_values.step_budget || checked_steps != completed_steps {
        return Err(Rejection::new(
            "long_run_incomplete",
            format!(
                "long run completed {completed_steps} steps and checked {checked_steps}; expected {}",
                policy_values.step_budget
            ),
        ));
    }
    let expected_profile_identity = blake3_identity(profile);
    if hash(raw, "parity_profile_identity")? != expected_profile_identity {
        return Err(Rejection::new(
            "stability_profile_mismatch",
            "long run does not bind the active parity profile",
        ));
    }
    let machine_facts_identity = validate_machine_facts(raw)?;
    let trajectory = validate_trajectory(raw, &policy_values)?;
    let thermal = validate_thermal(raw, profile, &policy_values)?;
    let (heldout_eval, _ratio) = validate_heldout(raw, &policy_values)?;
    Ok(json!({
        "schema_version": STABILITY_SCHEMA_VERSION,
        "kind": STABILITY_KIND,
        "profile_id": policy_values.profile_id,
        "parity_profile_identity": expected_profile_identity,
        "policy_identity": blake3_identity(policy),
        "machine_facts_identity": machine_facts_identity,
        "completed_steps": completed_steps,
        "trajectory": trajectory,
        "thermal": thermal,
        "heldout_eval": heldout_eval,
        "verdict": "pass",
    }))
}

pub fn admit_stability_receipt(
    receipt: &Value,
    raw: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<StabilityAdmission, Rejection> {
    exact_fields(receipt, "stability_receipt", &RECEIPT_FIELDS)?;
    let expected = summarize_stability(raw, profile, policy)?;
    if receipt != &expected {
        return Err(Rejection::new(
            "stability_receipt_mismatch",
            "stability receipt does not match the deterministic summary",
        ));
    }
    if string(receipt, "kind")? != STABILITY_KIND || string(receipt, "verdict")? != "pass" {
        return Err(Rejection::new(
            "long_run_incomplete",
            "stability receipt is not a passing receipt",
        ));
    }
    let trajectory_samples = field(receipt, "trajectory")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_stability", "trajectory is not an array"))?
        .len();
    let thermal_samples = field(field(receipt, "thermal")?, "samples")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_stability", "thermal samples is not an array"))?
        .len();
    let heldout = field(receipt, "heldout_eval")?;
    let eval_loss_ratio =
        finite(heldout, "after_loss", false)? / finite(heldout, "before_loss", true)?;
    Ok(StabilityAdmission {
        identity_blake3: blake3_identity(receipt),
        completed_steps: unsigned(receipt, "completed_steps")?,
        trajectory_samples: u64::try_from(trajectory_samples).map_err(|_| {
            Rejection::new(
                "stability_sample_count_overflow",
                "trajectory sample count does not fit in u64",
            )
        })?,
        thermal_samples: u64::try_from(thermal_samples).map_err(|_| {
            Rejection::new(
                "stability_sample_count_overflow",
                "thermal sample count does not fit in u64",
            )
        })?,
        eval_loss_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP_BUDGET: u64 = 64;
    const CADENCE: u64 = 16;
    const DEVICE_ID: u64 = 0;
    const TEMPERATURE_CELSIUS: f64 = 70.0;
    const POWER_WATTS: f64 = 100.0;

    fn profile() -> Value {
        json!({
            "profile_id": "test-parity",
            "device": {"physical_device_ids": [DEVICE_ID]},
        })
    }

    fn policy() -> Value {
        json!({
            "schema_version": STABILITY_SCHEMA_VERSION,
            "profile_id": "test-parity-stability",
            "parity_profile_id": "test-parity",
            "step_budget": STEP_BUDGET,
            "trajectory_cadence_steps": CADENCE,
            "thermal_cadence_steps": CADENCE,
            "maximum_loss": 10.0,
            "maximum_gradient_norm": 100.0,
            "maximum_temperature_celsius": 90.0,
            "maximum_power_watts": 200.0,
            "maximum_eval_loss_ratio": 0.95,
            "thermal_source": "tt-smi-jsonl",
            "generalization_non_claim": "held-out loss does not prove generalization beyond the bound split",
            "hash_algorithm": "blake3",
        })
    }

    fn raw() -> Value {
        let facts = bind_machine_facts(&json!({
            "board_identity": "p150a-test",
            "firmware_version": "firmware-test",
            "mesh_descriptor_identity": "mesh-test",
            "pcie_link": "gen4-x16",
            "power_state": "performance",
        }))
        .unwrap();
        let steps = [CADENCE, CADENCE * 2, CADENCE * 3, STEP_BUDGET];
        let trajectory = steps
            .iter()
            .map(|step| json!({"step": step, "loss": 2.0, "gradient_norm": 1.0}))
            .collect::<Vec<_>>();
        let thermal_samples = steps
            .iter()
            .map(|step| {
                json!({
                    "step": step,
                    "devices": [{
                        "device_id": DEVICE_ID,
                        "temperature_celsius": TEMPERATURE_CELSIUS,
                        "power_watts": POWER_WATTS,
                    }],
                })
            })
            .collect::<Vec<_>>();
        json!({
            "status": "completed",
            "completed_steps": STEP_BUDGET,
            "checked_steps": STEP_BUDGET,
            "parity_profile_identity": blake3_identity(&profile()),
            "machine_facts": facts,
            "trajectory": trajectory,
            "thermal": {"source": "tt-smi-jsonl", "samples": thermal_samples},
            "heldout_eval": {
                "split_identity": "a".repeat(BLAKE3_HEX_LENGTH),
                "reference_batch_identity": "b".repeat(BLAKE3_HEX_LENGTH),
                "split_role": "held_out",
                "metric": "loss",
                "before_loss": 4.0,
                "after_loss": 3.0,
                "generalization_non_claim": "held-out loss does not prove generalization beyond the bound split",
            },
        })
    }

    #[test]
    fn tt_smi_snapshot_is_normalized_for_selected_devices() {
        let snapshot = json!({
            "device_info": [{
                "telemetry": {"asic_temperature": " 70.5", "power": "100.0"}
            }]
        });
        let normalized = normalize_tt_smi_snapshot(&snapshot, &profile()).unwrap();
        assert_eq!(normalized["devices"][0]["device_id"], DEVICE_ID);
        assert_eq!(normalized["devices"][0]["temperature_celsius"], 70.5);
        assert_eq!(normalized["devices"][0]["power_watts"], POWER_WATTS);
    }

    #[test]
    fn tt_smi_snapshot_rejects_a_missing_selected_device() {
        let rejection =
            normalize_tt_smi_snapshot(&json!({"device_info": []}), &profile()).unwrap_err();
        assert_eq!(rejection.code, "missing_thermal_device");
        assert!(rejection.detail.contains("device 0"));
    }

    #[test]
    fn stable_long_run_is_deterministic_and_admitted() {
        let first = summarize_stability(&raw(), &profile(), &policy()).unwrap();
        let second = summarize_stability(&raw(), &profile(), &policy()).unwrap();
        assert_eq!(first, second);
        let admission = admit_stability_receipt(&first, &raw(), &profile(), &policy()).unwrap();
        assert_eq!(admission.completed_steps, STEP_BUDGET);
        assert_eq!(admission.trajectory_samples, STEP_BUDGET / CADENCE);
        assert_eq!(admission.thermal_samples, STEP_BUDGET / CADENCE);
    }

    #[test]
    fn divergence_names_the_step() {
        let mut raw = raw();
        raw["trajectory"][1]["gradient_norm"] = json!(101.0);
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "long_run_divergence");
        assert!(rejection.detail.contains("step 32"));
    }

    #[test]
    fn non_finite_trajectory_names_the_step() {
        let mut raw = raw();
        raw["trajectory"][2]["loss"] = json!("NaN");
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "invalid_stability_metric");
        assert!(rejection.detail.contains("step 48"));
    }

    #[test]
    fn non_finite_thermal_names_device_and_step() {
        let mut raw = raw();
        raw["thermal"]["samples"][1]["devices"][0]["power_watts"] = json!("NaN");
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "invalid_thermal_sample");
        assert!(rejection.detail.contains("step 32, device 0"));
    }

    #[test]
    fn thermal_limit_is_enforced() {
        let mut raw = raw();
        raw["thermal"]["samples"][0]["devices"][0]["temperature_celsius"] = json!(91.0);
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "thermal_bound_exceeded");
    }

    #[test]
    fn reference_batch_is_not_held_out() {
        let mut raw = raw();
        raw["heldout_eval"]["split_identity"] =
            raw["heldout_eval"]["reference_batch_identity"].clone();
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "eval_not_held_out");
    }

    #[test]
    fn heldout_eval_must_improve() {
        let mut raw = raw();
        raw["heldout_eval"]["after_loss"] = json!(4.0);
        let rejection = summarize_stability(&raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "heldout_eval_no_improvement");
    }

    #[test]
    fn changed_receipt_is_rejected() {
        let raw = raw();
        let mut receipt = summarize_stability(&raw, &profile(), &policy()).unwrap();
        receipt["completed_steps"] = json!(1);
        let rejection = admit_stability_receipt(&receipt, &raw, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "stability_receipt_mismatch");
    }
}
