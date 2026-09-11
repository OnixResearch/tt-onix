//! Pure checked-baseline, regression-gate, determinism, and metric-sanity logic.
//!
//! r[impl tenstorrent.harness.baseline_store]
//! r[impl tenstorrent.harness.regression_gate]
//! r[impl tenstorrent.harness.determinism]
//! r[impl tenstorrent.harness.metric_sanity]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::{Rejection, admit_measurement_receipt, blake3_identity};

const REGRESSION_SCHEMA_VERSION: u64 = 1;
const REGRESSION_GATE_KIND: &str = "rwkv7_lora_regression_gate";
const DETERMINISM_KIND: &str = "rwkv7_lora_determinism";
const DIRECTION_LOWER_IS_BETTER: &str = "lower_is_better";
const BASELINE_STORE_FIELDS: [&str; 2] = ["schema_version", "records"];
const BASELINE_BODY_FIELDS: [&str; 9] = [
    "schema_version",
    "profile_identity",
    "measurement_profile_id",
    "machine_facts_identity",
    "run_identity",
    "repeat_count",
    "policy_identity",
    "metrics",
    "determinism_tolerances",
];
const BASELINE_RECORD_FIELDS: [&str; 10] = [
    "schema_version",
    "profile_identity",
    "measurement_profile_id",
    "machine_facts_identity",
    "run_identity",
    "repeat_count",
    "policy_identity",
    "metrics",
    "determinism_tolerances",
    "identity_blake3",
];
const POLICY_FIELDS: [&str; 5] = [
    "schema_version",
    "profile_id",
    "metric_bounds",
    "determinism_tolerances",
    "hash_algorithm",
];
const METRIC_BOUND_FIELDS: [&str; 3] = ["metric", "direction", "maximum_regression_fraction"];
const METRIC_FIELDS: [&str; 4] = ["name", "value", "direction", "maximum_regression_fraction"];
const DETERMINISM_TOLERANCE_FIELDS: [&str; 2] = ["time_seconds", "loss"];
const REGRESSION_RECEIPT_FIELDS: [&str; 8] = [
    "schema_version",
    "kind",
    "profile_identity",
    "machine_facts_identity",
    "measurement_identity",
    "baseline_identity",
    "comparisons",
    "verdict",
];
const COMPARISON_FIELDS: [&str; 6] = [
    "metric",
    "baseline_value",
    "current_value",
    "maximum_regression_fraction",
    "maximum_value",
    "passed",
];
const DETERMINISM_RECEIPT_FIELDS: [&str; 8] = [
    "schema_version",
    "kind",
    "profile_identity",
    "left_measurement_identity",
    "right_measurement_identity",
    "machine_facts_identity",
    "metrics",
    "verdict",
];
const DETERMINISM_METRIC_FIELDS: [&str; 3] = ["maximum_difference", "tolerance", "passed"];
const REQUIRED_METRICS: [&str; 3] = ["time_seconds.mean", "time_seconds.p99", "loss.mean"];
const BLAKE3_HEX_LENGTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionAdmission {
    pub identity_blake3: String,
    pub baseline_identity: String,
    pub comparison_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeterminismAdmission {
    pub identity_blake3: String,
    pub maximum_time_difference: f64,
    pub maximum_loss_difference: f64,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, Rejection> {
    value.as_object().ok_or_else(|| {
        Rejection::new(
            "malformed_regression",
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
            "unexpected_regression_field",
            format!("field '{name}' has fields {actual:?}; expected {expected:?}"),
        ));
    }
    Ok(())
}

fn required_field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Rejection> {
    value.get(key).ok_or_else(|| {
        Rejection::new(
            "missing_regression_field",
            format!("missing regression field '{key}'"),
        )
    })
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    let text = required_field(value, key)?.as_str().ok_or_else(|| {
        Rejection::new(
            "malformed_regression",
            format!("regression field '{key}' is not a string"),
        )
    })?;
    if text.is_empty() {
        return Err(Rejection::new(
            "malformed_regression",
            format!("regression field '{key}' is empty"),
        ));
    }
    Ok(text)
}

fn required_hash<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    let identity = required_string(value, key)?;
    let valid = identity.len() == BLAKE3_HEX_LENGTH
        && identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(Rejection::new(
            "invalid_regression_identity",
            format!("field '{key}' is not a lowercase BLAKE3 identity"),
        ));
    }
    Ok(identity)
}

fn required_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    required_field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "malformed_regression",
            format!("regression field '{key}' is not an unsigned integer"),
        )
    })
}

fn finite_number(value: &Value, key: &str, positive: bool) -> Result<f64, Rejection> {
    let number = required_field(value, key)?.as_f64().ok_or_else(|| {
        Rejection::new(
            "invalid_regression_metric",
            format!("metric '{key}' is not a finite number"),
        )
    })?;
    if !number.is_finite() {
        return Err(Rejection::new(
            "invalid_regression_metric",
            format!("metric '{key}' is not finite"),
        ));
    }
    if positive && number <= 0.0 {
        return Err(Rejection::new(
            "invalid_regression_metric",
            format!("metric '{key}' must be positive, got {number}"),
        ));
    }
    if !positive && number < 0.0 {
        return Err(Rejection::new(
            "invalid_regression_metric",
            format!("metric '{key}' must be non-negative, got {number}"),
        ));
    }
    Ok(number)
}

fn measurement_profile_id(profile: &Value) -> Result<&str, Rejection> {
    required_string(
        required_field(profile, "measurement_profile")?,
        "profile_id",
    )
}

fn repeat_count(profile: &Value) -> Result<u64, Rejection> {
    required_u64(
        required_field(profile, "measurement_profile")?,
        "repeat_count",
    )
}

fn machine_facts_identity(receipt: &Value) -> Result<String, Rejection> {
    Ok(blake3_identity(required_field(receipt, "machine_facts")?))
}

fn metric_value(receipt: &Value, name: &str) -> Result<f64, Rejection> {
    let statistics = required_field(receipt, "statistics")?;
    match name {
        "time_seconds.mean" => {
            finite_number(required_field(statistics, "time_seconds")?, "mean", true)
        }
        "time_seconds.p99" => {
            finite_number(required_field(statistics, "time_seconds")?, "p99", true)
        }
        "loss.mean" => finite_number(required_field(statistics, "loss")?, "mean", false),
        _ => Err(Rejection::new(
            "unsupported_regression_metric",
            format!("metric '{name}' is not supported"),
        )),
    }
}

fn validate_policy<'a>(policy: &'a Value, profile: &Value) -> Result<&'a Value, Rejection> {
    exact_fields(policy, "regression_policy", &POLICY_FIELDS)?;
    let schema_version = required_u64(policy, "schema_version")?;
    if schema_version != REGRESSION_SCHEMA_VERSION {
        return Err(Rejection::new(
            "regression_policy_schema_mismatch",
            format!("policy schema is {schema_version}, expected {REGRESSION_SCHEMA_VERSION}"),
        ));
    }
    let policy_profile_id = required_string(policy, "profile_id")?;
    let expected_profile_id = measurement_profile_id(profile)?;
    if policy_profile_id != expected_profile_id {
        return Err(Rejection::new(
            "regression_profile_mismatch",
            format!("policy profile is '{policy_profile_id}', expected '{expected_profile_id}'"),
        ));
    }
    if required_string(policy, "hash_algorithm")? != "blake3" {
        return Err(Rejection::new(
            "regression_hash_mismatch",
            "regression policy must use BLAKE3",
        ));
    }
    let tolerances = required_field(policy, "determinism_tolerances")?;
    exact_fields(
        tolerances,
        "determinism_tolerances",
        &DETERMINISM_TOLERANCE_FIELDS,
    )?;
    finite_number(tolerances, "time_seconds", false)?;
    finite_number(tolerances, "loss", false)?;
    Ok(policy)
}

fn metric_policy(policy: &Value) -> Result<Vec<Value>, Rejection> {
    let bounds = required_field(policy, "metric_bounds")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_regression", "metric_bounds is not an array"))?;
    let mut actual = Vec::with_capacity(bounds.len());
    let mut names = BTreeSet::new();
    for bound in bounds {
        exact_fields(bound, "metric_bound", &METRIC_BOUND_FIELDS)?;
        let name = required_string(bound, "metric")?;
        let direction = required_string(bound, "direction")?;
        if direction != DIRECTION_LOWER_IS_BETTER {
            return Err(Rejection::new(
                "unsupported_regression_direction",
                format!("metric '{name}' direction is '{direction}'"),
            ));
        }
        finite_number(bound, "maximum_regression_fraction", false)?;
        if !names.insert(name) {
            return Err(Rejection::new(
                "duplicate_regression_metric",
                format!("metric '{name}' appears more than once"),
            ));
        }
        actual.push(name);
    }
    if actual != REQUIRED_METRICS {
        return Err(Rejection::new(
            "regression_metric_set_mismatch",
            format!("policy metrics are {actual:?}; expected {REQUIRED_METRICS:?}"),
        ));
    }
    Ok(bounds.clone())
}

pub fn create_baseline(
    measurement_receipt: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    admit_measurement_receipt(measurement_receipt, profile)?;
    validate_policy(policy, profile)?;
    let bounds = metric_policy(policy)?;
    let mut metrics = Vec::with_capacity(bounds.len());
    for bound in bounds {
        let name = required_string(&bound, "metric")?;
        metrics.push(json!({
            "name": name,
            "value": metric_value(measurement_receipt, name)?,
            "direction": required_string(&bound, "direction")?,
            "maximum_regression_fraction": finite_number(
                &bound,
                "maximum_regression_fraction",
                false,
            )?,
        }));
    }
    let body = json!({
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "profile_identity": blake3_identity(profile),
        "measurement_profile_id": measurement_profile_id(profile)?,
        "machine_facts_identity": machine_facts_identity(measurement_receipt)?,
        "run_identity": blake3_identity(measurement_receipt),
        "repeat_count": repeat_count(profile)?,
        "policy_identity": blake3_identity(policy),
        "metrics": metrics,
        "determinism_tolerances": required_field(policy, "determinism_tolerances")?,
    });
    let mut record = object(&body, "baseline_body")?.clone();
    record.insert("identity_blake3".to_string(), json!(blake3_identity(&body)));
    Ok(Value::Object(record))
}

fn validate_baseline_record(record: &Value) -> Result<(), Rejection> {
    exact_fields(record, "baseline_record", &BASELINE_RECORD_FIELDS)?;
    let mut body = object(record, "baseline_record")?.clone();
    let identity = required_string(record, "identity_blake3")?;
    body.remove("identity_blake3");
    let body = Value::Object(body);
    exact_fields(&body, "baseline_body", &BASELINE_BODY_FIELDS)?;
    if identity != blake3_identity(&body) {
        return Err(Rejection::new(
            "baseline_identity_mismatch",
            "baseline BLAKE3 identity does not bind its record",
        ));
    }
    if required_u64(record, "schema_version")? != REGRESSION_SCHEMA_VERSION
        || required_u64(record, "repeat_count")? == 0
    {
        return Err(Rejection::new(
            "baseline_schema_mismatch",
            "baseline schema or repeat count is unsupported",
        ));
    }
    for key in [
        "profile_identity",
        "machine_facts_identity",
        "run_identity",
        "policy_identity",
        "identity_blake3",
    ] {
        required_hash(record, key)?;
    }
    let tolerances = required_field(record, "determinism_tolerances")?;
    exact_fields(
        tolerances,
        "baseline_determinism_tolerances",
        &DETERMINISM_TOLERANCE_FIELDS,
    )?;
    finite_number(tolerances, "time_seconds", false)?;
    finite_number(tolerances, "loss", false)?;
    let metrics = required_field(record, "metrics")?
        .as_array()
        .ok_or_else(|| {
            Rejection::new("malformed_regression", "baseline metrics is not an array")
        })?;
    let mut names = Vec::with_capacity(metrics.len());
    for metric in metrics {
        exact_fields(metric, "baseline_metric", &METRIC_FIELDS)?;
        let name = required_string(metric, "name")?;
        names.push(name);
        finite_number(metric, "value", false)?;
        if required_string(metric, "direction")? != DIRECTION_LOWER_IS_BETTER {
            return Err(Rejection::new(
                "unsupported_regression_direction",
                format!("baseline metric '{name}' has an unsupported direction"),
            ));
        }
        finite_number(metric, "maximum_regression_fraction", false)?;
    }
    if names != REQUIRED_METRICS {
        return Err(Rejection::new(
            "regression_metric_set_mismatch",
            format!("baseline metrics are {names:?}; expected {REQUIRED_METRICS:?}"),
        ));
    }
    Ok(())
}

pub fn admit_baseline_store(
    store: &Value,
    profile: &Value,
    measurement_receipt: &Value,
) -> Result<Value, Rejection> {
    admit_measurement_receipt(measurement_receipt, profile)?;
    exact_fields(store, "baseline_store", &BASELINE_STORE_FIELDS)?;
    if required_u64(store, "schema_version")? != REGRESSION_SCHEMA_VERSION {
        return Err(Rejection::new(
            "baseline_store_schema_mismatch",
            "baseline store schema version is unsupported",
        ));
    }
    let records = required_field(store, "records")?
        .as_array()
        .ok_or_else(|| {
            Rejection::new("malformed_regression", "baseline records is not an array")
        })?;
    let expected_profile = blake3_identity(profile);
    let expected_machine = machine_facts_identity(measurement_receipt)?;
    let mut selected = None;
    let mut keys = BTreeSet::new();
    for record in records {
        validate_baseline_record(record)?;
        let profile_identity = required_string(record, "profile_identity")?;
        let machine_identity = required_string(record, "machine_facts_identity")?;
        let key = (profile_identity, machine_identity);
        if !keys.insert(key) {
            return Err(Rejection::new(
                "duplicate_baseline_key",
                format!("baseline key {key:?} appears more than once"),
            ));
        }
        if profile_identity == expected_profile && machine_identity == expected_machine {
            selected = Some(record.clone());
        }
    }
    let selected = selected.ok_or_else(|| {
        Rejection::new(
            "baseline_key_mismatch",
            format!(
                "no baseline matches profile '{expected_profile}' and machine facts '{expected_machine}'"
            ),
        )
    })?;
    if required_string(&selected, "measurement_profile_id")? != measurement_profile_id(profile)?
        || required_u64(&selected, "repeat_count")? != repeat_count(profile)?
    {
        return Err(Rejection::new(
            "baseline_profile_mismatch",
            "baseline measurement profile or repeat count does not match the active profile",
        ));
    }
    Ok(selected)
}

fn baseline_metric_map(record: &Value) -> Result<BTreeMap<String, (f64, f64)>, Rejection> {
    let metrics = required_field(record, "metrics")?
        .as_array()
        .expect("validated baseline metrics is an array");
    let mut output = BTreeMap::new();
    for metric in metrics {
        output.insert(
            required_string(metric, "name")?.to_string(),
            (
                finite_number(metric, "value", false)?,
                finite_number(metric, "maximum_regression_fraction", false)?,
            ),
        );
    }
    Ok(output)
}

pub fn gate_regression(
    measurement_receipt: &Value,
    profile: &Value,
    baseline_store: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    admit_measurement_receipt(measurement_receipt, profile)?;
    validate_policy(policy, profile)?;
    metric_policy(policy)?;
    let baseline = admit_baseline_store(baseline_store, profile, measurement_receipt)?;
    if required_string(&baseline, "policy_identity")? != blake3_identity(policy) {
        return Err(Rejection::new(
            "regression_policy_mismatch",
            "baseline binds a different regression policy",
        ));
    }
    let baseline_metrics = baseline_metric_map(&baseline)?;
    let mut comparisons = Vec::with_capacity(REQUIRED_METRICS.len());
    for name in REQUIRED_METRICS {
        let current = metric_value(measurement_receipt, name)?;
        let (baseline_value, maximum_fraction) = baseline_metrics[name];
        let maximum_value = baseline_value * (1.0 + maximum_fraction);
        if current > maximum_value {
            return Err(Rejection::new(
                "regression_gate_failure",
                format!(
                    "metric '{name}' is {current}, baseline {baseline_value}, maximum regression fraction {maximum_fraction}, maximum value {maximum_value}"
                ),
            ));
        }
        comparisons.push(json!({
            "metric": name,
            "baseline_value": baseline_value,
            "current_value": current,
            "maximum_regression_fraction": maximum_fraction,
            "maximum_value": maximum_value,
            "passed": true,
        }));
    }
    Ok(json!({
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "kind": REGRESSION_GATE_KIND,
        "profile_identity": blake3_identity(profile),
        "machine_facts_identity": machine_facts_identity(measurement_receipt)?,
        "measurement_identity": blake3_identity(measurement_receipt),
        "baseline_identity": required_string(&baseline, "identity_blake3")?,
        "comparisons": comparisons,
        "verdict": "pass",
    }))
}

pub fn admit_regression_receipt(receipt: &Value) -> Result<RegressionAdmission, Rejection> {
    exact_fields(receipt, "regression_receipt", &REGRESSION_RECEIPT_FIELDS)?;
    if required_u64(receipt, "schema_version")? != REGRESSION_SCHEMA_VERSION
        || required_string(receipt, "kind")? != REGRESSION_GATE_KIND
        || required_string(receipt, "verdict")? != "pass"
    {
        return Err(Rejection::new(
            "regression_gate_failure",
            "regression receipt is not a passing gate receipt",
        ));
    }
    for key in [
        "profile_identity",
        "machine_facts_identity",
        "measurement_identity",
        "baseline_identity",
    ] {
        required_hash(receipt, key)?;
    }
    let comparisons = required_field(receipt, "comparisons")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_regression", "comparisons is not an array"))?;
    if comparisons.len() != REQUIRED_METRICS.len() {
        return Err(Rejection::new(
            "regression_metric_set_mismatch",
            "regression receipt has the wrong comparison count",
        ));
    }
    for (comparison, expected_name) in comparisons.iter().zip(REQUIRED_METRICS) {
        exact_fields(comparison, "regression_comparison", &COMPARISON_FIELDS)?;
        if required_string(comparison, "metric")? != expected_name {
            return Err(Rejection::new(
                "regression_metric_set_mismatch",
                format!("comparison metric does not match '{expected_name}'"),
            ));
        }
        let baseline_value = finite_number(comparison, "baseline_value", false)?;
        let current_value = finite_number(comparison, "current_value", false)?;
        let maximum_fraction = finite_number(comparison, "maximum_regression_fraction", false)?;
        let maximum_value = finite_number(comparison, "maximum_value", false)?;
        let expected_maximum = baseline_value * (1.0 + maximum_fraction);
        if maximum_value != expected_maximum
            || current_value > maximum_value
            || required_field(comparison, "passed")?.as_bool() != Some(true)
        {
            return Err(Rejection::new(
                "regression_gate_failure",
                format!("comparison for metric '{expected_name}' is not a valid pass"),
            ));
        }
    }
    let comparison_count = u64::try_from(comparisons.len()).map_err(|_| {
        Rejection::new(
            "comparison_count_overflow",
            "comparison count does not fit in u64",
        )
    })?;
    Ok(RegressionAdmission {
        identity_blake3: blake3_identity(receipt),
        baseline_identity: required_hash(receipt, "baseline_identity")?.to_string(),
        comparison_count,
    })
}

fn run_values(receipt: &Value, metric: &str) -> Result<Vec<f64>, Rejection> {
    let runs = required_field(receipt, "runs")?.as_array().ok_or_else(|| {
        Rejection::new("malformed_regression", "measurement runs is not an array")
    })?;
    let mut output = Vec::with_capacity(runs.len());
    for (index, run) in runs.iter().enumerate() {
        let value = finite_number(run, metric, metric == "time_seconds").map_err(|error| {
            Rejection::new(error.code, format!("run {index}: {}", error.detail))
        })?;
        output.push(value);
    }
    Ok(output)
}

fn maximum_difference(left: &[f64], right: &[f64], metric: &str) -> Result<f64, Rejection> {
    if left.len() != right.len() {
        return Err(Rejection::new(
            "determinism_repeat_mismatch",
            format!(
                "metric '{metric}' has {} left runs and {} right runs",
                left.len(),
                right.len()
            ),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max))
}

pub fn compare_determinism(
    left_receipt: &Value,
    right_receipt: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    admit_measurement_receipt(left_receipt, profile)?;
    admit_measurement_receipt(right_receipt, profile)?;
    validate_policy(policy, profile)?;
    if machine_facts_identity(left_receipt)? != machine_facts_identity(right_receipt)? {
        return Err(Rejection::new(
            "determinism_machine_mismatch",
            "determinism receipts bind different machine facts",
        ));
    }
    let tolerances = required_field(policy, "determinism_tolerances")?;
    let time_tolerance = finite_number(tolerances, "time_seconds", false)?;
    let loss_tolerance = finite_number(tolerances, "loss", false)?;
    let maximum_time_difference = maximum_difference(
        &run_values(left_receipt, "time_seconds")?,
        &run_values(right_receipt, "time_seconds")?,
        "time_seconds",
    )?;
    let maximum_loss_difference = maximum_difference(
        &run_values(left_receipt, "loss")?,
        &run_values(right_receipt, "loss")?,
        "loss",
    )?;
    if maximum_time_difference > time_tolerance {
        return Err(Rejection::new(
            "determinism_divergence",
            format!(
                "time_seconds maximum difference {maximum_time_difference} exceeds tolerance {time_tolerance}"
            ),
        ));
    }
    if maximum_loss_difference > loss_tolerance {
        return Err(Rejection::new(
            "determinism_divergence",
            format!(
                "loss maximum difference {maximum_loss_difference} exceeds tolerance {loss_tolerance}"
            ),
        ));
    }
    Ok(json!({
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "kind": DETERMINISM_KIND,
        "profile_identity": blake3_identity(profile),
        "left_measurement_identity": blake3_identity(left_receipt),
        "right_measurement_identity": blake3_identity(right_receipt),
        "machine_facts_identity": machine_facts_identity(left_receipt)?,
        "metrics": {
            "time_seconds": {
                "maximum_difference": maximum_time_difference,
                "tolerance": time_tolerance,
                "passed": true,
            },
            "loss": {
                "maximum_difference": maximum_loss_difference,
                "tolerance": loss_tolerance,
                "passed": true,
            },
        },
        "verdict": "pass",
    }))
}

pub fn admit_determinism_receipt(receipt: &Value) -> Result<DeterminismAdmission, Rejection> {
    exact_fields(receipt, "determinism_receipt", &DETERMINISM_RECEIPT_FIELDS)?;
    if required_u64(receipt, "schema_version")? != REGRESSION_SCHEMA_VERSION
        || required_string(receipt, "kind")? != DETERMINISM_KIND
        || required_string(receipt, "verdict")? != "pass"
    {
        return Err(Rejection::new(
            "determinism_divergence",
            "determinism receipt is not a passing determinism receipt",
        ));
    }
    for key in [
        "profile_identity",
        "left_measurement_identity",
        "right_measurement_identity",
        "machine_facts_identity",
    ] {
        required_hash(receipt, key)?;
    }
    let metrics = required_field(receipt, "metrics")?;
    exact_fields(
        metrics,
        "determinism_metrics",
        &DETERMINISM_TOLERANCE_FIELDS,
    )?;
    let time = required_field(metrics, "time_seconds")?;
    let loss = required_field(metrics, "loss")?;
    let maximum_time_difference = admit_determinism_metric(time, "time_seconds")?;
    let maximum_loss_difference = admit_determinism_metric(loss, "loss")?;
    Ok(DeterminismAdmission {
        identity_blake3: blake3_identity(receipt),
        maximum_time_difference,
        maximum_loss_difference,
    })
}

fn admit_determinism_metric(metric: &Value, name: &str) -> Result<f64, Rejection> {
    exact_fields(metric, name, &DETERMINISM_METRIC_FIELDS)?;
    let difference = finite_number(metric, "maximum_difference", false)?;
    let tolerance = finite_number(metric, "tolerance", false)?;
    if difference > tolerance || required_field(metric, "passed")?.as_bool() != Some(true) {
        return Err(Rejection::new(
            "determinism_divergence",
            format!("determinism metric '{name}' is not a valid pass"),
        ));
    }
    Ok(difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HEX_HASH_LEN, bind_machine_facts, summarize_measurement};

    const REPEAT_COUNT: u64 = 3;

    fn profile() -> Value {
        json!({
            "profile_id": "parity-profile",
            "parity_profile": {"profile_id": "parity-profile"},
            "measurement_profile": {
                "profile_id": "measurement-profile",
                "parity_profile_id": "parity-profile",
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

    fn policy() -> Value {
        json!({
            "schema_version": REGRESSION_SCHEMA_VERSION,
            "profile_id": "measurement-profile",
            "metric_bounds": [
                {
                    "metric": "time_seconds.mean",
                    "direction": DIRECTION_LOWER_IS_BETTER,
                    "maximum_regression_fraction": 0.1,
                },
                {
                    "metric": "time_seconds.p99",
                    "direction": DIRECTION_LOWER_IS_BETTER,
                    "maximum_regression_fraction": 0.1,
                },
                {
                    "metric": "loss.mean",
                    "direction": DIRECTION_LOWER_IS_BETTER,
                    "maximum_regression_fraction": 0.1,
                },
            ],
            "determinism_tolerances": {"time_seconds": 0.01, "loss": 0.01},
            "hash_algorithm": "blake3",
        })
    }

    fn receipt(time: f64, loss: f64) -> Value {
        let facts = bind_machine_facts(&json!({
            "board_identity": "p150a",
            "firmware_version": "firmware-1",
            "mesh_descriptor_identity": "mesh-1",
            "pcie_link": "gen4-x16",
            "power_state": "performance",
        }))
        .unwrap();
        let raw = json!({
            "stage_times_seconds": {
                "launch": 1.0,
                "compile": 2.0,
                "step": time,
                "checkpoint": 0.0,
                "teardown": 1.0,
                "end_to_end": time + 4.0,
            },
            "cpu_baseline": {
                "identity_blake3": "a".repeat(HEX_HASH_LEN),
                "step_seconds": 12.0,
            },
            "machine_facts": facts,
            "runs": [
                {"time_seconds": time, "loss": loss},
                {"time_seconds": time, "loss": loss},
                {"time_seconds": time, "loss": loss},
            ],
        });
        summarize_measurement(&raw, &profile()).unwrap()
    }

    fn store(baseline: &Value) -> Value {
        json!({"schema_version": REGRESSION_SCHEMA_VERSION, "records": [baseline]})
    }

    #[test]
    fn baseline_is_deterministic_and_admitted() {
        let receipt = receipt(1.0, 2.0);
        let first = create_baseline(&receipt, &profile(), &policy()).unwrap();
        let second = create_baseline(&receipt, &profile(), &policy()).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            admit_baseline_store(&store(&first), &profile(), &receipt).unwrap(),
            first
        );
    }

    #[test]
    fn mismatched_baseline_identity_is_rejected() {
        let receipt = receipt(1.0, 2.0);
        let mut baseline = create_baseline(&receipt, &profile(), &policy()).unwrap();
        baseline["machine_facts_identity"] = json!("b".repeat(HEX_HASH_LEN));
        let rejection = admit_baseline_store(&store(&baseline), &profile(), &receipt).unwrap_err();
        assert_eq!(rejection.code, "baseline_identity_mismatch");
    }

    #[test]
    fn metric_within_bound_passes_gate() {
        let baseline_receipt = receipt(1.0, 2.0);
        let baseline = create_baseline(&baseline_receipt, &profile(), &policy()).unwrap();
        let current = receipt(1.05, 2.05);
        let gate = gate_regression(&current, &profile(), &store(&baseline), &policy()).unwrap();
        let admission = admit_regression_receipt(&gate).unwrap();
        assert_eq!(admission.comparison_count, REQUIRED_METRICS.len() as u64);
    }

    #[test]
    fn forged_regression_pass_is_rejected() {
        let baseline_receipt = receipt(1.0, 2.0);
        let baseline = create_baseline(&baseline_receipt, &profile(), &policy()).unwrap();
        let mut gate =
            gate_regression(&baseline_receipt, &profile(), &store(&baseline), &policy()).unwrap();
        gate["comparisons"][0]["passed"] = json!(false);
        let rejection = admit_regression_receipt(&gate).unwrap_err();
        assert_eq!(rejection.code, "regression_gate_failure");
    }

    #[test]
    fn changed_policy_is_rejected() {
        let baseline_receipt = receipt(1.0, 2.0);
        let baseline = create_baseline(&baseline_receipt, &profile(), &policy()).unwrap();
        let mut changed_policy = policy();
        changed_policy["metric_bounds"][0]["maximum_regression_fraction"] = json!(1.0);
        let rejection = gate_regression(
            &baseline_receipt,
            &profile(),
            &store(&baseline),
            &changed_policy,
        )
        .unwrap_err();
        assert_eq!(rejection.code, "regression_policy_mismatch");
    }

    #[test]
    fn metric_beyond_bound_fails_gate() {
        let baseline_receipt = receipt(1.0, 2.0);
        let baseline = create_baseline(&baseline_receipt, &profile(), &policy()).unwrap();
        let current = receipt(1.2, 2.0);
        let rejection =
            gate_regression(&current, &profile(), &store(&baseline), &policy()).unwrap_err();
        assert_eq!(rejection.code, "regression_gate_failure");
        assert!(rejection.detail.contains("time_seconds.mean"));
    }

    #[test]
    fn identical_runs_agree() {
        let receipt = receipt(1.0, 2.0);
        let report = compare_determinism(&receipt, &receipt, &profile(), &policy()).unwrap();
        let admission = admit_determinism_receipt(&report).unwrap();
        assert_eq!(admission.maximum_time_difference, 0.0);
        assert_eq!(admission.maximum_loss_difference, 0.0);
    }

    #[test]
    fn forged_determinism_pass_is_rejected() {
        let receipt = receipt(1.0, 2.0);
        let mut report = compare_determinism(&receipt, &receipt, &profile(), &policy()).unwrap();
        report["metrics"]["loss"]["passed"] = json!(false);
        let rejection = admit_determinism_receipt(&report).unwrap_err();
        assert_eq!(rejection.code, "determinism_divergence");
    }

    #[test]
    fn divergent_runs_are_rejected() {
        let left = receipt(1.0, 2.0);
        let right = receipt(1.02, 2.0);
        let rejection = compare_determinism(&left, &right, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "determinism_divergence");
        assert!(rejection.detail.contains("time_seconds"));
    }

    #[test]
    fn invalid_metric_is_rejected_before_gate() {
        let baseline_receipt = receipt(1.0, 2.0);
        let baseline = create_baseline(&baseline_receipt, &profile(), &policy()).unwrap();
        let mut current = receipt(1.0, 2.0);
        current["runs"][0]["time_seconds"] = json!("NaN");
        let rejection =
            gate_regression(&current, &profile(), &store(&baseline), &policy()).unwrap_err();
        assert_eq!(rejection.code, "invalid_metric");
    }

    #[test]
    fn repeat_inconsistent_metric_is_rejected() {
        let left = receipt(1.0, 2.0);
        let mut right = receipt(1.0, 2.0);
        right["runs"].as_array_mut().unwrap().pop();
        let rejection = compare_determinism(&left, &right, &profile(), &policy()).unwrap_err();
        assert_eq!(rejection.code, "repeat_count_mismatch");
    }
}
