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
use serde_json::json;
use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// r[impl tenstorrent.qwen38_p150x2.host_timeline_profile]
// r[verify tenstorrent.qwen38_p150x2.host_timeline_profile]
const PROGRAM_NAME: &str = "qwen36-host-timeline-receipt.rs";
const EAGER_ROOT_FLAG: &str = "--eager-root";
const CANDIDATE_ROOT_FLAG: &str = "--candidate-root";
const EXECUTION_ORDER_FLAG: &str = "--execution-order";
const OUTPUT_FLAG: &str = "--output";
const SELF_TEST_FLAG: &str = "--self-test";
const IDENTITY_FILE: &str = "identity.json";
const ZONES_FILE: &str = "zones.csv";
const IDENTITY_SCHEMA: &str = "qwen36.host-timeline-root.v1";
const RECEIPT_SCHEMA: &str = "qwen36.host-timeline-attribution.v1";
const EXPECTED_ROOTS_PER_MODE: usize = 2;
const EXPECTED_DEVICE_COUNT: usize = 2;
const EXPECTED_DEVICE_IDS: [i64; EXPECTED_DEVICE_COUNT] = [0, 1];
const EXPECTED_MESH_DIMENSION_COUNT: usize = 2;
const EXPECTED_MESH_COLUMNS: usize = 2;
const EXPECTED_MESH_SHAPE: [usize; EXPECTED_MESH_DIMENSION_COUNT] = [1, EXPECTED_MESH_COLUMNS];
const EXPECTED_ORDER_COUNT: usize = 4;
const EXPECTED_ORDER: [&str; EXPECTED_ORDER_COUNT] = ["eager", "candidate", "candidate", "eager"];
const FLAG_WIDTH: usize = 2;
const PAIR_WINDOW: usize = 2;
const MEDIAN_DIVISOR_USIZE: usize = 2;
const MEDIAN_DIVISOR: f64 = 2.0;
const ZERO_NS: i64 = 0;
const SUCCESS: u8 = 0;
const FAILURE: u8 = 1;
const SELF_TEST_EAGER_GAP_NS: i64 = 10;
const SELF_TEST_CANDIDATE_GAP_NS: i64 = 30;
const SELF_TEST_UNSTABLE_GAP_NS: i64 = 100;
const SELF_TEST_EXPECTED_DELTA_NS: f64 =
    (SELF_TEST_CANDIDATE_GAP_NS - SELF_TEST_EAGER_GAP_NS) as f64;
const SELF_TEST_PREFILL_START_NS: i64 = 100;
const SELF_TEST_PREFILL_DURATION_NS: i64 = 200;
const SELF_TEST_FIRST_ENQUEUE_START_NS: i64 = 110;
const SELF_TEST_ENQUEUE_DURATION_NS: i64 = 10;
const SELF_TEST_SECOND_ENQUEUE_BASE_NS: i64 = 120;
const SELF_TEST_SYNC_START_NS: i64 = 180;
const SELF_TEST_SYNC_DURATION_NS: i64 = 20;
const ZONE_NAME_COLUMN: usize = 0;
const ZONE_START_COLUMN: usize = 5;
const ZONE_DURATION_COLUMN: usize = 6;
const ZONE_THREAD_COLUMN: usize = 7;
const ZONE_REQUIRED_COLUMN_COUNT: usize = ZONE_THREAD_COLUMN + 1;
const MAX_DUPLICATE_RATIO: f64 = 1.25;

#[derive(Debug)]
struct Config {
    eager_roots: Vec<PathBuf>,
    candidate_roots: Vec<PathBuf>,
    execution_order: Vec<String>,
    output: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RootIdentity {
    schema: String,
    decode_mode: String,
    root_index: usize,
    runtime_identity: String,
    package_identity: String,
    model_identity: String,
    physical_pcie_device_ids: Vec<i64>,
    mesh_shape: Vec<usize>,
    simulated: bool,
    profiled: bool,
}

#[derive(Clone, Debug)]
struct Zone {
    name: String,
    start_ns: i64,
    duration_ns: i64,
    thread: String,
}

impl Zone {
    fn end_ns(&self) -> Result<i64, String> {
        self.start_ns
            .checked_add(self.duration_ns)
            .ok_or_else(|| format!("zone timestamp overflow for {}", self.name))
    }
}

#[derive(Clone, Debug, Serialize)]
struct RootSummary {
    root: String,
    identity_blake3: String,
    zones_blake3: String,
    prefill_wall_ns: i64,
    enqueue_count: usize,
    enqueue_exec_total_ns: i64,
    inter_enqueue_gap_total_ns: i64,
    largest_inter_enqueue_gap_ns: i64,
    leading_gap_ns: i64,
    trailing_gap_ns: i64,
    synchronization_total_ns: i64,
    non_enqueue_span_ns: i64,
}

#[derive(Clone, Debug, Serialize)]
struct ModeComparability {
    prefill_wall_ratio: f64,
    enqueue_exec_ratio: f64,
    inter_enqueue_gap_ratio: f64,
    valid: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ModeMedian {
    prefill_wall_ns: f64,
    enqueue_count: f64,
    enqueue_exec_total_ns: f64,
    inter_enqueue_gap_total_ns: f64,
    largest_inter_enqueue_gap_ns: f64,
    leading_gap_ns: f64,
    trailing_gap_ns: f64,
    synchronization_total_ns: f64,
    non_enqueue_span_ns: f64,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::from(SUCCESS),
        Err(error) => {
            eprintln!("{PROGRAM_NAME}: {error}");
            ExitCode::from(FAILURE)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args == [SELF_TEST_FLAG] {
        self_test()?;
        println!("{PROGRAM_NAME}: self-test: PASS");
        return Ok(());
    }
    let config = parse_config(&args)?;
    let receipt = build_receipt(&config)?;
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&config.output, encoded).map_err(|error| error.to_string())?;
    println!("{PROGRAM_NAME}: PASS");
    Ok(())
}

fn parse_config(args: &[String]) -> Result<Config, String> {
    let mut eager_roots = Vec::new();
    let mut candidate_roots = Vec::new();
    let mut execution_order: Option<Vec<String>> = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        if index + 1 >= args.len() {
            return Err(format!("missing value for {flag}"));
        }
        let value = &args[index + 1];
        match flag.as_str() {
            EAGER_ROOT_FLAG => eager_roots.push(PathBuf::from(value)),
            CANDIDATE_ROOT_FLAG => candidate_roots.push(PathBuf::from(value)),
            EXECUTION_ORDER_FLAG => {
                execution_order = Some(value.split(',').map(String::from).collect())
            }
            OUTPUT_FLAG => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += FLAG_WIDTH;
    }
    if eager_roots.len() != EXPECTED_ROOTS_PER_MODE
        || candidate_roots.len() != EXPECTED_ROOTS_PER_MODE
    {
        return Err("exactly two eager and two candidate roots are required".into());
    }
    let execution_order = execution_order.ok_or("missing --execution-order")?;
    if execution_order.as_slice() != EXPECTED_ORDER {
        return Err("execution order must be eager,candidate,candidate,eager".into());
    }
    Ok(Config {
        eager_roots,
        candidate_roots,
        execution_order,
        output: output.ok_or("missing --output")?,
    })
}

fn build_receipt(config: &Config) -> Result<serde_json::Value, String> {
    let mut eager = Vec::new();
    let mut candidate = Vec::new();
    let mut identity_reference: Option<RootIdentity> = None;

    for (mode, roots, summaries) in [
        ("eager", &config.eager_roots, &mut eager),
        ("candidate", &config.candidate_roots, &mut candidate),
    ] {
        for (root_index, root) in roots.iter().enumerate() {
            let identity = read_identity(root)?;
            validate_identity(&identity, mode, root_index, identity_reference.as_ref())?;
            if identity_reference.is_none() {
                identity_reference = Some(identity.clone());
            }
            summaries.push(summarize_root(root)?);
        }
    }

    let eager_median = mode_median(&eager);
    let candidate_median = mode_median(&candidate);
    let eager_comparability = mode_comparability(&eager);
    let candidate_comparability = mode_comparability(&candidate);
    let comparable = eager_comparability.valid && candidate_comparability.valid;
    let differences = json!({
        "prefill_wall_ns": difference(eager_median.prefill_wall_ns, candidate_median.prefill_wall_ns),
        "enqueue_exec_total_ns": difference(eager_median.enqueue_exec_total_ns, candidate_median.enqueue_exec_total_ns),
        "inter_enqueue_gap_total_ns": difference(eager_median.inter_enqueue_gap_total_ns, candidate_median.inter_enqueue_gap_total_ns),
        "largest_inter_enqueue_gap_ns": difference(eager_median.largest_inter_enqueue_gap_ns, candidate_median.largest_inter_enqueue_gap_ns),
        "leading_gap_ns": difference(eager_median.leading_gap_ns, candidate_median.leading_gap_ns),
        "trailing_gap_ns": difference(eager_median.trailing_gap_ns, candidate_median.trailing_gap_ns),
        "synchronization_total_ns": difference(eager_median.synchronization_total_ns, candidate_median.synchronization_total_ns),
        "non_enqueue_span_ns": difference(eager_median.non_enqueue_span_ns, candidate_median.non_enqueue_span_ns),
        "enqueue_count": difference(eager_median.enqueue_count, candidate_median.enqueue_count),
    });

    let identity = identity_reference.ok_or("no identity")?;
    Ok(json!({
        "schema": RECEIPT_SCHEMA,
        "verdict": if comparable { "diagnostic-only" } else { "rejected" },
        "rejection_reason": if comparable { serde_json::Value::Null } else { json!("duplicate-root-instability") },
        "maximum_duplicate_ratio": MAX_DUPLICATE_RATIO,
        "execution_order": config.execution_order,
        "identity": {
            "runtime_identity": identity.runtime_identity,
            "package_identity": identity.package_identity,
            "model_identity": identity.model_identity,
            "physical_pcie_device_ids": identity.physical_pcie_device_ids,
            "mesh_shape": identity.mesh_shape,
        },
        "eager": { "roots": eager, "median": eager_median, "comparability": eager_comparability },
        "candidate": { "roots": candidate, "median": candidate_median, "comparability": candidate_comparability },
        "candidate_minus_eager": differences,
        "non_claims": [
            "host-timeline evidence is diagnostic and is ineligible for TTFT admission",
            "the receipt does not admit traced decode or select an optimization",
            "capture overhead is not production performance evidence",
            "production remains eager"
        ]
    }))
}

fn read_identity(root: &Path) -> Result<RootIdentity, String> {
    let path = root.join(IDENTITY_FILE);
    let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn validate_identity(
    identity: &RootIdentity,
    expected_mode: &str,
    expected_index: usize,
    reference: Option<&RootIdentity>,
) -> Result<(), String> {
    if identity.schema != IDENTITY_SCHEMA
        || identity.decode_mode != expected_mode
        || identity.root_index != expected_index
        || identity.physical_pcie_device_ids != EXPECTED_DEVICE_IDS
        || identity.mesh_shape != EXPECTED_MESH_SHAPE
        || identity.simulated
        || !identity.profiled
    {
        return Err(format!("invalid {expected_mode} root identity"));
    }
    for (name, value) in [
        ("runtime_identity", identity.runtime_identity.as_str()),
        ("package_identity", identity.package_identity.as_str()),
        ("model_identity", identity.model_identity.as_str()),
    ] {
        if value.is_empty() {
            return Err(format!("empty {name}"));
        }
    }
    if let Some(reference) = reference {
        if identity.runtime_identity != reference.runtime_identity
            || identity.package_identity != reference.package_identity
            || identity.model_identity != reference.model_identity
            || identity.physical_pcie_device_ids != reference.physical_pcie_device_ids
            || identity.mesh_shape != reference.mesh_shape
        {
            return Err("root identity drift".into());
        }
    }
    Ok(())
}

fn summarize_root(root: &Path) -> Result<RootSummary, String> {
    let zones_path = root.join(ZONES_FILE);
    let identity_path = root.join(IDENTITY_FILE);
    let zones_bytes =
        fs::read(&zones_path).map_err(|error| format!("{}: {error}", zones_path.display()))?;
    let identity_bytes = fs::read(&identity_path)
        .map_err(|error| format!("{}: {error}", identity_path.display()))?;
    let zones = parse_zones(&zones_bytes)?;
    let prefill = zones
        .iter()
        .filter(|zone| zone.name.contains("_profiled_prefill_device"))
        .max_by_key(|zone| zone.start_ns)
        .or_else(|| {
            zones
                .iter()
                .filter(|zone| zone.name.contains("_prefill_first_token"))
                .max_by_key(|zone| zone.start_ns)
        })
        .ok_or("capture has no profiled prefill zone")?;
    let prefill_end = prefill.end_ns()?;

    let mut enqueues: Vec<&Zone> = zones
        .iter()
        .filter(|zone| {
            zone.name == "EnqueueProgram"
                && zone.start_ns >= prefill.start_ns
                && zone.end_ns().is_ok_and(|end| end <= prefill_end)
        })
        .collect();
    if enqueues.is_empty() {
        return Err("prefill has no EnqueueProgram zones".into());
    }
    enqueues.sort_by_key(|zone| (zone.thread.as_str(), zone.start_ns));

    let enqueue_exec_total_ns = checked_sum(enqueues.iter().map(|zone| zone.duration_ns))?;
    let mut gap_total = ZERO_NS;
    let mut largest_gap = ZERO_NS;
    for pair in enqueues.windows(PAIR_WINDOW) {
        if pair[0].thread != pair[1].thread {
            continue;
        }
        let prior_end = pair[0].end_ns()?;
        if pair[1].start_ns < prior_end {
            return Err("overlapping EnqueueProgram zones on one thread".into());
        }
        let gap = pair[1].start_ns - prior_end;
        gap_total = gap_total.checked_add(gap).ok_or("gap total overflow")?;
        largest_gap = largest_gap.max(gap);
    }

    let first_start = enqueues
        .iter()
        .map(|zone| zone.start_ns)
        .min()
        .ok_or("no enqueue")?;
    let last_end = enqueues
        .iter()
        .map(|zone| zone.end_ns())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no enqueue")?;
    let leading_gap_ns = first_start - prefill.start_ns;
    let trailing_gap_ns = prefill_end - last_end;
    if leading_gap_ns < ZERO_NS || trailing_gap_ns < ZERO_NS {
        return Err("enqueue lies outside prefill zone".into());
    }

    let synchronization_total_ns = checked_sum(zones.iter().filter_map(|zone| {
        let in_prefill =
            zone.start_ns >= prefill.start_ns && zone.end_ns().is_ok_and(|end| end <= prefill_end);
        let lower = zone.name.to_ascii_lowercase();
        (in_prefill && (lower.contains("finish") || lower.contains("synchroniz")))
            .then_some(zone.duration_ns)
    }))?;
    let non_enqueue_span_ns = prefill
        .duration_ns
        .checked_sub(enqueue_exec_total_ns)
        .ok_or("enqueue durations exceed prefill wall span")?;

    Ok(RootSummary {
        root: root.display().to_string(),
        identity_blake3: blake3::hash(&identity_bytes).to_hex().to_string(),
        zones_blake3: blake3::hash(&zones_bytes).to_hex().to_string(),
        prefill_wall_ns: prefill.duration_ns,
        enqueue_count: enqueues.len(),
        enqueue_exec_total_ns,
        inter_enqueue_gap_total_ns: gap_total,
        largest_inter_enqueue_gap_ns: largest_gap,
        leading_gap_ns,
        trailing_gap_ns,
        synchronization_total_ns,
        non_enqueue_span_ns,
    })
}

fn parse_zones(bytes: &[u8]) -> Result<Vec<Zone>, String> {
    let text = String::from_utf8_lossy(bytes);
    let mut zones = Vec::new();
    for line in text.lines() {
        // tracy-csvexport does not escape every diagnostic newline in zone
        // text. Parse the fixed leading columns directly. Rows needed for
        // attribution have empty zone-name/text fields and a stable shape.
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < ZONE_REQUIRED_COLUMN_COUNT
            || fields[ZONE_NAME_COLUMN] == "name"
            || fields[ZONE_NAME_COLUMN].is_empty()
        {
            continue;
        }
        let Ok(start_ns) =
            parse_non_negative(fields.get(ZONE_START_COLUMN).copied(), "ns_since_start")
        else {
            continue;
        };
        let Ok(duration_ns) =
            parse_non_negative(fields.get(ZONE_DURATION_COLUMN).copied(), "exec_time_ns")
        else {
            continue;
        };
        if duration_ns == ZERO_NS {
            continue;
        }
        zones.push(Zone {
            name: fields[ZONE_NAME_COLUMN].to_string(),
            start_ns,
            duration_ns,
            thread: fields[ZONE_THREAD_COLUMN].to_string(),
        });
    }
    if zones.is_empty() {
        return Err("zones CSV is empty".into());
    }
    Ok(zones)
}

fn parse_non_negative(value: Option<&str>, name: &str) -> Result<i64, String> {
    let parsed = value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<i64>()
        .map_err(|error| format!("invalid {name}: {error}"))?;
    if parsed < ZERO_NS {
        return Err(format!("negative {name}"));
    }
    Ok(parsed)
}

fn checked_sum(mut values: impl Iterator<Item = i64>) -> Result<i64, String> {
    values.try_fold(ZERO_NS, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| "duration total overflow".into())
    })
}

fn mode_comparability(roots: &[RootSummary]) -> ModeComparability {
    let prefill_wall_ratio = duplicate_ratio(roots.iter().map(|root| root.prefill_wall_ns));
    let enqueue_exec_ratio = duplicate_ratio(roots.iter().map(|root| root.enqueue_exec_total_ns));
    let inter_enqueue_gap_ratio =
        duplicate_ratio(roots.iter().map(|root| root.inter_enqueue_gap_total_ns));
    ModeComparability {
        prefill_wall_ratio,
        enqueue_exec_ratio,
        inter_enqueue_gap_ratio,
        valid: [
            prefill_wall_ratio,
            enqueue_exec_ratio,
            inter_enqueue_gap_ratio,
        ]
        .into_iter()
        .all(|ratio| ratio <= MAX_DUPLICATE_RATIO),
    }
}

fn duplicate_ratio(values: impl Iterator<Item = i64>) -> f64 {
    let values: Vec<i64> = values.collect();
    let minimum = *values.iter().min().expect("duplicate roots exist");
    let maximum = *values.iter().max().expect("duplicate roots exist");
    if minimum == ZERO_NS {
        f64::INFINITY
    } else {
        maximum as f64 / minimum as f64
    }
}

fn mode_median(roots: &[RootSummary]) -> ModeMedian {
    ModeMedian {
        prefill_wall_ns: median(roots.iter().map(|root| root.prefill_wall_ns as f64)),
        enqueue_count: median(roots.iter().map(|root| root.enqueue_count as f64)),
        enqueue_exec_total_ns: median(roots.iter().map(|root| root.enqueue_exec_total_ns as f64)),
        inter_enqueue_gap_total_ns: median(
            roots
                .iter()
                .map(|root| root.inter_enqueue_gap_total_ns as f64),
        ),
        largest_inter_enqueue_gap_ns: median(
            roots
                .iter()
                .map(|root| root.largest_inter_enqueue_gap_ns as f64),
        ),
        leading_gap_ns: median(roots.iter().map(|root| root.leading_gap_ns as f64)),
        trailing_gap_ns: median(roots.iter().map(|root| root.trailing_gap_ns as f64)),
        synchronization_total_ns: median(
            roots
                .iter()
                .map(|root| root.synchronization_total_ns as f64),
        ),
        non_enqueue_span_ns: median(roots.iter().map(|root| root.non_enqueue_span_ns as f64)),
    }
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = values.len() / MEDIAN_DIVISOR_USIZE;
    if values.len() % MEDIAN_DIVISOR_USIZE == 0 {
        (values[middle - 1] + values[middle]) / MEDIAN_DIVISOR
    } else {
        values[middle]
    }
}

fn difference(eager: f64, candidate: f64) -> serde_json::Value {
    json!({
        "delta": candidate - eager,
        "ratio": if eager == 0.0 { serde_json::Value::Null } else { json!(candidate / eager) },
    })
}

fn self_test() -> Result<(), String> {
    let root = env::temp_dir().join(format!(
        "qwen36-host-timeline-self-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    let eager_one = root.join("eager-1");
    let eager_two = root.join("eager-2");
    let candidate_one = root.join("candidate-1");
    let candidate_two = root.join("candidate-2");
    write_fixture(&eager_one, "eager", 0, SELF_TEST_EAGER_GAP_NS)?;
    write_fixture(&eager_two, "eager", 1, SELF_TEST_EAGER_GAP_NS)?;
    write_fixture(&candidate_one, "candidate", 0, SELF_TEST_CANDIDATE_GAP_NS)?;
    write_fixture(&candidate_two, "candidate", 1, SELF_TEST_CANDIDATE_GAP_NS)?;
    let config = Config {
        eager_roots: vec![eager_one.clone(), eager_two],
        candidate_roots: vec![candidate_one.clone(), candidate_two],
        execution_order: EXPECTED_ORDER
            .iter()
            .map(|value| String::from(*value))
            .collect(),
        output: root.join("receipt.json"),
    };
    let receipt = build_receipt(&config)?;
    let delta = receipt["candidate_minus_eager"]["inter_enqueue_gap_total_ns"]["delta"]
        .as_f64()
        .ok_or("self-test receipt missing gap delta")?;
    if delta != SELF_TEST_EXPECTED_DELTA_NS {
        return Err(format!(
            "positive self-test expected gap delta {SELF_TEST_EXPECTED_DELTA_NS}, got {delta}"
        ));
    }
    if receipt["verdict"] != "diagnostic-only" {
        return Err("positive self-test rejected stable duplicate roots".into());
    }

    write_fixture(&candidate_one, "candidate", 0, SELF_TEST_UNSTABLE_GAP_NS)?;
    let unstable_receipt = build_receipt(&config)?;
    if unstable_receipt["verdict"] != "rejected"
        || unstable_receipt["rejection_reason"] != "duplicate-root-instability"
    {
        return Err("negative self-test accepted unstable duplicate roots".into());
    }
    write_fixture(&candidate_one, "candidate", 0, SELF_TEST_CANDIDATE_GAP_NS)?;

    let mut bad_identity: RootIdentity = serde_json::from_slice(
        &fs::read(candidate_one.join(IDENTITY_FILE)).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    bad_identity.runtime_identity = "drifted-runtime".into();
    fs::write(
        candidate_one.join(IDENTITY_FILE),
        serde_json::to_vec_pretty(&bad_identity).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if build_receipt(&config).is_ok() {
        return Err("negative self-test accepted identity drift".into());
    }

    let bad_args = vec![
        EXECUTION_ORDER_FLAG.into(),
        "candidate,eager,eager,candidate".into(),
    ];
    if parse_config(&bad_args).is_ok() {
        return Err("negative self-test accepted wrong execution order".into());
    }
    fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
    Ok(())
}

fn write_fixture(root: &Path, mode: &str, root_index: usize, gap_ns: i64) -> Result<(), String> {
    fs::create_dir_all(root).map_err(|error| error.to_string())?;
    let identity = RootIdentity {
        schema: IDENTITY_SCHEMA.into(),
        decode_mode: mode.into(),
        root_index,
        runtime_identity: "runtime".into(),
        package_identity: "package".into(),
        model_identity: "model".into(),
        physical_pcie_device_ids: EXPECTED_DEVICE_IDS.into(),
        mesh_shape: EXPECTED_MESH_SHAPE.into(),
        simulated: false,
        profiled: true,
    };
    fs::write(
        root.join(IDENTITY_FILE),
        serde_json::to_vec_pretty(&identity).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let second_start = SELF_TEST_SECOND_ENQUEUE_BASE_NS + gap_ns;
    let csv = format!(
        "name,src_file,src_line,zone_name,zone_text,ns_since_start,exec_time_ns,thread,special_parent_text\n_prefill_first_token,server.py,1,,,{SELF_TEST_PREFILL_START_NS},{SELF_TEST_PREFILL_DURATION_NS},main,\nEnqueueProgram,fd.cpp,1,,,{SELF_TEST_FIRST_ENQUEUE_START_NS},{SELF_TEST_ENQUEUE_DURATION_NS},main,\nEnqueueProgram,fd.cpp,1,,,{second_start},{SELF_TEST_ENQUEUE_DURATION_NS},main,\nFDMeshCommandQueue::finish,fd.cpp,1,,,{SELF_TEST_SYNC_START_NS},{SELF_TEST_SYNC_DURATION_NS},main,\n"
    );
    fs::write(root.join(ZONES_FILE), csv).map_err(|error| error.to_string())?;
    Ok(())
}
