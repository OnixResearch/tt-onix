//! Pure admission core for the dual-P150A TT-XLA training platform.
//!
//! r[impl tenstorrent.training_platform.runtime_bundle]
//! r[impl tenstorrent.training_platform.configuration]
//! r[impl tenstorrent.training_platform.preflight]
//! r[impl tenstorrent.training_platform.launch]
//! r[impl tenstorrent.training_platform.evidence]
//!
//! Every function is deterministic and free of I/O, clocks, and process
//! state. The CLI shell in `main.rs` owns all filesystem and process work.

use serde_json::{Map, Value, json};

const HEX_DIGITS: &str = "0123456789abcdef";
const GIT_REVISION_LEN: usize = 40;
const HASH_HEX_LEN: usize = 64;
const IMAGE_DIGEST_PREFIX: &str = "sha256:";
const REQUIRED_DEVICE_COUNT: u64 = 2;
const REQUIRED_ARCHITECTURE: &str = "blackhole";
const REQUIRED_BOARD_TYPE: &str = "p150a";
const REQUIRED_MESH_NAME: &str = "p150_x2";
const REQUIRED_MESH_ROWS: u64 = 1;
const REQUIRED_MESH_COLUMNS: u64 = 2;
const REQUIRED_HUGEPAGE_SIZE: &str = "1G";
const HASH_ALGORITHM: &str = "blake3";
const MUTABLE_TAG: &str = "latest";
const MAX_CONTAINER_DESCRIPTOR_PATH: &str = "/mesh/p150_x2_mesh_graph_descriptor.textproto";
const MESH_DESCRIPTOR_ENV: &str = "TT_MESH_GRAPH_DESC_PATH";

const RECEIPT_STAGES: [&str; 6] = [
    "health_before",
    "mesh_open",
    "smoke_execute",
    "mesh_close",
    "restore",
    "health_after",
];

const REQUIRED_NON_CLAIMS: [&str; 5] = [
    "no training correctness claim",
    "no gradient parity claim",
    "no convergence claim",
    "no speedup claim",
    "no thermal endurance claim",
];

const FORBIDDEN_RECEIPT_KEYS: [&str; 6] = [
    "password",
    "secret",
    "token",
    "credential",
    "serial_number",
    "raw_log",
];

const REJECTED_AMBIENT_ENV: [&str; 4] = [
    "PYTHONPATH",
    "PYTHONHOME",
    "TT_METAL_HOME",
    "TT_METAL_RUNTIME_ROOT",
];

/// One stable admission failure. The `code` is machine-stable; `detail`
/// names the offending component and value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub code: &'static str,
    pub detail: String,
}

impl Rejection {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Rejection {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

/// Bundle admitted for launch. Carries the deterministic BLAKE3 identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedBundle {
    pub bundle_id: String,
    pub identity_blake3: String,
    pub image_reference: String,
    pub pjrt_wheel_version: String,
    pub pjrt_wheel_sha256: String,
    pub tt_blacksmith_revision: String,
    pub driver_version: String,
    pub firmware_floor: String,
}

/// Platform configuration admitted for launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedConfig {
    pub identity_blake3: String,
    pub bundle_id: String,
    pub device_root: String,
    pub mesh_name: String,
    pub descriptor_relative_path: String,
    pub hugepage_mount: String,
    pub required_free_hugepages: u64,
    pub evidence_root: String,
    pub smoke_timeout_seconds: u64,
}

/// Deterministic launch plan consumed by the thin launcher shell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchPlan {
    pub identity_blake3: String,
    pub plan: Value,
}

/// Facts admitted from one device-free preflight snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightFacts {
    pub device_board_ids: Vec<String>,
    pub firmware_bundle: String,
    pub driver_version: String,
}

/// Serialize with sorted object keys (serde_json::Map is a BTreeMap) and no
/// insignificant whitespace. This is the canonical identity form.
pub fn canonical_json(value: &Value) -> String {
    value.to_string()
}

pub fn blake3_hex(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.chars().all(|c| HEX_DIGITS.contains(c))
}

fn require_object<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>, Rejection> {
    value
        .as_object()
        .ok_or_else(|| Rejection::new("type", format!("{what} must be a JSON object")))
}

fn require_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    what: &str,
) -> Result<&'a str, Rejection> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| Rejection::new("missing_field", format!("{what}.{field} must be a string")))
}

fn require_u64(object: &Map<String, Value>, field: &str, what: &str) -> Result<u64, Rejection> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        Rejection::new(
            "missing_field",
            format!("{what}.{field} must be an unsigned integer"),
        )
    })
}

fn reject_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    what: &str,
) -> Result<(), Rejection> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Rejection::new(
                "unknown_field",
                format!("{what} has unknown field {key}"),
            ));
        }
    }
    Ok(())
}

fn check_exact_version(version: &str, what: &str) -> Result<(), Rejection> {
    if version.is_empty() {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} version is empty"),
        ));
    }
    if version == MUTABLE_TAG
        || version.contains('>')
        || version.contains('<')
        || version.contains('~')
        || version.contains('*')
    {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} version {version} is not an exact version"),
        ));
    }
    Ok(())
}

fn check_hash_hex(hash: &str, what: &str) -> Result<(), Rejection> {
    if !is_lower_hex(hash, HASH_HEX_LEN) {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} hash {hash} is not a 64-digit lowercase hex identity"),
        ));
    }
    Ok(())
}

fn check_git_revision(revision: &str, what: &str) -> Result<(), Rejection> {
    if !is_lower_hex(revision, GIT_REVISION_LEN) {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} revision {revision} is not a full commit hash"),
        ));
    }
    Ok(())
}

fn check_image_digest(digest: &str, what: &str) -> Result<(), Rejection> {
    let Some(hex) = digest.strip_prefix(IMAGE_DIGEST_PREFIX) else {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} image reference {digest} lacks an immutable sha256 digest"),
        ));
    };
    if !is_lower_hex(hex, HASH_HEX_LEN) {
        return Err(Rejection::new(
            "mutable_identity",
            format!("{what} image digest {digest} is malformed"),
        ));
    }
    Ok(())
}

/// r[impl tenstorrent.training_platform.runtime_bundle]
/// Admit one compatibility manifest. Every component must carry an immutable
/// identity: a sha256 image digest, a full git revision, an exact version, or
/// a package hash. Returns the deterministic BLAKE3 bundle identity.
pub fn admit_bundle(manifest: &Value) -> Result<AdmittedBundle, Rejection> {
    let root = require_object(manifest, "manifest")?;
    reject_unknown_fields(
        root,
        &[
            "schema_version",
            "bundle_id",
            "recorded_date",
            "runtime_image",
            "pjrt_plugin",
            "tt_xla_source",
            "tt_blacksmith_source",
            "pytorch",
            "pytorch_xla",
            "jax",
            "jaxlib",
            "transformers",
            "driver",
            "firmware",
        ],
        "manifest",
    )?;

    let bundle_id = require_string(root, "bundle_id", "manifest")?.to_string();

    let image = require_object(
        root.get("runtime_image")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.runtime_image missing"))?,
        "manifest.runtime_image",
    )?;
    let image_repo = require_string(image, "repository", "manifest.runtime_image")?;
    let image_digest = require_string(image, "digest", "manifest.runtime_image")?;
    check_image_digest(image_digest, "runtime_image")?;
    if image_repo.contains(":latest") || image_repo.ends_with(":latest") {
        return Err(Rejection::new(
            "mutable_identity",
            "runtime_image repository carries a mutable latest tag",
        ));
    }

    let pjrt = require_object(
        root.get("pjrt_plugin")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.pjrt_plugin missing"))?,
        "manifest.pjrt_plugin",
    )?;
    let pjrt_version = require_string(pjrt, "version", "manifest.pjrt_plugin")?;
    check_exact_version(pjrt_version, "pjrt_plugin")?;
    let pjrt_hash = require_string(pjrt, "wheel_sha256", "manifest.pjrt_plugin")?;
    check_hash_hex(pjrt_hash, "pjrt_plugin")?;

    for source_name in ["tt_xla_source", "tt_blacksmith_source"] {
        let source = require_object(
            root.get(source_name).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{source_name} missing"))
            })?,
            source_name,
        )?;
        let revision = require_string(source, "revision", source_name)?;
        check_git_revision(revision, source_name)?;
    }
    let blacksmith = require_object(
        root.get("tt_blacksmith_source").ok_or_else(|| {
            Rejection::new("missing_field", "manifest.tt_blacksmith_source missing")
        })?,
        "manifest.tt_blacksmith_source",
    )?;
    let blacksmith_revision =
        require_string(blacksmith, "revision", "tt_blacksmith_source")?.to_string();

    for wheel_name in ["pytorch", "pytorch_xla"] {
        let wheel = require_object(
            root.get(wheel_name).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{wheel_name} missing"))
            })?,
            wheel_name,
        )?;
        check_exact_version(require_string(wheel, "version", wheel_name)?, wheel_name)?;
        check_hash_hex(
            require_string(wheel, "wheel_sha256", wheel_name)?,
            wheel_name,
        )?;
    }

    for package_name in ["jax", "jaxlib", "transformers"] {
        let package = require_object(
            root.get(package_name).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{package_name} missing"))
            })?,
            package_name,
        )?;
        check_exact_version(
            require_string(package, "version", package_name)?,
            package_name,
        )?;
    }

    let driver = require_object(
        root.get("driver")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.driver missing"))?,
        "manifest.driver",
    )?;
    let driver_version = require_string(driver, "version", "manifest.driver")?.to_string();

    let firmware = require_object(
        root.get("firmware")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.firmware missing"))?,
        "manifest.firmware",
    )?;
    let firmware_floor =
        require_string(firmware, "floor_version", "manifest.firmware")?.to_string();
    check_exact_version(&firmware_floor, "firmware.floor_version")?;

    let identity = blake3_hex(&canonical_json(manifest));
    Ok(AdmittedBundle {
        bundle_id,
        identity_blake3: identity,
        image_reference: format!("{image_repo}@{image_digest}"),
        pjrt_wheel_version: pjrt_version.to_string(),
        pjrt_wheel_sha256: pjrt_hash.to_string(),
        tt_blacksmith_revision: blacksmith_revision,
        driver_version,
        firmware_floor,
    })
}

/// r[impl tenstorrent.training_platform.configuration]
/// Validate one exported platform configuration against an admitted bundle.
pub fn validate_config(
    config: &Value,
    expected_bundle_id: &str,
) -> Result<ValidatedConfig, Rejection> {
    let root = require_object(config, "config")?;
    reject_unknown_fields(
        root,
        &[
            "schema_version",
            "bundle_id",
            "devices",
            "mesh",
            "hugepages",
            "cache",
            "output",
            "evidence",
            "smoke",
        ],
        "config",
    )?;

    let bundle_id = require_string(root, "bundle_id", "config")?;
    if bundle_id != expected_bundle_id {
        return Err(Rejection::new(
            "bundle_mismatch",
            format!("config selects bundle {bundle_id}, expected {expected_bundle_id}"),
        ));
    }

    let devices = require_object(
        root.get("devices")
            .ok_or_else(|| Rejection::new("missing_field", "config.devices missing"))?,
        "config.devices",
    )?;
    let count = require_u64(devices, "count", "config.devices")?;
    if count != REQUIRED_DEVICE_COUNT {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.devices.count {count} is not {REQUIRED_DEVICE_COUNT}"),
        ));
    }
    let arch = require_string(devices, "architecture", "config.devices")?;
    if arch != REQUIRED_ARCHITECTURE {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.devices.architecture {arch} is not {REQUIRED_ARCHITECTURE}"),
        ));
    }
    let device_root = require_string(devices, "device_root", "config.devices")?.to_string();
    check_absolute_bounded_path(&device_root, "config.devices.device_root")?;

    let mesh = require_object(
        root.get("mesh")
            .ok_or_else(|| Rejection::new("missing_field", "config.mesh missing"))?,
        "config.mesh",
    )?;
    let mesh_name = require_string(mesh, "name", "config.mesh")?.to_string();
    if mesh_name != REQUIRED_MESH_NAME {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.mesh.name {mesh_name} is not {REQUIRED_MESH_NAME}"),
        ));
    }
    let rows = require_u64(mesh, "rows", "config.mesh")?;
    let columns = require_u64(mesh, "columns", "config.mesh")?;
    if rows != REQUIRED_MESH_ROWS || columns != REQUIRED_MESH_COLUMNS {
        return Err(Rejection::new(
            "unsupported_combination",
            format!(
                "config.mesh shape {rows}x{columns} is not {REQUIRED_MESH_ROWS}x{REQUIRED_MESH_COLUMNS}"
            ),
        ));
    }
    let descriptor_relative_path =
        require_string(mesh, "descriptor_relative_path", "config.mesh")?.to_string();
    if descriptor_relative_path.starts_with('/') || descriptor_relative_path.contains("..") {
        return Err(Rejection::new(
            "invalid_path",
            "config.mesh.descriptor_relative_path must be a bounded relative path",
        ));
    }

    let hugepages = require_object(
        root.get("hugepages")
            .ok_or_else(|| Rejection::new("missing_field", "config.hugepages missing"))?,
        "config.hugepages",
    )?;
    let page_size = require_string(hugepages, "page_size", "config.hugepages")?;
    if page_size != REQUIRED_HUGEPAGE_SIZE {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.hugepages.page_size {page_size} is not {REQUIRED_HUGEPAGE_SIZE}"),
        ));
    }
    let hugepage_mount = require_string(hugepages, "mount", "config.hugepages")?.to_string();
    check_absolute_bounded_path(&hugepage_mount, "config.hugepages.mount")?;
    let required_free_hugepages =
        require_u64(hugepages, "required_free_pages", "config.hugepages")?;
    if required_free_hugepages == 0 {
        return Err(Rejection::new(
            "unsupported_combination",
            "config.hugepages.required_free_pages must be positive",
        ));
    }

    let output = require_object(
        root.get("output")
            .ok_or_else(|| Rejection::new("missing_field", "config.output missing"))?,
        "config.output",
    )?;
    let evidence_root = require_string(output, "evidence_root", "config.output")?.to_string();
    check_absolute_bounded_path(&evidence_root, "config.output.evidence_root")?;

    let evidence = require_object(
        root.get("evidence")
            .ok_or_else(|| Rejection::new("missing_field", "config.evidence missing"))?,
        "config.evidence",
    )?;
    let hash_algorithm = require_string(evidence, "hash_algorithm", "config.evidence")?;
    if hash_algorithm != HASH_ALGORITHM {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.evidence.hash_algorithm {hash_algorithm} is not {HASH_ALGORITHM}"),
        ));
    }

    let smoke = require_object(
        root.get("smoke")
            .ok_or_else(|| Rejection::new("missing_field", "config.smoke missing"))?,
        "config.smoke",
    )?;
    let smoke_timeout_seconds = require_u64(smoke, "timeout_seconds", "config.smoke")?;
    if smoke_timeout_seconds == 0 {
        return Err(Rejection::new(
            "unsupported_combination",
            "config.smoke.timeout_seconds must be positive",
        ));
    }

    let identity = blake3_hex(&canonical_json(config));
    Ok(ValidatedConfig {
        identity_blake3: identity,
        bundle_id: bundle_id.to_string(),
        device_root,
        mesh_name,
        descriptor_relative_path,
        hugepage_mount,
        required_free_hugepages,
        evidence_root,
        smoke_timeout_seconds,
    })
}

fn check_absolute_bounded_path(path: &str, what: &str) -> Result<(), Rejection> {
    if !path.starts_with('/') {
        return Err(Rejection::new(
            "invalid_path",
            format!("{what} {path} is not absolute"),
        ));
    }
    if path.contains("..") {
        return Err(Rejection::new(
            "invalid_path",
            format!("{what} {path} contains a parent traversal"),
        ));
    }
    Ok(())
}

/// r[impl tenstorrent.training_platform.launch]
/// Build the deterministic launch plan: exact image reference, exact mounts,
/// exact environment, and the ambient overrides the shell must reject.
pub fn build_launch_plan(
    bundle: &AdmittedBundle,
    config: &ValidatedConfig,
    descriptor_host_path: &str,
) -> Result<LaunchPlan, Rejection> {
    check_absolute_bounded_path(descriptor_host_path, "descriptor_host_path")?;
    let plan = json!({
        "schema_version": 1,
        "bundle_identity": bundle.identity_blake3,
        "config_identity": config.identity_blake3,
        "image": bundle.image_reference,
        "runtime": "docker",
        "devices": [config.device_root],
        "mounts": [
            {"host": config.hugepage_mount, "container": config.hugepage_mount, "read_only": false},
            {"host": descriptor_host_path, "container": MAX_CONTAINER_DESCRIPTOR_PATH, "read_only": true},
        ],
        "environment": {
            MESH_DESCRIPTOR_ENV: MAX_CONTAINER_DESCRIPTOR_PATH,
        },
        "rejected_ambient_environment": REJECTED_AMBIENT_ENV,
        "pjrt_plugin": {
            "version": bundle.pjrt_wheel_version,
            "wheel_sha256": bundle.pjrt_wheel_sha256,
        },
        "tt_blacksmith_revision": bundle.tt_blacksmith_revision,
        "mesh": {
            "name": config.mesh_name,
            "rows": REQUIRED_MESH_ROWS,
            "columns": REQUIRED_MESH_COLUMNS,
        },
        "evidence_root": config.evidence_root,
        "smoke_timeout_seconds": config.smoke_timeout_seconds,
    });
    let identity = blake3_hex(&canonical_json(&plan));
    Ok(LaunchPlan {
        identity_blake3: identity,
        plan,
    })
}

/// r[impl tenstorrent.training_platform.preflight]
/// Admit one preflight snapshot. The snapshot is a device-free JSON document
/// assembled by the thin shell from tt-smi, sysfs, and ownership probes.
pub fn admit_preflight_snapshot(
    snapshot: &Value,
    bundle: &AdmittedBundle,
    config: &ValidatedConfig,
) -> Result<PreflightFacts, Rejection> {
    let root = require_object(snapshot, "preflight_snapshot")?;
    reject_unknown_fields(
        root,
        &["driver_version", "devices", "hugepages", "ownership"],
        "preflight_snapshot",
    )?;

    let driver_version = require_string(root, "driver_version", "preflight_snapshot")?;
    if driver_version != bundle.driver_version {
        return Err(Rejection::new(
            "driver_mismatch",
            format!(
                "host driver {driver_version} does not match manifest driver {}",
                bundle.driver_version
            ),
        ));
    }

    let devices = root
        .get("devices")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Rejection::new(
                "missing_field",
                "preflight_snapshot.devices must be an array",
            )
        })?;
    if devices.len() as u64 != REQUIRED_DEVICE_COUNT {
        return Err(Rejection::new(
            "device_count",
            format!(
                "preflight snapshot lists {} devices, expected {REQUIRED_DEVICE_COUNT}",
                devices.len()
            ),
        ));
    }

    let mut board_ids = Vec::with_capacity(REQUIRED_DEVICE_COUNT as usize);
    let mut firmware_bundle: Option<String> = None;
    for (index, device) in devices.iter().enumerate() {
        let device = require_object(device, "preflight_snapshot.devices[]")?;
        let board_type = require_string(device, "board_type", "preflight_snapshot.devices[]")?;
        if board_type != REQUIRED_BOARD_TYPE {
            return Err(Rejection::new(
                "wrong_architecture",
                format!("device {index} board_type {board_type} is not {REQUIRED_BOARD_TYPE}"),
            ));
        }
        let dram_ok = device
            .get("dram_status")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                Rejection::new("missing_field", "devices[].dram_status must be a bool")
            })?;
        if !dram_ok {
            return Err(Rejection::new(
                "unhealthy_device",
                format!("device {index} reports failed DRAM status"),
            ));
        }
        let bundle_version =
            require_string(device, "fw_bundle_version", "preflight_snapshot.devices[]")?;
        if compare_versions(bundle_version, &bundle.firmware_floor) == std::cmp::Ordering::Less {
            return Err(Rejection::new(
                "old_firmware",
                format!(
                    "device {index} firmware bundle {bundle_version} is below floor {}",
                    bundle.firmware_floor
                ),
            ));
        }
        match &firmware_bundle {
            None => firmware_bundle = Some(bundle_version.to_string()),
            Some(existing) if existing != bundle_version => {
                return Err(Rejection::new(
                    "firmware_mismatch",
                    format!("devices run mixed firmware bundles {existing} and {bundle_version}"),
                ));
            }
            _ => {}
        }
        board_ids
            .push(require_string(device, "board_id", "preflight_snapshot.devices[]")?.to_string());
    }

    let hugepages = require_object(
        root.get("hugepages").ok_or_else(|| {
            Rejection::new("missing_field", "preflight_snapshot.hugepages missing")
        })?,
        "preflight_snapshot.hugepages",
    )?;
    let mount = require_string(hugepages, "mount", "preflight_snapshot.hugepages")?;
    if mount != config.hugepage_mount {
        return Err(Rejection::new(
            "hugepage_mismatch",
            format!(
                "hugepage mount {mount} does not match configured {}",
                config.hugepage_mount
            ),
        ));
    }
    let free_pages = require_u64(hugepages, "free_pages", "preflight_snapshot.hugepages")?;
    if free_pages < config.required_free_hugepages {
        return Err(Rejection::new(
            "missing_hugepages",
            format!(
                "free 1G hugepages {free_pages} below required {}",
                config.required_free_hugepages
            ),
        ));
    }

    let ownership = require_object(
        root.get("ownership").ok_or_else(|| {
            Rejection::new("missing_field", "preflight_snapshot.ownership missing")
        })?,
        "preflight_snapshot.ownership",
    )?;
    let devices_free = ownership
        .get("devices_free")
        .and_then(Value::as_bool)
        .ok_or_else(|| Rejection::new("missing_field", "ownership.devices_free must be a bool"))?;
    if !devices_free {
        return Err(Rejection::new(
            "busy_device",
            "another process owns a selected device",
        ));
    }

    Ok(PreflightFacts {
        device_board_ids: board_ids,
        firmware_bundle: firmware_bundle
            .ok_or_else(|| Rejection::new("missing_field", "no device firmware recorded"))?,
        driver_version: driver_version.to_string(),
    })
}

/// Compare two dotted numeric versions component by component.
pub fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    let mut left_parts = left.split('.').map(|part| part.parse::<u64>().unwrap_or(0));
    let mut right_parts = right
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0));
    loop {
        match (left_parts.next(), right_parts.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (Some(l), None) => {
                if l != 0 {
                    return std::cmp::Ordering::Greater;
                }
            }
            (None, Some(r)) => {
                if r != 0 {
                    return std::cmp::Ordering::Less;
                }
            }
            (Some(l), Some(r)) => match l.cmp(&r) {
                std::cmp::Ordering::Equal => {}
                other => return other,
            },
        }
    }
}

/// r[impl tenstorrent.training_platform.evidence]
/// Validate one hardware smoke receipt against expected identities.
pub fn validate_hardware_receipt(
    receipt: &Value,
    expected_bundle_identity: &str,
    expected_config_identity: &str,
    expected_plan_identity: &str,
) -> Result<(), Rejection> {
    let root = require_object(receipt, "receipt")?;
    reject_unknown_fields(
        root,
        &[
            "schema_version",
            "kind",
            "bundle_identity",
            "config_identity",
            "launch_plan_identity",
            "stages",
            "artifacts",
            "non_claims",
        ],
        "receipt",
    )?;

    let kind = require_string(root, "kind", "receipt")?;
    if kind != "dual_p150a_platform_smoke" {
        return Err(Rejection::new(
            "receipt_kind",
            format!("receipt kind {kind} is not dual_p150a_platform_smoke"),
        ));
    }

    for (field, expected) in [
        ("bundle_identity", expected_bundle_identity),
        ("config_identity", expected_config_identity),
        ("launch_plan_identity", expected_plan_identity),
    ] {
        let actual = require_string(root, field, "receipt")?;
        check_hash_hex(actual, field)?;
        if actual != expected {
            return Err(Rejection::new(
                "stale_identity",
                format!("receipt {field} does not match the current admitted identity"),
            ));
        }
    }

    let stages = root
        .get("stages")
        .and_then(Value::as_array)
        .ok_or_else(|| Rejection::new("missing_field", "receipt.stages must be an array"))?;
    if stages.len() != RECEIPT_STAGES.len() {
        return Err(Rejection::new(
            "missing_stage",
            format!(
                "receipt lists {} stages, expected {}",
                stages.len(),
                RECEIPT_STAGES.len()
            ),
        ));
    }
    for (index, expected_stage) in RECEIPT_STAGES.iter().enumerate() {
        let stage = require_object(&stages[index], "receipt.stages[]")?;
        let name = require_string(stage, "name", "receipt.stages[]")?;
        if name != *expected_stage {
            return Err(Rejection::new(
                "missing_stage",
                format!("receipt stage {index} is {name}, expected {expected_stage}"),
            ));
        }
        let status = require_string(stage, "status", "receipt.stages[]")?;
        if status != "pass" {
            return Err(Rejection::new(
                "failed_stage",
                format!("receipt stage {name} has status {status}"),
            ));
        }
    }

    let artifacts = require_object(
        root.get("artifacts")
            .ok_or_else(|| Rejection::new("missing_field", "receipt.artifacts missing"))?,
        "receipt.artifacts",
    )?;
    if artifacts.is_empty() {
        return Err(Rejection::new(
            "invalid_artifact",
            "receipt binds no artifacts",
        ));
    }
    for (name, hash) in artifacts {
        let hash = hash.as_str().ok_or_else(|| {
            Rejection::new(
                "invalid_artifact",
                format!("artifact {name} hash is not a string"),
            )
        })?;
        check_hash_hex(hash, "receipt.artifacts")?;
    }

    let non_claims = root
        .get("non_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| Rejection::new("missing_field", "receipt.non_claims must be an array"))?;
    let non_claims: Vec<&str> = non_claims.iter().filter_map(Value::as_str).collect();
    for required in REQUIRED_NON_CLAIMS {
        if !non_claims.contains(&required) {
            return Err(Rejection::new(
                "overclaiming",
                format!("receipt lacks the required non-claim: {required}"),
            ));
        }
    }

    reject_unsafe_fields(receipt, "$receipt")?;
    Ok(())
}

/// Reject credentials, raw logs, serial numbers, and absolute private paths
/// anywhere inside a receipt.
fn reject_unsafe_fields(value: &Value, path: &str) -> Result<(), Rejection> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let lowered = key.to_ascii_lowercase();
                for forbidden in FORBIDDEN_RECEIPT_KEYS {
                    if lowered.contains(forbidden) {
                        return Err(Rejection::new(
                            "unsafe_field",
                            format!(
                                "receipt field {path}.{key} is excluded from portable receipts"
                            ),
                        ));
                    }
                }
                reject_unsafe_fields(child, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_unsafe_fields(child, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        Value::String(text) => {
            if text.starts_with("/home/")
                || text.starts_with("/root")
                || text.starts_with("/Users/")
            {
                return Err(Rejection::new(
                    "unsafe_field",
                    format!("receipt value at {path} is an absolute private path"),
                ));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> Value {
        json!({
            "schema_version": 1,
            "bundle_id": "tt-xla-training-2026-08-04",
            "recorded_date": "2026-08-04",
            "runtime_image": {
                "repository": "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04",
                "digest": "sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503",
                "created": "2026-08-03T12:07:05Z",
                "base_os": "ubuntu-24.04"
            },
            "pjrt_plugin": {
                "package": "pjrt-plugin-tt",
                "version": "1.5.0.dev20260804002041",
                "wheel_sha256": "ae57e7a2e4f7c1c9533135e14db5a33201caa3ff1aa2278de9a88ebc9d4d3fe1",
                "python_abi": "cp312",
                "index": "https://pypi.eng.aws.tenstorrent.com"
            },
            "tt_xla_source": {
                "repository": "https://github.com/tenstorrent/tt-xla",
                "revision": "10b111f3885a598958b0c4e393a07d672584736a",
                "revision_date": "2026-08-04T15:31:46Z"
            },
            "tt_blacksmith_source": {
                "repository": "https://github.com/tenstorrent/tt-blacksmith",
                "revision": "6e44259c7f4342709ae7977cc3715cf5e2e19dac",
                "revision_date": "2026-08-04T11:46:42Z"
            },
            "pytorch": {
                "package": "torch",
                "version": "2.11.0+cpu",
                "wheel_sha256": "f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338",
                "python_abi": "cp312"
            },
            "pytorch_xla": {
                "package": "torch-xla",
                "version": "2.9.0+gitb230007",
                "wheel_sha256": "4318d8cce4f9de38d497712b9e80cb74746dc16dc8e2a15e555c417711757481",
                "python_abi": "cp312"
            },
            "jax": {"package": "jax", "version": "0.7.1"},
            "jaxlib": {"package": "jaxlib", "version": "0.7.1"},
            "transformers": {"package": "transformers", "version": "4.57.1"},
            "driver": {"name": "TT-KMD", "version": "2.10.1-pre"},
            "firmware": {"bundle_version": "18.8.0.0", "floor_version": "18.8.0.0"}
        })
    }

    fn valid_config() -> Value {
        json!({
            "schema_version": 1,
            "bundle_id": "tt-xla-training-2026-08-04",
            "devices": {
                "count": 2,
                "architecture": "blackhole",
                "board_class": "p150",
                "device_root": "/dev/tenstorrent"
            },
            "mesh": {
                "name": "p150_x2",
                "rows": 1,
                "columns": 2,
                "fabric_channels": 4,
                "descriptor_relative_path": "libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto"
            },
            "hugepages": {"mount": "/dev/hugepages-1G", "page_size": "1G", "required_free_pages": 1},
            "cache": {"root": "/var/lib/tenstorrent-training/cache"},
            "output": {"evidence_root": "/var/lib/tenstorrent-training/evidence"},
            "evidence": {
                "hash_algorithm": "blake3",
                "exclude": ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"]
            },
            "smoke": {"timeout_seconds": 900, "collective_free": true}
        })
    }

    fn valid_snapshot() -> Value {
        json!({
            "driver_version": "2.10.1-pre",
            "devices": [
                {"board_type": "p150a", "board_id": "0000040331921011", "dram_status": true, "fw_bundle_version": "18.8.0.0"},
                {"board_type": "p150a", "board_id": "000004033192100e", "dram_status": true, "fw_bundle_version": "18.8.0.0"}
            ],
            "hugepages": {"mount": "/dev/hugepages-1G", "free_pages": 6},
            "ownership": {"devices_free": true}
        })
    }

    fn admitted() -> (AdmittedBundle, ValidatedConfig) {
        let bundle = admit_bundle(&valid_manifest()).expect("valid manifest admits");
        let config =
            validate_config(&valid_config(), &bundle.bundle_id).expect("valid config admits");
        (bundle, config)
    }

    // r[verify tenstorrent.training_platform.runtime_bundle.valid]
    #[test]
    fn valid_bundle_admits_with_deterministic_identity() {
        let first = admit_bundle(&valid_manifest()).expect("admits");
        let second = admit_bundle(&valid_manifest()).expect("admits again");
        assert_eq!(first.identity_blake3, second.identity_blake3);
        assert!(is_lower_hex(&first.identity_blake3, HASH_HEX_LEN));
        assert_eq!(
            first.image_reference,
            "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04@sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503"
        );
    }

    // r[verify tenstorrent.training_platform.runtime_bundle.invalid]
    #[test]
    fn mutable_image_tag_is_rejected() {
        let mut manifest = valid_manifest();
        manifest["runtime_image"]["digest"] = json!("latest");
        let error = admit_bundle(&manifest).expect_err("mutable tag must fail");
        assert_eq!(error.code, "mutable_identity");
        assert!(error.detail.contains("runtime_image"));
    }

    // r[verify tenstorrent.training_platform.runtime_bundle.invalid]
    #[test]
    fn branch_revision_is_rejected() {
        let mut manifest = valid_manifest();
        manifest["tt_xla_source"]["revision"] = json!("main");
        let error = admit_bundle(&manifest).expect_err("branch must fail");
        assert_eq!(error.code, "mutable_identity");
    }

    // r[verify tenstorrent.training_platform.runtime_bundle.invalid]
    #[test]
    fn version_range_is_rejected() {
        let mut manifest = valid_manifest();
        manifest["transformers"]["version"] = json!(">=4.0");
        let error = admit_bundle(&manifest).expect_err("range must fail");
        assert_eq!(error.code, "mutable_identity");
    }

    // r[verify tenstorrent.training_platform.runtime_bundle.invalid]
    #[test]
    fn unknown_manifest_field_is_rejected() {
        let mut manifest = valid_manifest();
        manifest["surprise"] = json!(true);
        let error = admit_bundle(&manifest).expect_err("unknown field must fail");
        assert_eq!(error.code, "unknown_field");
    }

    // r[verify tenstorrent.training_platform.configuration.valid]
    #[test]
    fn valid_config_admits_with_deterministic_identity() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let first = validate_config(&valid_config(), &bundle.bundle_id).expect("admits");
        let second = validate_config(&valid_config(), &bundle.bundle_id).expect("admits again");
        assert_eq!(first.identity_blake3, second.identity_blake3);
    }

    // r[verify tenstorrent.training_platform.configuration.invalid]
    #[test]
    fn wrong_device_count_is_rejected() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let mut config = valid_config();
        config["devices"]["count"] = json!(1);
        let error = validate_config(&config, &bundle.bundle_id).expect_err("one device must fail");
        assert_eq!(error.code, "unsupported_combination");
    }

    // r[verify tenstorrent.training_platform.configuration.invalid]
    #[test]
    fn wrong_mesh_is_rejected() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let mut config = valid_config();
        config["mesh"]["name"] = json!("p150_x4");
        let error = validate_config(&config, &bundle.bundle_id).expect_err("wrong mesh must fail");
        assert_eq!(error.code, "unsupported_combination");
    }

    // r[verify tenstorrent.training_platform.configuration.invalid]
    #[test]
    fn relative_cache_path_is_rejected() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let mut config = valid_config();
        config["devices"]["device_root"] = json!("dev/tenstorrent");
        let error =
            validate_config(&config, &bundle.bundle_id).expect_err("relative path must fail");
        assert_eq!(error.code, "invalid_path");
    }

    // r[verify tenstorrent.training_platform.configuration.invalid]
    #[test]
    fn stale_bundle_selection_is_rejected() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let mut config = valid_config();
        config["bundle_id"] = json!("tt-xla-training-1999-01-01");
        let error =
            validate_config(&config, &bundle.bundle_id).expect_err("stale bundle must fail");
        assert_eq!(error.code, "bundle_mismatch");
    }

    // r[verify tenstorrent.training_platform.launch.isolated]
    #[test]
    fn launch_plan_binds_exact_image_and_rejects_ambient_env() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        assert_eq!(plan.plan["image"], json!(bundle.image_reference));
        assert_eq!(
            plan.plan["environment"]["TT_MESH_GRAPH_DESC_PATH"],
            json!(MAX_CONTAINER_DESCRIPTOR_PATH)
        );
        let rejected = plan.plan["rejected_ambient_environment"]
            .as_array()
            .expect("array");
        assert!(rejected.contains(&json!("PYTHONPATH")));
        assert!(rejected.contains(&json!("TT_METAL_HOME")));
        let again = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds again");
        assert_eq!(plan.identity_blake3, again.identity_blake3);
    }

    // r[verify tenstorrent.training_platform.preflight.valid]
    #[test]
    fn valid_preflight_snapshot_admits() {
        let (bundle, config) = admitted();
        let facts =
            admit_preflight_snapshot(&valid_snapshot(), &bundle, &config).expect("snapshot admits");
        assert_eq!(facts.device_board_ids.len(), 2);
        assert_eq!(facts.firmware_bundle, "18.8.0.0");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn one_device_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["devices"] = json!([
            {"board_type": "p150a", "board_id": "x", "dram_status": true, "fw_bundle_version": "18.8.0.0"}
        ]);
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("one device must fail");
        assert_eq!(error.code, "device_count");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn busy_device_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["ownership"]["devices_free"] = json!(false);
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("busy device must fail");
        assert_eq!(error.code, "busy_device");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn old_firmware_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["devices"][0]["fw_bundle_version"] = json!("17.9.0.0");
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("old firmware must fail");
        assert_eq!(error.code, "old_firmware");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn missing_hugepages_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["hugepages"]["free_pages"] = json!(0);
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("no hugepages must fail");
        assert_eq!(error.code, "missing_hugepages");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn wrong_board_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["devices"][1]["board_type"] = json!("n300");
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("wrong board must fail");
        assert_eq!(error.code, "wrong_architecture");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn mixed_firmware_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["devices"][1]["fw_bundle_version"] = json!("18.9.0.0");
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("mixed firmware must fail");
        assert_eq!(error.code, "firmware_mismatch");
    }

    // r[verify tenstorrent.training_platform.preflight.invalid]
    #[test]
    fn driver_mismatch_snapshot_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        snapshot["driver_version"] = json!("2.9.0");
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("driver mismatch must fail");
        assert_eq!(error.code, "driver_mismatch");
    }

    fn valid_receipt(
        bundle: &AdmittedBundle,
        config: &ValidatedConfig,
        plan: &LaunchPlan,
    ) -> Value {
        json!({
            "schema_version": 1,
            "kind": "dual_p150a_platform_smoke",
            "bundle_identity": bundle.identity_blake3,
            "config_identity": config.identity_blake3,
            "launch_plan_identity": plan.identity_blake3,
            "stages": [
                {"name": "health_before", "status": "pass"},
                {"name": "mesh_open", "status": "pass"},
                {"name": "smoke_execute", "status": "pass"},
                {"name": "mesh_close", "status": "pass"},
                {"name": "restore", "status": "pass"},
                {"name": "health_after", "status": "pass"}
            ],
            "artifacts": {
                "smoke_output": "2c6f51c4f3e2a5f6d7a7e38fdde0b63f0e2be3d4566f6c0c38139cb2c46d2ab5"
            },
            "non_claims": [
                "no training correctness claim",
                "no gradient parity claim",
                "no convergence claim",
                "no speedup claim",
                "no thermal endurance claim"
            ]
        })
    }

    // r[verify tenstorrent.training_platform.evidence.valid]
    #[test]
    fn valid_hardware_receipt_admits() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let receipt = valid_receipt(&bundle, &config, &plan);
        validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect("receipt admits");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn stale_receipt_identity_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["bundle_identity"] = json!("0".repeat(HASH_HEX_LEN));
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("stale identity must fail");
        assert_eq!(error.code, "stale_identity");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn missing_stage_receipt_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["stages"] = json!([{"name": "health_before", "status": "pass"}]);
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("missing stages must fail");
        assert_eq!(error.code, "missing_stage");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn failed_stage_receipt_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["stages"][3]["status"] = json!("fail");
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("failed stage must fail");
        assert_eq!(error.code, "failed_stage");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn receipt_with_serial_number_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["artifacts"]["device_serial_number"] = json!("0".repeat(HASH_HEX_LEN));
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("serial number must fail");
        assert_eq!(error.code, "unsafe_field");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn receipt_with_private_path_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["stages"][0]["log_path"] = json!("/home/operator/smoke.log");
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("private path must fail");
        assert_eq!(error.code, "unsafe_field");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn receipt_missing_non_claim_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["non_claims"] = json!(["no speedup claim"]);
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("missing non-claims must fail");
        assert_eq!(error.code, "overclaiming");
    }

    // r[verify tenstorrent.training_platform.evidence.invalid]
    #[test]
    fn receipt_with_invalid_artifact_hash_is_rejected() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["artifacts"]["smoke_output"] = json!("not-a-hash");
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("bad artifact hash must fail");
        assert_eq!(error.code, "mutable_identity");
    }

    #[test]
    fn version_comparison_orders_components() {
        assert_eq!(
            compare_versions("18.8.0.0", "18.8.0.0"),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_versions("18.9.0.0", "18.8.0.0"),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            compare_versions("17.10.0.0", "18.8.0.0"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_versions("18.8", "18.8.0.0"),
            std::cmp::Ordering::Equal
        );
    }
}
