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
const STABLE_BOARD_ID_LEN: usize = 16;
const IMAGE_DIGEST_PREFIX: &str = "sha256:";
const ADMISSION_SCHEMA_VERSION: u64 = 1;
const REQUIRED_BUNDLE_ID: &str = "tt-xla-training-2026-08-04";
const REQUIRED_IMAGE_REPOSITORY: &str = "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04";
const REQUIRED_BASE_OS: &str = "ubuntu-24.04";
const REQUIRED_PJRT_PACKAGE: &str = "pjrt-plugin-tt";
const REQUIRED_PYTHON_ABI: &str = "cp312";
const REQUIRED_TT_XLA_REPOSITORY: &str = "https://github.com/tenstorrent/tt-xla";
const REQUIRED_TT_BLACKSMITH_REPOSITORY: &str = "https://github.com/tenstorrent/tt-blacksmith";
const REQUIRED_PYTORCH_PACKAGE: &str = "torch";
const REQUIRED_PYTORCH_XLA_PACKAGE: &str = "torch-xla";
const REQUIRED_JAX_PACKAGE: &str = "jax";
const REQUIRED_JAXLIB_PACKAGE: &str = "jaxlib";
const REQUIRED_TRANSFORMERS_PACKAGE: &str = "transformers";
const REQUIRED_DRIVER_NAME: &str = "TT-KMD";
const REQUIRED_DEVICE_COUNT: u64 = 2;
const REQUIRED_ARCHITECTURE: &str = "blackhole";
const REQUIRED_CONFIG_BOARD_CLASS: &str = "p150";
const REQUIRED_BOARD_TYPE: &str = "p150a";
const REQUIRED_DEVICE_ROOT: &str = "/dev/tenstorrent";
const REQUIRED_MESH_NAME: &str = "p150_x2";
const REQUIRED_MESH_ROWS: u64 = 1;
const REQUIRED_MESH_COLUMNS: u64 = 2;
const REQUIRED_DESCRIPTOR_RELATIVE_PATH: &str = "libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto";
const REQUIRED_HUGEPAGE_SIZE: &str = "1G";
const HASH_ALGORITHM: &str = "blake3";
const MUTABLE_TAG: &str = "latest";
const MAX_CONTAINER_DESCRIPTOR_PATH: &str = "/mesh/p150_x2_mesh_graph_descriptor.textproto";
const WHEELHOUSE_CONTAINER_PATH: &str = "/opt/tenstorrent-wheelhouse";
const MESH_DESCRIPTOR_ENV: &str = "TT_MESH_GRAPH_DESC_PATH";
const REQUIRED_EVIDENCE_EXCLUSIONS: [&str; 4] = [
    "credentials",
    "raw_logs",
    "serial_numbers",
    "absolute_private_paths",
];

const RECEIPT_STAGES: [&str; 7] = [
    "health_before",
    "preflight",
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
    pub firmware_bundle: String,
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

/// Package-owned paths and content identities admitted for production launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeAuthority {
    pub bundle: AdmittedBundle,
    pub config: ValidatedConfig,
    pub descriptor_path: String,
    pub wheelhouse_path: String,
    pub requirements_path: String,
    pub identity_blake3: String,
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

fn require_bool(object: &Map<String, Value>, field: &str, what: &str) -> Result<bool, Rejection> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| Rejection::new("missing_field", format!("{what}.{field} must be a bool")))
}

fn require_schema_version(object: &Map<String, Value>, what: &str) -> Result<(), Rejection> {
    let schema_version = require_u64(object, "schema_version", what)?;
    if schema_version != ADMISSION_SCHEMA_VERSION {
        return Err(Rejection::new(
            "schema_version",
            format!("{what}.schema_version {schema_version} is not {ADMISSION_SCHEMA_VERSION}"),
        ));
    }
    Ok(())
}

fn require_exact_string(
    object: &Map<String, Value>,
    field: &str,
    what: &str,
    expected: &str,
) -> Result<(), Rejection> {
    let actual = require_string(object, field, what)?;
    if actual != expected {
        return Err(Rejection::new(
            "unsupported_identity",
            format!("{what}.{field} {actual} is not {expected}"),
        ));
    }
    Ok(())
}

fn require_https_url(value: &str, what: &str) -> Result<(), Rejection> {
    if !value.starts_with("https://") {
        return Err(Rejection::new(
            "invalid_url",
            format!("{what} must use https"),
        ));
    }
    Ok(())
}

fn require_iso_date(value: &str, what: &str) -> Result<(), Rejection> {
    const DATE_LENGTH: usize = 10;
    const YEAR_END: usize = 4;
    const MONTH_START: usize = 5;
    const MONTH_END: usize = 7;
    const DAY_START: usize = 8;
    const DAY_END: usize = 10;
    const MIN_MONTH: u32 = 1;
    const MAX_MONTH: u32 = 12;
    const MIN_DAY: u32 = 1;
    const MAX_DAY: u32 = 31;

    let Some(date) = value.get(..DATE_LENGTH) else {
        return Err(Rejection::new(
            "invalid_date",
            format!("{what} is not an ISO date"),
        ));
    };
    let bytes = date.as_bytes();
    if bytes.get(YEAR_END) != Some(&b'-') || bytes.get(MONTH_END) != Some(&b'-') {
        return Err(Rejection::new(
            "invalid_date",
            format!("{what} is not an ISO date"),
        ));
    }
    let year = date[..YEAR_END].parse::<u32>();
    let month = date[MONTH_START..MONTH_END].parse::<u32>();
    let day = date[DAY_START..DAY_END].parse::<u32>();
    if year.is_err()
        || !matches!(month, Ok(MIN_MONTH..=MAX_MONTH))
        || !matches!(day, Ok(MIN_DAY..=MAX_DAY))
    {
        return Err(Rejection::new(
            "invalid_date",
            format!("{what} is not an ISO date"),
        ));
    }
    Ok(())
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
    require_schema_version(root, "manifest")?;
    require_exact_string(root, "bundle_id", "manifest", REQUIRED_BUNDLE_ID)?;
    require_iso_date(
        require_string(root, "recorded_date", "manifest")?,
        "manifest.recorded_date",
    )?;
    let bundle_id = REQUIRED_BUNDLE_ID.to_string();

    let image = require_object(
        root.get("runtime_image")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.runtime_image missing"))?,
        "manifest.runtime_image",
    )?;
    reject_unknown_fields(
        image,
        &["repository", "digest", "created", "base_os"],
        "manifest.runtime_image",
    )?;
    require_exact_string(
        image,
        "repository",
        "manifest.runtime_image",
        REQUIRED_IMAGE_REPOSITORY,
    )?;
    require_exact_string(image, "base_os", "manifest.runtime_image", REQUIRED_BASE_OS)?;
    require_iso_date(
        require_string(image, "created", "manifest.runtime_image")?,
        "manifest.runtime_image.created",
    )?;
    let image_digest = require_string(image, "digest", "manifest.runtime_image")?;
    check_image_digest(image_digest, "runtime_image")?;

    let pjrt = require_object(
        root.get("pjrt_plugin")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.pjrt_plugin missing"))?,
        "manifest.pjrt_plugin",
    )?;
    reject_unknown_fields(
        pjrt,
        &["package", "version", "wheel_sha256", "python_abi", "index"],
        "manifest.pjrt_plugin",
    )?;
    require_exact_string(
        pjrt,
        "package",
        "manifest.pjrt_plugin",
        REQUIRED_PJRT_PACKAGE,
    )?;
    require_exact_string(
        pjrt,
        "python_abi",
        "manifest.pjrt_plugin",
        REQUIRED_PYTHON_ABI,
    )?;
    require_https_url(
        require_string(pjrt, "index", "manifest.pjrt_plugin")?,
        "manifest.pjrt_plugin.index",
    )?;
    let pjrt_version = require_string(pjrt, "version", "manifest.pjrt_plugin")?;
    check_exact_version(pjrt_version, "pjrt_plugin")?;
    let pjrt_hash = require_string(pjrt, "wheel_sha256", "manifest.pjrt_plugin")?;
    check_hash_hex(pjrt_hash, "pjrt_plugin")?;

    let sources = [
        ("tt_xla_source", REQUIRED_TT_XLA_REPOSITORY),
        ("tt_blacksmith_source", REQUIRED_TT_BLACKSMITH_REPOSITORY),
    ];
    for (source_name, expected_repository) in sources {
        let source = require_object(
            root.get(source_name).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{source_name} missing"))
            })?,
            source_name,
        )?;
        reject_unknown_fields(
            source,
            &["repository", "revision", "revision_date"],
            source_name,
        )?;
        require_exact_string(source, "repository", source_name, expected_repository)?;
        check_git_revision(
            require_string(source, "revision", source_name)?,
            source_name,
        )?;
        require_iso_date(
            require_string(source, "revision_date", source_name)?,
            &format!("manifest.{source_name}.revision_date"),
        )?;
    }
    let blacksmith = require_object(
        root.get("tt_blacksmith_source").ok_or_else(|| {
            Rejection::new("missing_field", "manifest.tt_blacksmith_source missing")
        })?,
        "manifest.tt_blacksmith_source",
    )?;
    let blacksmith_revision =
        require_string(blacksmith, "revision", "tt_blacksmith_source")?.to_string();

    let hashed_packages = [
        ("pytorch", REQUIRED_PYTORCH_PACKAGE),
        ("pytorch_xla", REQUIRED_PYTORCH_XLA_PACKAGE),
    ];
    for (field, expected_package) in hashed_packages {
        let wheel = require_object(
            root.get(field).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{field} missing"))
            })?,
            field,
        )?;
        reject_unknown_fields(
            wheel,
            &["package", "version", "wheel_sha256", "python_abi"],
            field,
        )?;
        require_exact_string(wheel, "package", field, expected_package)?;
        require_exact_string(wheel, "python_abi", field, REQUIRED_PYTHON_ABI)?;
        check_exact_version(require_string(wheel, "version", field)?, field)?;
        check_hash_hex(require_string(wheel, "wheel_sha256", field)?, field)?;
    }

    let versioned_packages = [
        ("jax", REQUIRED_JAX_PACKAGE),
        ("jaxlib", REQUIRED_JAXLIB_PACKAGE),
        ("transformers", REQUIRED_TRANSFORMERS_PACKAGE),
    ];
    for (field, expected_package) in versioned_packages {
        let package = require_object(
            root.get(field).ok_or_else(|| {
                Rejection::new("missing_field", format!("manifest.{field} missing"))
            })?,
            field,
        )?;
        reject_unknown_fields(package, &["package", "version", "python_abi"], field)?;
        require_exact_string(package, "package", field, expected_package)?;
        if package
            .get("python_abi")
            .is_some_and(|abi| abi.as_str() != Some(REQUIRED_PYTHON_ABI))
        {
            return Err(Rejection::new(
                "unsupported_identity",
                format!("manifest.{field}.python_abi is not {REQUIRED_PYTHON_ABI}"),
            ));
        }
        check_exact_version(require_string(package, "version", field)?, field)?;
    }

    let driver = require_object(
        root.get("driver")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.driver missing"))?,
        "manifest.driver",
    )?;
    reject_unknown_fields(driver, &["name", "version"], "manifest.driver")?;
    require_exact_string(driver, "name", "manifest.driver", REQUIRED_DRIVER_NAME)?;
    let driver_version = require_string(driver, "version", "manifest.driver")?.to_string();
    check_exact_version(&driver_version, "manifest.driver.version")?;

    let firmware = require_object(
        root.get("firmware")
            .ok_or_else(|| Rejection::new("missing_field", "manifest.firmware missing"))?,
        "manifest.firmware",
    )?;
    reject_unknown_fields(
        firmware,
        &["bundle_version", "floor_version"],
        "manifest.firmware",
    )?;
    let firmware_bundle =
        require_string(firmware, "bundle_version", "manifest.firmware")?.to_string();
    let firmware_floor =
        require_string(firmware, "floor_version", "manifest.firmware")?.to_string();
    check_exact_version(&firmware_bundle, "firmware.bundle_version")?;
    check_exact_version(&firmware_floor, "firmware.floor_version")?;
    if compare_versions(&firmware_bundle, &firmware_floor) == std::cmp::Ordering::Less {
        return Err(Rejection::new(
            "unsupported_identity",
            "manifest firmware bundle is below its floor",
        ));
    }

    let identity = blake3_hex(&canonical_json(manifest));
    Ok(AdmittedBundle {
        bundle_id,
        identity_blake3: identity,
        image_reference: format!("{REQUIRED_IMAGE_REPOSITORY}@{image_digest}"),
        pjrt_wheel_version: pjrt_version.to_string(),
        pjrt_wheel_sha256: pjrt_hash.to_string(),
        tt_blacksmith_revision: blacksmith_revision,
        driver_version,
        firmware_bundle,
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
    require_schema_version(root, "config")?;

    let bundle_id = require_string(root, "bundle_id", "config")?;
    if bundle_id != expected_bundle_id || bundle_id != REQUIRED_BUNDLE_ID {
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
    reject_unknown_fields(
        devices,
        &["count", "architecture", "board_class", "device_root"],
        "config.devices",
    )?;
    let count = require_u64(devices, "count", "config.devices")?;
    if count != REQUIRED_DEVICE_COUNT {
        return Err(Rejection::new(
            "unsupported_combination",
            format!("config.devices.count {count} is not {REQUIRED_DEVICE_COUNT}"),
        ));
    }
    require_exact_string(
        devices,
        "architecture",
        "config.devices",
        REQUIRED_ARCHITECTURE,
    )?;
    require_exact_string(
        devices,
        "board_class",
        "config.devices",
        REQUIRED_CONFIG_BOARD_CLASS,
    )?;
    let device_root = require_string(devices, "device_root", "config.devices")?.to_string();
    check_absolute_bounded_path(&device_root, "config.devices.device_root")?;
    if device_root != REQUIRED_DEVICE_ROOT {
        return Err(Rejection::new(
            "unsupported_identity",
            format!("config.devices.device_root {device_root} is not {REQUIRED_DEVICE_ROOT}"),
        ));
    }

    let mesh = require_object(
        root.get("mesh")
            .ok_or_else(|| Rejection::new("missing_field", "config.mesh missing"))?,
        "config.mesh",
    )?;
    reject_unknown_fields(
        mesh,
        &[
            "name",
            "rows",
            "columns",
            "fabric_channels",
            "descriptor_relative_path",
        ],
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
    let fabric_channels = require_u64(mesh, "fabric_channels", "config.mesh")?;
    if fabric_channels == 0 {
        return Err(Rejection::new(
            "unsupported_combination",
            "config.mesh.fabric_channels must be positive",
        ));
    }
    require_exact_string(
        mesh,
        "descriptor_relative_path",
        "config.mesh",
        REQUIRED_DESCRIPTOR_RELATIVE_PATH,
    )?;
    let descriptor_relative_path = REQUIRED_DESCRIPTOR_RELATIVE_PATH.to_string();

    let hugepages = require_object(
        root.get("hugepages")
            .ok_or_else(|| Rejection::new("missing_field", "config.hugepages missing"))?,
        "config.hugepages",
    )?;
    reject_unknown_fields(
        hugepages,
        &["mount", "page_size", "required_free_pages"],
        "config.hugepages",
    )?;
    require_exact_string(
        hugepages,
        "page_size",
        "config.hugepages",
        REQUIRED_HUGEPAGE_SIZE,
    )?;
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

    let cache = require_object(
        root.get("cache")
            .ok_or_else(|| Rejection::new("missing_field", "config.cache missing"))?,
        "config.cache",
    )?;
    reject_unknown_fields(cache, &["root"], "config.cache")?;
    check_absolute_bounded_path(
        require_string(cache, "root", "config.cache")?,
        "config.cache.root",
    )?;

    let output = require_object(
        root.get("output")
            .ok_or_else(|| Rejection::new("missing_field", "config.output missing"))?,
        "config.output",
    )?;
    reject_unknown_fields(output, &["evidence_root"], "config.output")?;
    let evidence_root = require_string(output, "evidence_root", "config.output")?.to_string();
    check_absolute_bounded_path(&evidence_root, "config.output.evidence_root")?;

    let evidence = require_object(
        root.get("evidence")
            .ok_or_else(|| Rejection::new("missing_field", "config.evidence missing"))?,
        "config.evidence",
    )?;
    reject_unknown_fields(evidence, &["hash_algorithm", "exclude"], "config.evidence")?;
    require_exact_string(
        evidence,
        "hash_algorithm",
        "config.evidence",
        HASH_ALGORITHM,
    )?;
    let exclusions = evidence
        .get("exclude")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Rejection::new("missing_field", "config.evidence.exclude must be an array")
        })?;
    if exclusions.len() != REQUIRED_EVIDENCE_EXCLUSIONS.len() {
        return Err(Rejection::new(
            "incomplete_exclusions",
            "config.evidence.exclude does not contain every required exclusion",
        ));
    }
    for required in REQUIRED_EVIDENCE_EXCLUSIONS {
        if !exclusions
            .iter()
            .any(|item| item.as_str() == Some(required))
        {
            return Err(Rejection::new(
                "incomplete_exclusions",
                format!("config.evidence.exclude lacks {required}"),
            ));
        }
    }

    let smoke = require_object(
        root.get("smoke")
            .ok_or_else(|| Rejection::new("missing_field", "config.smoke missing"))?,
        "config.smoke",
    )?;
    reject_unknown_fields(
        smoke,
        &["timeout_seconds", "collective_free"],
        "config.smoke",
    )?;
    let smoke_timeout_seconds = require_u64(smoke, "timeout_seconds", "config.smoke")?;
    if smoke_timeout_seconds == 0 {
        return Err(Rejection::new(
            "unsupported_combination",
            "config.smoke.timeout_seconds must be positive",
        ));
    }
    if !require_bool(smoke, "collective_free", "config.smoke")? {
        return Err(Rejection::new(
            "unsupported_combination",
            "config.smoke.collective_free must be true",
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

/// Admit the Nix-owned production authority after the imperative shell hashes
/// its descriptor and requirements lock. The authority embeds the exact
/// manifest and configuration exported from Nickel.
pub fn admit_runtime_authority(
    authority: &Value,
    actual_descriptor_blake3: &str,
    actual_requirements_blake3: &str,
) -> Result<RuntimeAuthority, Rejection> {
    let root = require_object(authority, "authority")?;
    reject_unknown_fields(
        root,
        &[
            "schema_version",
            "manifest",
            "configuration",
            "descriptor",
            "wheelhouse",
        ],
        "authority",
    )?;
    require_schema_version(root, "authority")?;

    let manifest = root
        .get("manifest")
        .ok_or_else(|| Rejection::new("missing_field", "authority.manifest missing"))?;
    let configuration = root
        .get("configuration")
        .ok_or_else(|| Rejection::new("missing_field", "authority.configuration missing"))?;
    let bundle = admit_bundle(manifest)?;
    let config = validate_config(configuration, &bundle.bundle_id)?;

    let descriptor = require_object(
        root.get("descriptor")
            .ok_or_else(|| Rejection::new("missing_field", "authority.descriptor missing"))?,
        "authority.descriptor",
    )?;
    reject_unknown_fields(descriptor, &["path", "blake3"], "authority.descriptor")?;
    let descriptor_path = require_string(descriptor, "path", "authority.descriptor")?.to_string();
    check_absolute_bounded_path(&descriptor_path, "authority.descriptor.path")?;
    let expected_descriptor_blake3 = require_string(descriptor, "blake3", "authority.descriptor")?;
    check_hash_hex(expected_descriptor_blake3, "authority.descriptor.blake3")?;
    check_hash_hex(actual_descriptor_blake3, "actual_descriptor_blake3")?;
    if actual_descriptor_blake3 != expected_descriptor_blake3 {
        return Err(Rejection::new(
            "stale_identity",
            "descriptor content does not match the package-owned authority",
        ));
    }

    let wheelhouse = require_object(
        root.get("wheelhouse")
            .ok_or_else(|| Rejection::new("missing_field", "authority.wheelhouse missing"))?,
        "authority.wheelhouse",
    )?;
    reject_unknown_fields(
        wheelhouse,
        &["path", "requirements_path", "requirements_blake3"],
        "authority.wheelhouse",
    )?;
    let wheelhouse_path = require_string(wheelhouse, "path", "authority.wheelhouse")?.to_string();
    let requirements_path =
        require_string(wheelhouse, "requirements_path", "authority.wheelhouse")?.to_string();
    check_absolute_bounded_path(&wheelhouse_path, "authority.wheelhouse.path")?;
    check_absolute_bounded_path(&requirements_path, "authority.wheelhouse.requirements_path")?;
    let expected_requirements_blake3 =
        require_string(wheelhouse, "requirements_blake3", "authority.wheelhouse")?;
    check_hash_hex(
        expected_requirements_blake3,
        "authority.wheelhouse.requirements_blake3",
    )?;
    check_hash_hex(actual_requirements_blake3, "actual_requirements_blake3")?;
    if actual_requirements_blake3 != expected_requirements_blake3 {
        return Err(Rejection::new(
            "stale_identity",
            "wheel requirements do not match the package-owned authority",
        ));
    }

    Ok(RuntimeAuthority {
        bundle,
        config,
        descriptor_path,
        wheelhouse_path,
        requirements_path,
        identity_blake3: blake3_hex(&canonical_json(authority)),
    })
}

/// Build the production plan from one admitted package-owned authority.
pub fn build_authorized_launch_plan(authority: &RuntimeAuthority) -> Result<LaunchPlan, Rejection> {
    let base = build_launch_plan(
        &authority.bundle,
        &authority.config,
        &authority.descriptor_path,
    )?;
    let mut plan = base.plan;
    let mounts = plan
        .get_mut("mounts")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| Rejection::new("internal", "launch plan mounts are not an array"))?;
    mounts.push(json!({
        "host": authority.wheelhouse_path,
        "container": WHEELHOUSE_CONTAINER_PATH,
        "read_only": true,
    }));
    let object = plan
        .as_object_mut()
        .ok_or_else(|| Rejection::new("internal", "launch plan is not an object"))?;
    object.insert(
        "runtime_authority_identity".to_string(),
        json!(authority.identity_blake3),
    );
    object.insert(
        "requirements_path".to_string(),
        json!(authority.requirements_path),
    );
    let identity_blake3 = blake3_hex(&canonical_json(&plan));
    Ok(LaunchPlan {
        identity_blake3,
        plan,
    })
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
        reject_unknown_fields(
            device,
            &["board_type", "board_id", "dram_status", "fw_bundle_version"],
            "preflight_snapshot.devices[]",
        )?;
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
        if bundle_version != bundle.firmware_bundle {
            return Err(Rejection::new(
                "firmware_mismatch",
                format!(
                    "device {index} firmware bundle {bundle_version} does not match admitted {}",
                    bundle.firmware_bundle
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
        let board_id = require_string(device, "board_id", "preflight_snapshot.devices[]")?;
        if board_id.is_empty() {
            return Err(Rejection::new(
                "missing_field",
                format!("device {index} has an empty board_id"),
            ));
        }
        if board_ids.iter().any(|existing| existing == board_id) {
            return Err(Rejection::new(
                "duplicate_device",
                format!("device {index} repeats board_id {board_id}"),
            ));
        }
        board_ids.push(board_id.to_string());
    }

    let hugepages = require_object(
        root.get("hugepages").ok_or_else(|| {
            Rejection::new("missing_field", "preflight_snapshot.hugepages missing")
        })?,
        "preflight_snapshot.hugepages",
    )?;
    reject_unknown_fields(
        hugepages,
        &["mount", "free_pages"],
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
    reject_unknown_fields(ownership, &["devices_free"], "preflight_snapshot.ownership")?;
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

/// Classify the first failed smoke stage from newline-delimited JSON output.
/// Fail closed at `mesh_open` when no supported failure record is present.
pub fn classify_smoke_failure(output: &str) -> &'static str {
    let mut failed_stage = "mesh_open";
    for line in output.lines() {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if record.get("status").and_then(Value::as_str) != Some("fail") {
            continue;
        }
        failed_stage = match record.get("stage").and_then(Value::as_str) {
            Some("smoke_execute") => "smoke_execute",
            Some("mesh_open") => "mesh_open",
            _ => failed_stage,
        };
    }
    failed_stage
}

/// Classify every terminal stage after success or one named failure.
pub fn terminal_stage_states(failed_stage: Option<&str>) -> Result<Value, Rejection> {
    let failure_index = match failed_stage {
        Some(name) => Some(
            RECEIPT_STAGES
                .iter()
                .position(|stage| *stage == name)
                .ok_or_else(|| {
                    Rejection::new("unknown_stage", format!("unknown terminal stage {name}"))
                })?,
        ),
        None => None,
    };
    let stages = RECEIPT_STAGES
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let status = match failure_index {
                None => "pass",
                Some(failed) if index < failed => "pass",
                Some(failed) if index == failed => "fail",
                Some(_) => "not_observed",
            };
            json!({"name": name, "status": status})
        })
        .collect::<Vec<Value>>();
    Ok(Value::Array(stages))
}

/// r[impl tenstorrent.training_platform.evidence]
/// Validate one terminal receipt. Returns `true` for a complete success and
/// `false` for a well-formed terminal failure.
pub fn validate_terminal_receipt(
    receipt: &Value,
    expected_bundle_identity: &str,
    expected_config_identity: &str,
    expected_plan_identity: &str,
) -> Result<bool, Rejection> {
    let root = require_object(receipt, "receipt")?;
    reject_unsafe_fields(receipt, "$receipt")?;
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
    require_schema_version(root, "receipt")?;
    require_exact_string(root, "kind", "receipt", "dual_p150a_platform_smoke")?;

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
    let mut failure_seen = false;
    for (index, expected_stage) in RECEIPT_STAGES.iter().enumerate() {
        let stage = require_object(&stages[index], "receipt.stages[]")?;
        reject_unknown_fields(stage, &["name", "status"], "receipt.stages[]")?;
        let name = require_string(stage, "name", "receipt.stages[]")?;
        if name != *expected_stage {
            return Err(Rejection::new(
                "missing_stage",
                format!("receipt stage {index} is {name}, expected {expected_stage}"),
            ));
        }
        let status = require_string(stage, "status", "receipt.stages[]")?;
        if failure_seen {
            if status != "not_observed" {
                return Err(Rejection::new(
                    "invalid_stage_state",
                    format!("receipt stage {name} must be not_observed after failure"),
                ));
            }
        } else {
            match status {
                "pass" => {}
                "fail" => failure_seen = true,
                "not_observed" => {
                    return Err(Rejection::new(
                        "invalid_stage_state",
                        format!("receipt stage {name} is not_observed before a failure"),
                    ));
                }
                other => {
                    return Err(Rejection::new(
                        "invalid_stage_state",
                        format!("receipt stage {name} has unsupported status {other}"),
                    ));
                }
            }
        }
    }

    let artifacts = require_object(
        root.get("artifacts")
            .ok_or_else(|| Rejection::new("missing_field", "receipt.artifacts missing"))?,
        "receipt.artifacts",
    )?;
    if artifacts.is_empty() && !failure_seen {
        return Err(Rejection::new(
            "invalid_artifact",
            "successful receipt binds no artifacts",
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

    Ok(!failure_seen)
}

/// Validate a successful hardware receipt for downstream prerequisite use.
pub fn validate_hardware_receipt(
    receipt: &Value,
    expected_bundle_identity: &str,
    expected_config_identity: &str,
    expected_plan_identity: &str,
) -> Result<(), Rejection> {
    let success = validate_terminal_receipt(
        receipt,
        expected_bundle_identity,
        expected_config_identity,
        expected_plan_identity,
    )?;
    if !success {
        return Err(Rejection::new(
            "failed_stage",
            "receipt records a terminal workload failure",
        ));
    }
    Ok(())
}

/// Validate one portable receipt or artifact as typed aggregate content.
pub fn validate_portable_artifact(value: &Value) -> Result<(), Rejection> {
    reject_unsafe_fields(value, "$artifact")
}

/// Reject credentials, raw logs, serial numbers, stable board identities,
/// and absolute private paths anywhere in portable content.
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
                    format!("portable value at {path} is an absolute private path"),
                ));
            }
            if text.len() == STABLE_BOARD_ID_LEN
                && text.chars().all(|character| character.is_ascii_hexdigit())
            {
                return Err(Rejection::new(
                    "unsafe_field",
                    format!("portable value at {path} has a stable board-identity shape"),
                ));
            }
            if text.contains('\n')
                || text.contains(" | info ")
                || text.contains(" | warning ")
                || text.contains(".cpp:")
            {
                return Err(Rejection::new(
                    "unsafe_field",
                    format!("portable value at {path} has raw-log content"),
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

    mod runtime_identity;

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
            "driver": {"name": "TT-KMD", "version": "2.10.0"},
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
            "driver_version": "2.10.0",
            "devices": [
                {"board_type": "p150a", "board_id": "0000040331921011", "dram_status": true, "fw_bundle_version": "18.8.0.0"},
                {"board_type": "p150a", "board_id": "000004033192100e", "dram_status": true, "fw_bundle_version": "18.8.0.0"}
            ],
            "hugepages": {"mount": "/dev/hugepages-1G", "free_pages": 6},
            "ownership": {"devices_free": true}
        })
    }

    fn valid_authority() -> Value {
        let descriptor_hash = "a".repeat(HASH_HEX_LEN);
        let requirements_hash = "b".repeat(HASH_HEX_LEN);
        json!({
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "manifest": valid_manifest(),
            "configuration": valid_config(),
            "descriptor": {
                "path": "/nix/store/descriptor.textproto",
                "blake3": descriptor_hash,
            },
            "wheelhouse": {
                "path": "/nix/store/wheelhouse",
                "requirements_path": "/nix/store/wheelhouse/requirements.lock",
                "requirements_blake3": requirements_hash,
            },
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

    #[test]
    fn invalid_nested_manifest_fields_are_rejected() {
        let mutations = [
            ("schema", json!(999)),
            ("package", json!("not-pjrt-plugin-tt")),
            ("abi", json!("cp311")),
            ("date", json!("not-a-date")),
            ("driver", json!("not-tt-kmd")),
        ];
        for (name, replacement) in mutations {
            let mut manifest = valid_manifest();
            match name {
                "schema" => manifest["schema_version"] = replacement,
                "package" => manifest["pjrt_plugin"]["package"] = replacement,
                "abi" => manifest["pytorch"]["python_abi"] = replacement,
                "date" => manifest["recorded_date"] = replacement,
                "driver" => manifest["driver"]["name"] = replacement,
                _ => unreachable!("fixture names are exhaustive"),
            }
            assert!(admit_bundle(&manifest).is_err(), "{name} must be rejected");
        }

        let mut unknown = valid_manifest();
        unknown["runtime_image"]["unknown"] = json!(true);
        let error = admit_bundle(&unknown).expect_err("nested unknown field must fail");
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

    #[test]
    fn invalid_nested_config_fields_are_rejected() {
        let bundle = admit_bundle(&valid_manifest()).expect("admits");
        let mutations = [
            ("schema", json!(999)),
            ("board", json!("n300")),
            ("fabric", json!(0)),
            ("cache", json!("relative/cache")),
            ("collective", json!(false)),
            ("exclude", json!([])),
        ];
        for (name, replacement) in mutations {
            let mut config = valid_config();
            match name {
                "schema" => config["schema_version"] = replacement,
                "board" => config["devices"]["board_class"] = replacement,
                "fabric" => config["mesh"]["fabric_channels"] = replacement,
                "cache" => config["cache"]["root"] = replacement,
                "collective" => config["smoke"]["collective_free"] = replacement,
                "exclude" => config["evidence"]["exclude"] = replacement,
                _ => unreachable!("fixture names are exhaustive"),
            }
            assert!(
                validate_config(&config, &bundle.bundle_id).is_err(),
                "{name} must be rejected"
            );
        }

        let mut unknown = valid_config();
        unknown["devices"]["unreviewed_device_override"] = json!(true);
        let error = validate_config(&unknown, &bundle.bundle_id)
            .expect_err("nested unknown field must fail");
        assert_eq!(error.code, "unknown_field");
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

    #[test]
    fn runtime_authority_binds_descriptor_and_wheel_requirements() {
        let descriptor_hash = "a".repeat(HASH_HEX_LEN);
        let requirements_hash = "b".repeat(HASH_HEX_LEN);
        let authority =
            admit_runtime_authority(&valid_authority(), &descriptor_hash, &requirements_hash)
                .expect("authority admits");
        let plan = build_authorized_launch_plan(&authority).expect("authorized plan builds");
        assert_eq!(
            plan.plan["runtime_authority_identity"],
            json!(authority.identity_blake3)
        );
        assert_eq!(
            plan.plan["mounts"][2]["container"],
            json!(WHEELHOUSE_CONTAINER_PATH)
        );
    }

    #[test]
    fn stale_runtime_authority_content_is_rejected() {
        let error = admit_runtime_authority(
            &valid_authority(),
            &"c".repeat(HASH_HEX_LEN),
            &"b".repeat(HASH_HEX_LEN),
        )
        .expect_err("stale descriptor must fail");
        assert_eq!(error.code, "stale_identity");
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

    #[test]
    fn duplicate_device_identity_is_rejected() {
        let (bundle, config) = admitted();
        let mut snapshot = valid_snapshot();
        let repeated_board_id = snapshot["devices"][0]["board_id"].clone();
        snapshot["devices"][1]["board_id"] = repeated_board_id;
        let error = admit_preflight_snapshot(&snapshot, &bundle, &config)
            .expect_err("duplicate board identity must fail");
        assert_eq!(error.code, "duplicate_device");
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
                {"name": "preflight", "status": "pass"},
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

    #[test]
    fn smoke_failure_classifier_uses_observed_execution_stage() {
        let output = concat!(
            "not-json\n",
            "{\"status\":\"pass\",\"stage\":\"mesh_open\"}\n",
            "{\"status\":\"fail\",\"stage\":\"smoke_execute\"}\n",
        );
        assert_eq!(classify_smoke_failure(output), "smoke_execute");
    }

    #[test]
    fn smoke_failure_classifier_fails_closed_for_unsupported_output() {
        let output = "{\"status\":\"fail\",\"stage\":\"untrusted-stage\"}\n";
        assert_eq!(classify_smoke_failure(output), "mesh_open");
        assert_eq!(classify_smoke_failure(""), "mesh_open");
    }

    #[test]
    fn terminal_stage_classifier_marks_unobserved_tail() {
        let stages = terminal_stage_states(Some("mesh_open")).expect("known stage classifies");
        assert_eq!(stages[0]["status"], json!("pass"));
        assert_eq!(stages[1]["status"], json!("pass"));
        assert_eq!(stages[2]["status"], json!("fail"));
        assert_eq!(stages[3]["status"], json!("not_observed"));
        let error = terminal_stage_states(Some("unknown")).expect_err("unknown stage must fail");
        assert_eq!(error.code, "unknown_stage");
    }

    #[test]
    fn terminal_failure_receipt_is_validated() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["stages"] = json!([
            {"name": "health_before", "status": "pass"},
            {"name": "preflight", "status": "pass"},
            {"name": "mesh_open", "status": "fail"},
            {"name": "smoke_execute", "status": "not_observed"},
            {"name": "mesh_close", "status": "not_observed"},
            {"name": "restore", "status": "not_observed"},
            {"name": "health_after", "status": "not_observed"}
        ]);
        receipt["artifacts"] = json!({});
        let success = validate_terminal_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect("terminal failure validates");
        assert!(!success);
        let error = validate_hardware_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("failed receipt cannot satisfy success admission");
        assert_eq!(error.code, "failed_stage");
    }

    #[test]
    fn stage_after_failure_cannot_claim_success() {
        let (bundle, config) = admitted();
        let plan = build_launch_plan(&bundle, &config, "/nix/store/tt-metal/descriptor.textproto")
            .expect("plan builds");
        let mut receipt = valid_receipt(&bundle, &config, &plan);
        receipt["stages"][1]["status"] = json!("fail");
        let error = validate_terminal_receipt(
            &receipt,
            &bundle.identity_blake3,
            &config.identity_blake3,
            &plan.identity_blake3,
        )
        .expect_err("later pass after failure must fail");
        assert_eq!(error.code, "invalid_stage_state");
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
        receipt["stages"][4]["status"] = json!("not_observed");
        receipt["stages"][5]["status"] = json!("not_observed");
        receipt["stages"][6]["status"] = json!("not_observed");
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

    #[test]
    fn sanitized_portable_artifact_is_valid() {
        let artifact = json!({
            "status": "pass",
            "device_count": REQUIRED_DEVICE_COUNT,
            "board_type": REQUIRED_BOARD_TYPE,
        });
        validate_portable_artifact(&artifact).expect("aggregate artifact is portable");
    }

    #[test]
    fn stable_board_identity_and_raw_log_are_rejected() {
        let board_identity = json!({"device": "0000040331921011"});
        let error = validate_portable_artifact(&board_identity)
            .expect_err("stable board identity must fail");
        assert_eq!(error.code, "unsafe_field");

        let raw_log = json!({"message": "runtime.cpp:10 | info opening device"});
        let error = validate_portable_artifact(&raw_log).expect_err("raw log must fail");
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
