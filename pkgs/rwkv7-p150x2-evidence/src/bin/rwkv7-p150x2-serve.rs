use rwkv7_p150x2_evidence::serve::{
    ServeMode, ServeProfile, build_completion_response, build_fast_worker_request,
    interpret_worker_receipt, parse_completion_request, parse_serve_profile,
    prepare_completion_batch, select_least_loaded_device,
};
use serde::Serialize;
use serde_json::json;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const VOCABULARY_PATH: Option<&str> = option_env!("RWKV7_TOKENIZER_VOCABULARY");
const DEVICE_RUNTIME_PATH: Option<&str> = option_env!("RWKV7_DEVICE_RUNTIME");
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const SERVE_MODE: &str = "serve";
const SELF_TEST_MODE: &str = "--self-test";
const PROFILE_OPTION: &str = "--profile";
const STATE_ROOT_OPTION: &str = "--state-root";
const SERVE_ARGUMENT_COUNT: usize = 5;
const SELF_TEST_ARGUMENT_COUNT: usize = 1;
const MODE_INDEX: usize = 0;
const PROFILE_OPTION_INDEX: usize = 1;
const PROFILE_VALUE_INDEX: usize = 2;
const STATE_ROOT_OPTION_INDEX: usize = 3;
const STATE_ROOT_VALUE_INDEX: usize = 4;
const HTTP_READ_CHUNK_BYTES: usize = 8192;
const HTTP_HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";
const HTTP_VERSION: &str = "HTTP/1.1";
const CONTENT_LENGTH_HEADER: &str = "content-length";
const CONTENT_TYPE_HEADER: &str = "content-type";
const TRANSFER_ENCODING_HEADER: &str = "transfer-encoding";
const JSON_CONTENT_TYPE: &str = "application/json";
const GET_METHOD: &str = "GET";
const POST_METHOD: &str = "POST";
const MAXIMUM_STATUS_CAPABILITY_HEX_DIGITS: usize = 32;
const CAPABILITY_NICE_BIT: u32 = 23;
const PRIVATE_DIRECTORY_MODE: u32 = 0o700;
const WORKER_REQUEST_FILE: &str = "request.json";
const WORKER_RECEIPT_FILE: &str = "receipt.json";
const WORKER_STDOUT_FILE: &str = "stdout.log";
const WORKER_STDERR_FILE: &str = "stderr.log";
const INITIALIZATION_READY_BARRIER: &str = "initialization-ready";
const WARMUP_START_BARRIER: &str = "warmup-start";
const TIMED_READY_BARRIER: &str = "timed-ready";
const TIMED_START_BARRIER: &str = "timed-start";
const WORKER_WKV_POLICY: &str = "production";
const PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANDIDATE";
const PRIMARY_THREAD_CPU_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_CPU";
const PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_FIFO_PRIORITY";
const MAXIMUM_WORKER_FAILURE_LOG_BYTES: usize = 8192;
const MAXIMUM_WORKER_FAILURE_LOG_CHARACTERS: usize = 1024;
const REDACTED_REQUEST_ROOT: &str = "<request-root>";

#[derive(Clone, Debug)]
struct ServeArguments {
    profile_path: PathBuf,
    state_root: PathBuf,
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct ParsedHttpHead {
    method: String,
    path: String,
    body_length: usize,
}

struct WorkerPaths {
    request: PathBuf,
    receipt: PathBuf,
    stdout: PathBuf,
    stderr: PathBuf,
    initialization_ready: PathBuf,
    warmup_start: PathBuf,
    timed_ready: PathBuf,
    timed_start: PathBuf,
}

struct ServiceState {
    profile: ServeProfile,
    vocabulary: Vec<u8>,
    runtime: PathBuf,
    state_root: PathBuf,
    request_counter: AtomicU64,
    active_connections: AtomicU32,
    telemetry: Mutex<AggregateTelemetry>,
    device_pool: DevicePool,
    has_fifo_capability: bool,
}

struct DevicePool {
    loads: Arc<Mutex<Vec<u32>>>,
    locks: Vec<Mutex<()>>,
}

struct DeviceReservation {
    device_index: usize,
    loads: Arc<Mutex<Vec<u32>>>,
}

struct RequestRoot {
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AggregateTelemetry {
    completed_batches: u64,
    failed_batches: u64,
    generated_tokens: u64,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv7-p150x2-serve: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .map(require_utf8_argument)
        .collect::<Result<Vec<_>, _>>()?;
    if arguments.len() == SELF_TEST_ARGUMENT_COUNT && arguments[MODE_INDEX] == SELF_TEST_MODE {
        run_self_test()?;
        return Ok(());
    }
    let arguments = parse_serve_arguments(&arguments).map_err(invalid_input)?;
    let profile = parse_serve_profile(&fs::read(&arguments.profile_path)?).map_err(invalid_data)?;
    let vocabulary = read_embedded_vocabulary()?;
    let runtime = embedded_runtime()?;
    let state_root = prepare_state_root(&arguments.state_root)?;
    let device_count = profile.devices.len();
    let state = Arc::new(ServiceState {
        profile,
        vocabulary,
        runtime,
        state_root,
        request_counter: AtomicU64::new(0),
        active_connections: AtomicU32::new(0),
        telemetry: Mutex::new(AggregateTelemetry::default()),
        device_pool: DevicePool::new(device_count)?,
        has_fifo_capability: process_has_nice_capability()?,
    });
    serve(state)?;
    Ok(())
}

fn parse_serve_arguments(arguments: &[String]) -> Result<ServeArguments, String> {
    if arguments.len() != SERVE_ARGUMENT_COUNT
        || arguments[MODE_INDEX] != SERVE_MODE
        || arguments[PROFILE_OPTION_INDEX] != PROFILE_OPTION
        || arguments[STATE_ROOT_OPTION_INDEX] != STATE_ROOT_OPTION
    {
        return Err(format!(
            "{SERVE_MODE} requires {PROFILE_OPTION} PATH {STATE_ROOT_OPTION} PATH"
        ));
    }
    let profile_path = require_absolute_existing_file(&arguments[PROFILE_VALUE_INDEX])?;
    let state_root = PathBuf::from(&arguments[STATE_ROOT_VALUE_INDEX]);
    if !state_root.is_absolute() || state_root.starts_with("/nix/store/") {
        return Err("state root must be an absolute non-store path".to_owned());
    }
    Ok(ServeArguments {
        profile_path,
        state_root,
    })
}

fn serve(state: Arc<ServiceState>) -> Result<(), Box<dyn Error>> {
    let listener = TcpListener::bind(&state.profile.server.listen_address)?;
    for _connection_index in 0..state.profile.limits.maximum_lifetime_connections {
        let (stream, _peer) = listener.accept()?;
        let previous = state.active_connections.fetch_add(1, Ordering::AcqRel);
        if previous >= state.profile.limits.maximum_active_connections {
            state.active_connections.fetch_sub(1, Ordering::AcqRel);
            write_error_response(
                stream,
                "503 Service Unavailable",
                "server connection capacity is full",
                "capacity_exceeded",
            )?;
            continue;
        }
        let connection_state = Arc::clone(&state);
        let handle = thread::Builder::new()
            .name("rwkv7-http-connection".to_owned())
            .spawn(move || {
                let result = handle_connection(stream, &connection_state);
                connection_state
                    .active_connections
                    .fetch_sub(1, Ordering::AcqRel);
                if let Err(error) = result {
                    eprintln!("rwkv7-p150x2-serve: connection failed: {error}");
                }
            })?;
        drop(handle);
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: &ServiceState) -> Result<(), Box<dyn Error>> {
    let io_timeout = Duration::from_secs(state.profile.limits.connection_io_timeout_seconds);
    stream.set_read_timeout(Some(io_timeout))?;
    stream.set_write_timeout(Some(io_timeout))?;
    let request = match read_http_request(
        &mut stream,
        state.profile.limits.maximum_http_header_bytes,
        state.profile.limits.maximum_http_body_bytes,
    ) {
        Ok(request) => request,
        Err(error) => {
            write_error_response(stream, "400 Bad Request", &error, "invalid_http_request")?;
            return Ok(());
        }
    };
    if request.method == GET_METHOD && request.path == state.profile.server.health_path {
        return write_health_response(stream, state);
    }
    if request.method != POST_METHOD || request.path != state.profile.server.openai_completions_path
    {
        write_error_response(
            stream,
            "404 Not Found",
            "route not found",
            "route_not_found",
        )?;
        return Ok(());
    }
    handle_completion_request(stream, state, &request.body)
}

fn write_health_response(stream: TcpStream, state: &ServiceState) -> Result<(), Box<dyn Error>> {
    let device_loads = state.device_pool.load_snapshot().map_err(invalid_data)?;
    let telemetry = telemetry_snapshot(state).map_err(invalid_data)?;
    let body = json!({
        "status": "ok",
        "model": state.profile.model,
        "strict_decode_windows": state.profile.limits.strict_decode_windows,
        "physical_device_ids": state.profile.devices.iter().map(|device| device.physical_device_id).collect::<Vec<_>>(),
        "fused_scheduler_capability": state.has_fifo_capability,
        "telemetry": {
            "active_connections": state.active_connections.load(Ordering::Acquire),
            "active_batches_per_device": device_loads,
            "completed_batches": telemetry.completed_batches,
            "failed_batches": telemetry.failed_batches,
            "generated_tokens": telemetry.generated_tokens,
        },
    });
    write_json_response(stream, "200 OK", &body)
}

fn handle_completion_request(
    stream: TcpStream,
    state: &ServiceState,
    body: &[u8],
) -> Result<(), Box<dyn Error>> {
    let completion_request = match parse_completion_request(body) {
        Ok(request) => request,
        Err(error) => {
            write_error_response(stream, "400 Bad Request", &error, "invalid_request")?;
            return Ok(());
        }
    };
    let prepared =
        match prepare_completion_batch(&state.profile, &state.vocabulary, completion_request) {
            Ok(prepared) => prepared,
            Err(error) => {
                write_error_response(stream, "400 Bad Request", &error, "invalid_request")?;
                return Ok(());
            }
        };
    let request_sequence = state.request_counter.fetch_add(1, Ordering::AcqRel);
    let request_id = format!("cmpl-rwkv7-{request_sequence}");
    let result = match execute_worker_batch(state, &prepared, &request_id) {
        Ok(result) => result,
        Err(error) => {
            if let Err(counter_error) = record_failed_batch(state) {
                eprintln!("rwkv7-p150x2-serve: cannot record failed batch: {counter_error}");
            }
            write_error_response(
                stream,
                "503 Service Unavailable",
                &error,
                "device_worker_failed",
            )?;
            return Ok(());
        }
    };
    record_completed_batch(state, &result).map_err(invalid_data)?;
    let created_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| invalid_data(format!("system clock precedes the Unix epoch: {error}")))?
        .as_secs();
    let response =
        build_completion_response(&state.profile, request_id, created_unix_seconds, result)
            .map_err(invalid_data)?;
    write_json_response(stream, "200 OK", &response)
}

fn execute_worker_batch(
    state: &ServiceState,
    prepared: &rwkv7_p150x2_evidence::serve::PreparedBatch,
    request_id: &str,
) -> Result<rwkv7_p150x2_evidence::serve::CompletionResult, String> {
    let reservation = state.device_pool.reserve()?;
    let device = state
        .profile
        .devices
        .get(reservation.device_index)
        .ok_or_else(|| "reserved device index is outside the profile".to_owned())?;
    let _device_lock = state.device_pool.lock_device(reservation.device_index)?;
    let worker_request = build_fast_worker_request(prepared, request_id)?;
    let request_root = create_request_root(&state.state_root, request_id)?;
    let receipt_path = launch_worker(
        state,
        device,
        prepared.mode,
        &worker_request,
        &request_root.path,
    )?;
    let receipt_bytes = fs::read(receipt_path)
        .map_err(|error| format!("cannot read the bounded worker receipt: {error}"))?;
    interpret_worker_receipt(
        &receipt_bytes,
        &state.vocabulary,
        prepared,
        &worker_request,
        device.physical_device_id,
    )
}

fn launch_worker(
    state: &ServiceState,
    device: &rwkv7_p150x2_evidence::serve::DeviceProfile,
    mode: ServeMode,
    worker_request: &rwkv7_p150x2_evidence::serve::FastWorkerRequest,
    request_root: &Path,
) -> Result<PathBuf, String> {
    let paths = worker_paths(request_root);
    write_new_json(&paths.request, worker_request)?;
    let mut command = worker_command(state, device, mode, request_root, &paths)?;
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot launch bounded device worker: {error}"))?;
    if let Err(error) = execute_worker_protocol(state, &mut child, &paths) {
        log_worker_failure(&paths, request_root, &error);
        return Err(error);
    }
    require(paths.receipt.is_file(), "device worker omitted its receipt")?;
    Ok(paths.receipt)
}

fn log_worker_failure(paths: &WorkerPaths, request_root: &Path, protocol_error: &str) {
    let summary = read_worker_log_tail(&paths.stderr)
        .or_else(|| read_worker_log_tail(&paths.stdout))
        .and_then(|tail| summarize_worker_failure(&tail, request_root));
    match summary {
        Some(summary) => eprintln!("bounded device worker failed: {protocol_error}; {summary}"),
        None => eprintln!("bounded device worker failed: {protocol_error}; no worker log summary"),
    }
}

fn read_worker_log_tail(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let length = file.metadata().ok()?.len();
    let maximum_bytes = u64::try_from(MAXIMUM_WORKER_FAILURE_LOG_BYTES).ok()?;
    let offset = length.saturating_sub(maximum_bytes);
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut bytes = Vec::with_capacity(MAXIMUM_WORKER_FAILURE_LOG_BYTES);
    file.read_to_end(&mut bytes).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn summarize_worker_failure(log_tail: &str, request_root: &Path) -> Option<String> {
    let line = log_tail
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())?;
    let redacted = line.replace(
        &request_root.to_string_lossy().to_string(),
        REDACTED_REQUEST_ROOT,
    );
    Some(
        redacted
            .chars()
            .take(MAXIMUM_WORKER_FAILURE_LOG_CHARACTERS)
            .collect(),
    )
}

fn worker_paths(root: &Path) -> WorkerPaths {
    WorkerPaths {
        request: root.join(WORKER_REQUEST_FILE),
        receipt: root.join(WORKER_RECEIPT_FILE),
        stdout: root.join(WORKER_STDOUT_FILE),
        stderr: root.join(WORKER_STDERR_FILE),
        initialization_ready: root.join(INITIALIZATION_READY_BARRIER),
        warmup_start: root.join(WARMUP_START_BARRIER),
        timed_ready: root.join(TIMED_READY_BARRIER),
        timed_start: root.join(TIMED_START_BARRIER),
    }
}

fn worker_command(
    state: &ServiceState,
    device: &rwkv7_p150x2_evidence::serve::DeviceProfile,
    mode: ServeMode,
    request_root: &Path,
    paths: &WorkerPaths,
) -> Result<Command, String> {
    let stdout = File::create(&paths.stdout)
        .map_err(|error| format!("cannot create bounded worker stdout: {error}"))?;
    let stderr = File::create(&paths.stderr)
        .map_err(|error| format!("cannot create bounded worker stderr: {error}"))?;
    let kernel_cache = state
        .state_root
        .join("cache")
        .join(format!("card-{}", device.physical_device_id));
    fs::create_dir_all(&kernel_cache)
        .map_err(|error| format!("cannot create card-local kernel cache: {error}"))?;
    let mut command = Command::new(&state.runtime);
    command
        .args(["--fast-worker", "--device"])
        .arg(device.physical_device_id.to_string())
        .arg("--requests")
        .arg(&paths.request)
        .arg("--output")
        .arg(&paths.receipt)
        .arg("--wkv-policy")
        .arg(WORKER_WKV_POLICY)
        .arg("--initialization-ready-barrier")
        .arg(&paths.initialization_ready)
        .arg("--warmup-start-barrier")
        .arg(&paths.warmup_start)
        .arg("--timed-ready-barrier")
        .arg(&paths.timed_ready)
        .arg("--timed-start-barrier")
        .arg(&paths.timed_start)
        .env("TT_METAL_LOGS_PATH", request_root.join("logs"))
        .env("TT_METAL_PROFILER_DIR", request_root.join("profiler"))
        .env("TT_METAL_CACHE", kernel_cache)
        .env("TT_METAL_INSPECTOR", "0")
        .env("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "0")
        .env("TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES", "0")
        .env("TT_METAL_INSPECTOR_RPC", "0")
        .env(
            PRIMARY_THREAD_CPU_ENVIRONMENT,
            device.primary_thread_cpu.to_string(),
        )
        .env_remove("TT_METAL_WATCHER")
        .env_remove("TT_METAL_DPRINT_CORES")
        .env_remove("TT_METAL_DEVICE_PROFILER")
        .env_remove("TT_METAL_PROFILER_MID_RUN_DUMP")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    configure_worker_policy(state, mode, &mut command);
    Ok(command)
}

fn configure_worker_policy(state: &ServiceState, mode: ServeMode, command: &mut Command) {
    if state.has_fifo_capability {
        command.env(
            PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT,
            state.profile.primary_thread_fifo_priority.to_string(),
        );
    } else {
        command.env_remove(PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT);
    }
    command.env(
        PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT,
        persistent_candidate_value(mode),
    );
}

fn persistent_candidate_value(mode: ServeMode) -> &'static str {
    match mode {
        ServeMode::Strict => "1",
        ServeMode::Experimental => "0",
    }
}

fn execute_worker_protocol(
    state: &ServiceState,
    child: &mut Child,
    paths: &WorkerPaths,
) -> Result<(), String> {
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(
            state.profile.limits.worker_timeout_seconds,
        ))
        .ok_or_else(|| "worker deadline overflows monotonic time".to_owned())?;
    let poll_interval = Duration::from_millis(state.profile.limits.barrier_poll_milliseconds);
    wait_for_worker_barrier(child, &paths.initialization_ready, deadline, poll_interval)?;
    signal_worker_barrier(&paths.warmup_start)?;
    wait_for_worker_barrier(child, &paths.timed_ready, deadline, poll_interval)?;
    signal_worker_barrier(&paths.timed_start)?;
    wait_for_worker_exit(child, deadline, poll_interval)
}

fn wait_for_worker_barrier(
    child: &mut Child,
    barrier: &Path,
    deadline: Instant,
    poll_interval: Duration,
) -> Result<(), String> {
    while Instant::now() < deadline {
        if classify_worker_barrier(barrier.is_dir(), barrier.exists())? {
            return Ok(());
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("cannot poll device worker: {error}"))?
        {
            return Err(format!(
                "device worker exited before a barrier with status {status}"
            ));
        }
        thread::sleep(poll_interval);
    }
    terminate_worker(child)?;
    Err("device worker timed out before a barrier".to_owned())
}

fn wait_for_worker_exit(
    child: &mut Child,
    deadline: Instant,
    poll_interval: Duration,
) -> Result<(), String> {
    while Instant::now() < deadline {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("cannot poll device worker exit: {error}"))?
        {
            return require(status.success(), "device worker returned a failure status");
        }
        thread::sleep(poll_interval);
    }
    terminate_worker(child)?;
    Err("device worker timed out".to_owned())
}

fn terminate_worker(child: &mut Child) -> Result<(), String> {
    child
        .kill()
        .map_err(|error| format!("cannot terminate timed-out device worker: {error}"))?;
    child
        .wait()
        .map_err(|error| format!("cannot reap terminated device worker: {error}"))?;
    Ok(())
}

fn classify_worker_barrier(is_directory: bool, exists: bool) -> Result<bool, String> {
    if is_directory {
        return Ok(true);
    }
    require(!exists, "worker barrier exists but is not a directory")?;
    Ok(false)
}

fn signal_worker_barrier(path: &Path) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| format!("cannot signal worker barrier: {error}"))
}

fn read_http_request(
    stream: &mut TcpStream,
    maximum_header_bytes: u32,
    maximum_body_bytes: u32,
) -> Result<HttpRequest, String> {
    let maximum_header_bytes =
        usize::try_from(maximum_header_bytes).map_err(|_| "HTTP header bound overflows usize")?;
    let maximum_body_bytes =
        usize::try_from(maximum_body_bytes).map_err(|_| "HTTP body bound overflows usize")?;
    let (mut bytes, header_end) = read_http_head(stream, maximum_header_bytes, maximum_body_bytes)?;
    let head = parse_http_head(&bytes, header_end, maximum_body_bytes)?;
    read_http_body(stream, &mut bytes, header_end, head.body_length)?;
    Ok(HttpRequest {
        method: head.method,
        path: head.path,
        body: bytes[header_end..].to_vec(),
    })
}

fn read_http_head(
    stream: &mut TcpStream,
    maximum_header_bytes: usize,
    maximum_body_bytes: usize,
) -> Result<(Vec<u8>, usize), String> {
    let maximum_total_bytes = maximum_header_bytes
        .checked_add(maximum_body_bytes)
        .ok_or_else(|| "HTTP request bound overflows usize".to_owned())?;
    let mut bytes = Vec::with_capacity(maximum_header_bytes);
    let mut chunk = [0_u8; HTTP_READ_CHUNK_BYTES];
    while bytes.len() < maximum_header_bytes {
        let count = stream
            .read(&mut chunk)
            .map_err(|error| format!("cannot read HTTP request: {error}"))?;
        require(count > 0, "HTTP connection closed before its headers")?;
        bytes.extend_from_slice(&chunk[..count]);
        require(
            bytes.len() <= maximum_total_bytes,
            "HTTP request exceeds the configured bound",
        )?;
        if let Some(index) = find_subsequence(&bytes, HTTP_HEADER_TERMINATOR) {
            let header_end = index + HTTP_HEADER_TERMINATOR.len();
            require(
                header_end <= maximum_header_bytes,
                "HTTP headers exceed the configured bound",
            )?;
            return Ok((bytes, header_end));
        }
    }
    Err("HTTP headers exceed the configured bound".to_owned())
}

fn parse_http_head(
    bytes: &[u8],
    header_end: usize,
    maximum_body_bytes: usize,
) -> Result<ParsedHttpHead, String> {
    let header_text = std::str::from_utf8(&bytes[..header_end])
        .map_err(|error| format!("HTTP headers are not UTF-8: {error}"))?;
    let content_end = header_text.len() - HTTP_HEADER_TERMINATOR.len();
    let mut lines = header_text[..content_end].split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "HTTP request line is missing".to_owned())?;
    let request_parts = request_line.split_whitespace().collect::<Vec<_>>();
    require(
        request_parts.len() == 3,
        "HTTP request line must have three fields",
    )?;
    require(request_parts[2] == HTTP_VERSION, "HTTP version must be 1.1")?;
    let method = request_parts[0].to_owned();
    let path = request_parts[1].to_owned();
    let (content_length, content_type) = parse_http_headers(lines)?;
    let body_length = validate_http_body_headers(
        &method,
        content_length,
        content_type.as_deref(),
        maximum_body_bytes,
    )?;
    Ok(ParsedHttpHead {
        method,
        path,
        body_length,
    })
}

fn parse_http_headers<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> Result<(Option<usize>, Option<String>), String> {
    let mut content_length = None;
    let mut content_type = None;
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| "HTTP header lacks a colon".to_owned())?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        require(
            name != TRANSFER_ENCODING_HEADER,
            "HTTP transfer encoding is unsupported",
        )?;
        if name == CONTENT_LENGTH_HEADER {
            require(
                content_length.is_none(),
                "HTTP content length is duplicated",
            )?;
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|error| format!("HTTP content length is invalid: {error}"))?,
            );
        }
        if name == CONTENT_TYPE_HEADER {
            content_type = Some(value.to_ascii_lowercase());
        }
    }
    Ok((content_length, content_type))
}

fn validate_http_body_headers(
    method: &str,
    content_length: Option<usize>,
    content_type: Option<&str>,
    maximum_body_bytes: usize,
) -> Result<usize, String> {
    if method != POST_METHOD {
        return Ok(content_length.unwrap_or(0));
    }
    let length = content_length.ok_or_else(|| "HTTP POST lacks content length".to_owned())?;
    require(
        length <= maximum_body_bytes,
        "HTTP body exceeds the configured bound",
    )?;
    require(
        content_type == Some(JSON_CONTENT_TYPE),
        "HTTP POST content type must be application/json",
    )?;
    Ok(length)
}

fn read_http_body(
    stream: &mut TcpStream,
    bytes: &mut Vec<u8>,
    header_end: usize,
    body_length: usize,
) -> Result<(), String> {
    let expected_total = header_end
        .checked_add(body_length)
        .ok_or_else(|| "HTTP body extent overflows usize".to_owned())?;
    let mut chunk = [0_u8; HTTP_READ_CHUNK_BYTES];
    while bytes.len() < expected_total {
        let read_limit = (expected_total - bytes.len()).min(HTTP_READ_CHUNK_BYTES);
        let count = stream
            .read(&mut chunk[..read_limit])
            .map_err(|error| format!("cannot read HTTP body: {error}"))?;
        require(count > 0, "HTTP connection closed before its body")?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    require(
        bytes.len() == expected_total,
        "HTTP request has trailing bytes",
    )
}

fn find_subsequence(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || bytes.len() < needle.len() {
        return None;
    }
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_json_response(
    mut stream: TcpStream,
    status: &str,
    body: &impl Serialize,
) -> Result<(), Box<dyn Error>> {
    let body = serde_json::to_vec(body)?;
    let headers = format!(
        "{HTTP_VERSION} {status}\r\nContent-Type: {JSON_CONTENT_TYPE}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn write_error_response(
    stream: TcpStream,
    status: &str,
    message: &str,
    code: &'static str,
) -> Result<(), Box<dyn Error>> {
    let body = ErrorEnvelope {
        error: ErrorBody {
            message: message.to_owned(),
            r#type: "invalid_request_error",
            code,
        },
    };
    write_json_response(stream, status, &body)
}

impl DevicePool {
    fn new(device_count: usize) -> Result<Self, String> {
        require(device_count > 0, "device pool is empty")?;
        let locks = (0..device_count).map(|_| Mutex::new(())).collect();
        Ok(Self {
            loads: Arc::new(Mutex::new(vec![0; device_count])),
            locks,
        })
    }

    fn reserve(&self) -> Result<DeviceReservation, String> {
        let mut loads = self
            .loads
            .lock()
            .map_err(|_| "device load lock is poisoned".to_owned())?;
        let device_index = select_least_loaded_device(&loads)?;
        loads[device_index] = loads[device_index]
            .checked_add(1)
            .ok_or_else(|| "device load count overflows u32".to_owned())?;
        drop(loads);
        Ok(DeviceReservation {
            device_index,
            loads: Arc::clone(&self.loads),
        })
    }

    fn lock_device(&self, index: usize) -> Result<MutexGuard<'_, ()>, String> {
        let lock = self
            .locks
            .get(index)
            .ok_or_else(|| "device lock index is outside the pool".to_owned())?;
        lock.lock()
            .map_err(|_| "device execution lock is poisoned".to_owned())
    }

    fn load_snapshot(&self) -> Result<Vec<u32>, String> {
        self.loads
            .lock()
            .map(|loads| loads.clone())
            .map_err(|_| "device load lock is poisoned".to_owned())
    }
}

fn record_completed_batch(
    state: &ServiceState,
    result: &rwkv7_p150x2_evidence::serve::CompletionResult,
) -> Result<(), String> {
    let generated_tokens = result
        .generated_token_ids
        .iter()
        .try_fold(0_u64, |total, tokens| {
            let count =
                u64::try_from(tokens.len()).map_err(|_| "generated-token count overflows u64")?;
            total
                .checked_add(count)
                .ok_or_else(|| "generated-token total overflows u64".to_owned())
        })?;
    update_telemetry(state, 1, 0, generated_tokens)
}

fn record_failed_batch(state: &ServiceState) -> Result<(), String> {
    update_telemetry(state, 0, 1, 0)
}

fn update_telemetry(
    state: &ServiceState,
    completed_batches: u64,
    failed_batches: u64,
    generated_tokens: u64,
) -> Result<(), String> {
    let mut telemetry = state
        .telemetry
        .lock()
        .map_err(|_| "aggregate telemetry lock is poisoned".to_owned())?;
    *telemetry = checked_telemetry_update(
        *telemetry,
        completed_batches,
        failed_batches,
        generated_tokens,
    )?;
    Ok(())
}

fn checked_telemetry_update(
    current: AggregateTelemetry,
    completed_batches: u64,
    failed_batches: u64,
    generated_tokens: u64,
) -> Result<AggregateTelemetry, String> {
    Ok(AggregateTelemetry {
        completed_batches: current
            .completed_batches
            .checked_add(completed_batches)
            .ok_or_else(|| "completed-batch counter overflowed".to_owned())?,
        failed_batches: current
            .failed_batches
            .checked_add(failed_batches)
            .ok_or_else(|| "failed-batch counter overflowed".to_owned())?,
        generated_tokens: current
            .generated_tokens
            .checked_add(generated_tokens)
            .ok_or_else(|| "generated-token counter overflowed".to_owned())?,
    })
}

fn telemetry_snapshot(state: &ServiceState) -> Result<AggregateTelemetry, String> {
    state
        .telemetry
        .lock()
        .map(|telemetry| *telemetry)
        .map_err(|_| "aggregate telemetry lock is poisoned".to_owned())
}

impl Drop for DeviceReservation {
    fn drop(&mut self) {
        let mut loads = match self.loads.lock() {
            Ok(loads) => loads,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(load) = loads.get_mut(self.device_index) {
            assert!(*load > 0, "device reservation load underflow");
            *load -= 1;
        }
    }
}

impl Drop for RequestRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!("rwkv7-p150x2-serve: cannot remove bounded request root: {error}");
        }
    }
}

fn create_request_root(state_root: &Path, request_id: &str) -> Result<RequestRoot, String> {
    let path = state_root.join("requests").join(request_id);
    fs::create_dir(&path)
        .map_err(|error| format!("cannot create bounded request root: {error}"))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE))
        .map_err(|error| format!("cannot restrict bounded request root permissions: {error}"))?;
    require(path.is_dir(), "bounded request root is not a directory")?;
    Ok(RequestRoot { path })
}

fn prepare_state_root(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(path)?;
    let canonical = fs::canonicalize(path)?;
    if canonical.starts_with("/nix/store/") {
        return Err(invalid_input("state root must be outside the Nix store").into());
    }
    let requests = canonical.join("requests");
    let cache = canonical.join("cache");
    fs::create_dir_all(&requests)?;
    fs::create_dir_all(&cache)?;
    fs::set_permissions(
        &requests,
        fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE),
    )?;
    fs::set_permissions(&cache, fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE))?;
    if fs::read_dir(&requests)?.next().is_some() {
        return Err(invalid_input("state root contains an unfinished request").into());
    }
    Ok(canonical)
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create bounded worker request: {error}"))?;
    serde_json::to_writer(file, value)
        .map_err(|error| format!("cannot serialize bounded worker request: {error}"))
}

fn require_absolute_existing_file(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err("profile path must be absolute".to_owned());
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize profile path: {error}"))?;
    if !canonical.is_file() {
        return Err("profile path is not a regular file".to_owned());
    }
    Ok(canonical)
}

fn read_embedded_vocabulary() -> Result<Vec<u8>, Box<dyn Error>> {
    let configured = VOCABULARY_PATH.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "RWKV7_TOKENIZER_VOCABULARY was not embedded at build time",
        )
    })?;
    let canonical = fs::canonicalize(configured)?;
    if !canonical.is_file() || !canonical.starts_with("/nix/store/") {
        return Err(invalid_data("embedded vocabulary is not a Nix store file").into());
    }
    Ok(fs::read(canonical)?)
}

fn embedded_runtime() -> Result<PathBuf, Box<dyn Error>> {
    let configured = DEVICE_RUNTIME_PATH.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "RWKV7_DEVICE_RUNTIME was not embedded at build time",
        )
    })?;
    let runtime = Path::new(configured).join("bin/rwkv7-p150x2-runtime");
    if !runtime.is_file() || !runtime.starts_with("/nix/store/") {
        return Err(invalid_data("embedded runtime is not a Nix store executable").into());
    }
    Ok(runtime)
}

fn process_has_nice_capability() -> Result<bool, Box<dyn Error>> {
    let status = fs::read_to_string("/proc/self/status")?;
    let effective = status
        .lines()
        .find_map(|line| line.strip_prefix("CapEff:\t"))
        .ok_or_else(|| invalid_data("process status omits CapEff"))?;
    if effective.is_empty() || effective.len() > MAXIMUM_STATUS_CAPABILITY_HEX_DIGITS {
        return Err(invalid_data("CapEff has an invalid width").into());
    }
    let bits = u128::from_str_radix(effective, 16)
        .map_err(|error| invalid_data(format!("CapEff is not hexadecimal: {error}")))?;
    Ok(bits & (1_u128 << CAPABILITY_NICE_BIT) != 0)
}

fn require_utf8_argument(argument: OsString) -> Result<String, io::Error> {
    argument
        .into_string()
        .map_err(|_| invalid_input("arguments must be valid UTF-8"))
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn run_self_test() -> Result<(), Box<dyn Error>> {
    let valid_arguments = vec![
        SERVE_MODE.to_owned(),
        PROFILE_OPTION.to_owned(),
        "/tmp/profile.json".to_owned(),
        STATE_ROOT_OPTION.to_owned(),
        "/var/tmp/rwkv7-serve".to_owned(),
    ];
    if parse_serve_arguments(&valid_arguments).is_ok() {
        return Err(invalid_data("self-test accepted a missing profile path").into());
    }
    if find_subsequence(b"abc\r\n\r\ndef", HTTP_HEADER_TERMINATOR) != Some(3) {
        return Err(invalid_data("self-test HTTP terminator search changed").into());
    }
    if find_subsequence(b"abc", HTTP_HEADER_TERMINATOR).is_some() {
        return Err(invalid_data("self-test found an absent HTTP terminator").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Shutdown;

    const HEADER_BOUND: u32 = 1024;
    const BODY_BOUND: u32 = 4096;

    fn read_from_request(request: &[u8]) -> Result<HttpRequest, String> {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have an address");
        let request = request.to_vec();
        let writer = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).expect("test client must connect");
            stream.write_all(&request).expect("test client must write");
            stream
                .shutdown(Shutdown::Write)
                .expect("test client must close writes");
        });
        let (mut stream, _) = listener.accept().expect("test listener must accept");
        let parsed = read_http_request(&mut stream, HEADER_BOUND, BODY_BOUND);
        writer.join().expect("test client thread must join");
        parsed
    }

    #[test]
    fn reads_bounded_json_post() {
        let body = br#"{"model":"m","prompt":"p","max_tokens":2,"mode":"strict"}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            std::str::from_utf8(body).expect("test body must be UTF-8")
        );
        let parsed = read_from_request(request.as_bytes()).expect("valid HTTP request must parse");
        assert_eq!(parsed.method, POST_METHOD);
        assert_eq!(parsed.path, "/v1/completions");
        assert_eq!(parsed.body, body);
    }

    #[test]
    fn rejects_chunked_and_oversized_requests() {
        let chunked = b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n";
        assert!(read_from_request(chunked).is_err());
        let oversized = b"POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 5000\r\n\r\n";
        assert!(read_from_request(oversized).is_err());
    }

    #[test]
    fn classifies_directory_barriers_and_rejects_other_objects() {
        assert_eq!(classify_worker_barrier(true, true), Ok(true));
        assert_eq!(classify_worker_barrier(false, false), Ok(false));
        assert!(classify_worker_barrier(false, true).is_err());
    }

    #[test]
    fn selects_persistent_dispatch_only_for_strict_mode() {
        assert_eq!(persistent_candidate_value(ServeMode::Strict), "1");
        assert_eq!(persistent_candidate_value(ServeMode::Experimental), "0");
    }

    #[test]
    fn summarizes_one_redacted_worker_failure_line() {
        let root = Path::new("/var/tmp/private-request");
        let log = "first line\nerror in /var/tmp/private-request/request.json\n";
        assert_eq!(
            summarize_worker_failure(log, root),
            Some("error in <request-root>/request.json".to_owned())
        );
        assert_eq!(summarize_worker_failure("\n", root), None);
    }

    #[test]
    fn telemetry_update_is_transactional_and_rejects_overflow() {
        let current = AggregateTelemetry {
            completed_batches: 4,
            failed_batches: 1,
            generated_tokens: 8,
        };
        assert_eq!(
            checked_telemetry_update(current, 1, 2, 3),
            Ok(AggregateTelemetry {
                completed_batches: 5,
                failed_batches: 3,
                generated_tokens: 11,
            })
        );
        let full = AggregateTelemetry {
            completed_batches: u64::MAX,
            ..AggregateTelemetry::default()
        };
        assert!(checked_telemetry_update(full, 1, 0, 0).is_err());
    }

    #[test]
    fn device_pool_balances_and_releases_load() {
        let pool = DevicePool::new(2).expect("test device pool must build");
        let first = pool.reserve().expect("first reservation must succeed");
        let second = pool.reserve().expect("second reservation must succeed");
        assert_eq!(first.device_index, 0);
        assert_eq!(second.device_index, 1);
        drop(first);
        let third = pool.reserve().expect("third reservation must succeed");
        assert_eq!(third.device_index, 0);
        drop(second);
        drop(third);
        assert_eq!(
            pool.load_snapshot().expect("test load snapshot must work"),
            vec![0, 0]
        );
    }
}
