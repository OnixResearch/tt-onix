use rwkv7_p150x2_evidence::{
    MAXIMUM_GENERATED_TOKEN_COUNT, MAXIMUM_PROMPT_TOKEN_COUNT, SCHEMA_VERSION, TokenizedRequest,
    tokenize_request,
};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const VOCABULARY_PATH: Option<&str> = option_env!("RWKV7_TOKENIZER_VOCABULARY");
const DEVICE_RUNTIME_PATH: Option<&str> = option_env!("RWKV7_DEVICE_RUNTIME");
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const TOKENIZE_MODE: &str = "tokenize";
const RUN_DEVICE_MODE: &str = "run-device";
const PUBLISH_MODE: &str = "publish";
const SELF_TEST_MODE: &str = "--self-test";
const MESSAGE_OPTION: &str = "--message";
const PROMPT_BOUND_OPTION: &str = "--max-prompt-tokens";
const GENERATION_BOUND_OPTION: &str = "--max-new-tokens";
const OUTPUT_OPTION: &str = "--output";
const REQUEST_OPTION: &str = "--request";
const EVIDENCE_OPTION: &str = "--evidence-root";
const REFERENCE_OPTION: &str = "--reference-root";
const RUNTIME_OPTION: &str = "--runtime-root";
const SAFETY_OPTION: &str = "--safety-receipt";
const TOKENIZE_ARGUMENT_COUNT: usize = 9;
const RUN_DEVICE_ARGUMENT_COUNT: usize = 5;
const PUBLISH_ARGUMENT_COUNT: usize = 9;
const SELF_TEST_ARGUMENT_COUNT: usize = 1;
const MODE_INDEX: usize = 0;
const FIRST_OPTION_INDEX: usize = 1;
const FIRST_VALUE_INDEX: usize = 2;
const SECOND_OPTION_INDEX: usize = 3;
const SECOND_VALUE_INDEX: usize = 4;
const THIRD_OPTION_INDEX: usize = 5;
const THIRD_VALUE_INDEX: usize = 6;
const FOURTH_OPTION_INDEX: usize = 7;
const FOURTH_VALUE_INDEX: usize = 8;
const TOKEN_LIST_SEPARATOR: &str = ",";
const LOGS_PATH_ENVIRONMENT: &str = "TT_METAL_LOGS_PATH";
const PROFILER_DIRECTORY_ENVIRONMENT: &str = "TT_METAL_PROFILER_DIR";
const INSPECTOR_ENVIRONMENT: &str = "TT_METAL_INSPECTOR";
const INSPECTOR_INITIALIZATION_ENVIRONMENT: &str = "TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT";
const INSPECTOR_RUNTIME_ENTRIES_ENVIRONMENT: &str = "TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES";
const INSPECTOR_RPC_ENVIRONMENT: &str = "TT_METAL_INSPECTOR_RPC";
const DEVICE_PROFILER_ENVIRONMENT: &str = "TT_METAL_DEVICE_PROFILER";
const MID_RUN_DUMP_ENVIRONMENT: &str = "TT_METAL_PROFILER_MID_RUN_DUMP";
const WATCHER_ENVIRONMENT: &str = "TT_METAL_WATCHER";
const DPRINT_CORES_ENVIRONMENT: &str = "TT_METAL_DPRINT_CORES";
const ENABLED_ENVIRONMENT_VALUE: &str = "1";
const DISABLED_ENVIRONMENT_VALUE: &str = "0";
const LOGS_DIRECTORY: &str = "logs";
const PROFILER_DIRECTORY: &str = "profiler";

#[derive(Clone, Debug, Eq, PartialEq)]
struct EnvironmentMutation {
    name: &'static str,
    value: Option<OsString>,
}

fn set_environment(name: &'static str, value: impl Into<OsString>) -> EnvironmentMutation {
    EnvironmentMutation {
        name,
        value: Some(value.into()),
    }
}

fn remove_environment(name: &'static str) -> EnvironmentMutation {
    EnvironmentMutation { name, value: None }
}

// r[impl tenstorrent.rwkv7_p150x2.profiler_isolation]
fn profiler_environment_policy(evidence_root: &Path) -> Vec<EnvironmentMutation> {
    vec![
        set_environment(
            LOGS_PATH_ENVIRONMENT,
            evidence_root.join(LOGS_DIRECTORY).into_os_string(),
        ),
        set_environment(
            PROFILER_DIRECTORY_ENVIRONMENT,
            evidence_root.join(PROFILER_DIRECTORY).into_os_string(),
        ),
        set_environment(INSPECTOR_ENVIRONMENT, ENABLED_ENVIRONMENT_VALUE),
        set_environment(
            INSPECTOR_INITIALIZATION_ENVIRONMENT,
            ENABLED_ENVIRONMENT_VALUE,
        ),
        set_environment(
            INSPECTOR_RUNTIME_ENTRIES_ENVIRONMENT,
            ENABLED_ENVIRONMENT_VALUE,
        ),
        set_environment(INSPECTOR_RPC_ENVIRONMENT, DISABLED_ENVIRONMENT_VALUE),
        set_environment(DEVICE_PROFILER_ENVIRONMENT, ENABLED_ENVIRONMENT_VALUE),
        // The device CSV only appears mid-run when mid-run dumps are enabled;
        // the runtime requires the file synchronously after each flush.
        set_environment(MID_RUN_DUMP_ENVIRONMENT, ENABLED_ENVIRONMENT_VALUE),
        remove_environment(WATCHER_ENVIRONMENT),
        remove_environment(DPRINT_CORES_ENVIRONMENT),
    ]
}

fn unique_environment_mutation<'a>(
    policy: &'a [EnvironmentMutation],
    name: &str,
) -> Result<&'a EnvironmentMutation, String> {
    let mut matching = policy.iter().filter(|mutation| mutation.name == name);
    let mutation = matching
        .next()
        .ok_or_else(|| format!("profiler environment policy lacks {name}"))?;
    if matching.next().is_some() {
        return Err(format!("profiler environment policy repeats {name}"));
    }
    Ok(mutation)
}

fn require_environment_set(
    policy: &[EnvironmentMutation],
    name: &str,
    expected: &OsStr,
) -> Result<(), String> {
    let mutation = unique_environment_mutation(policy, name)?;
    if mutation.value.as_deref() != Some(expected) {
        return Err(format!(
            "profiler environment policy must set {name} to {}",
            expected.to_string_lossy()
        ));
    }
    Ok(())
}

fn require_environment_removed(policy: &[EnvironmentMutation], name: &str) -> Result<(), String> {
    let mutation = unique_environment_mutation(policy, name)?;
    if mutation.value.is_some() {
        return Err(format!(
            "profiler environment policy must remove {name} instead of assigning a value"
        ));
    }
    Ok(())
}

fn validate_profiler_environment_policy(policy: &[EnvironmentMutation]) -> Result<(), String> {
    require_environment_set(
        policy,
        DEVICE_PROFILER_ENVIRONMENT,
        OsStr::new(ENABLED_ENVIRONMENT_VALUE),
    )?;
    require_environment_set(
        policy,
        MID_RUN_DUMP_ENVIRONMENT,
        OsStr::new(ENABLED_ENVIRONMENT_VALUE),
    )?;
    require_environment_removed(policy, WATCHER_ENVIRONMENT)?;
    require_environment_removed(policy, DPRINT_CORES_ENVIRONMENT)?;
    Ok(())
}

fn apply_environment_policy(command: &mut Command, policy: &[EnvironmentMutation]) {
    for mutation in policy {
        match &mutation.value {
            Some(value) => {
                command.env(mutation.name, value);
            }
            None => {
                command.env_remove(mutation.name);
            }
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("rwkv7-p150x2-command: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let arguments = std::env::args_os()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .map(require_utf8_argument)
        .collect::<Result<Vec<_>, _>>()?;
    match arguments.get(MODE_INDEX).map(String::as_str) {
        Some(TOKENIZE_MODE) => {
            run_tokenize(&arguments)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(RUN_DEVICE_MODE) => run_device(&arguments),
        Some(PUBLISH_MODE) => run_publish(&arguments),
        Some(SELF_TEST_MODE) if arguments.len() == SELF_TEST_ARGUMENT_COUNT => {
            run_self_test()?;
            Ok(ExitCode::SUCCESS)
        }
        _ => Err(invalid_input("unknown or missing command mode").into()),
    }
}

fn run_tokenize(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments.len() != TOKENIZE_ARGUMENT_COUNT
        || arguments[FIRST_OPTION_INDEX] != MESSAGE_OPTION
        || arguments[SECOND_OPTION_INDEX] != PROMPT_BOUND_OPTION
        || arguments[THIRD_OPTION_INDEX] != GENERATION_BOUND_OPTION
        || arguments[FOURTH_OPTION_INDEX] != OUTPUT_OPTION
    {
        return Err(invalid_input(format!(
            "{TOKENIZE_MODE} requires {MESSAGE_OPTION} TEXT {PROMPT_BOUND_OPTION} COUNT {GENERATION_BOUND_OPTION} COUNT {OUTPUT_OPTION} PATH"
        ))
        .into());
    }
    let max_prompt_tokens =
        parse_positive_bound(&arguments[SECOND_VALUE_INDEX], PROMPT_BOUND_OPTION)?;
    let max_new_tokens =
        parse_positive_bound(&arguments[THIRD_VALUE_INDEX], GENERATION_BOUND_OPTION)?;
    let output = require_new_absolute_path(&arguments[FOURTH_VALUE_INDEX])?;
    let vocabulary = read_embedded_vocabulary()?;
    let request = tokenize_request(
        &vocabulary,
        arguments[FIRST_VALUE_INDEX].clone(),
        max_prompt_tokens,
        max_new_tokens,
    )
    .map_err(invalid_input)?;
    write_new_json(&output, &request)?;
    println!("{}", output.display());
    Ok(())
}

fn run_device(arguments: &[String]) -> Result<ExitCode, Box<dyn Error>> {
    if arguments.len() != RUN_DEVICE_ARGUMENT_COUNT
        || arguments[FIRST_OPTION_INDEX] != REQUEST_OPTION
        || arguments[SECOND_OPTION_INDEX] != EVIDENCE_OPTION
    {
        return Err(invalid_input(format!(
            "{RUN_DEVICE_MODE} requires {REQUEST_OPTION} PATH {EVIDENCE_OPTION} PATH"
        ))
        .into());
    }
    let request_path = require_existing_file(&arguments[FIRST_VALUE_INDEX])?;
    let evidence_root = require_empty_directory(&arguments[SECOND_VALUE_INDEX])?;
    let request: TokenizedRequest = serde_json::from_slice(&fs::read(request_path)?)
        .map_err(|error| invalid_data(format!("invalid tokenized request: {error}")))?;
    validate_tokenized_request(&request)?;
    let runtime = DEVICE_RUNTIME_PATH.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "RWKV7_DEVICE_RUNTIME was not embedded at build time",
        )
    })?;
    let runtime = Path::new(runtime).join("bin/rwkv7-p150x2-runtime");
    if !runtime.starts_with("/nix/store/") || !runtime.is_file() {
        return Err(invalid_data("embedded device runtime is not a Nix store executable").into());
    }
    let token_list = request
        .prompt_token_ids
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(TOKEN_LIST_SEPARATOR);
    let environment_policy = profiler_environment_policy(&evidence_root);
    validate_profiler_environment_policy(&environment_policy).map_err(invalid_data)?;
    let mut command = Command::new(runtime);
    command
        .arg("--tokens")
        .arg(token_list)
        .arg("--max-new")
        .arg(request.max_new_tokens.to_string())
        .arg("--evidence-root")
        .arg(&evidence_root);
    apply_environment_policy(&mut command, &environment_policy);
    let error = command.exec();
    Err(io::Error::new(
        error.kind(),
        format!("failed to exec the one permitted device runtime: {error}"),
    )
    .into())
}

fn run_publish(arguments: &[String]) -> Result<ExitCode, Box<dyn Error>> {
    if arguments.len() != PUBLISH_ARGUMENT_COUNT
        || arguments[FIRST_OPTION_INDEX] != REFERENCE_OPTION
        || arguments[SECOND_OPTION_INDEX] != RUNTIME_OPTION
        || arguments[THIRD_OPTION_INDEX] != SAFETY_OPTION
        || arguments[FOURTH_OPTION_INDEX] != OUTPUT_OPTION
    {
        return Err(invalid_input(format!(
            "{PUBLISH_MODE} requires {REFERENCE_OPTION} PATH {RUNTIME_OPTION} PATH {SAFETY_OPTION} PATH {OUTPUT_OPTION} PATH"
        ))
        .into());
    }
    let current = std::env::current_exe()?;
    let classifier = current
        .parent()
        .ok_or_else(|| invalid_data("command executable has no parent directory"))?
        .join("rwkv7-p150x2-evidence");
    if !classifier.is_file() {
        return Err(invalid_data("sibling evidence classifier is unavailable").into());
    }
    let status = Command::new(classifier)
        .arg(REFERENCE_OPTION)
        .arg(&arguments[FIRST_VALUE_INDEX])
        .arg(RUNTIME_OPTION)
        .arg(&arguments[SECOND_VALUE_INDEX])
        .arg(SAFETY_OPTION)
        .arg(&arguments[THIRD_VALUE_INDEX])
        .arg(OUTPUT_OPTION)
        .arg(&arguments[FOURTH_VALUE_INDEX])
        .status()?;
    Ok(if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(
            status
                .code()
                .and_then(|code| u8::try_from(code).ok())
                .unwrap_or(1),
        )
    })
}

fn validate_tokenized_request(request: &TokenizedRequest) -> Result<(), Box<dyn Error>> {
    if request.schema_version != SCHEMA_VERSION
        || request.prompt_token_ids.is_empty()
        || request.prompt_token_ids.len() > request.max_prompt_tokens
        || request.max_prompt_tokens > MAXIMUM_PROMPT_TOKEN_COUNT
        || request.max_new_tokens > MAXIMUM_GENERATED_TOKEN_COUNT
    {
        return Err(invalid_data("tokenized request violates package bounds").into());
    }
    let vocabulary = read_embedded_vocabulary()?;
    let reproduced = tokenize_request(
        &vocabulary,
        request.user_message.clone(),
        request.max_prompt_tokens,
        request.max_new_tokens,
    )
    .map_err(invalid_data)?;
    if &reproduced != request {
        return Err(invalid_data("tokenized request does not reproduce exactly").into());
    }
    Ok(())
}

fn parse_positive_bound(text: &str, name: &str) -> Result<usize, io::Error> {
    let value = text
        .parse::<usize>()
        .map_err(|error| invalid_input(format!("{name} must be an unsigned integer: {error}")))?;
    if value == 0 {
        return Err(invalid_input(format!("{name} must be positive")));
    }
    Ok(value)
}

fn require_existing_file(text: &str) -> Result<PathBuf, io::Error> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err(invalid_input("request path must be absolute"));
    }
    let canonical = fs::canonicalize(path)?;
    if !canonical.is_file() || canonical.starts_with("/nix/store/") {
        return Err(invalid_input(
            "request path must be a non-store regular file",
        ));
    }
    Ok(canonical)
}

fn require_empty_directory(text: &str) -> Result<PathBuf, io::Error> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err(invalid_input("evidence root must be absolute"));
    }
    let canonical = fs::canonicalize(&path)?;
    if canonical != path || !canonical.is_dir() || canonical.starts_with("/nix/store/") {
        return Err(invalid_input(
            "evidence root must be a canonical non-store directory",
        ));
    }
    if fs::read_dir(&canonical)?.next().is_some() {
        return Err(invalid_input("evidence root must be empty"));
    }
    Ok(canonical)
}

fn require_new_absolute_path(text: &str) -> Result<PathBuf, io::Error> {
    let path = PathBuf::from(text);
    if !path.is_absolute() || path.starts_with("/nix/store/") || path.exists() {
        return Err(invalid_input(
            "output must be a new absolute non-store path",
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| invalid_input("output path has no parent"))?;
    let canonical_parent = fs::canonicalize(parent)?;
    let filename = path
        .file_name()
        .ok_or_else(|| invalid_input("output path has no filename"))?;
    Ok(canonical_parent.join(filename))
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

fn write_new_json(path: &Path, value: &TokenizedRequest) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn run_self_test() -> Result<(), Box<dyn Error>> {
    let error = parse_positive_bound("0", GENERATION_BOUND_OPTION)
        .expect_err("zero generation bound must fail");
    if !error.to_string().contains("positive") {
        return Err(invalid_data("self-test returned wrong bound diagnostic").into());
    }
    let policy = profiler_environment_policy(Path::new("/tmp/rwkv7-p150x2-self-test"));
    validate_profiler_environment_policy(&policy).map_err(invalid_data)?;
    let mut invalid = policy.clone();
    let watcher = invalid
        .iter_mut()
        .find(|mutation| mutation.name == WATCHER_ENVIRONMENT)
        .ok_or_else(|| invalid_data("self-test policy lacks Watcher removal"))?;
    watcher.value = Some(DISABLED_ENVIRONMENT_VALUE.into());
    let error = validate_profiler_environment_policy(&invalid)
        .expect_err("zero-valued Watcher assignment must fail");
    if !error.contains("must remove TT_METAL_WATCHER") {
        return Err(invalid_data("self-test returned wrong Watcher diagnostic").into());
    }
    println!("rwkv7 P150x2 bounded command self-test: PASS");
    Ok(())
}

fn require_utf8_argument(argument: OsString) -> Result<String, io::Error> {
    argument
        .into_string()
        .map_err(|_| invalid_input("rwkv7-p150x2-command arguments must be valid UTF-8"))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        DISABLED_ENVIRONMENT_VALUE, DPRINT_CORES_ENVIRONMENT, EnvironmentMutation,
        GENERATION_BOUND_OPTION, MID_RUN_DUMP_ENVIRONMENT, WATCHER_ENVIRONMENT,
        parse_positive_bound, profiler_environment_policy, require_new_absolute_path,
        validate_profiler_environment_policy,
    };
    use std::ffi::OsString;
    use std::path::Path;

    const TEST_POSITIVE_BOUND: usize = 2;

    fn assign_environment(
        policy: &mut [EnvironmentMutation],
        name: &str,
        value: impl Into<OsString>,
    ) {
        policy
            .iter_mut()
            .find(|mutation| mutation.name == name)
            .expect("test policy must contain environment mutation")
            .value = Some(value.into());
    }

    #[test]
    fn bounds_accept_positive_and_reject_zero_or_malformed() {
        assert_eq!(
            parse_positive_bound(&TEST_POSITIVE_BOUND.to_string(), GENERATION_BOUND_OPTION)
                .expect("positive bound must parse"),
            TEST_POSITIVE_BOUND
        );
        assert!(parse_positive_bound("0", GENERATION_BOUND_OPTION).is_err());
        assert!(parse_positive_bound("no", GENERATION_BOUND_OPTION).is_err());
        assert!(require_new_absolute_path("relative").is_err());
    }

    // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.clean]
    #[test]
    fn profiler_policy_removes_debug_instrumentation() {
        let policy = profiler_environment_policy(Path::new("/tmp/evidence"));
        validate_profiler_environment_policy(&policy)
            .expect("clean profiler environment policy must pass");
    }

    #[test]
    fn profiler_policy_rejects_zero_or_disabled_mid_run_dump() {
        for value in [DISABLED_ENVIRONMENT_VALUE, "0"] {
            let mut policy = profiler_environment_policy(Path::new("/tmp/evidence"));
            assign_environment(&mut policy, MID_RUN_DUMP_ENVIRONMENT, value);
            assert!(
                validate_profiler_environment_policy(&policy)
                    .expect_err("disabled mid-run dump must fail")
                    .contains("must set TT_METAL_PROFILER_MID_RUN_DUMP")
            );
        }
        let mut policy = profiler_environment_policy(Path::new("/tmp/evidence"));
        let mutation = policy
            .iter_mut()
            .find(|mutation| mutation.name == MID_RUN_DUMP_ENVIRONMENT)
            .expect("test policy must contain the mid-run dump mutation");
        mutation.value = None;
        assert!(
            validate_profiler_environment_policy(&policy)
                .expect_err("missing mid-run dump must fail")
                .contains("must set TT_METAL_PROFILER_MID_RUN_DUMP")
        );
    }

    // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.watcher_zero]
    #[test]
    fn profiler_policy_rejects_zero_or_enabled_watcher_assignments() {
        for value in [DISABLED_ENVIRONMENT_VALUE, "1"] {
            let mut policy = profiler_environment_policy(Path::new("/tmp/evidence"));
            assign_environment(&mut policy, WATCHER_ENVIRONMENT, value);
            assert!(
                validate_profiler_environment_policy(&policy)
                    .expect_err("present Watcher must fail")
                    .contains("must remove TT_METAL_WATCHER")
            );
        }
    }

    // r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.dprint_present]
    #[test]
    fn profiler_policy_rejects_empty_or_enabled_dprint_assignments() {
        for value in ["", "all"] {
            let mut policy = profiler_environment_policy(Path::new("/tmp/evidence"));
            assign_environment(&mut policy, DPRINT_CORES_ENVIRONMENT, value);
            assert!(
                validate_profiler_environment_policy(&policy)
                    .expect_err("present DPrint must fail")
                    .contains("must remove TT_METAL_DPRINT_CORES")
            );
        }
    }
}
