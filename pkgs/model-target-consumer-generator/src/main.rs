use model_target_consumer_generator::{find_drift, render_consumer_files};
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_ARGUMENT_COUNT: usize = 1;
const MODE_INDEX: usize = 0;
const INPUT_INDEX: usize = 1;
const ROOT_INDEX: usize = 2;
const EXPECTED_ARGUMENT_COUNT: usize = 3;
const GENERATE_MODE: &str = "generate";
const CHECK_MODE: &str = "check";
const SELF_TEST_MODE: &str = "--self-test";
const SELF_TEST_ARGUMENT_COUNT: usize = 1;
const MAXIMUM_INPUT_BYTES: u64 = 1_048_576;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("model-target-consumer-generator: {error}");
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
        println!("model target consumer generator self-test: PASS");
        return Ok(());
    }
    if arguments.len() != EXPECTED_ARGUMENT_COUNT {
        return Err(invalid_input("requires: generate|check PROFILES.json REPOSITORY_ROOT").into());
    }
    let mode = &arguments[MODE_INDEX];
    if mode != GENERATE_MODE && mode != CHECK_MODE {
        return Err(invalid_input("mode must be generate or check").into());
    }
    let input_path = require_regular_file(&arguments[INPUT_INDEX])?;
    let repository_root = require_directory(&arguments[ROOT_INDEX])?;
    let metadata = fs::metadata(&input_path)?;
    if metadata.len() > MAXIMUM_INPUT_BYTES {
        return Err(invalid_data(format!(
            "profile JSON byte count {} exceeds {MAXIMUM_INPUT_BYTES}",
            metadata.len()
        ))
        .into());
    }
    let rendered = render_consumer_files(&fs::read(input_path)?).map_err(invalid_data)?;
    if mode == GENERATE_MODE {
        write_outputs(&rendered, &repository_root)?;
        println!(
            "generated {} model target consumer projections",
            rendered.len()
        );
        return Ok(());
    }
    let drifted = find_drift(&rendered, &repository_root)?;
    if !drifted.is_empty() {
        let profiles = arguments[INPUT_INDEX].as_str();
        let outputs = drifted
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(invalid_data(format!(
            "stale generated consumer projection from {profiles}: {outputs}"
        ))
        .into());
    }
    println!("model target consumer projections are fresh");
    Ok(())
}

fn write_outputs(
    rendered: &[model_target_consumer_generator::RenderedFile],
    repository_root: &Path,
) -> Result<(), io::Error> {
    for file in rendered {
        let path = repository_root.join(&file.path);
        let parent = path
            .parent()
            .ok_or_else(|| invalid_input("generated output must have a parent directory"))?;
        fs::create_dir_all(parent)?;
        fs::write(path, &file.bytes)?;
    }
    Ok(())
}

fn require_regular_file(text: &str) -> Result<PathBuf, io::Error> {
    let path = fs::canonicalize(text)?;
    if !path.is_file() {
        return Err(invalid_input("profile input must be a regular file"));
    }
    Ok(path)
}

fn require_directory(text: &str) -> Result<PathBuf, io::Error> {
    let path = fs::canonicalize(text)?;
    if !path.is_dir() {
        return Err(invalid_input("repository root must be a directory"));
    }
    Ok(path)
}

fn require_utf8_argument(argument: OsString) -> Result<String, io::Error> {
    argument
        .into_string()
        .map_err(|_| invalid_input("arguments must be valid UTF-8"))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
