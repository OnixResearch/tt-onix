use rwkv7_p150x2_evidence::classify_attempt3;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_ARGUMENT_COUNT: usize = 1;
const EXPECTED_ARGUMENT_COUNT: usize = 2;
const FIXTURE_ROOT_OPTION_INDEX: usize = 0;
const FIXTURE_ROOT_VALUE_INDEX: usize = 1;
const FIXTURE_ROOT_OPTION: &str = "--fixture-root";
const HARDWARE_STDOUT_FILE: &str = "hardware.stdout";
const HARDWARE_STDERR_FILE: &str = "hardware.stderr";
const SAFETY_RECEIPT_FILE: &str = "safety-receipt.json";
const TERMINAL_OUTCOME_FILE: &str = "terminal-outcome.json";
const FORBIDDEN_RUNTIME_RECEIPT: &str = "runtime/runtime-receipt.json";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv7-p150x2-attempt3: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .map(require_utf8_argument)
        .collect::<Result<Vec<_>, _>>()?;
    let fixture_root = parse_arguments(&arguments).map_err(invalid_input)?;
    let fixture_root = fs::canonicalize(fixture_root)?;
    if !fixture_root.is_dir() {
        return Err(invalid_input("fixture root must be a directory").into());
    }
    if fixture_root.join(FORBIDDEN_RUNTIME_RECEIPT).exists() {
        return Err(
            invalid_data("attempt-3 fixture must not contain a runtime success receipt").into(),
        );
    }

    let hardware_stdout = read_utf8(&fixture_root.join(HARDWARE_STDOUT_FILE))?;
    let hardware_stderr = read_utf8(&fixture_root.join(HARDWARE_STDERR_FILE))?;
    let safety = fs::read(fixture_root.join(SAFETY_RECEIPT_FILE))?;
    let terminal = fs::read(fixture_root.join(TERMINAL_OUTCOME_FILE))?;
    let classification = classify_attempt3(&hardware_stdout, &hardware_stderr, &safety, &terminal)
        .map_err(invalid_data)?;

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &classification)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn parse_arguments(arguments: &[String]) -> Result<PathBuf, String> {
    if arguments.len() != EXPECTED_ARGUMENT_COUNT
        || arguments[FIXTURE_ROOT_OPTION_INDEX] != FIXTURE_ROOT_OPTION
    {
        return Err(format!("requires {FIXTURE_ROOT_OPTION} PATH"));
    }
    let root = PathBuf::from(&arguments[FIXTURE_ROOT_VALUE_INDEX]);
    if !root.is_absolute() {
        return Err("fixture root must be absolute".to_owned());
    }
    Ok(root)
}

fn read_utf8(path: &Path) -> Result<String, Box<dyn Error>> {
    String::from_utf8(fs::read(path)?)
        .map_err(|error| invalid_data(format!("{} is not UTF-8: {error}", path.display())).into())
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

#[cfg(test)]
mod tests {
    use super::{FIXTURE_ROOT_VALUE_INDEX, parse_arguments};

    fn valid_arguments() -> Vec<String> {
        vec!["--fixture-root".to_owned(), "/tmp/attempt-3".to_owned()]
    }

    #[test]
    fn accepts_an_absolute_fixture_root() {
        assert_eq!(
            parse_arguments(&valid_arguments()).expect("valid arguments must parse"),
            std::path::Path::new("/tmp/attempt-3")
        );
    }

    #[test]
    fn rejects_relative_or_incomplete_arguments() {
        let mut relative = valid_arguments();
        relative[FIXTURE_ROOT_VALUE_INDEX] = "relative".to_owned();
        assert!(parse_arguments(&relative).is_err());
        assert!(parse_arguments(&[]).is_err());
    }
}
