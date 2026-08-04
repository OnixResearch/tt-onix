use model_target_core::{CandidateFacts, CandidateStatus, evaluate_candidate};
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROGRAM_ARGUMENT_COUNT: usize = 1;
const REPORT_MODE: &str = "report";
const SELF_TEST_MODE: &str = "--self-test";
const REPORT_ARGUMENT_COUNT: usize = 2;
const SELF_TEST_ARGUMENT_COUNT: usize = 1;
const MODE_INDEX: usize = 0;
const INPUT_INDEX: usize = 1;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("model-target-report: {error}");
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
    if arguments.len() != REPORT_ARGUMENT_COUNT || arguments[MODE_INDEX] != REPORT_MODE {
        return Err(invalid_input("requires: report CANDIDATE.json").into());
    }
    let input = require_regular_file(&arguments[INPUT_INDEX])?;
    let facts: CandidateFacts = serde_json::from_slice(&fs::read(input)?)
        .map_err(|error| invalid_data(format!("invalid candidate facts: {error}")))?;
    let report = evaluate_candidate(&facts);
    serde_json::to_writer_pretty(io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

fn require_regular_file(text: &str) -> Result<PathBuf, io::Error> {
    let path = Path::new(text);
    let canonical = fs::canonicalize(path)?;
    if !canonical.is_file() {
        return Err(invalid_input("candidate input must be a regular file"));
    }
    Ok(canonical)
}

fn run_self_test() -> Result<(), Box<dyn Error>> {
    let facts = CandidateFacts {
        candidate_name: "self-test".to_owned(),
        upstream_identity: None,
        revision: None,
        license: None,
        artifacts: None,
        tokenizer: None,
        state_kind: None,
        required_operators: None,
        memory_bounds: None,
        reference_authority: None,
        evidence_adapter: None,
        topology_compatibility: None,
    };
    let report = evaluate_candidate(&facts);
    if report.status != CandidateStatus::Unsupported
        || report.simulator_change_eligible
        || report.runnable_output_exposed
        || report.blockers.is_empty()
    {
        return Err(invalid_data("self-test accepted incomplete candidate facts").into());
    }
    println!("model target candidate report self-test: PASS");
    Ok(())
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
    use super::{REPORT_MODE, require_regular_file, run_self_test};

    #[test]
    fn self_test_rejects_incomplete_candidate_facts() {
        run_self_test().expect("self-test must pass");
        assert_eq!(REPORT_MODE, "report");
    }

    #[test]
    fn file_validator_rejects_a_missing_input() {
        assert!(require_regular_file("/definitely/missing/model-candidate.json").is_err());
    }
}
