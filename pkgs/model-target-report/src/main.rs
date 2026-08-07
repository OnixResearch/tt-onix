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
// Generated from quality/policy.ncl. r[impl tenstorrent.quality.parser_robustness]
const CANDIDATE_JSON_MAX_BYTES: u64 = 1_048_576;

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
    let metadata = fs::metadata(&input)?;
    if metadata.len() > CANDIDATE_JSON_MAX_BYTES {
        return Err(invalid_data(format!(
            "candidate JSON byte count {} exceeds {CANDIDATE_JSON_MAX_BYTES}",
            metadata.len()
        ))
        .into());
    }
    let facts = parse_candidate_facts(&fs::read(input)?)?;
    let report = evaluate_candidate(&facts);
    serde_json::to_writer_pretty(io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

fn parse_candidate_facts(bytes: &[u8]) -> Result<CandidateFacts, io::Error> {
    if bytes.len() as u64 > CANDIDATE_JSON_MAX_BYTES {
        return Err(invalid_data(format!(
            "candidate JSON byte count {} exceeds {CANDIDATE_JSON_MAX_BYTES}",
            bytes.len()
        )));
    }
    serde_json::from_slice(bytes)
        .map_err(|error| invalid_data(format!("invalid candidate facts: {error}")))
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
    use super::{
        CANDIDATE_JSON_MAX_BYTES, REPORT_MODE, parse_candidate_facts, require_regular_file,
        run_self_test,
    };

    const VALID_CANDIDATE: &[u8] = br#"{
        "candidate_name":"fixture",
        "upstream_identity":null,
        "revision":null,
        "license":null,
        "artifacts":null,
        "tokenizer":null,
        "state_kind":null,
        "required_operators":null,
        "memory_bounds":null,
        "reference_authority":null,
        "evidence_adapter":null,
        "topology_compatibility":null
    }"#;

    #[test]
    fn self_test_rejects_incomplete_candidate_facts() {
        run_self_test().expect("self-test must pass");
        assert_eq!(REPORT_MODE, "report");
    }

    #[test]
    fn candidate_parser_accepts_bounded_input_and_rejects_invalid_input() {
        let facts = parse_candidate_facts(VALID_CANDIDATE).expect("bounded candidate must parse");
        assert_eq!(facts.candidate_name, "fixture");
        assert!(parse_candidate_facts(b"{").is_err());
        let oversized = vec![b' '; CANDIDATE_JSON_MAX_BYTES as usize + 1];
        assert!(parse_candidate_facts(&oversized).is_err());
    }

    #[test]
    fn file_validator_rejects_a_missing_input() {
        assert!(require_regular_file("/definitely/missing/model-candidate.json").is_err());
    }
}
