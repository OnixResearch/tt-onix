use rwkv7_p150x2_evidence::persistent_decode_contract::{
    PersistentReceiptPolicy, validate_persistent_ab_receipt, validate_persistent_receipt,
    validate_persistent_telemetry_receipt, validate_persistent_terminal_probe_receipt,
};
use std::path::{Path, PathBuf};

const ARGUMENT_COUNT: usize = 3;
const POLICY_ARGUMENT_INDEX: usize = 1;
const RECEIPT_ARGUMENT_INDEX: usize = 2;
const USAGE: &str = "usage: rwkv7-p150x2-persistent-decode-evidence candidate|production|telemetry|physical-acceptance|terminal-cancel|terminal-timeout|terminal-failure RECEIPT_JSON";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationMode {
    Worker(PersistentReceiptPolicy),
    Telemetry,
    PhysicalAcceptance,
    TerminalProbe(&'static str),
}

fn parse_arguments(arguments: &[String]) -> Result<(ValidationMode, PathBuf), String> {
    if arguments.len() != ARGUMENT_COUNT {
        return Err(USAGE.to_owned());
    }
    let mode = match arguments[POLICY_ARGUMENT_INDEX].as_str() {
        "candidate" => ValidationMode::Worker(PersistentReceiptPolicy::Candidate),
        "production" => ValidationMode::Worker(PersistentReceiptPolicy::Production),
        "telemetry" => ValidationMode::Telemetry,
        "physical-acceptance" => ValidationMode::PhysicalAcceptance,
        "terminal-cancel" => ValidationMode::TerminalProbe("cancel"),
        "terminal-timeout" => ValidationMode::TerminalProbe("timeout"),
        "terminal-failure" => ValidationMode::TerminalProbe("failure"),
        _ => return Err(USAGE.to_owned()),
    };
    let receipt_path = Path::new(&arguments[RECEIPT_ARGUMENT_INDEX]);
    if !receipt_path.is_absolute() {
        return Err("worker receipt path must be absolute".to_owned());
    }
    Ok((mode, receipt_path.to_path_buf()))
}

fn run(arguments: &[String]) -> Result<(), String> {
    let (mode, receipt_path) = parse_arguments(arguments)?;
    let receipt = std::fs::read_to_string(&receipt_path)
        .map_err(|error| format!("failed to read receipt: {error}"))?;
    match mode {
        ValidationMode::Worker(policy) => {
            validate_persistent_receipt(&receipt, policy).map_err(|error| error.to_string())?;
        }
        ValidationMode::Telemetry => {
            validate_persistent_telemetry_receipt(&receipt).map_err(|error| error.to_string())?;
        }
        ValidationMode::PhysicalAcceptance => {
            validate_persistent_ab_receipt(&receipt).map_err(|error| error.to_string())?;
        }
        ValidationMode::TerminalProbe(probe) => {
            validate_persistent_terminal_probe_receipt(&receipt, probe)
                .map_err(|error| error.to_string())?;
        }
    }
    println!("persistent decode receipt: PASS");
    Ok(())
}

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    if let Err(error) = run(&arguments) {
        eprintln!("rwkv7-p150x2-persistent-decode-evidence: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_candidate_receipt() {
        let arguments = vec![
            "validator".to_owned(),
            "candidate".to_owned(),
            "/tmp/receipt.json".to_owned(),
        ];
        let (mode, path) = parse_arguments(&arguments).expect("valid candidate arguments");
        assert_eq!(
            mode,
            ValidationMode::Worker(PersistentReceiptPolicy::Candidate)
        );
        assert_eq!(path, PathBuf::from("/tmp/receipt.json"));

        let telemetry = vec![
            "validator".to_owned(),
            "telemetry".to_owned(),
            "/tmp/telemetry.json".to_owned(),
        ];
        let (mode, path) = parse_arguments(&telemetry).expect("telemetry arguments");
        assert_eq!(mode, ValidationMode::Telemetry);
        assert_eq!(path, PathBuf::from("/tmp/telemetry.json"));

        let physical = vec![
            "validator".to_owned(),
            "physical-acceptance".to_owned(),
            "/tmp/physical.json".to_owned(),
        ];
        let (mode, path) = parse_arguments(&physical).expect("physical acceptance arguments");
        assert_eq!(mode, ValidationMode::PhysicalAcceptance);
        assert_eq!(path, PathBuf::from("/tmp/physical.json"));

        let terminal = vec![
            "validator".to_owned(),
            "terminal-timeout".to_owned(),
            "/tmp/terminal.json".to_owned(),
        ];
        let (mode, path) = parse_arguments(&terminal).expect("terminal probe arguments");
        assert_eq!(mode, ValidationMode::TerminalProbe("timeout"));
        assert_eq!(path, PathBuf::from("/tmp/terminal.json"));
    }

    #[test]
    fn rejects_unknown_policy_and_relative_path() {
        let unknown_policy = vec![
            "validator".to_owned(),
            "unknown".to_owned(),
            "/tmp/receipt.json".to_owned(),
        ];
        assert_eq!(
            parse_arguments(&unknown_policy).expect_err("unknown policy"),
            USAGE
        );

        let relative_path = vec![
            "validator".to_owned(),
            "production".to_owned(),
            "receipt.json".to_owned(),
        ];
        assert_eq!(
            parse_arguments(&relative_path).expect_err("relative path"),
            "worker receipt path must be absolute"
        );
    }
}
