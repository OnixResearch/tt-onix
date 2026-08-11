use rwkv7_p150x2_evidence::persistent_decode_contract::{
    PersistentReceiptPolicy, validate_persistent_ab_receipt, validate_persistent_receipt,
};
use std::path::{Path, PathBuf};

const ARGUMENT_COUNT: usize = 3;
const POLICY_ARGUMENT_INDEX: usize = 1;
const RECEIPT_ARGUMENT_INDEX: usize = 2;
const USAGE: &str = "usage: rwkv7-p150x2-persistent-decode-evidence candidate|production|physical-acceptance RECEIPT_JSON";

fn parse_arguments(
    arguments: &[String],
) -> Result<(Option<PersistentReceiptPolicy>, PathBuf), String> {
    if arguments.len() != ARGUMENT_COUNT {
        return Err(USAGE.to_owned());
    }
    let policy = match arguments[POLICY_ARGUMENT_INDEX].as_str() {
        "candidate" => Some(PersistentReceiptPolicy::Candidate),
        "production" => Some(PersistentReceiptPolicy::Production),
        "physical-acceptance" => None,
        _ => return Err(USAGE.to_owned()),
    };
    let receipt_path = Path::new(&arguments[RECEIPT_ARGUMENT_INDEX]);
    if !receipt_path.is_absolute() {
        return Err("worker receipt path must be absolute".to_owned());
    }
    Ok((policy, receipt_path.to_path_buf()))
}

fn run(arguments: &[String]) -> Result<(), String> {
    let (policy, receipt_path) = parse_arguments(arguments)?;
    let receipt = std::fs::read_to_string(&receipt_path)
        .map_err(|error| format!("failed to read receipt: {error}"))?;
    match policy {
        Some(worker_policy) => {
            validate_persistent_receipt(&receipt, worker_policy)
                .map_err(|error| error.to_string())?;
        }
        None => {
            validate_persistent_ab_receipt(&receipt).map_err(|error| error.to_string())?;
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
        let (policy, path) = parse_arguments(&arguments).expect("valid candidate arguments");
        assert_eq!(policy, Some(PersistentReceiptPolicy::Candidate));
        assert_eq!(path, PathBuf::from("/tmp/receipt.json"));

        let physical = vec![
            "validator".to_owned(),
            "physical-acceptance".to_owned(),
            "/tmp/physical.json".to_owned(),
        ];
        let (policy, path) = parse_arguments(&physical).expect("physical acceptance arguments");
        assert_eq!(policy, None);
        assert_eq!(path, PathBuf::from("/tmp/physical.json"));
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
