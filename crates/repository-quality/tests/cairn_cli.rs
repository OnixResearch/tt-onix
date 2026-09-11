use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../quality/fixtures/cairn")
        .join(name)
}

fn run(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_repository-quality"))
        .arg("cairn-verdict")
        .arg(path)
        .output()
        .expect("fixture command must start")
}

#[test]
fn cli_accepts_valid_receipt_and_rejects_false_success() {
    let accepted = run(&fixture("valid.json"));
    assert!(accepted.status.success(), "{accepted:?}");
    assert!(String::from_utf8_lossy(&accepted.stdout).contains("Cairn verdict: PASS"));
    for (name, diagnostic) in [
        ("nested-success.json", "top-level valid must be true"),
        ("duplicate-verdict.json", "duplicate JSON key"),
        ("missing-receipt.json", "failed to inspect receipt"),
    ] {
        let rejected = run(&fixture(name));
        assert!(!rejected.status.success(), "accepted {name}");
        assert!(
            rejected.stdout.is_empty(),
            "printed a success message for {name}"
        );
        assert!(
            String::from_utf8_lossy(&rejected.stderr).contains(diagnostic),
            "{rejected:?}"
        );
    }
    assert!(!run(&fixture(".")).status.success());
}

#[test]
fn cli_rejects_wrong_command_and_argument_count() {
    for args in [
        vec![],
        vec!["cairn-verdict"],
        vec!["unknown", "receipt"],
        vec!["cairn-verdict", "receipt", "extra"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_repository-quality"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
    }
}
