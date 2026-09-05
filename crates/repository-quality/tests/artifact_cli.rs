use repository_quality::ArtifactManifest;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
const MANIFEST: &str = "manifest.json";
const RECEIPT: &str = "docs/research/run/receipt.json";
const VALID_BYTES: &[u8] = br#"{"valid":true}"#;
const NON_UTF8_BYTE: u8 = 0xff;

struct Tree(PathBuf);

impl Tree {
    fn new() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "repository-quality-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }

    fn write(&self, path: &str, bytes: &[u8]) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn manifest(&self) -> ArtifactManifest {
        let output = run(&[Path::new("manifest"), &self.0]);
        assert!(output.status.success(), "{output:?}");
        self.write(MANIFEST, &output.stdout);
        serde_json::from_slice(&output.stdout).unwrap()
    }

    fn check(&self) -> Output {
        run(&[Path::new("check"), &self.0, &self.0.join(MANIFEST)])
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn run(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_repository-quality"))
        .args(args)
        .output()
        .unwrap()
}

fn rejected(output: &Output, diagnostic: &str) {
    assert!(
        !output.status.success(),
        "accepted invalid tree: {output:?}"
    );
    assert!(
        output.stdout.is_empty(),
        "printed success for an invalid tree"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(diagnostic),
        "{output:?}"
    );
}

#[test]
fn research_receipt_is_manifested_and_private_content_is_rejected() {
    for suffix in [
        "receipt.json",
        "renamed.txt",
        "target/receipt.json",
        ".pi/receipt.json",
        "node_modules/receipt.json",
    ] {
        let tree = Tree::new();
        let path = format!("docs/research/{suffix}");
        tree.write(&path, VALID_BYTES);
        let manifest = tree.manifest();
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.entries[0].path, path);
        assert_eq!(manifest.entries[0].class, "portable_evidence");
        assert!(tree.check().status.success());
        tree.write(&path, b"credential=fixture-only");
        tree.manifest();
        rejected(&tree.check(), "contains private content");
    }
}

#[test]
fn binary_suffix_does_not_hide_a_private_marker() {
    let tree = Tree::new();
    let mut private = b"credential=fixture-only".to_vec();
    private.push(NON_UTF8_BYTE);
    tree.write(RECEIPT, &private);
    tree.manifest();
    rejected(&tree.check(), "contains private content");
}

#[test]
fn stale_missing_and_relocated_receipts_fail_manifest_admission() {
    for mutation in ["change", "remove", "relocate", "add"] {
        let tree = Tree::new();
        tree.write(RECEIPT, VALID_BYTES);
        tree.manifest();
        assert!(tree.check().status.success());
        match mutation {
            "change" => tree.write(RECEIPT, br#"{"valid":null}"#),
            "remove" => fs::remove_file(tree.0.join(RECEIPT)).unwrap(),
            "relocate" => {
                tree.write("docs/receipt.json", VALID_BYTES);
                fs::remove_file(tree.0.join(RECEIPT)).unwrap();
            }
            "add" => tree.write("docs/research/new.json", VALID_BYTES),
            _ => unreachable!("closed mutation fixture"),
        }
        rejected(&tree.check(), "artifact manifest");
    }
}

#[test]
fn adjacent_documentation_is_not_registered_evidence() {
    let tree = Tree::new();
    tree.write("docs/research-more/receipt.json", VALID_BYTES);
    tree.write("docs/guide.md", b"Guide");
    assert!(tree.manifest().entries.is_empty());
    assert!(tree.check().status.success());
}

#[cfg(unix)]
#[test]
fn registered_symlinks_fail_without_following_external_content() {
    let tree = Tree::new();
    let outside = Tree::new();
    outside.write("private.txt", b"credential=fixture-only");
    tree.write("docs/research/guide.md", b"Guide");
    tree.manifest();
    fs::create_dir_all(tree.0.join("docs/research/run")).unwrap();
    std::os::unix::fs::symlink(outside.0.join("private.txt"), tree.0.join(RECEIPT)).unwrap();
    rejected(
        &tree.check(),
        "portable evidence must not be a symbolic link",
    );
    assert_eq!(
        fs::read(outside.0.join("private.txt")).unwrap(),
        b"credential=fixture-only"
    );
}
