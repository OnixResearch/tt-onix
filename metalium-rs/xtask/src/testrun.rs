//! Building, listing and running test binaries one test per process: what
//! `cargo xtask silicon` and `cargo xtask conformance` share.
//!
//! The binaries are built once and run directly, not through `cargo test`, so a
//! run does not rebuild between tests and a per-test timeout measures the test
//! rather than cargo. One test per process because a card is one physical
//! resource and the simulator is a once-per-process singleton.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::json::{self, Value};

/// Polling cadence while waiting for a test process to finish.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Unix `ECHILD` errno value (`No child processes` on Linux/BSD/macOS).
#[cfg(unix)]
const ERRNO_ECHILD: i32 = 10;

/// A test binary `cargo test --no-run` produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built {
    /// The cargo target name (`step11_burn`, `tensor`).
    pub name: String,
    pub exe: PathBuf,
    /// The directory `cargo test` would run it from: its package's.
    pub dir: PathBuf,
}

/// Build test binaries and return what cargo reports.
///
/// `select` is the package and feature selection, as `cargo test` arguments
/// (`-p tt-tests --features tt-tests/silicon`). Decodes cargo's JSON messages
/// with the dependency-free RFC 8259 parser in [`crate::json`].
pub fn build(root: &Path, release: bool, select: &[String]) -> Result<Vec<Built>, String> {
    build_with_env(root, release, select, &[])
}

fn build_with_env(
    root: &Path,
    release: bool,
    select: &[String],
    env: &[(&str, &Path)],
) -> Result<Vec<Built>, String> {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("test")
        .args(select)
        // Keep artifact JSON for discovery, but render compiler diagnostics to
        // inherited stderr instead of discarding them with captured stdout.
        .args(["--no-run", "--message-format=json-render-diagnostics"])
        .args(release.then_some("--release"))
        .current_dir(root)
        .stderr(Stdio::inherit());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !out.status.success() {
        return Err("building the test binaries failed".into());
    }
    let stdout = std::str::from_utf8(&out.stdout)
        .map_err(|e| format!("cargo emitted non-UTF-8 JSON output: {e}"))?;
    parse_cargo_artifacts(stdout)
}

/// Parse newline-delimited Cargo `--message-format=json*` messages and extract
/// executable compiler artifacts.
///
/// Rejects malformed JSON lines, malformed `compiler-artifact` messages, and an
/// unexpectedly empty artifact set so discovery failures never masquerade as an
/// empty success.
fn parse_cargo_artifacts(stdout: &str) -> Result<Vec<Built>, String> {
    let mut v = Vec::new();
    for (idx, raw_line) in stdout.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let val = json::parse(line)
            .map_err(|e| format!("malformed cargo JSON message on line {line_no}: {e}"))?;
        let obj = val
            .as_object()
            .ok_or_else(|| format!("cargo JSON message on line {line_no} is not an object"))?;
        let reason = obj
            .get("reason")
            .ok_or_else(|| format!("cargo JSON message on line {line_no}: missing `reason`"))?
            .as_str()
            .ok_or_else(|| {
                format!("cargo JSON message on line {line_no}: `reason` must be a string")
            })?;
        if reason != "compiler-artifact" {
            continue;
        }

        let exe = match obj.get("executable") {
            None | Some(Value::Null) => continue,
            Some(Value::String(path)) if !path.is_empty() => PathBuf::from(path),
            Some(Value::String(_)) => {
                return Err(format!(
                    "cargo compiler-artifact on line {line_no}: `executable` is empty"
                ));
            }
            Some(_) => {
                return Err(format!(
                    "cargo compiler-artifact on line {line_no}: `executable` must be a string or null"
                ));
            }
        };

        let target = obj
            .get("target")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                format!(
                    "cargo compiler-artifact on line {line_no}: missing or invalid `target` object"
                )
            })?;
        let name = target
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                format!(
                    "cargo compiler-artifact on line {line_no}: missing or invalid `target.name`"
                )
            })?
            .to_string();

        let manifest_path = obj
            .get("manifest_path")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                format!(
                    "cargo compiler-artifact on line {line_no}: missing or invalid `manifest_path`"
                )
            })?;
        let dir = Path::new(manifest_path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or_else(|| {
                format!(
                    "cargo compiler-artifact on line {line_no}: `manifest_path` `{manifest_path}` has no parent directory"
                )
            })?
            .to_path_buf();

        v.push(Built { name, exe, dir });
    }
    if v.is_empty() {
        return Err("cargo test --no-run produced no executable test artifacts".into());
    }
    v.sort_by(|a, b| (&a.name, &a.exe).cmp(&(&b.name, &b.exe)));
    Ok(v)
}

/// The tests in a binary, without the `#[ignore]` ones unless `include_ignored`.
pub fn list(exe: &Path, include_ignored: bool) -> Result<Vec<String>, String> {
    let mut all = run_list(exe, false)?;
    if !include_ignored {
        let ignored = run_list(exe, true)?;
        all.retain(|t| !ignored.contains(t));
    }
    Ok(all)
}

fn run_list(exe: &Path, ignored_only: bool) -> Result<Vec<String>, String> {
    let mut cmd = Command::new(exe);
    cmd.args(["--list", "--format", "terse"]);
    if ignored_only {
        cmd.arg("--ignored");
    }
    let out = cmd
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_suffix(": test"))
        .map(str::to_string)
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Timeout,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Timeout => "TIMEOUT",
        })
    }
}

/// How the spawned test process group was accounted for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessOutcome {
    /// The test process exited on its own and was reaped cleanly. The RAII guard
    /// disarmed before dropping so it never signals a reaped PID/PGID.
    Exited,
    /// The harness actively terminated the process group (`SIGKILL` to `-pgid`)
    /// and reaped the direct child (for example after a timeout or polling error).
    Terminated,
    /// Terminating or reaping the process group failed, or wait status was
    /// ambiguous (`ECHILD`), leaving containment unverified.
    ContainmentFailed(String),
}

impl ProcessOutcome {
    /// Short machine-readable tag recorded in logs and summaries.
    pub fn tag(&self) -> &'static str {
        match self {
            ProcessOutcome::Exited => "exited",
            ProcessOutcome::Terminated => "terminated",
            ProcessOutcome::ContainmentFailed(_) => "containment-failed",
        }
    }
}

impl fmt::Display for ProcessOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessOutcome::Exited => f.write_str("exited"),
            ProcessOutcome::Terminated => f.write_str("terminated"),
            ProcessOutcome::ContainmentFailed(detail) => {
                write!(f, "containment-failed ({detail})")
            }
        }
    }
}

/// Result of a test invocation whose process group was accounted for cleanly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunReport {
    pub verdict: Verdict,
    pub outcome: ProcessOutcome,
}

/// Harness failure while running a test, retaining both the primary failure and
/// process-group containment accounting when a child was spawned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunError {
    pub primary: String,
    pub outcome: Option<ProcessOutcome>,
}

impl RunError {
    /// Short outcome tag for log lines.
    pub fn outcome_tag(&self) -> &'static str {
        match &self.outcome {
            Some(outcome) => outcome.tag(),
            None => "spawn-failed",
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.outcome {
            Some(outcome) => write!(f, "{} [process: {outcome}]", self.primary),
            None => f.write_str(&self.primary),
        }
    }
}

/// Operations on a spawned child process that leads its own process group
/// (`pgid == pid`). Abstracted solely so unit tests can inject polling errors,
/// unwinding panics, and cleanup failures against real child processes.
trait ProcessGroupChild {
    fn pgid(&self) -> u32;
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>>;
    fn kill_group(&self) -> Result<(), String>;
    fn kill_direct(&mut self) -> std::io::Result<()>;
    fn wait(&mut self) -> std::io::Result<ExitStatus>;
}

struct OsProcessGroupChild(Child);

impl ProcessGroupChild for OsProcessGroupChild {
    fn pgid(&self) -> u32 {
        self.0.id()
    }

    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        self.0.try_wait()
    }

    fn kill_group(&self) -> Result<(), String> {
        kill_group_pgid(self.0.id())
    }

    fn kill_direct(&mut self) -> std::io::Result<()> {
        self.0.kill()
    }

    fn wait(&mut self) -> std::io::Result<ExitStatus> {
        self.0.wait()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuardState {
    /// Child has not been reaped; PID/PGID is pinned in our child table and
    /// `Drop` will terminate and reap the process group if not disarmed first.
    Armed,
    /// Child has been reaped or explicitly handed off; `Drop` must never signal
    /// `pgid` again because the OS may immediately recycle the PID/PGID.
    Disarmed,
}

/// Local RAII owner for a spawned test process group.
///
/// Ownership begins immediately after `spawn()` succeeds, before any fallible
/// polling work. While armed, the direct child has not been reaped by this
/// parent process, which pins its PID/PGID in the kernel so `-pgid` cannot be
/// reused by an unrelated process. Once the child is reaped (normal exit) or
/// explicitly terminated/disarmed, the guard transitions to `Disarmed` so it
/// never signals a stale or recycled PID/PGID.
///
/// On panic unwinding while armed, `Drop` terminates and reaps the group
/// without ever panicking.
struct ProcessGroupGuard<C: ProcessGroupChild> {
    child: C,
    pgid: u32,
    state: GuardState,
}

impl<C: ProcessGroupChild> ProcessGroupGuard<C> {
    fn new(child: C) -> Self {
        let pgid = child.pgid();
        Self {
            child,
            pgid,
            state: GuardState::Armed,
        }
    }

    /// Non-blocking poll for child exit. Disarms the guard immediately when the
    /// OS reaps the child so no subsequent operation can signal a reaped PID.
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        debug_assert_eq!(self.state, GuardState::Armed);
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.state = GuardState::Disarmed;
        }
        Ok(status)
    }

    /// Disarm without signaling after an ambiguous wait error (such as
    /// `ECHILD`), where the kernel reports the child is no longer an unreaped
    /// child of this process and PID/PGID reuse cannot be ruled out.
    fn disarm_ambiguous(&mut self, reason: String) -> ProcessOutcome {
        self.state = GuardState::Disarmed;
        ProcessOutcome::ContainmentFailed(reason)
    }

    /// Terminate the entire process group (`SIGKILL` to `-pgid`) while the
    /// direct child is still unreaped, then reap the direct child.
    ///
    /// Transitions to `Disarmed` before returning so `Drop` never repeats
    /// signaling after reaping. Never panics.
    fn terminate_and_reap(&mut self) -> ProcessOutcome {
        if self.state != GuardState::Armed {
            return ProcessOutcome::Exited;
        }
        self.state = GuardState::Disarmed;

        let mut failures = Vec::new();
        // Signal the group BEFORE reaping the direct child so PID/PGID `pgid`
        // remains pinned in our process table and cannot be recycled mid-kill.
        if let Err(err) = self.child.kill_group() {
            failures.push(err);
        }
        if let Err(err) = self.child.kill_direct() {
            failures.push(format!("direct kill of pid {} failed: {err}", self.pgid));
        }
        if let Err(err) = self.child.wait() {
            failures.push(format!("reaping pid {} failed: {err}", self.pgid));
        }

        if failures.is_empty() {
            ProcessOutcome::Terminated
        } else {
            ProcessOutcome::ContainmentFailed(failures.join("; "))
        }
    }
}

impl<C: ProcessGroupChild> Drop for ProcessGroupGuard<C> {
    fn drop(&mut self) {
        if self.state != GuardState::Armed {
            return;
        }
        // Cleanup during stack unwinding MUST NEVER panic (a second panic
        // aborts the process). `writeln!` ignores I/O errors, unlike `eprintln!`.
        if let ProcessOutcome::ContainmentFailed(detail) = self.terminate_and_reap() {
            let _ = writeln!(
                std::io::stderr(),
                "xtask: containment-failed while dropping test process group {}: {detail}",
                self.pgid
            );
        }
    }
}

/// Returns `true` when a `try_wait` error is ambiguous with respect to whether
/// the direct child is still an unreaped child of this process (`ECHILD`).
///
/// On `ECHILD`, the kernel reports that `pid` is not an unreaped child of this
/// parent (for example it was already reaped outside this guard). Signaling
/// `-pgid` in that state risks hitting an unrelated process that recycled the
/// PID/PGID, so we fail containment explicitly rather than signaling blindly.
fn is_ambiguous_wait_error(err: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        err.raw_os_error() == Some(ERRNO_ECHILD)
    }
    #[cfg(not(unix))]
    {
        let _ = err;
        false
    }
}

fn drive_process<C: ProcessGroupChild>(
    mut guard: ProcessGroupGuard<C>,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<RunReport, RunError> {
    let deadline = Instant::now().checked_add(timeout);
    loop {
        match guard.try_wait() {
            Ok(Some(status)) => {
                let verdict = if status.success() {
                    Verdict::Pass
                } else {
                    Verdict::Fail
                };
                return Ok(RunReport {
                    verdict,
                    outcome: ProcessOutcome::Exited,
                });
            }
            Ok(None) => {}
            Err(err) => {
                let primary = format!("polling test process {} failed: {err}", guard.pgid);
                let outcome = if is_ambiguous_wait_error(&err) {
                    guard.disarm_ambiguous(format!(
                        "ambiguous wait error (ECHILD) for pgid {}; skipped signaling to avoid hitting a reused PID/PGID",
                        guard.pgid
                    ))
                } else {
                    guard.terminate_and_reap()
                };
                return Err(RunError {
                    primary,
                    outcome: Some(outcome),
                });
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            let outcome = guard.terminate_and_reap();
            return match outcome {
                ProcessOutcome::Terminated => Ok(RunReport {
                    verdict: Verdict::Timeout,
                    outcome,
                }),
                ProcessOutcome::ContainmentFailed(_) => Err(RunError {
                    primary: format!("test timed out after {}ms", timeout.as_millis()),
                    outcome: Some(outcome),
                }),
                ProcessOutcome::Exited => Ok(RunReport {
                    verdict: Verdict::Timeout,
                    outcome,
                }),
            };
        }
        std::thread::sleep(poll_interval);
    }
}

/// Run one test of `exe` in its own process, on one thread, from `dir`, with
/// `env` added; stdout and stderr go to `out_path`.
pub fn run_one(
    exe: &Path,
    test: &str,
    dir: &Path,
    env: &[(&str, String)],
    timeout: Duration,
    out_path: &Path,
) -> Result<RunReport, RunError> {
    let out = File::create(out_path).map_err(|e| RunError {
        primary: format!("{}: {e}", out_path.display()),
        outcome: None,
    })?;
    let err = out.try_clone().map_err(|e| RunError {
        primary: e.to_string(),
        outcome: None,
    })?;
    // Its own process group: the gates do their device work in a forked child
    // (`tt_ttsim::fork_scope`), and a timeout that killed only the process it spawned left that
    // child running, still holding the card, until the next run found it "in use".
    use std::os::unix::process::CommandExt;
    let child = Command::new(exe)
        .process_group(0)
        .args([
            test,
            "--exact",
            "--test-threads=1",
            "--include-ignored",
            "--nocapture",
        ])
        .envs(env.iter().map(|(k, v)| (*k, v)))
        .current_dir(dir)
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| RunError {
            primary: format!("{}: {e}", exe.display()),
            outcome: None,
        })?;
    // Establish RAII ownership immediately after spawn, before any fallible
    // polling or deadline arithmetic.
    let guard = ProcessGroupGuard::new(OsProcessGroupChild(child));
    let result = drive_process(guard, timeout, POLL_INTERVAL);
    if let Err(ref run_err) = result {
        if let Ok(mut f) = OpenOptions::new().append(true).open(out_path) {
            let _ = writeln!(f, "\n[xtask] harness error: {run_err}");
        }
    }
    if let Ok(f) = File::open(out_path) {
        let _ = f.sync_all();
    }
    result
}

/// Send `SIGKILL` to every process in the group `pgid` leads (the test and any
/// child it forked).
///
/// Must only be called while the direct child (`pid == pgid`) is still an
/// unreaped child of this process so the kernel cannot recycle `pgid`.
fn kill_group_pgid(pgid: u32) -> Result<(), String> {
    if pgid == 0 {
        return Err("refusing to signal process group 0".into());
    }
    let out = Command::new("kill")
        .args(["-KILL", "--"])
        .arg(format!("-{pgid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to execute `kill -KILL -- -{pgid}`: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            Err(format!(
                "`kill -KILL -- -{pgid}` exited with {}",
                out.status
            ))
        } else {
            Err(format!(
                "`kill -KILL -- -{pgid}` failed ({}): {detail}",
                out.status
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("{prefix}-{}-{seq}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_exec_script(path: &Path, body: &str) {
        // Write and chmod the script in a short-lived helper process so this
        // multi-threaded test binary never holds a writable FD to `path` across
        // another test thread's `fork()` (which otherwise races into `ETXTBSY`).
        let mut child = Command::new("/bin/sh")
            .args(["-c", "cat > \"$1\" && chmod 0755 \"$1\"", "sh"])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(body.as_bytes())
            .unwrap();
        let status = child.wait().unwrap();
        assert!(
            status.success(),
            "failed to write script {}",
            path.display()
        );
    }

    fn spawn_script_in_group(script: &Path, dir: &Path) -> OsProcessGroupChild {
        use std::os::unix::process::CommandExt;
        let child = Command::new(script)
            .process_group(0)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        OsProcessGroupChild(child)
    }

    fn is_pid_alive(pid: u32) -> bool {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    fn wait_until_dead(pid: u32, max_wait: Duration) -> bool {
        let deadline = Instant::now() + max_wait;
        while Instant::now() < deadline {
            if !is_pid_alive(pid) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        !is_pid_alive(pid)
    }

    fn read_pid_when_ready(pid_file: &Path, max_wait: Duration) -> u32 {
        let deadline = Instant::now() + max_wait;
        while Instant::now() < deadline {
            if let Ok(text) = std::fs::read_to_string(pid_file) {
                if let Ok(pid) = text.trim().parse::<u32>() {
                    return pid;
                }
            }
            std::thread::sleep(Duration::from_millis(15));
        }
        panic!("forked child did not write {}", pid_file.display());
    }

    // -------------------------------------------------------------------------
    // tt-onix#40: RFC 8259 Cargo artifact JSON decoding tests
    // -------------------------------------------------------------------------

    #[test]
    fn parses_ordinary_spaces_escapes_unicode_reordered_and_whitespace() {
        let messages = [
            // Non-artifact messages must be skipped.
            r#"{"reason":"compiler-message","message":{"level":"warning","rendered":"warn"}}"#,
            r#"{"reason":"build-script-executed","package_id":"foo 0.1.0","cfgs":[]}"#,
            // Artifact with executable: null (e.g. library crate) must be skipped.
            r#"{"reason":"compiler-artifact","executable":null,"target":{"name":"lib_only"},"manifest_path":"/work/Cargo.toml"}"#,
            // Ordinary path.
            r#"{"reason":"compiler-artifact","manifest_path":"/work/pkg/Cargo.toml","target":{"name":"step11_burn","kind":["test"]},"executable":"/work/target/debug/deps/step11_burn-deadbeef"}"#,
            // Spaces, escaped quotes, backslashes, Unicode (\uXXXX, surrogate pair, raw UTF-8),
            // reordered fields, arbitrary single-line whitespace, and a distracting top-level `"name"` field.
            " \t{ \t \"executable\" \t: \"/tmp/dir with spaces/quote\\\"dir/back\\\\slash/caf\\u00E9-\\uD83D\\uDE80-🦀/my_test-1234\" , \"name\" : \"distractor\" , \"manifest_path\" : \"/tmp/dir with spaces/pkg \\\"one\\\"/Cargo.toml\" , \"target\" : { \"kind\" : [ \"test\" ] , \"name\" : \"quoted_\\u0074est\" } , \"reason\" : \"compiler-artifact\" }\t ",
            r#"{"reason":"build-finished","success":true}"#,
        ]
        .join("\n");

        let built = parse_cargo_artifacts(&messages).unwrap();
        assert_eq!(built.len(), 2);
        // Sorted by (name, exe): "quoted_test" < "step11_burn".
        assert_eq!(built[0].name, "quoted_test");
        assert_eq!(
            built[0].exe,
            PathBuf::from("/tmp/dir with spaces/quote\"dir/back\\slash/café-🚀-🦀/my_test-1234")
        );
        assert_eq!(
            built[0].dir,
            PathBuf::from("/tmp/dir with spaces/pkg \"one\"")
        );

        assert_eq!(built[1].name, "step11_burn");
        assert_eq!(
            built[1].exe,
            PathBuf::from("/work/target/debug/deps/step11_burn-deadbeef")
        );
        assert_eq!(built[1].dir, PathBuf::from("/work/pkg"));
    }

    #[test]
    fn parses_multiple_artifacts_independent_of_dir_and_hash() {
        let messages = [
            r#"{"reason":"compiler-artifact","executable":"/custom/out/zeta-aaaa","target":{"name":"zeta"},"manifest_path":"/repo/b/Cargo.toml"}"#,
            r#"{"reason":"compiler-artifact","executable":"/other/path/alpha-9999","target":{"name":"alpha"},"manifest_path":"/repo/a/Cargo.toml"}"#,
            r#"{"reason":"compiler-artifact","executable":"/other/path/alpha-1111","target":{"name":"alpha"},"manifest_path":"/repo/a/Cargo.toml"}"#,
        ]
        .join("\n");

        let built = parse_cargo_artifacts(&messages).unwrap();
        let summary: Vec<(&str, &Path)> = built
            .iter()
            .map(|b| (b.name.as_str(), b.exe.as_path()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("alpha", Path::new("/other/path/alpha-1111")),
                ("alpha", Path::new("/other/path/alpha-9999")),
                ("zeta", Path::new("/custom/out/zeta-aaaa")),
            ]
        );
    }

    #[test]
    fn rejects_malformed_cargo_json_and_empty_artifact_sets() {
        // Empty stdout -> explicit error, never empty success.
        let err = parse_cargo_artifacts("").unwrap_err();
        assert!(err.contains("no executable test artifacts"), "{err}");

        // Only non-artifact / executable:null messages -> explicit error.
        let only_non_exec = [
            r#"{"reason":"compiler-artifact","executable":null,"target":{"name":"lib"},"manifest_path":"/a/Cargo.toml"}"#,
            r#"{"reason":"build-finished","success":true}"#,
        ]
        .join("\n");
        let err = parse_cargo_artifacts(&only_non_exec).unwrap_err();
        assert!(err.contains("no executable test artifacts"), "{err}");

        // Truncated / malformed JSON.
        let err =
            parse_cargo_artifacts(r#"{"reason":"compiler-artifact","executable":"/a"#).unwrap_err();
        assert!(err.contains("malformed cargo JSON"), "{err}");

        // Non-object JSON.
        let err = parse_cargo_artifacts("[1, 2, 3]").unwrap_err();
        assert!(err.contains("not an object"), "{err}");

        // Missing reason.
        let err = parse_cargo_artifacts(r#"{"executable":"/a"}"#).unwrap_err();
        assert!(err.contains("missing `reason`"), "{err}");

        // Non-string executable on compiler-artifact.
        let err = parse_cargo_artifacts(
            r#"{"reason":"compiler-artifact","executable":42,"target":{"name":"t"},"manifest_path":"/a/Cargo.toml"}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("`executable` must be a string or null"),
            "{err}"
        );

        // Missing target.name on executable compiler-artifact.
        let err = parse_cargo_artifacts(
            r#"{"reason":"compiler-artifact","executable":"/a/bin","target":{},"manifest_path":"/a/Cargo.toml"}"#,
        )
        .unwrap_err();
        assert!(err.contains("`target.name`"), "{err}");

        // Missing manifest_path on executable compiler-artifact.
        let err = parse_cargo_artifacts(
            r#"{"reason":"compiler-artifact","executable":"/a/bin","target":{"name":"t"}}"#,
        )
        .unwrap_err();
        assert!(err.contains("`manifest_path`"), "{err}");
    }

    /// Deterministic device-free integration test: builds a tiny temporary
    /// fixture crate with `cargo` (both clean build and warm build) into a
    /// non-default target dir under `/tmp`, discovers the test executable via
    /// Cargo JSON, lists its tests, and executes the discovered test binary.
    #[test]
    fn cargo_build_discovers_and_runs_fixture_crate_clean_and_warm() {
        let root = TempDir::new("xtask-cargo-fixture");
        let crate_dir = root.path().join("crate with spaces");
        let target_dir = root.path().join("custom target dir");
        std::fs::create_dir_all(crate_dir.join("src")).unwrap();
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            r#"[package]
name = "fixture_probe"
version = "0.1.0"
edition = "2021"
"#,
        )
        .unwrap();
        std::fs::write(
            crate_dir.join("src/lib.rs"),
            r#"#[test]
fn discovers_and_passes() {
    assert_eq!(2 + 2, 4);
}
"#,
        )
        .unwrap();

        let env = [("CARGO_TARGET_DIR", target_dir.as_path())];
        // Clean build.
        let clean = build_with_env(&crate_dir, false, &[], &env).unwrap();
        assert_eq!(clean.len(), 1);
        assert_eq!(clean[0].name, "fixture_probe");
        assert!(clean[0].exe.starts_with(&target_dir));
        assert!(clean[0].exe.is_file());
        assert_eq!(clean[0].dir, crate_dir);

        // Warm build (Cargo emits `"fresh": true`).
        let warm = build_with_env(&crate_dir, false, &[], &env).unwrap();
        assert_eq!(warm, clean);

        // List and run the discovered test executable end-to-end.
        let tests = list(&warm[0].exe, false).unwrap();
        assert_eq!(tests, vec!["discovers_and_passes"]);
        let out_path = root.path().join("fixture.out");
        let report = run_one(
            &warm[0].exe,
            &tests[0],
            &warm[0].dir,
            &[],
            Duration::from_secs(10),
            &out_path,
        )
        .unwrap();
        assert_eq!(
            report,
            RunReport {
                verdict: Verdict::Pass,
                outcome: ProcessOutcome::Exited,
            }
        );
    }

    // -------------------------------------------------------------------------
    // tt-onix#41: Process-group accounting, polling errors, and unwinding tests
    // -------------------------------------------------------------------------

    /// A timeout must take the test's forked children with it: the gates fork their device work, and
    /// a survivor keeps the card "in use" for the next run.
    #[test]
    fn a_timeout_kills_the_forked_child_too() {
        let dir = TempDir::new("xtask-timeout");
        let script = dir.path().join("fake-test.sh");
        let pid_file = dir.path().join("child.pid");
        let pid_tmp = dir.path().join("child.pid.tmp");
        write_exec_script(
            &script,
            &format!(
                "#!/bin/sh\nsleep 300 &\necho $! > '{}'\nmv '{}' '{}'\nsleep 300\n",
                pid_tmp.display(),
                pid_tmp.display(),
                pid_file.display()
            ),
        );

        let report = run_one(
            &script,
            "ignored",
            dir.path(),
            &[],
            Duration::from_millis(1500),
            &dir.path().join("out.txt"),
        )
        .unwrap();
        assert_eq!(
            report,
            RunReport {
                verdict: Verdict::Timeout,
                outcome: ProcessOutcome::Terminated,
            }
        );

        let child = read_pid_when_ready(&pid_file, Duration::from_secs(1));
        assert!(
            wait_until_dead(child, Duration::from_secs(2)),
            "the forked child {child} survived the timeout"
        );
    }

    struct RecordingChild {
        inner: OsProcessGroupChild,
        kill_group_calls: Arc<AtomicUsize>,
        kill_direct_calls: Arc<AtomicUsize>,
        wait_calls: Arc<AtomicUsize>,
    }

    impl ProcessGroupChild for RecordingChild {
        fn pgid(&self) -> u32 {
            self.inner.pgid()
        }

        fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
            self.inner.try_wait()
        }

        fn kill_group(&self) -> Result<(), String> {
            self.kill_group_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.kill_group()
        }

        fn kill_direct(&mut self) -> std::io::Result<()> {
            self.kill_direct_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.kill_direct()
        }

        fn wait(&mut self) -> std::io::Result<ExitStatus> {
            self.wait_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.wait()
        }
    }

    #[test]
    fn normal_success_and_failure_exit_disarm_without_signaling_after_reap() {
        let dir = TempDir::new("xtask-normal-exit");

        for (name, exit_code, expected_verdict) in
            [("pass.sh", 0, Verdict::Pass), ("fail.sh", 7, Verdict::Fail)]
        {
            let script = dir.path().join(name);
            write_exec_script(&script, &format!("#!/bin/sh\nexit {exit_code}\n"));
            let kill_group_calls = Arc::new(AtomicUsize::new(0));
            let kill_direct_calls = Arc::new(AtomicUsize::new(0));
            let wait_calls = Arc::new(AtomicUsize::new(0));

            let child = RecordingChild {
                inner: spawn_script_in_group(&script, dir.path()),
                kill_group_calls: Arc::clone(&kill_group_calls),
                kill_direct_calls: Arc::clone(&kill_direct_calls),
                wait_calls: Arc::clone(&wait_calls),
            };
            let report = drive_process(
                ProcessGroupGuard::new(child),
                Duration::from_secs(5),
                Duration::from_millis(10),
            )
            .unwrap();
            assert_eq!(
                report,
                RunReport {
                    verdict: expected_verdict,
                    outcome: ProcessOutcome::Exited,
                }
            );
            assert_eq!(
                kill_group_calls.load(Ordering::SeqCst),
                0,
                "must not signal process group after normal reap"
            );
            assert_eq!(
                kill_direct_calls.load(Ordering::SeqCst),
                0,
                "must not signal direct pid after normal reap"
            );
            assert_eq!(
                wait_calls.load(Ordering::SeqCst),
                0,
                "must not wait again after try_wait reaped the child"
            );
        }
    }

    struct InjectOnReadyChild {
        inner: OsProcessGroupChild,
        pid_file: PathBuf,
        mode: InjectMode,
        kill_group_calls: Arc<AtomicUsize>,
    }

    enum InjectMode {
        PollError(fn() -> std::io::Error),
        Panic(&'static str),
    }

    impl ProcessGroupChild for InjectOnReadyChild {
        fn pgid(&self) -> u32 {
            self.inner.pgid()
        }

        fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
            if self.pid_file.exists() {
                match self.mode {
                    InjectMode::PollError(make_err) => return Err(make_err()),
                    InjectMode::Panic(msg) => panic!("{msg}"),
                }
            }
            self.inner.try_wait()
        }

        fn kill_group(&self) -> Result<(), String> {
            self.kill_group_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.kill_group()
        }

        fn kill_direct(&mut self) -> std::io::Result<()> {
            self.inner.kill_direct()
        }

        fn wait(&mut self) -> std::io::Result<ExitStatus> {
            self.inner.wait()
        }
    }

    #[test]
    fn polling_error_terminates_direct_process_and_forked_child() {
        let dir = TempDir::new("xtask-poll-err");
        let script = dir.path().join("forker.sh");
        let pid_file = dir.path().join("child.pid");
        let pid_tmp = dir.path().join("child.pid.tmp");
        write_exec_script(
            &script,
            &format!(
                "#!/bin/sh\nsleep 300 &\necho $! > '{}'\nmv '{}' '{}'\nsleep 300\n",
                pid_tmp.display(),
                pid_tmp.display(),
                pid_file.display()
            ),
        );

        let inner = spawn_script_in_group(&script, dir.path());
        let direct_pid = inner.pgid();
        let kill_group_calls = Arc::new(AtomicUsize::new(0));
        let wrapped = InjectOnReadyChild {
            inner,
            pid_file: pid_file.clone(),
            mode: InjectMode::PollError(|| std::io::Error::other("injected try_wait EIO")),
            kill_group_calls: Arc::clone(&kill_group_calls),
        };

        let err = drive_process(
            ProcessGroupGuard::new(wrapped),
            Duration::from_secs(5),
            Duration::from_millis(10),
        )
        .unwrap_err();
        assert!(err.primary.contains("injected try_wait EIO"), "{err}");
        assert_eq!(err.outcome, Some(ProcessOutcome::Terminated));
        assert_eq!(err.outcome_tag(), "terminated");
        assert_eq!(kill_group_calls.load(Ordering::SeqCst), 1);

        let forked_pid = read_pid_when_ready(&pid_file, Duration::from_secs(1));
        assert!(
            wait_until_dead(direct_pid, Duration::from_secs(2)),
            "direct process {direct_pid} survived polling error"
        );
        assert!(
            wait_until_dead(forked_pid, Duration::from_secs(2)),
            "forked child {forked_pid} survived polling error"
        );
    }

    #[test]
    fn ambiguous_echild_polling_error_does_not_signal_and_reports_containment_failed() {
        let dir = TempDir::new("xtask-echild");
        let script = dir.path().join("sleeper.sh");
        let pid_file = dir.path().join("ready.pid");
        let pid_tmp = dir.path().join("ready.pid.tmp");
        write_exec_script(
            &script,
            &format!(
                "#!/bin/sh\necho $$ > '{}'\nmv '{}' '{}'\nsleep 300\n",
                pid_tmp.display(),
                pid_tmp.display(),
                pid_file.display()
            ),
        );

        let inner = spawn_script_in_group(&script, dir.path());
        let direct_pid = inner.pgid();
        // Keep a teardown handle so the real sleeper does not leak after the test
        // confirms the guard skipped signaling on ECHILD.
        let inner = Arc::new(Mutex::new(Some(inner)));
        struct EchildChild {
            inner: Arc<Mutex<Option<OsProcessGroupChild>>>,
            pgid: u32,
            pid_file: PathBuf,
            kill_group_calls: Arc<AtomicUsize>,
            kill_direct_calls: Arc<AtomicUsize>,
        }
        impl ProcessGroupChild for EchildChild {
            fn pgid(&self) -> u32 {
                self.pgid
            }
            fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
                if self.pid_file.exists() {
                    return Err(std::io::Error::from_raw_os_error(ERRNO_ECHILD));
                }
                self.inner.lock().unwrap().as_mut().unwrap().try_wait()
            }
            fn kill_group(&self) -> Result<(), String> {
                self.kill_group_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn kill_direct(&mut self) -> std::io::Result<()> {
                self.kill_direct_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn wait(&mut self) -> std::io::Result<ExitStatus> {
                self.inner.lock().unwrap().as_mut().unwrap().wait()
            }
        }

        let kill_group_calls = Arc::new(AtomicUsize::new(0));
        let kill_direct_calls = Arc::new(AtomicUsize::new(0));
        let err = drive_process(
            ProcessGroupGuard::new(EchildChild {
                inner: Arc::clone(&inner),
                pgid: direct_pid,
                pid_file,
                kill_group_calls: Arc::clone(&kill_group_calls),
                kill_direct_calls: Arc::clone(&kill_direct_calls),
            }),
            Duration::from_secs(5),
            Duration::from_millis(10),
        )
        .unwrap_err();

        // Clean up the real child in test teardown.
        if let Some(mut real) = inner.lock().unwrap().take() {
            let _ = real.kill_group();
            let _ = real.kill_direct();
            let _ = real.wait();
        }

        assert_eq!(kill_group_calls.load(Ordering::SeqCst), 0);
        assert_eq!(kill_direct_calls.load(Ordering::SeqCst), 0);
        assert_eq!(err.outcome_tag(), "containment-failed");
        assert!(matches!(
            err.outcome,
            Some(ProcessOutcome::ContainmentFailed(ref msg)) if msg.contains("ECHILD")
        ));
    }

    #[test]
    fn panic_unwinding_cleans_up_process_group_without_double_panic() {
        let dir = TempDir::new("xtask-unwind");
        let script = dir.path().join("forker.sh");
        let pid_file = dir.path().join("child.pid");
        let pid_tmp = dir.path().join("child.pid.tmp");
        write_exec_script(
            &script,
            &format!(
                "#!/bin/sh\nsleep 300 &\necho $! > '{}'\nmv '{}' '{}'\nsleep 300\n",
                pid_tmp.display(),
                pid_tmp.display(),
                pid_file.display()
            ),
        );

        let inner = spawn_script_in_group(&script, dir.path());
        let direct_pid = inner.pgid();
        let kill_group_calls = Arc::new(AtomicUsize::new(0));
        let wrapped = InjectOnReadyChild {
            inner,
            pid_file: pid_file.clone(),
            mode: InjectMode::Panic("simulated polling panic"),
            kill_group_calls: Arc::clone(&kill_group_calls),
        };

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = drive_process(
                ProcessGroupGuard::new(wrapped),
                Duration::from_secs(5),
                Duration::from_millis(10),
            );
        }));
        assert!(unwound.is_err(), "expected simulated panic to unwind");
        assert_eq!(
            kill_group_calls.load(Ordering::SeqCst),
            1,
            "RAII Drop must terminate process group once on unwind"
        );

        let forked_pid = read_pid_when_ready(&pid_file, Duration::from_secs(1));
        assert!(
            wait_until_dead(direct_pid, Duration::from_secs(2)),
            "direct process {direct_pid} survived unwind"
        );
        assert!(
            wait_until_dead(forked_pid, Duration::from_secs(2)),
            "forked child {forked_pid} survived unwind"
        );
    }

    #[test]
    fn panic_unwinding_with_failing_cleanup_does_not_double_panic() {
        struct PanicAndFailCleanup;

        impl ProcessGroupChild for PanicAndFailCleanup {
            fn pgid(&self) -> u32 {
                99999
            }
            fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
                panic!("simulated poll panic before failing cleanup");
            }
            fn kill_group(&self) -> Result<(), String> {
                Err("injected unwind kill_group failure".into())
            }
            fn kill_direct(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("injected unwind kill_direct failure"))
            }
            fn wait(&mut self) -> std::io::Result<ExitStatus> {
                Err(std::io::Error::other("injected unwind wait failure"))
            }
        }

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = drive_process(
                ProcessGroupGuard::new(PanicAndFailCleanup),
                Duration::from_secs(5),
                Duration::from_millis(10),
            );
        }));
        assert!(
            unwound.is_err(),
            "panic must unwind cleanly without aborting from Drop"
        );
    }

    #[test]
    fn cleanup_failure_preserves_primary_error_and_containment_diagnostics() {
        struct FailingCleanupChild {
            pgid: u32,
            fail_poll: bool,
            polled: Cell<bool>,
        }

        impl ProcessGroupChild for FailingCleanupChild {
            fn pgid(&self) -> u32 {
                self.pgid
            }

            fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
                if self.fail_poll {
                    return Err(std::io::Error::other("primary poll failure"));
                }
                self.polled.set(true);
                Ok(None)
            }

            fn kill_group(&self) -> Result<(), String> {
                Err("injected kill -KILL EPERM".into())
            }

            fn kill_direct(&mut self) -> std::io::Result<()> {
                Ok(())
            }

            fn wait(&mut self) -> std::io::Result<ExitStatus> {
                Err(std::io::Error::other("injected wait failure"))
            }
        }

        // Case 1: polling error + cleanup failure retains BOTH diagnostics.
        let err = drive_process(
            ProcessGroupGuard::new(FailingCleanupChild {
                pgid: 4242,
                fail_poll: true,
                polled: Cell::new(false),
            }),
            Duration::from_secs(5),
            Duration::from_millis(10),
        )
        .unwrap_err();
        assert!(err.primary.contains("primary poll failure"), "{err}");
        assert_eq!(err.outcome_tag(), "containment-failed");
        let rendered = err.to_string();
        assert!(rendered.contains("primary poll failure"), "{rendered}");
        assert!(rendered.contains("containment-failed"), "{rendered}");
        assert!(rendered.contains("injected kill -KILL EPERM"), "{rendered}");
        assert!(rendered.contains("injected wait failure"), "{rendered}");

        // Case 2: timeout + cleanup failure must NOT masquerade as clean Timeout or Pass.
        let timeout_err = drive_process(
            ProcessGroupGuard::new(FailingCleanupChild {
                pgid: 4243,
                fail_poll: false,
                polled: Cell::new(false),
            }),
            Duration::from_millis(0),
            Duration::from_millis(10),
        )
        .unwrap_err();
        assert!(timeout_err.primary.contains("timed out"), "{timeout_err}");
        assert_eq!(timeout_err.outcome_tag(), "containment-failed");
        let rendered = timeout_err.to_string();
        assert!(rendered.contains("injected kill -KILL EPERM"), "{rendered}");
        assert!(rendered.contains("injected wait failure"), "{rendered}");
    }
}
