//! Bounded parent-process execution shared by host MPI integration tests.

use std::env;
use std::ffi::OsStr;
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CHILD_TIMEOUT: Duration = Duration::from_secs(180);
const CHILD_OUTPUT_LIMIT: usize = 64 * 1024;

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

pub(crate) fn assert_mpi_test(
    ranks: usize,
    child_test: &str,
    child_env: &str,
    extra_environment: &[(&str, &OsStr)],
) {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    let mut child = command
        .args(["-n", &ranks.to_string()])
        .arg(executable)
        .args(["--exact", child_test, "--nocapture"])
        .env(child_env, "1")
        .envs(extra_environment.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("registered MPI evidence requires mpirun on PATH");
    let stdout = child.stdout.take().expect("MPI child stdout is captured");
    let stderr = child.stderr.take().expect("MPI child stderr is captured");
    let stdout_reader = thread::spawn(move || drain_bounded(stdout, CHILD_OUTPUT_LIMIT));
    let stderr_reader = thread::spawn(move || drain_bounded(stderr, CHILD_OUTPUT_LIMIT));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("MPI child status is readable") {
            break (status, false);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            break (
                child.wait().expect("the killed MPI launcher is reaped"),
                true,
            );
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .expect("MPI stdout reader does not panic")
        .expect("MPI child stdout remains readable");
    let stderr = stderr_reader
        .join()
        .expect("MPI stderr reader does not panic")
        .expect("MPI child stderr remains readable");
    if timed_out {
        panic!(
            "{ranks}-rank {child_test} MPI child exceeded {CHILD_TIMEOUT:?}\nstdout{}:\n{}\nstderr{}:\n{}",
            truncation_marker(&stdout),
            String::from_utf8_lossy(&stdout.bytes),
            truncation_marker(&stderr),
            String::from_utf8_lossy(&stderr.bytes),
        );
    }
    assert!(
        status.success(),
        "{ranks}-rank {child_test} MPI child failed\nstdout{}:\n{}\nstderr{}:\n{}",
        truncation_marker(&stdout),
        String::from_utf8_lossy(&stdout.bytes),
        truncation_marker(&stderr),
        String::from_utf8_lossy(&stderr.bytes),
    );
}

fn launcher_accepts_oversubscribe(launcher: &std::ffi::OsStr) -> bool {
    Command::new(launcher)
        .args(["--oversubscribe", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn drain_bounded(mut reader: impl Read, maximum: usize) -> std::io::Result<BoundedOutput> {
    let mut bytes = Vec::with_capacity(maximum);
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let retained = count.min(maximum.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained != count;
    }
    Ok(BoundedOutput { bytes, truncated })
}

fn truncation_marker(output: &BoundedOutput) -> &'static str {
    if output.truncated { " (truncated)" } else { "" }
}
