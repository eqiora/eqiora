//! Invoke the sole independent mathematical owner and require its complete report.
use crate::numerical_admission::tests::artifact_capture::{digest, json_bytes, write};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CHECKS: [&str; 9] = [
    "full_operator",
    "full_load",
    "physical_free_residual",
    "every_coefficient",
    "reference_free_residual",
    "every_reaction_vs_exported_physical_operator",
    "every_reaction_vs_reference_full_vector",
    "every_reaction_vs_exact_target",
    "reaction_total",
];

pub(super) struct Oracle {
    python: OsString,
    script: PathBuf,
    script_sha256: String,
}

impl Oracle {
    pub(super) fn preflight(root: &Path) -> Self {
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../verify/numerics/cartesian-robin-q1/reference/robin.py");
        assert!(
            script.is_file(),
            "Robin canonical reference is missing: {}",
            script.display()
        );
        let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
        let output = Command::new(&python).arg("--version").output()
            .unwrap_or_else(|error| panic!("Robin comparison requires Python 3 and its standard library; cannot launch {:?}: {error}; artifacts retained at {}", python,root.display()));
        write(&root.join("python-version.stdout"), &output.stdout);
        write(&root.join("python-version.stderr"), &output.stderr);
        let version = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success() && version.starts_with("Python 3."),
            "Robin comparison requires working Python 3; artifacts retained at {}",
            root.display()
        );
        Self {
            python,
            script_sha256: digest(&std::fs::read(&script).unwrap()),
            script,
        }
    }

    pub(super) fn compare(&self, case: &str, directory: &Path, input_sha256: &str) {
        assert_eq!(
            digest(&std::fs::read(&self.script).unwrap()),
            self.script_sha256
        );
        let payload = directory.join("robin-input.json");
        let report = directory.join("comparison.json");
        assert!(!report.exists(), "preserve the earlier comparison report");
        let child = Command::new(&self.python)
            .arg(&self.script)
            .arg("compare")
            .arg(&payload)
            .arg("--output")
            .arg(&report)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| {
                panic!(
                    "cannot launch independent Robin comparator: {error}; artifacts retained at {}",
                    directory.display()
                )
            });
        let pid = child.id();
        let output = child.wait_with_output().unwrap();
        write(&directory.join("comparison.stdout.log"), &output.stdout);
        write(&directory.join("comparison.stderr.log"), &output.stderr);
        let bytes = std::fs::read(&report);
        write(
            &directory.join("comparison-process.json"),
            &json_bytes(&serde_json::json!({
                "pid":pid,"exit_code":output.status.code(),"success":output.status.success(),
                "python":self.python.to_string_lossy(),"script":self.script,"reference_sha256":self.script_sha256,
                "case":case,"input_sha256":input_sha256,
                "report_sha256":bytes.as_ref().ok().map(|bytes|digest(bytes)),
                "stdout_sha256":digest(&output.stdout),"stderr_sha256":digest(&output.stderr),
            })),
        );
        let bytes=bytes.unwrap_or_else(|error|panic!("independent Robin comparator did not write its report: {error}; artifacts retained at {}",directory.display()));
        let checked = validate_report(output.status.success(), &bytes, case, input_sha256);
        assert!(
            checked.is_ok(),
            "independent Robin comparison failed: {:?}; inspect {}",
            checked.err(),
            directory.display()
        );
        assert_eq!(
            digest(&std::fs::read(&self.script).unwrap()),
            self.script_sha256
        );
    }
}

fn validate_report(
    success: bool,
    bytes: &[u8],
    case: &str,
    input_sha256: &str,
) -> Result<(), String> {
    if !success {
        return Err("comparator process failed".into());
    }
    let report: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("invalid comparator JSON: {e}"))?;
    if report["schema"] != 1
        || report["accepted"] != true
        || report["case"] != case
        || report["input_sha256"] != input_sha256
    {
        return Err("comparator did not accept this exact case/input".into());
    }
    let checks = report["checks"]
        .as_object()
        .ok_or("missing comparator checks")?;
    if checks.keys().map(String::as_str).collect::<BTreeSet<_>>() != CHECKS.into_iter().collect()
        || checks.values().any(|value| value != true)
    {
        return Err("comparator lacks all nine successful independent checks".into());
    }
    Ok(())
}

#[test]
fn comparator_report_controls_reject_failed_missing_or_changed_acceptance() {
    let good = serde_json::json!({"schema":1,"accepted":true,"case":"robin","input_sha256":"input",
        "checks":CHECKS.into_iter().map(|name|(name,true)).collect::<std::collections::BTreeMap<_,_>>()});
    let raw = serde_json::to_vec(&good).unwrap();
    validate_report(true, &raw, "robin", "input").unwrap();
    assert!(validate_report(false, &raw, "robin", "input").is_err());
    assert!(validate_report(true, b"", "robin", "input").is_err());
    assert!(validate_report(true, b"{", "robin", "input").is_err());
    for changed in ["accepted", "case", "input_sha256", "checks"] {
        let mut bad = good.clone();
        bad[changed] = serde_json::Value::Null;
        assert!(
            validate_report(true, &serde_json::to_vec(&bad).unwrap(), "robin", "input").is_err()
        );
    }
    let mut missing = good.clone();
    missing["checks"].as_object_mut().unwrap().remove(CHECKS[0]);
    assert!(
        validate_report(
            true,
            &serde_json::to_vec(&missing).unwrap(),
            "robin",
            "input"
        )
        .is_err()
    );
    let mut false_check = good;
    false_check["checks"][CHECKS[0]] = false.into();
    assert!(
        validate_report(
            true,
            &serde_json::to_vec(&false_check).unwrap(),
            "robin",
            "input"
        )
        .is_err()
    );
}
