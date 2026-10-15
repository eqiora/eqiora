//! A reused verifier executable must inspect the repository selected at runtime.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Repositories(PathBuf);

impl Repositories {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("eqiora-cli-roots-{}-{nonce}", std::process::id()));
        for name in ["alpha", "beta"] {
            let repository = root.join(name);
            let case = repository.join("verify/cache").join(name);
            for directory in ["models", "references", "expected"] {
                fs::create_dir_all(case.join(directory)).unwrap();
            }
            fs::write(
                case.join("README.md"),
                "Runtime repository selection fixture.\n",
            )
            .unwrap();
            fs::write(case.join("case.toml"), format!(
                "id = \"cache.{name}\"\nstatus = \"specified\"\nreference_kind = \"fixture\"\ncapabilities = [\"runtime-root\"]\n"
            )).unwrap();
            fs::create_dir_all(repository.join("src")).unwrap();
            fs::write(repository.join("src/lib.rs"), "").unwrap();
            fs::write(repository.join("Cargo.toml"), "[package]\nname = \"root-fixture\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n").unwrap();
            fs::write(
                repository.join("Cargo.lock"),
                "version = 4\n[[package]]\nname = \"root-fixture\"\nversion = \"0.0.0\"\n",
            )
            .unwrap();
        }
        Self(root)
    }
}

impl Drop for Repositories {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn list(directory: &Path, root: Option<&str>) -> serde_json::Value {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eqiora-verify"));
    command
        .current_dir(directory)
        .args(["list", "--format", "json"]);
    if let Some(root) = root {
        command.args(["--root", root]);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn one_binary_follows_the_runtime_repository_without_a_build_path_fallback() {
    let repositories = Repositories::new();
    for name in ["alpha", "beta"] {
        let report = list(&repositories.0.join(name), None);
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(
            cases.len(),
            1,
            "only the selected repository belongs in the report"
        );
        assert_eq!(cases[0]["id"], format!("cache.{name}"));
    }
    let report = list(&repositories.0.join("alpha"), Some("../beta"));
    assert_eq!(report["cases"][0]["id"], "cache.beta");
    assert_eq!(report["cases"].as_array().unwrap().len(), 1);
    for root in ["../missing", ".."] {
        let output = Command::new(env!("CARGO_BIN_EXE_eqiora-verify"))
            .current_dir(repositories.0.join("alpha"))
            .args(["list", "--root", root])
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "invalid root must never fall back to the build checkout"
        );
    }
}
