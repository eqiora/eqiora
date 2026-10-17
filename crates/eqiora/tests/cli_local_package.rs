use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn create() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqiora-cli-local-package-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn write(path: impl AsRef<Path>, contents: &str) {
    let path = path.as_ref();
    fs::create_dir_all(path.parent().expect("file parent")).expect("create file parent");
    fs::write(path, contents).expect("write fixture file");
}

#[cfg(target_os = "linux")]
#[test]
fn cli_git_package_locks_fetches_and_compiles_offline() {
    let fixture = TestDirectory::create();
    let repo = fixture.0.join("repository");
    let project = fixture.0.join("project");
    let store = fixture.0.join("store");
    fs::create_dir(&store).unwrap();
    write(
        repo.join("eqiora.toml"),
        "[package]\nname=\"org.example.Git\"\nversion=\"1.0.0\"\nentry=\"main\"\n",
    );
    write(
        repo.join("src/main.eqi"),
        "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    );
    for args in [
        vec!["init", "--initial-branch=main"],
        vec!["add", "."],
        vec!["commit", "-m", "first"],
    ] {
        assert!(
            Command::new("/usr/bin/git")
                .args([
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "commit.gpgsign=false"
                ])
                .args(args)
                .current_dir(&repo)
                .env_clear()
                .env("HOME", &repo)
                .env("PATH", "/usr/bin:/bin")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    write(
        project.join("eqiora.toml"),
        "[package]\nname=\"org.example.Root\"\nversion=\"1.0.0\"\nentry=\"main\"\n",
    );
    write(
        project.join("src/main.eqi"),
        "import org.example.Git.main as library; model Main() {}",
    );
    let add = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "add"])
        .arg(&project)
        .args(["org.example.Git", "--version", "1.0.0", "--git"])
        .arg(&repo)
        .args(["--rev", "refs/heads/main", "--store"])
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );
    let lock = fs::read(project.join("eqiora.lock")).unwrap();
    let fetch = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "fetch"])
        .arg(&project)
        .arg("--store")
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        fetch.status.success(),
        "{}",
        String::from_utf8_lossy(&fetch.stderr)
    );
    fs::remove_dir_all(repo).unwrap();
    let check = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "check"])
        .arg(&project)
        .args(["--entry-model", "library.Shared", "--store"])
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert_eq!(fs::read(project.join("eqiora.lock")).unwrap(), lock);
}

#[test]
fn cli_bundled_vendor_fetch_update_and_offline_check_share_project_owner() {
    let fixture = TestDirectory::create();
    let project = fixture.0.join("project");
    let store = fixture.0.join("store");
    let vendor = fixture.0.join("vendor");
    fs::create_dir(&store).unwrap();
    fs::create_dir(&vendor).unwrap();
    write(
        project.join("eqiora.toml"),
        "[package]\nname = \"org.example.Portable\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    );
    write(
        project.join("src/main.eqi"),
        "model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    );
    let run = |operation: &str, store: &Path, extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_eqiora"))
            .args(["package", operation])
            .arg(&project)
            .args(extra)
            .arg("--store")
            .arg(store)
            .output()
            .unwrap()
    };
    let external = fixture.0.join("external");
    write(
        external.join("eqiora.toml"),
        "[package]\nname = \"org.example.External\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    );
    write(external.join("src/main.eqi"), "public model Shared() {}");
    assert!(
        run(
            "add",
            &store,
            &[
                "org.example.External",
                "--version",
                "1.0.0",
                "--path",
                "../external"
            ]
        )
        .status
        .success()
    );
    let added = run(
        "add",
        &store,
        &[
            "Eqiora.Fluid.Incompressible",
            "--version",
            "0.6.0",
            "--bundled",
        ],
    );
    assert!(
        added.status.success(),
        "{}",
        String::from_utf8_lossy(&added.stderr)
    );
    let lock = fs::read(project.join("eqiora.lock")).unwrap();
    let vendored = run(
        "vendor",
        &store,
        &["--destination", vendor.to_str().unwrap()],
    );
    assert!(
        vendored.status.success(),
        "{}",
        String::from_utf8_lossy(&vendored.stderr)
    );
    assert!(run("fetch", &vendor, &[]).status.success());
    fs::remove_dir_all(&store).unwrap();
    fs::remove_dir_all(&external).unwrap();
    assert!(
        run("check", &vendor, &["--entry-model", "Main"])
            .status
            .success()
    );
    assert_eq!(fs::read(project.join("eqiora.lock")).unwrap(), lock);
    fs::create_dir(&external).unwrap();
    write(
        external.join("eqiora.toml"),
        "[package]\nname = \"org.example.External\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    );
    write(external.join("src/main.eqi"), "public model Shared() {}");
    write(
        project.join("src/main.eqi"),
        "model Main() { parameter gain: 1 = 3; relation law { gain - 3 = 0; } }",
    );
    assert!(
        !run("check", &vendor, &["--entry-model", "Main"])
            .status
            .success()
    );
    assert!(!run("fetch", &vendor, &[]).status.success());
    assert_eq!(fs::read(project.join("eqiora.lock")).unwrap(), lock);
    assert!(run("update", &vendor, &[]).status.success());
    assert!(
        run("check", &vendor, &["--entry-model", "Main"])
            .status
            .success()
    );
    assert!(
        run("remove", &vendor, &["Eqiora.Fluid.Incompressible"])
            .status
            .success()
    );
}

#[test]
fn cli_locks_and_checks_the_same_local_package_project_offline() {
    let fixture = TestDirectory::create();
    let project = fixture.0.join("project");
    let dependency = fixture.0.join("dependency");
    let store = fixture.0.join("store");
    fs::create_dir_all(&store).expect("create package store");

    write(
        project.join("eqiora.toml"),
        "[package]\nname = \"org.example.Root\"\nversion = \"1.0.0\"\nentry = \"main\"\n\n[dependencies.\"org.example.Library\"]\nversion = \"1.0.0\"\nsources = [{ path = \"../dependency\" }]\n",
    );
    write(
        project.join("src/main.eqi"),
        "import org.example.Library.main as library; model Local() {}",
    );
    write(
        dependency.join("eqiora.toml"),
        "[package]\nname = \"org.example.Library\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    );
    write(
        dependency.join("src/main.eqi"),
        "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    );

    let lock = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "lock"])
        .arg(&project)
        .arg("--store")
        .arg(&store)
        .output()
        .expect("run package lock");
    assert!(
        lock.status.success(),
        "package lock failed: {}",
        String::from_utf8_lossy(&lock.stderr)
    );
    assert!(lock.stdout.starts_with(b"locked "));
    let accepted_lock = fs::read(project.join("eqiora.lock")).expect("read accepted lock");

    let hostile = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "add"])
        .arg(&project)
        .arg("org.example.\u{1b}[31m")
        .args(["--version", "1.0.0", "--path", "../dependency", "--store"])
        .arg(&store)
        .output()
        .unwrap();
    assert!(!hostile.status.success());
    assert!(!hostile.stderr.contains(&0x1b));
    assert!(String::from_utf8_lossy(&hostile.stderr).contains("\\u{1b}"));
    assert_eq!(
        fs::read(project.join("eqiora.lock")).unwrap(),
        accepted_lock
    );

    let remove = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "remove"])
        .arg(&project)
        .arg("org.example.Library")
        .arg("--store")
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        !remove.status.success(),
        "cannot remove an imported dependency"
    );
    assert_eq!(
        fs::read(project.join("eqiora.lock")).unwrap(),
        accepted_lock
    );

    let add = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "add"])
        .arg(&project)
        .arg("org.example.Library")
        .args(["--version", "1.0.0", "--path", "../dependency", "--store"])
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );
    assert_eq!(
        fs::read(project.join("eqiora.lock")).unwrap(),
        accepted_lock
    );

    fs::remove_dir_all(&dependency).expect("remove dependency source after locking");
    let check = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "check"])
        .arg(&project)
        .arg("--store")
        .arg(&store)
        .args(["--entry-model", "library.Shared"])
        .output()
        .expect("run package check");
    assert!(
        check.status.success(),
        "package check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(check.stdout.starts_with(b"accepted "));
    fs::remove_file(project.join("src/main.eqi")).expect("remove root source after locking");
    let missing_root = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "check"])
        .arg(&project)
        .arg("--store")
        .arg(&store)
        .args(["--entry-model", "library.Shared"])
        .output()
        .unwrap();
    assert!(
        !missing_root.status.success(),
        "project reopening validates authored root sources"
    );
    assert_eq!(
        fs::read(project.join("eqiora.lock")).expect("reread accepted lock"),
        accepted_lock
    );
}

#[test]
fn cli_preview_and_update_share_the_native_request_selection_proposal() {
    let fixture = TestDirectory::create();
    let project = fixture.0.join("project");
    let store = fixture.0.join("store");
    fs::create_dir(&store).unwrap();
    write(
        project.join("eqiora.toml"),
        "[package]\nname='Root'\nversion='1.0.0'\nentry='main'\n[dependencies.Library]\nversion='1'\nsources=[{path='../old'}, {path='../new'}]\n",
    );
    write(
        project.join("src/main.eqi"),
        "model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    );
    for (path, version) in [("old", "1.9.0"), ("new", "1.10.0")] {
        write(
            fixture.0.join(path).join("eqiora.toml"),
            &format!("[package]\nname='Library'\nversion='{version}'\nentry='main'\n"),
        );
        write(
            fixture.0.join(path).join("src/main.eqi"),
            "public model Shared() {}",
        );
    }
    let native =
        eqiora::package::PackagedModelDocument::preview_local_package_project_v1(&project).unwrap();
    let expected = native.lock_bytes().unwrap();
    let preview = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "preview"])
        .arg(&project)
        .output()
        .unwrap();
    assert!(
        preview.status.success(),
        "{}",
        String::from_utf8_lossy(&preview.stderr)
    );
    assert_eq!(preview.stdout, [expected.as_slice(), b"\n"].concat());
    assert!(!project.join("eqiora.lock").exists());
    let update = Command::new(env!("CARGO_BIN_EXE_eqiora"))
        .args(["package", "update"])
        .arg(&project)
        .arg("--store")
        .arg(&store)
        .output()
        .unwrap();
    assert!(
        update.status.success(),
        "{}",
        String::from_utf8_lossy(&update.stderr)
    );
    assert_eq!(fs::read(project.join("eqiora.lock")).unwrap(), expected);
    let lock: serde_json::Value = serde_json::from_slice(&expected).unwrap();
    assert_eq!(lock["requests"][0]["request"], "1");
    assert_eq!(lock["requests"][0]["selected"], "1.10.0");
}
