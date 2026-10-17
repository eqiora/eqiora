use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use eqiora::package::PackageCompilationError;
use eqiora::package::{
    DirectoryPackageInstaller, DirectoryPackageStore, PackageInstallDisposition,
    PackageInstallError, PackageInstallReceipt, PackageReleaseV1, PackageStageCleanup,
    PackageStore, PackagedModelDocument, ResolutionError, ResolutionRecordV1, SourceBundleDigest,
    SourceFileV1, StoreError,
};

use super::{assert_package_semantics, library_release, root_release};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn create() -> Self {
        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqps-{}-{nonce:x}-{sequence:x}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create package-store test directory");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ExactFixture {
    library: PackageReleaseV1,
    root: PackageReleaseV1,
    resolution: ResolutionRecordV1,
    library_bytes: Vec<u8>,
    root_bytes: Vec<u8>,
}

fn exact_fixture() -> ExactFixture {
    let library = library_release();
    let root = root_release(&library);
    let resolution = ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&library))
        .expect("derive exact lock");
    let library_bytes = library.canonical_json().expect("canonical library release");
    let root_bytes = root.canonical_json().expect("canonical root release");
    ExactFixture {
        library,
        root,
        resolution,
        library_bytes,
        root_bytes,
    }
}

fn release_digests(resolution: &ResolutionRecordV1) -> (SourceBundleDigest, SourceBundleDigest) {
    let library = resolution
        .nodes()
        .iter()
        .find(|node| node.identity().name.as_str() == "Eqiora.Electrical.Basic")
        .expect("library node")
        .source_digest();
    let root = resolution
        .nodes()
        .iter()
        .find(|node| node.identity() == resolution.root())
        .expect("root node")
        .source_digest();
    (library, root)
}

fn entry_path(root: &Path, digest: SourceBundleDigest) -> PathBuf {
    root.join(format!("{digest}.json"))
}

fn assert_install(
    outcome: PackageInstallReceipt,
    digest: SourceBundleDigest,
    disposition: PackageInstallDisposition,
    cleanup: PackageStageCleanup,
) {
    assert_eq!(outcome.digest(), digest);
    assert_eq!(outcome.disposition(), disposition);
    assert_eq!(outcome.staging_cleanup(), cleanup);
}

#[test]
fn explicit_store_rejects_missing_substituted_malformed_and_oversize_entries() {
    let fixture = exact_fixture();
    let (library_digest, root_digest) = release_digests(&fixture.resolution);
    let directory = TestDirectory::create();
    let store = DirectoryPackageStore::open_ambient(&directory.0).expect("open empty store");
    assert!(matches!(
        PackagedModelDocument::compile_locked(
            &store,
            &fixture.resolution,
            "Main",
        ),
        Err(PackageCompilationError::Resolution(
            ResolutionError::MissingBundle(digest)
        )) if digest == library_digest
    ));

    fs::write(
        entry_path(&directory.0, library_digest),
        &fixture.root_bytes,
    )
    .expect("write digest-substituted library");
    fs::write(entry_path(&directory.0, root_digest), &fixture.root_bytes)
        .expect("write root release");
    assert!(PackagedModelDocument::compile_locked(&store, &fixture.resolution, "Main").is_err());

    fs::write(entry_path(&directory.0, library_digest), b"{not-json")
        .expect("write malformed library");
    assert!(PackagedModelDocument::compile_locked(&store, &fixture.resolution, "Main").is_err());

    fs::write(
        entry_path(&directory.0, library_digest),
        &fixture.library_bytes,
    )
    .expect("write exact library");
    fs::write(directory.0.join("unrelated.json"), b"untrusted decoy")
        .expect("write unrelated entry");
    PackagedModelDocument::compile_locked(&store, &fixture.resolution, "Main")
        .expect("unrelated entry cannot affect exact replay");
    assert!(matches!(
        store.load_exact(library_digest, fixture.library_bytes.len() - 1),
        Err(StoreError::ReleaseTooLarge {
            digest,
            observed,
            limit,
        }) if digest == library_digest
            && observed == u64::try_from(fixture.library_bytes.len()).expect("fixture length")
            && limit == u64::try_from(fixture.library_bytes.len() - 1).expect("fixture limit")
    ));
}

#[cfg(unix)]
#[test]
fn exact_replay_does_not_require_directory_enumeration() {
    use std::os::unix::fs::PermissionsExt;

    let fixture = exact_fixture();
    let (library_digest, root_digest) = release_digests(&fixture.resolution);
    let directory = TestDirectory::create();
    fs::write(
        entry_path(&directory.0, library_digest),
        &fixture.library_bytes,
    )
    .expect("write exact library release");
    fs::write(entry_path(&directory.0, root_digest), &fixture.root_bytes)
        .expect("write exact root release");
    let store = DirectoryPackageStore::open_ambient(&directory.0).expect("retain store root");

    let original_permissions = fs::metadata(&directory.0)
        .expect("store metadata")
        .permissions();
    let mut execute_only = original_permissions.clone();
    execute_only.set_mode(0o100);
    fs::set_permissions(&directory.0, execute_only).expect("remove directory-listing permission");

    let enumeration = fs::read_dir(&directory.0);
    let compiled = PackagedModelDocument::compile_locked(&store, &fixture.resolution, "Main");

    fs::set_permissions(&directory.0, original_permissions)
        .expect("restore test-directory permissions");
    assert!(
        enumeration.is_err(),
        "the test root must deny directory enumeration"
    );
    compiled.expect("exact digest lookup must not require directory enumeration");
}

#[test]
fn atomic_installer_publishes_the_exact_locked_restart() {
    let fixture = exact_fixture();
    let (library_digest, root_digest) = release_digests(&fixture.resolution);
    let directory = TestDirectory::create();
    let installer =
        DirectoryPackageInstaller::open_ambient(&directory.0).expect("open package installer");

    assert_install(
        installer
            .install(&fixture.library)
            .expect("install library"),
        library_digest,
        PackageInstallDisposition::Installed,
        PackageStageCleanup::Removed,
    );
    assert_install(
        installer.install(&fixture.root).expect("install root"),
        root_digest,
        PackageInstallDisposition::Installed,
        PackageStageCleanup::Removed,
    );
    assert_install(
        installer
            .install(&fixture.library)
            .expect("repeat library installation"),
        library_digest,
        PackageInstallDisposition::AlreadyPresent,
        PackageStageCleanup::NotNeeded,
    );

    drop(installer);
    let store = DirectoryPackageStore::open_ambient(&directory.0).expect("open installed store");
    assert_package_semantics(&store, &fixture.resolution);

    let substituted_files = fixture
        .library
        .source()
        .files()
        .iter()
        .map(|file| {
            let mut bytes = file.bytes().to_vec();
            bytes.extend_from_slice(b"\n");
            SourceFileV1::new(file.path().clone(), file.role(), bytes)
        })
        .collect();
    let substituted = PackageReleaseV1::new(
        fixture.library.manifest().clone(),
        fixture.library.semantic().clone(),
        substituted_files,
    )
    .expect("same-identity source substitution");
    let substituted_digest = substituted.source_digest().expect("substituted digest");
    assert_ne!(substituted_digest, library_digest);
    fs::write(
        entry_path(&directory.0, library_digest),
        substituted.canonical_json().expect("substituted wire"),
    )
    .expect("mutate installed library after accepted replay");
    assert!(matches!(
        PackagedModelDocument::compile_locked(
            &store,
            &fixture.resolution,
            "Main",
        ),
        Err(PackageCompilationError::Resolution(
            ResolutionError::SourceDigestMismatch { expected, actual }
        )) if expected == library_digest && actual == substituted_digest
    ));
}

#[test]
fn concurrent_equal_installers_converge_on_one_exact_release() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let fixture = exact_fixture();
    let (library_digest, _) = release_digests(&fixture.resolution);
    let library = fixture.library;
    let directory = TestDirectory::create();
    let installer =
        DirectoryPackageInstaller::open_ambient(&directory.0).expect("open package installer");
    let barrier = Arc::new(Barrier::new(2));
    let run =
        |installer: DirectoryPackageInstaller, release: PackageReleaseV1, barrier: Arc<Barrier>| {
            thread::spawn(move || {
                barrier.wait();
                installer.install(&release)
            })
        };
    let first = run(installer.clone(), library.clone(), Arc::clone(&barrier));
    let second = run(installer, library.clone(), barrier);
    let first = first
        .join()
        .expect("first installer")
        .expect("first result");
    let second = second
        .join()
        .expect("second installer")
        .expect("second result");
    assert_eq!(first.digest(), library_digest);
    assert_eq!(second.digest(), library_digest);
    assert!(matches!(
        (first.disposition(), second.disposition()),
        (
            PackageInstallDisposition::Installed,
            PackageInstallDisposition::AlreadyPresent
        ) | (
            PackageInstallDisposition::AlreadyPresent,
            PackageInstallDisposition::Installed
        )
    ));

    let store = DirectoryPackageStore::open_ambient(&directory.0).expect("open installed store");
    assert_eq!(
        store
            .load_exact(library_digest, usize::MAX)
            .expect("load converged release"),
        Some(library.canonical_json().expect("canonical library release"))
    );
}

#[cfg(unix)]
#[test]
fn atomic_installer_does_not_require_directory_enumeration() {
    use std::os::unix::fs::PermissionsExt;

    let fixture = exact_fixture();
    let directory = TestDirectory::create();
    let installer =
        DirectoryPackageInstaller::open_ambient(&directory.0).expect("retain installer root");

    let original_permissions = fs::metadata(&directory.0)
        .expect("store metadata")
        .permissions();
    let mut write_execute_only = original_permissions.clone();
    write_execute_only.set_mode(0o300);
    fs::set_permissions(&directory.0, write_execute_only)
        .expect("remove directory-listing permission");

    let enumeration = fs::read_dir(&directory.0);
    let library_result = installer.install(&fixture.library);
    let root_result = installer.install(&fixture.root);

    fs::set_permissions(&directory.0, original_permissions)
        .expect("restore test-directory permissions");
    assert!(
        enumeration.is_err(),
        "the test root must deny directory enumeration"
    );
    assert!(library_result.is_ok());
    assert!(root_result.is_ok());

    let store = DirectoryPackageStore::open_ambient(&directory.0).expect("open installed store");
    assert_package_semantics(&store, &fixture.resolution);
}

#[cfg(unix)]
#[test]
fn atomic_installer_rejects_non_regular_occupied_digest_entries() {
    use std::os::unix::fs::symlink;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use rustix::fs::{CWD, Mode, mkfifoat};

    let fixture = exact_fixture();
    let (library_digest, _) = release_digests(&fixture.resolution);
    let library = fixture.library;

    let directory_occupant = TestDirectory::create();
    fs::create_dir(entry_path(&directory_occupant.0, library_digest))
        .expect("create directory occupant");
    let directory_installer = DirectoryPackageInstaller::open_ambient(&directory_occupant.0)
        .expect("open directory-occupant installer");
    assert!(matches!(
        directory_installer.install(&library),
        Err(PackageInstallError::ExistingEntry {
            digest,
            source: StoreError::NonRegularEntry(actual),
        }) if digest == library_digest && actual == library_digest
    ));

    let symlink_occupant = TestDirectory::create();
    fs::write(
        symlink_occupant.0.join("target.json"),
        &fixture.library_bytes,
    )
    .expect("write symlink target");
    symlink(
        "target.json",
        entry_path(&symlink_occupant.0, library_digest),
    )
    .expect("create symlink occupant");
    let symlink_installer = DirectoryPackageInstaller::open_ambient(&symlink_occupant.0)
        .expect("open symlink-occupant installer");
    assert!(matches!(
        symlink_installer.install(&library),
        Err(PackageInstallError::ExistingEntry { digest, .. })
            if digest == library_digest
    ));

    let fifo_occupant = TestDirectory::create();
    mkfifoat(
        CWD,
        entry_path(&fifo_occupant.0, library_digest),
        Mode::RUSR | Mode::WUSR,
    )
    .expect("create FIFO occupant");
    let fifo_installer = DirectoryPackageInstaller::open_ambient(&fifo_occupant.0)
        .expect("open FIFO-occupant installer");
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || sender.send(fifo_installer.install(&library)));
    let fifo_result = receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("classifying an occupied FIFO must not block");
    reader
        .join()
        .expect("FIFO installer thread")
        .expect("send FIFO result");
    assert!(matches!(
        fifo_result,
        Err(PackageInstallError::ExistingEntry { digest, .. })
            if digest == library_digest
    ));
}

#[cfg(unix)]
#[test]
fn explicit_store_rejects_redirection_and_special_entries_and_retains_its_root() {
    use std::os::unix::fs::symlink;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use rustix::fs::{CWD, Mode, mkfifoat};

    let fixture = exact_fixture();
    let (library_digest, _) = release_digests(&fixture.resolution);

    let symlink_directory = TestDirectory::create();
    fs::write(
        symlink_directory.0.join("target.json"),
        &fixture.library_bytes,
    )
    .expect("write symlink target");
    symlink(
        "target.json",
        entry_path(&symlink_directory.0, library_digest),
    )
    .expect("create store-entry symlink");
    let symlink_store =
        DirectoryPackageStore::open_ambient(&symlink_directory.0).expect("open symlink store root");
    assert!(
        symlink_store
            .load_exact(library_digest, usize::MAX)
            .is_err()
    );

    let special_directory = TestDirectory::create();
    mkfifoat(
        CWD,
        entry_path(&special_directory.0, library_digest),
        Mode::RUSR | Mode::WUSR,
    )
    .expect("create FIFO store entry");
    let special_store =
        DirectoryPackageStore::open_ambient(&special_directory.0).expect("open special store root");
    let (sender, receiver) = mpsc::channel();
    let reader =
        thread::spawn(move || sender.send(special_store.load_exact(library_digest, usize::MAX)));
    let special_result = receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("opening a FIFO store entry must not block");
    reader
        .join()
        .expect("FIFO reader thread")
        .expect("send FIFO result");
    assert!(matches!(
        special_result,
        Err(StoreError::NonRegularEntry(digest))
            | Err(StoreError::EntryIo { digest, .. })
            if digest == library_digest
    ));

    let retained_directory = TestDirectory::create();
    fs::write(
        entry_path(&retained_directory.0, library_digest),
        &fixture.library_bytes,
    )
    .expect("write original release");
    let retained =
        DirectoryPackageStore::open_ambient(&retained_directory.0).expect("retain store root");
    let moved = retained_directory.0.with_extension("moved");
    fs::rename(&retained_directory.0, &moved).expect("move retained root");
    fs::create_dir(&retained_directory.0).expect("create replacement root");
    fs::write(
        entry_path(&retained_directory.0, library_digest),
        &fixture.root_bytes,
    )
    .expect("write replacement release");
    assert_eq!(
        retained
            .load_exact(library_digest, usize::MAX)
            .expect("load retained release"),
        Some(fixture.library_bytes)
    );

    fs::remove_dir_all(&retained_directory.0).expect("remove replacement root");
    fs::rename(moved, &retained_directory.0).expect("restore original root for cleanup");
}
