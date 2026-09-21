use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora_package::{
    BundleEntryV1, BundleRoleV1, DirectoryPackageStore, ExactVersion, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, SourceFileV1,
};

use super::*;
use crate::package::PackagedModelDocument;

#[cfg(target_os = "linux")]
mod git;
mod offline;
mod version_requests;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
const SOURCE_PATH: &str = "src/main.eqi";

pub(super) struct TestDirectory(pub(super) PathBuf);

impl TestDirectory {
    pub(super) fn create(name: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eqiora-local-package-{name}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }

    fn child(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).expect("create child directory");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn author_sources(
    name: &str,
    source: &str,
    dependencies: Vec<PackageDependencyV1>,
) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse(SOURCE_PATH).expect("source path");
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse("1.0.0").expect("version"),
        dependencies,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("package manifest");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("author sources")
}

fn exact_dependency(release: &PackageReleaseV1) -> PackageDependencyV1 {
    PackageDependencyV1::new(release.package_identity().expect("package identity"))
}

fn write_package(
    package_root: &Path,
    source_root: &str,
    sources: &PackageSourcesV1,
    dependencies: &[(&PackageReleaseV1, &str)],
) {
    let source_directory = package_root.join(source_root);
    fs::create_dir_all(&source_directory).expect("create source directory");
    fs::write(
        source_directory.join("main.eqi"),
        sources.files()[0].bytes(),
    )
    .expect("write source");
    let package = sources.manifest();
    let mut manifest = format!(
        "[package]\nname = \"{}\"\nversion = \"{}\"\nsource = \"{source_root}\"\nentry = \"main\"\n",
        package.name(),
        package.version()
    );
    for (dependency, dependency_path) in dependencies {
        let identity = dependency.package_identity().expect("dependency identity");
        manifest.push_str(&format!(
            "\n[dependencies.\"{}\"]\nversion = \"{}\"\nsources = [{{ path = \"{dependency_path}\" }}]\n",
            identity.name, identity.version
        ));
    }
    fs::write(package_root.join(PROJECT_MANIFEST), manifest).expect("write package manifest");
}

#[test]
fn local_project_locks_deterministically_and_reopens_offline() {
    let fixture = TestDirectory::create("complete");
    let library_path = fixture.child("library");
    let auxiliary_path = fixture.child("auxiliary");
    let first_store = fixture.child("store-first");
    let second_store = fixture.child("store-second");

    let library_sources = author_sources(
        "org.example.Library",
        "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
        vec![],
    );
    let library_release =
        prepare_package_release_v1(library_sources.clone(), &[]).expect("library release");
    write_package(&library_path, "src", &library_sources, &[]);

    let auxiliary_sources = author_sources(
        "org.example.Auxiliary",
        "public model Other() {}",
        vec![exact_dependency(&library_release)],
    );
    let auxiliary_release = prepare_package_release_v1(
        auxiliary_sources.clone(),
        std::slice::from_ref(&library_release),
    )
    .expect("auxiliary release");
    write_package(
        &auxiliary_path,
        "src",
        &auxiliary_sources,
        &[(&library_release, "../library")],
    );

    let root_sources = author_sources(
        "org.example.Root",
        "import org.example.Library.main as library; model Local() {}",
        vec![
            exact_dependency(&library_release),
            exact_dependency(&auxiliary_release),
        ],
    );
    write_package(
        &fixture.0,
        "root/src",
        &root_sources,
        &[
            (&auxiliary_release, "auxiliary"),
            (&library_release, "library"),
        ],
    );

    let first =
        resolve_local_package_project_v1(&fixture.0, &first_store).expect("first resolution");
    let lock_bytes = fs::read(fixture.0.join(PROJECT_LOCK)).expect("read exact lock");
    assert_eq!(
        lock::ProjectLock::decode(&lock_bytes).unwrap().resolution,
        first
    );
    let reopened = lock::ProjectLock::decode(&lock_bytes)
        .expect("reopen project lock")
        .resolution;
    let second =
        resolve_local_package_project_v1(&fixture.0, &second_store).expect("repeated resolution");
    assert_eq!(
        first.canonical_json().expect("first lock"),
        second.canonical_json().expect("second lock")
    );

    let store = DirectoryPackageStore::open_ambient(&first_store).expect("offline store");
    let model = PackagedModelDocument::compile_locked(&store, &reopened, "library.Shared")
        .expect("compile imported public Model");
    model
        .compilation()
        .validate_against(&first)
        .expect("exact local resolution lineage");
    assert!(model.model().aliases().contains_key("law"));
}

#[test]
fn local_project_editor_analysis_is_read_only_and_accepts_source_overrides() {
    let fixture = TestDirectory::create("editor");
    let library_path = fixture.child("library");

    let library_sources = author_sources(
        "org.example.EditorLibrary",
        "public component Resistor() {}",
        vec![],
    );
    let library_release =
        prepare_package_release_v1(library_sources.clone(), &[]).expect("library release");
    write_package(&library_path, "src", &library_sources, &[]);
    let root_source = "import org.example.EditorLibrary.main as library; model Main() { instance load: library.Resistor(); }";
    let root_sources = author_sources(
        "org.example.EditorRoot",
        root_source,
        vec![exact_dependency(&library_release)],
    );
    write_package(
        &fixture.0,
        "root/src",
        &root_sources,
        &[(&library_release, "library")],
    );

    let (workspace, paths) = analyze_local_package_editor_project_v1(
        41,
        &fixture.0,
        &BTreeMap::from([(
            PathBuf::from("root").join(SOURCE_PATH),
            format!("// unsaved\n{root_source}"),
        )]),
    )
    .expect("read-only editor analysis");
    let root_file = paths
        .iter()
        .find_map(|(file, path)| (path == &PathBuf::from("root").join(SOURCE_PATH)).then_some(file))
        .expect("root source location");
    let library_file = paths
        .iter()
        .find_map(|(file, path)| {
            (path == &PathBuf::from("library").join(SOURCE_PATH)).then_some(file)
        })
        .expect("library source location");
    assert_eq!(workspace.version(), 41);
    assert!(workspace.document(root_file).is_some());
    assert!(workspace.document(library_file).is_some());
    assert_eq!(workspace.references().len(), 1);
    assert_eq!(workspace.references()[0].definition().file(), library_file);
    assert!(!fixture.0.join(PROJECT_LOCK).exists());

    let incomplete = "import org.example.EditorLibrary.main as library; model Main() { instance load: library.Re";
    let (recovered, paths) = analyze_local_package_editor_project_v1(
        42,
        &fixture.0,
        &BTreeMap::from([
            (PathBuf::from("root").join(SOURCE_PATH), incomplete.into()),
            (
                PathBuf::from("library").join(SOURCE_PATH),
                "/// Newly authored name.\npublic component Revised() {".into(),
            ),
        ]),
    )
    .expect("completion retains admitted dependencies during incomplete edits");
    let root = paths
        .iter()
        .find_map(|(file, path)| (path == &PathBuf::from("root").join(SOURCE_PATH)).then_some(file))
        .unwrap();
    let (_, completion) = recovered.completion(root, incomplete.len() as u32).unwrap();
    assert_eq!(
        completion.iter().map(|c| c.name()).collect::<Vec<_>>(),
        ["library.Revised"]
    );
    assert!(
        completion[0]
            .documentation()
            .unwrap()
            .contains("Newly authored")
    );
    assert!(recovered.references().is_empty());
    assert!(recovered.definitions().is_empty());
    assert!(recovered.compile_model(root, "Main").is_err());
    assert_eq!(
        fs::read_to_string(fixture.0.join("root").join(SOURCE_PATH)).unwrap(),
        root_source
    );
    assert_eq!(
        fs::read_to_string(library_path.join(SOURCE_PATH)).unwrap(),
        "public component Resistor() {}"
    );
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}

#[test]
fn path_loaded_and_in_memory_source_have_identical_declaration_documentation() {
    let fixture = TestDirectory::create("documentation");
    let source = "// 🧪\r\n/// Model explanation.\r\nmodel Main() {\r\n/// State explanation.\r\nvariable x:1; relation balance { x=0; }\r\n}\r\n";
    let sources = author_sources("org.example.Documentation", source, vec![]);
    write_package(&fixture.0, "src", &sources, &[]);
    let (workspace, paths) =
        analyze_local_package_editor_project_v1(1, &fixture.0, &BTreeMap::new()).unwrap();
    let file = paths.keys().next().unwrap();
    let loaded = workspace.document(file).unwrap();
    let memory = crate::editor::EditorService::new(file, 1, source);
    assert_eq!(loaded.symbols(), memory.current().symbols());
    assert_eq!(loaded.formatted(), memory.current().formatted());
    let model = &loaded.symbols()[0];
    assert_eq!(model.doc_comment().unwrap().summary(), "Model explanation.");
    let doc = model.children()[0].doc_comment().unwrap();
    assert_eq!(doc.summary(), "State explanation.");
    assert_eq!(
        &source[doc.range().start() as usize..doc.range().end() as usize],
        "/// State explanation.\r"
    );
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}

#[test]
fn changed_local_content_generates_a_new_exact_identity() {
    let fixture = TestDirectory::create("mismatch");
    let dependency_path = fixture.child("dependency");
    let store = fixture.child("store");

    let admitted_sources = author_sources(
        "org.example.Dependency",
        "public model Shared() { parameter gain: 1 = 1; }",
        vec![],
    );
    let admitted_release =
        prepare_package_release_v1(admitted_sources, &[]).expect("expected dependency release");
    let root_sources = author_sources(
        "org.example.Root",
        "model Local() {}",
        vec![exact_dependency(&admitted_release)],
    );
    write_package(
        &fixture.0,
        "root/src",
        &root_sources,
        &[(&admitted_release, "dependency")],
    );
    let admitted_sources = author_sources(
        "org.example.Dependency",
        "public model Shared() { parameter gain: 1 = 1; }",
        vec![],
    );
    write_package(&dependency_path, "src", &admitted_sources, &[]);
    let accepted =
        resolve_local_package_project_v1(&fixture.0, &store).expect("initial exact lock");
    let previous_lock = accepted.canonical_json().expect("previous lock");

    let changed_sources = author_sources(
        "org.example.Dependency",
        "public model Shared() { parameter gain: 1 = 2; }",
        vec![],
    );
    fs::write(
        dependency_path.join(SOURCE_PATH),
        changed_sources.files()[0].bytes(),
    )
    .expect("change dependency source");

    let changed = resolve_local_package_project_v1(&fixture.0, &store)
        .expect("changed author source generates a new exact release");
    let changed_lock = changed.canonical_json().expect("changed lock");
    assert_ne!(changed_lock, previous_lock);
    assert_eq!(
        lock::ProjectLock::decode(&fs::read(fixture.0.join(PROJECT_LOCK)).expect("updated lock"))
            .unwrap()
            .resolution,
        changed
    );
}

#[test]
fn project_manifest_rejects_path_escape_and_unknown_fields_before_lock() {
    let fixture = TestDirectory::create("invalid-project");
    let store = fixture.child("store");
    fs::write(
        fixture.0.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Invalid\"\nversion = \"1.0.0\"\nsource = \"../root\"\nentry = \"main\"\nextra = true\n",
    )
    .expect("write invalid manifest");

    assert!(resolve_local_package_project_v1(&fixture.0, &store).is_err());
    assert!(!fixture.0.join(PROJECT_LOCK).exists());

    fs::write(
        fixture.0.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Invalid\"\nversion = \"1.0.0\"\nsource = \"../root\"\nentry = \"main\"\n",
    )
    .expect("write escaping manifest");
    assert!(resolve_local_package_project_v1(&fixture.0, &store).is_err());
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}

#[test]
fn dependency_paths_are_explicit_bounded_and_portable() {
    assert_eq!(
        resolve_dependency_path(Path::new(""), "../library").unwrap(),
        Path::new("../library")
    );
    assert_eq!(
        resolve_dependency_path(Path::new("../library"), "../other").unwrap(),
        Path::new("../other")
    );
    for invalid in [
        "/absolute",
        "C:/absolute",
        "library/../other",
        "./library",
        "library//child",
        "library\\child",
        "",
    ] {
        assert!(
            resolve_dependency_path(Path::new(""), invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
    assert!(resolve_dependency_path(Path::new(""), &"../".repeat(65)).is_err());
}

#[cfg(unix)]
#[test]
fn external_dependency_rejects_intermediate_symlinks() {
    let fixture = TestDirectory::create("external-symlink");
    let project = fixture.child("project");
    let library = fixture.child("library");
    let directory = open_project_root(&project).unwrap();
    std::os::unix::fs::symlink(&library, fixture.0.join("alias")).unwrap();
    assert!(open_dependency_directory(&directory, "../library").is_ok());
    assert!(open_dependency_directory(&directory, "../alias").is_err());
    assert!(open_dependency_directory(&directory, "../alias/child").is_err());
}

#[test]
fn failed_reresolution_preserves_the_accepted_lock() {
    let fixture = TestDirectory::create("preserve-lock");
    let store = fixture.child("store");
    let sources = author_sources("org.example.Root", "model Main() {}", vec![]);
    write_package(&fixture.0, "src", &sources, &[]);
    resolve_local_package_project_v1(&fixture.0, &store).expect("initial resolution");
    let accepted = fs::read(fixture.0.join(PROJECT_LOCK)).expect("accepted lock");

    fs::write(fixture.0.join("src/Main.eqi"), "model Other() {}")
        .expect("write portable case collision");
    assert!(resolve_local_package_project_v1(&fixture.0, &store).is_err());
    assert_eq!(
        fs::read(fixture.0.join(PROJECT_LOCK)).expect("preserved lock"),
        accepted
    );
}

#[test]
fn local_dependency_cycle_is_rejected_before_lock() {
    let fixture = TestDirectory::create("cycle");
    let child = fixture.child("child");
    let store = fixture.child("store");
    let root = author_sources("org.example.Root", "model Main() {}", vec![]);
    let dependency = author_sources("org.example.Child", "public model Shared() {}", vec![]);
    write_package(&fixture.0, "src", &root, &[]);
    write_package(&child, "src", &dependency, &[]);
    fs::write(
        fixture.0.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Root\"\nversion = \"1.0.0\"\nentry = \"main\"\n\n[dependencies.\"org.example.Child\"]\nversion = \"1.0.0\"\nsources = [{ path = \"child\" }]\n",
    )
    .expect("write root manifest");
    fs::write(
        child.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Child\"\nversion = \"1.0.0\"\nentry = \"main\"\n\n[dependencies.\"org.example.Root\"]\nversion = \"1.0.0\"\nsources = [{ path = \"..\" }]\n",
    )
    .expect("write child manifest");

    assert!(resolve_local_package_project_v1(&fixture.0, &store).is_err());
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}

#[test]
fn partial_lock_write_preserves_the_accepted_project_pair() {
    let fixture = TestDirectory::create("partial-lock-write");
    let store_path = fixture.child("store");
    let sources = author_sources(
        "org.example.Root",
        "model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
        vec![],
    );
    write_package(&fixture.0, "src", &sources, &[]);
    let accepted =
        resolve_local_package_project_v1(&fixture.0, &store_path).expect("initial project lock");
    let manifest = fs::read(fixture.0.join(PROJECT_MANIFEST)).expect("accepted manifest");
    let lock = fs::read(fixture.0.join(PROJECT_LOCK)).expect("accepted lock");
    fs::write(
        fixture.0.join("src/main.eqi"),
        "model Main() { parameter gain: 1 = 3; relation law { gain - 3 = 0; } }",
    )
    .expect("changed source");
    let candidate = prepare_local_package_project(
        open_project_root(&fixture.0).unwrap(),
        &fixture.0,
        LocalProjectOverrides::default(),
    )
    .expect("prepare changed project");
    let changed = ResolutionRecordV1::from_exact_releases(&candidate.root.release, &[])
        .expect("changed lock");
    assert_ne!(accepted, changed);
    let guard = transaction::write_guard(&candidate.project).unwrap();
    let failure = transaction::commit_using(
        &candidate.project,
        &manifest,
        &lock::ProjectLock::new(changed, vec![], vec![])
            .unwrap()
            .bytes()
            .unwrap(),
        || Ok(()),
        |file, bytes| {
            file.write_all(&bytes[..bytes.len() / 2])?;
            Err(std::io::Error::other(
                "injected failure after partial staging write",
            ))
        },
    )
    .expect_err("partial write must fail");
    drop(guard);
    assert!(failure.to_string().contains("injected failure"));
    assert_eq!(
        fs::read(fixture.0.join(PROJECT_MANIFEST)).unwrap(),
        manifest
    );
    assert_eq!(fs::read(fixture.0.join(PROJECT_LOCK)).unwrap(), lock);
    assert!(fs::read_dir(&fixture.0).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".eqiora-project-file-")
    }));
    let store = DirectoryPackageStore::open_ambient(&store_path).expect("accepted store");
    PackagedModelDocument::compile_locked(&store, &accepted, "Main")
        .expect("previous accepted project remains usable");
}

#[test]
fn proposed_dependency_changes_are_validated_without_publishing() {
    let fixture = TestDirectory::create("proposed-dependency");
    let store_path = fixture.child("store");
    let library_path = fixture.child("library");
    let root = author_sources(
        "org.example.Root",
        "model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
        vec![],
    );
    let library = author_sources(
        "org.example.Library",
        "public component Shared() {}",
        vec![],
    );
    write_package(&fixture.0, "src", &root, &[]);
    write_package(&library_path, "src", &library, &[]);
    let accepted =
        resolve_local_package_project_v1(&fixture.0, &store_path).expect("accepted project");
    let manifest_bytes = fs::read(fixture.0.join(PROJECT_MANIFEST)).unwrap();
    let lock_bytes = fs::read(fixture.0.join(PROJECT_LOCK)).unwrap();
    let directory = open_project_root(&fixture.0).unwrap();

    for version in ["1.0.0", "2.0.0"] {
        let mut manifest = read_project_manifest(&directory).unwrap();
        manifest.dependencies.insert(
            "org.example.Library".to_owned(),
            LocalProjectDependency {
                version: version.to_owned(),
                sources: vec![LocalDependencySource {
                    path: Some("library".to_owned()),
                    bundled: false,
                    git: None,
                }],
            },
        );
        let candidate = prepare_local_package_project(
            directory.try_clone().unwrap(),
            &fixture.0,
            LocalProjectOverrides {
                manifest: Some(manifest),
                sources: BTreeMap::new(),
                ..Default::default()
            },
        );
        if version == "1.0.0" {
            let candidate = candidate.expect("validate proposed dependency");
            assert_eq!(candidate.root.release.manifest().dependencies().len(), 1);
            assert_eq!(candidate.root.dependencies.len(), 1);
        } else {
            let error = candidate.err().expect("reject foreign dependency version");
            assert!(error.to_string().contains("no single release satisfies"));
        }
        assert_eq!(
            fs::read(fixture.0.join(PROJECT_MANIFEST)).unwrap(),
            manifest_bytes
        );
        assert_eq!(fs::read(fixture.0.join(PROJECT_LOCK)).unwrap(), lock_bytes);
    }
    let store = DirectoryPackageStore::open_ambient(&store_path).unwrap();
    PackagedModelDocument::compile_locked(&store, &accepted, "Main")
        .expect("accepted closure remains usable after candidate validation");

    assert!(
        PackagedModelDocument::add_local_package_dependency_v1(
            &fixture.0,
            &store_path,
            "org.example.Library",
            "2.0.0",
            "library",
        )
        .is_err()
    );
    assert_eq!(
        fs::read(fixture.0.join(PROJECT_MANIFEST)).unwrap(),
        manifest_bytes
    );
    assert_eq!(fs::read(fixture.0.join(PROJECT_LOCK)).unwrap(), lock_bytes);

    let added = PackagedModelDocument::add_local_package_dependency_v1(
        &fixture.0,
        &store_path,
        "org.example.Library",
        "1.0.0",
        "library",
    )
    .expect("publish dependency addition");
    assert_eq!(
        read_project_manifest(&directory)
            .unwrap()
            .dependencies
            .len(),
        1
    );
    assert_eq!(read_project_lock(&directory).unwrap().resolution, added);
    PackagedModelDocument::compile_locked(&store, &added, "Main").unwrap();

    let removed = PackagedModelDocument::remove_local_package_dependency_v1(
        &fixture.0,
        &store_path,
        "org.example.Library",
    )
    .expect("publish dependency removal");
    assert!(
        read_project_manifest(&directory)
            .unwrap()
            .dependencies
            .is_empty()
    );
    assert_eq!(read_project_lock(&directory).unwrap().resolution, removed);
    assert_eq!(removed, accepted);
    PackagedModelDocument::compile_locked(&store, &removed, "Main").unwrap();
}

#[test]
fn dependency_depth_is_bounded_before_reading_another_manifest() {
    let fixture = TestDirectory::create("depth");
    let directory = open_project_root(&fixture.0).expect("project root");
    let result = inventory::load(
        &directory,
        &fixture.0,
        PathBuf::new(),
        MAX_LOCAL_DEPENDENCY_DEPTH + 1,
        &mut LocalProjectOverrides::default(),
        &mut inventory::Frozen::default(),
    );
    assert!(
        result
            .expect_err("bounded depth")
            .to_string()
            .contains("depth bounds")
    );
}

#[test]
fn dependency_cannot_replace_an_ancestor_from_another_directory() {
    let fixture = TestDirectory::create("duplicate-ancestor");
    let child = fixture.child("child");
    let duplicate = fixture.child("duplicate");
    let store = fixture.child("store");
    let root_sources = author_sources("org.example.Root", "model Main() {}", vec![]);
    let child_sources = author_sources("org.example.Child", "model Main() {}", vec![]);
    let root_release = prepare_package_release_v1(root_sources.clone(), &[]).unwrap();
    let child_release = prepare_package_release_v1(child_sources.clone(), &[]).unwrap();
    write_package(
        &fixture.0,
        "src",
        &root_sources,
        &[(&child_release, "child")],
    );
    write_package(
        &child,
        "src",
        &child_sources,
        &[(&root_release, "../duplicate")],
    );
    write_package(&duplicate, "src", &root_sources, &[]);
    let error = resolve_local_package_project_v1(&fixture.0, &store)
        .expect_err("duplicate ancestor must fail before preparation");
    assert!(error.to_string().contains("conflicting content"));
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}

#[cfg(unix)]
#[test]
fn project_preparation_and_lock_publication_retain_the_opened_directory() {
    let fixture = TestDirectory::create("retained-project");
    let original = fixture.child("project");
    let moved = fixture.0.join("moved");
    let root = author_sources("org.example.Original", "model Main() {}", vec![]);
    write_package(&original, "src", &root, &[]);
    let directory = open_project_root(&original).unwrap();
    fs::rename(&original, &moved).unwrap();
    fs::create_dir(&original).unwrap();
    fs::write(original.join(PROJECT_MANIFEST), "replacement manifest").unwrap();
    let candidate =
        prepare_local_package_project(directory, &original, LocalProjectOverrides::default())
            .expect("prepare through retained root");
    let resolution = ResolutionRecordV1::from_exact_releases(&candidate.root.release, &[]).unwrap();
    assert_eq!(resolution.root().name.as_str(), "org.example.Original");
    let _guard = transaction::write_guard(&candidate.project).unwrap();
    let manifest = transaction::read(
        &candidate.project,
        PROJECT_MANIFEST,
        MAX_PROJECT_MANIFEST_BYTES,
    )
    .unwrap();
    transaction::commit(
        &candidate.project,
        &manifest,
        &lock::ProjectLock::new(resolution.clone(), vec![], vec![])
            .unwrap()
            .bytes()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        fs::read(moved.join(PROJECT_LOCK)).unwrap(),
        lock::ProjectLock::new(resolution, vec![], vec![])
            .unwrap()
            .bytes()
            .unwrap()
    );
    assert!(!original.join(PROJECT_LOCK).exists());
    assert_eq!(
        fs::read(original.join(PROJECT_MANIFEST)).unwrap(),
        b"replacement manifest"
    );
}

#[test]
fn manifest_entry_selects_a_nested_module_during_offline_replay() {
    let fixture = TestDirectory::create("nested-entry");
    let store_path = fixture.child("store");
    fs::create_dir_all(fixture.0.join("sources/models")).expect("source tree");
    fs::write(
        fixture.0.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Entry\"\nversion = \"1.0.0\"\nsource = \"sources\"\nentry = \"models.selected\"\n",
    ).expect("manifest");
    fs::write(
        fixture.0.join("sources/models/selected.eqi"),
        "model Main() { parameter selected: 1 = 2; relation law { selected - 2 = 0; } }",
    )
    .expect("selected module");
    fs::write(
        fixture.0.join("sources/main.eqi"),
        "model Main() { parameter decoy: 1 = 3; relation law { decoy - 3 = 0; } }",
    )
    .expect("decoy module");
    let resolution =
        resolve_local_package_project_v1(&fixture.0, &store_path).expect("resolve explicit entry");
    fs::remove_dir_all(fixture.0.join("sources")).expect("remove authored sources");
    let store = DirectoryPackageStore::open_ambient(&store_path).expect("offline store");
    let document = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile the locked entry");
    assert!(document.model().aliases().contains_key("selected"));
    assert!(!document.model().aliases().contains_key("decoy"));
}

#[cfg(unix)]
#[test]
fn source_root_rejects_an_intermediate_symlink() {
    let fixture = TestDirectory::create("source-link");
    let store = fixture.child("store");
    let sources = author_sources("org.example.Root", "model Main() {}", vec![]);
    write_package(&fixture.0, "actual/src", &sources, &[]);
    std::os::unix::fs::symlink("actual", fixture.0.join("linked"))
        .expect("create intermediate symlink");
    fs::write(
        fixture.0.join(PROJECT_MANIFEST),
        "[package]\nname = \"org.example.Root\"\nversion = \"1.0.0\"\nsource = \"linked/src\"\nentry = \"main\"\n",
    )
    .expect("write manifest");
    let error = resolve_local_package_project_v1(&fixture.0, &store)
        .expect_err("intermediate symlink must fail");
    assert!(error.to_string().contains("cannot open source root"));
    assert!(!fixture.0.join(PROJECT_LOCK).exists());
}
