use super::*;

fn fixture(name: &str) -> (TestDirectory, PathBuf, PathBuf, PathBuf) {
    let fixture = TestDirectory::create(name);
    let project = fixture.child("project");
    let external = fixture.child("external");
    let store = fixture.child("store");
    let sources = author_sources("org.example.External", "public model Shared() {}", vec![]);
    let release = prepare_package_release_v1(sources.clone(), &[]).unwrap();
    write_package(&external, "src", &sources, &[]);
    let root = author_sources(
        "org.example.Offline",
        "import org.example.External.main as external; model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
        vec![exact_dependency(&release)],
    );
    write_package(&project, "src", &root, &[(&release, "../external")]);
    (fixture, project, external, store)
}

#[test]
fn bundled_and_external_closure_moves_and_reopens_without_sources() {
    let (fixture, project, external, store) = fixture("portable");
    let lock = PackagedModelDocument::add_bundled_package_dependency_v1(
        &project,
        &store,
        "Eqiora.Fluid.Incompressible",
        "0.7.0",
    )
    .unwrap();
    let vendor = fixture.child("vendor");
    assert_eq!(
        PackagedModelDocument::vendor_local_package_project_v1(&project, &store, &vendor).unwrap(),
        lock
    );
    let original = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&store).unwrap(),
        &lock,
        "Main",
    )
    .unwrap();
    fs::remove_dir_all(external).unwrap();
    fs::remove_dir_all(store).unwrap();
    let moved = fixture.0.join("moved");
    fs::rename(project, &moved).unwrap();
    assert_eq!(
        PackagedModelDocument::open_local_package_project_v1(&moved, &vendor).unwrap(),
        lock
    );
    let replay = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&vendor).unwrap(),
        &lock,
        "Main",
    )
    .unwrap();
    assert_eq!(
        original.model().structural_fingerprint(),
        replay.model().structural_fingerprint()
    );
    fs::write(moved.join("src/main.eqi"), "model Changed() {}").unwrap();
    assert!(
        PackagedModelDocument::open_local_package_project_v1(&moved, &vendor)
            .unwrap_err()
            .to_string()
            .contains("accepted eqiora.lock")
    );
}

#[test]
fn fetch_preserves_lock_and_transport_does_not_change_identity() {
    let (fixture, project, external, store) = fixture("fetch");
    let lock = PackagedModelDocument::resolve_local_package_project_v1(&project, &store).unwrap();
    let manifest = fs::read_to_string(project.join(PROJECT_MANIFEST)).unwrap();
    let bytes = fs::read(project.join(PROJECT_LOCK)).unwrap();
    fs::rename(&external, fixture.0.join("relocated")).unwrap();
    fs::write(
        project.join(PROJECT_MANIFEST),
        manifest.replace("../external", "../relocated"),
    )
    .unwrap();
    let second = fixture.child("second");
    assert_eq!(
        PackagedModelDocument::fetch_local_package_project_v1(&project, &second).unwrap(),
        lock
    );
    assert_eq!(fs::read(project.join(PROJECT_LOCK)).unwrap(), bytes);
    let accepted_manifest = fs::read(project.join(PROJECT_MANIFEST)).unwrap();
    fs::write(
        fixture.0.join("relocated/src/main.eqi"),
        "public model Changed() {}",
    )
    .unwrap();
    assert!(PackagedModelDocument::fetch_local_package_project_v1(&project, &second).is_err());
    assert_eq!(fs::read(project.join(PROJECT_LOCK)).unwrap(), bytes);
    assert_eq!(
        fs::read(project.join(PROJECT_MANIFEST)).unwrap(),
        accepted_manifest
    );
    assert_ne!(
        PackagedModelDocument::resolve_local_package_project_v1(&project, &second).unwrap(),
        lock
    );
}

#[test]
fn unavailable_standard_and_failed_publication_preserve_accepted_pair() {
    let (fixture, project, _, store) = fixture("atomic");
    PackagedModelDocument::resolve_local_package_project_v1(&project, &store).unwrap();
    let manifest = fs::read(project.join(PROJECT_MANIFEST)).unwrap();
    let lock = fs::read(project.join(PROJECT_LOCK)).unwrap();
    assert!(
        PackagedModelDocument::add_bundled_package_dependency_v1(
            &project,
            &store,
            "Eqiora.Fluid.Incompressible",
            "99.0.0"
        )
        .is_err()
    );
    let file = fixture.0.join("not-a-store");
    fs::write(&file, "preserved").unwrap();
    assert!(
        PackagedModelDocument::add_bundled_package_dependency_v1(
            &project,
            &file,
            "Eqiora.Fluid.Incompressible",
            "0.7.0"
        )
        .is_err()
    );
    assert_eq!(fs::read_to_string(file).unwrap(), "preserved");
    assert_eq!(fs::read(project.join(PROJECT_MANIFEST)).unwrap(), manifest);
    assert_eq!(fs::read(project.join(PROJECT_LOCK)).unwrap(), lock);
}

#[test]
fn missing_or_modified_vendor_entries_fail_without_overwrite() {
    let (fixture, project, _, store) = fixture("damaged-vendor");
    let lock = PackagedModelDocument::resolve_local_package_project_v1(&project, &store).unwrap();
    let vendor = fixture.child("vendor");
    PackagedModelDocument::vendor_local_package_project_v1(&project, &store, &vendor).unwrap();
    let entry = fs::read_dir(&vendor)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .unwrap();
    let accepted = fs::read(&entry).unwrap();
    fs::remove_file(&entry).unwrap();
    assert!(PackagedModelDocument::open_local_package_project_v1(&project, &vendor).is_err());
    fs::write(&entry, b"modified").unwrap();
    assert!(PackagedModelDocument::open_local_package_project_v1(&project, &vendor).is_err());
    assert!(
        PackagedModelDocument::vendor_local_package_project_v1(&project, &store, &vendor).is_err()
    );
    assert_eq!(fs::read(&entry).unwrap(), b"modified");
    assert_eq!(
        read_project_lock(&open_project_root(&project).unwrap())
            .unwrap()
            .resolution,
        lock
    );
    fs::write(entry, accepted).unwrap();
    assert_eq!(
        PackagedModelDocument::open_local_package_project_v1(&project, &vendor).unwrap(),
        lock
    );
}

#[test]
fn conflicting_local_standard_release_and_invalid_transport_are_rejected() {
    let (fixture, project, _, store) = fixture("standard-conflict");
    let conflicting = fixture.child("mechanics");
    let sources = author_sources(
        "Eqiora.Mechanics.Interfaces",
        "public model Other() {}",
        vec![],
    );
    write_package(&conflicting, "src", &sources, &[]);
    let manifest = fs::read_to_string(conflicting.join(PROJECT_MANIFEST))
        .unwrap()
        .replace("1.0.0", "0.3.0");
    fs::write(conflicting.join(PROJECT_MANIFEST), manifest).unwrap();
    let original = fs::read_to_string(project.join(PROJECT_MANIFEST)).unwrap();
    fs::write(project.join(PROJECT_MANIFEST), format!("{original}\n[dependencies.\"Eqiora.Mechanics.Interfaces\"]\nversion = \"0.3.0\"\nsources = [{{ path = \"../mechanics\" }}]\n")).unwrap();
    PackagedModelDocument::resolve_local_package_project_v1(&project, &store).unwrap();
    let manifest = fs::read(project.join(PROJECT_MANIFEST)).unwrap();
    let lock = fs::read(project.join(PROJECT_LOCK)).unwrap();
    let error = PackagedModelDocument::add_bundled_package_dependency_v1(
        &project,
        &store,
        "Eqiora.Fluid.Incompressible",
        "0.7.0",
    )
    .unwrap_err();
    assert!(error.to_string().contains("conflicting content"), "{error}");
    assert_eq!(fs::read(project.join(PROJECT_MANIFEST)).unwrap(), manifest);
    assert_eq!(fs::read(project.join(PROJECT_LOCK)).unwrap(), lock);
    fs::write(
        project.join(PROJECT_MANIFEST),
        String::from_utf8(manifest).unwrap().replace(
            "path = \"../mechanics\"",
            "path = \"../mechanics\", bundled = true",
        ),
    )
    .unwrap();
    assert!(
        PackagedModelDocument::open_local_package_project_v1(&project, &store)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );
}
