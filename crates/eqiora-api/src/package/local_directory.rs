//! Authored project requests, deterministic selection and exact store preparation.

mod git;
mod inventory;
mod lock;
mod offline;
mod selection;
mod transaction;
mod update;

pub use update::ProjectUpdate;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
#[cfg(test)]
use std::io::Write;
use std::path::{Path, PathBuf};

use cap_fs_ext::{
    DirExt, FollowSymlinks, OpenOptionsFollowExt, OpenOptionsMaybeDirExt, OpenOptionsSyncExt,
};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, File, OpenOptions};
use eqiora_package::{
    BundleEntryV1, BundleRoleV1, DirectoryPackageInstaller, ExactResolver, ExactVersion,
    ModelPackageIdentityV1, NormalizedRelativePath, PackageDependencyV1, PackageDirectory,
    PackageManifestV1, PackageReleaseV1, PackageSourcesV1, QualifiedName, ResolutionRecordV1,
    SourceFileV1, VersionRequest,
};
use serde::{Deserialize, Serialize};

use super::{PackagePreparationError, PackagedModelDocument, prepare_package_release_v1};

const MAX_LOCAL_PACKAGE_DIRECTORIES_V1: usize = 65_536;
const MAX_LOCAL_DEPENDENCY_DEPTH: usize = 64;
const MAX_PROJECT_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_PACKAGE_README_BYTES: usize = 1024 * 1024;
const MAX_PROJECT_LOCK_BYTES: usize = 1024 * 1024 * 1024;
const PROJECT_MANIFEST: &str = "eqiora.toml";
const PROJECT_LOCK: &str = "eqiora.lock";

type PackageKey = (QualifiedName, ExactVersion);

#[derive(Clone)]
struct LocalPackageSource {
    path: PathBuf,
    relative_path: PathBuf,
    source_root: NormalizedRelativePath,
    name: QualifiedName,
    version: ExactVersion,
    entry: String,
    files: Vec<SourceFileV1>,
    dependencies: BTreeMap<QualifiedName, PackageKey>,
}

#[derive(Clone)]
struct PreparedLocalPackage {
    release: PackageReleaseV1,
    dependencies: BTreeMap<ModelPackageIdentityV1, PackageReleaseV1>,
}

struct PreparedLocalProject {
    project: Dir,
    root: PreparedLocalPackage,
    packages: BTreeMap<PackageKey, LocalPackageSource>,
    prepared: BTreeMap<PackageKey, PreparedLocalPackage>,
    git: Vec<lock::GitPin>,
    requests: Vec<lock::RequestPin>,
}

#[derive(Default)]
struct LocalProjectOverrides {
    manifest: Option<LocalProjectManifest>,
    sources: BTreeMap<PathBuf, String>,
    offline: bool,
    prepared: BTreeMap<PackageKey, PreparedLocalPackage>,
    allow_git: bool,
    confined: bool,
    locked_git: Option<Vec<lock::GitPin>>,
    locked_requests: Option<Vec<lock::RequestPin>>,
    locked_versions: Option<BTreeMap<QualifiedName, ExactVersion>>,
    git_stack: BTreeSet<String>,
    git_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalProjectManifest {
    package: LocalPackageManifest,
    #[serde(default)]
    dependencies: BTreeMap<String, LocalProjectDependency>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalPackageManifest {
    name: String,
    version: String,
    #[serde(default = "default_source_root")]
    source: String,
    entry: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalProjectDependency {
    version: String,
    sources: Vec<LocalDependencySource>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalDependencySource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    bundled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    git: Option<git::GitSource>,
}

fn default_source_root() -> String {
    "src".to_owned()
}

impl PackagedModelDocument {
    /// Maximum root-plus-dependency count admitted by local exact resolution.
    pub const MAX_LOCAL_PROJECT_PACKAGES_V1: usize = MAX_LOCAL_PACKAGE_DIRECTORIES_V1;

    /// Add or replace an authored local request and publish its validated exact selection.
    ///
    /// # Errors
    /// Returns an error if the request, dependency closure, installation, or publication fails.
    pub fn add_local_package_dependency_v1(
        project_root: impl Into<PathBuf>,
        store_root: impl Into<PathBuf>,
        name: &str,
        version: &str,
        path: &str,
    ) -> Result<ResolutionRecordV1, PackagePreparationError> {
        update_local_package_project(project_root.into(), store_root.into(), |manifest| {
            manifest.dependencies.insert(
                name.to_owned(),
                LocalProjectDependency {
                    version: version.to_owned(),
                    sources: vec![LocalDependencySource {
                        path: Some(path.to_owned()),
                        bundled: false,
                        git: None,
                    }],
                },
            );
            Ok(true)
        })
    }

    /// Remove a direct dependency and publish the validated manifest/lock pair.
    ///
    /// # Errors
    /// Returns an error for an absent dependency or an invalid remaining project, or if
    /// installation or publication fails. Remaining source imports must still resolve.
    pub fn remove_local_package_dependency_v1(
        project_root: impl Into<PathBuf>,
        store_root: impl Into<PathBuf>,
        name: &str,
    ) -> Result<ResolutionRecordV1, PackagePreparationError> {
        update_local_package_project(project_root.into(), store_root.into(), |manifest| {
            if manifest.dependencies.remove(name).is_none() {
                return Err(PackagePreparationError::LocalDirectoryGraph(format!(
                    "direct dependency `{name}` does not exist"
                )));
            }
            Ok(true)
        })
    }

    /// Resolve one local package project, write its exact lock, and populate an offline store.
    ///
    /// `eqiora.toml` owns the root package name, exact version, source root, entry module, and
    /// direct local dependencies. Dependency manifests use the same format. Resolution discovers
    /// bounded `.eqi` inventories, generates closed package manifests, validates the exact graph,
    /// and atomically replaces `eqiora.lock` with the current project provenance envelope.
    ///
    /// # Errors
    ///
    /// Returns a project, directory, graph, compiler-preparation, exact-identity, installation,
    /// or lock-publication failure. The previous lock remains usable on failure.
    pub fn resolve_local_package_project_v1<R, P>(
        project_root: R,
        store_root: P,
    ) -> Result<ResolutionRecordV1, PackagePreparationError>
    where
        R: Into<PathBuf>,
        P: Into<PathBuf>,
    {
        resolve_local_package_project_v1(project_root, store_root)
    }
}

fn resolve_local_package_project_v1<R, P>(
    project_root: R,
    store_root: P,
) -> Result<ResolutionRecordV1, PackagePreparationError>
where
    R: Into<PathBuf>,
    P: Into<PathBuf>,
{
    update_local_package_project(project_root.into(), store_root.into(), |_| Ok(false))
}

fn update_local_package_project(
    project_path: PathBuf,
    store_root: PathBuf,
    edit: impl FnOnce(&mut LocalProjectManifest) -> Result<bool, PackagePreparationError>,
) -> Result<ResolutionRecordV1, PackagePreparationError> {
    update::preview(project_path, edit)?.commit(store_root)
}

pub(crate) fn analyze_local_package_editor_project_v1(
    version: u64,
    project_root: impl Into<PathBuf>,
    overrides: &BTreeMap<PathBuf, String>,
) -> Result<
    (
        crate::editor::EditorWorkspaceSnapshot,
        BTreeMap<String, PathBuf>,
    ),
    PackagePreparationError,
> {
    let project_path = project_root.into();
    let project = open_project_root(&project_path)?;
    let _guard = transaction::read_guard(&project)
        .map_err(|error| PackagePreparationError::LocalDirectoryGraph(error.to_string()))?;
    transaction::require_complete(&project)
        .map_err(|error| PackagePreparationError::LocalDirectoryGraph(error.to_string()))?;
    let accepted = match project.symlink_metadata(PROJECT_LOCK) {
        Ok(_) => Some(read_project_lock(&project)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(git::error(&error.to_string())),
    };
    let settings = LocalProjectOverrides {
        manifest: None,
        sources: overrides.clone(),
        locked_versions: accepted.as_ref().map(lock::ProjectLock::versions),
        locked_requests: accepted.as_ref().map(|lock| lock.requests.clone()),
        ..Default::default()
    };
    let (prepared, recovery) = match prepare_local_package_project(project, &project_path, settings)
    {
        Ok(prepared) => (prepared, None),
        Err(error) if !overrides.is_empty() => {
            // Recover names from the admitted disk graph only. Unsaved invalid
            // bytes are not releases and must never authorize compilation.
            let prepared = prepare_local_package_project(
                open_project_root(&project_path)?,
                &project_path,
                LocalProjectOverrides {
                    locked_versions: accepted.as_ref().map(lock::ProjectLock::versions),
                    locked_requests: accepted.map(|lock| lock.requests),
                    ..Default::default()
                },
            )?;
            (
                prepared,
                Some(eqiora_core::Diagnostic::error(
                    eqiora_core::diagnostic::codes::LANGUAGE_LOWERING_ERROR,
                    format!("unsaved package source is incomplete or invalid: {error}"),
                )),
            )
        }
        Err(error) => return Err(error),
    };
    let dependencies = prepared
        .root
        .dependencies
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let resolution =
        ResolutionRecordV1::from_exact_releases(&prepared.root.release, &dependencies)?;
    let resolved = ExactResolver
        .resolve_releases(&resolution, &prepared.root.release, &dependencies)
        .map_err(PackagePreparationError::Resolution)?;
    let namespaces =
        super::compilation_namespaces(&resolved).map_err(super::map_compilation_preparation)?;
    let input = super::compiler_input(&resolved, &namespaces)
        .map_err(super::map_compilation_preparation)?;
    let mut relative_paths = BTreeMap::new();
    for (key, package) in &prepared.packages {
        let identity = prepared
            .prepared
            .get(key)
            .expect("every local package was prepared")
            .release
            .package_identity()?;
        let namespace = namespaces
            .get(&identity)
            .expect("every resolved local package has a compilation namespace");
        for file in package
            .files
            .iter()
            .filter(|file| file.role() == BundleRoleV1::ModelSource)
        {
            let source = std::str::from_utf8(file.bytes()).map_err(|error| {
                PackagePreparationError::LocalDirectoryGraph(format!(
                    "model source `{}` is not UTF-8: {error}",
                    file.path()
                ))
            })?;
            let unit = eqiora_compiler::ResolvedSourceUnit::new(
                namespace.clone(),
                file.path().as_str(),
                source,
            )?;
            relative_paths.insert(
                unit.diagnostic_file(),
                package
                    .relative_path
                    .join(package.source_root.as_str())
                    .join(
                        file.path()
                            .as_str()
                            .strip_prefix("src/")
                            .expect("generated model source path"),
                    ),
            );
        }
    }
    let mut snapshot = crate::editor::EditorWorkspaceSnapshot::analyze_modules(version, input);
    if let Some(diagnostic) = recovery {
        let sources = relative_paths
            .iter()
            .filter_map(|(file, path)| {
                overrides
                    .get(path)
                    .map(|source| (file.clone(), source.clone()))
            })
            .collect();
        snapshot = snapshot.recover_overrides(&sources, diagnostic);
    }
    Ok((snapshot, relative_paths))
}

fn prepare_local_package_project(
    project: Dir,
    project_path: &Path,
    mut overrides: LocalProjectOverrides,
) -> Result<PreparedLocalProject, PackagePreparationError> {
    let mut frozen = inventory::Frozen::default();
    let retained = overrides
        .prepared
        .values()
        .map(|package| package.release.clone())
        .collect::<Vec<_>>();
    frozen.retain_releases(&retained, overrides.locked_requests.as_deref())?;
    let root_key = inventory::load(
        &project,
        project_path,
        PathBuf::new(),
        0,
        &mut overrides,
        &mut frozen,
    )?;
    let selected = selection::solve(
        &root_key,
        &frozen.requests,
        overrides.locked_versions.as_ref(),
    )?;
    frozen
        .packages
        .retain(|key, _| selected.get(&key.0) == Some(key));
    frozen
        .prepared
        .retain(|key, _| selected.get(&key.0) == Some(key));
    let mut requests = Vec::new();
    for key in selected.values() {
        let authored = frozen
            .requests
            .get(key)
            .expect("selected candidate requests");
        let mut dependencies = BTreeMap::new();
        for (name, request) in authored {
            let target = selected.get(name).expect("complete selection").clone();
            requests.push(lock::RequestPin {
                declaring: key.0.clone(),
                declaring_version: key.1.clone(),
                dependency: name.clone(),
                request: request.clone(),
                selected: target.1.clone(),
            });
            dependencies.insert(name.clone(), target);
        }
        if let Some(package) = frozen.packages.get_mut(key) {
            package.dependencies = dependencies;
        }
    }
    if let Some(expected) = &overrides.locked_requests {
        lock::require_requests(expected, &requests)?;
    }
    frozen.git.retain(|pin| {
        selected.get(&pin.declaring)
            == Some(&(pin.declaring.clone(), pin.declaring_version.clone()))
            && selected.get(&pin.dependency) == Some(&(pin.dependency.clone(), pin.version.clone()))
    });
    let packages = frozen.packages;
    let mut visiting = BTreeSet::new();
    let mut prepared = frozen.prepared;
    let root_package = prepare_local_package(&root_key, &packages, &mut visiting, &mut prepared)?;
    if let Some(path) = overrides.sources.keys().next() {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "editor override `{}` is not a declared model source",
            path.display()
        )));
    }
    Ok(PreparedLocalProject {
        project,
        root: root_package,
        packages,
        prepared,
        git: frozen.git,
        requests,
    })
}

fn open_relative_directory(root: &Dir, path: &NormalizedRelativePath) -> std::io::Result<Dir> {
    let mut directory = root.try_clone()?;
    for segment in path.as_str().split('/') {
        directory = directory.open_dir_nofollow(segment)?;
    }
    Ok(directory)
}

fn open_dependency_directory(root: &Dir, path: &str) -> std::io::Result<Dir> {
    let mut directory = root.try_clone()?;
    for segment in path.split('/') {
        directory = if segment == ".." {
            // Only explicitly declared dependency paths grant parent traversal.
            // Sources and artifacts remain confined to the resulting directory.
            directory.open_parent_dir(ambient_authority())?
        } else {
            directory.open_dir_nofollow(segment)?
        };
    }
    Ok(directory)
}

fn resolve_dependency_path(
    declaring: &Path,
    value: &str,
) -> Result<PathBuf, PackagePreparationError> {
    if value.is_empty() || value.len() > 4096 || value.contains('\\') {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "invalid local dependency path `{value}`"
        )));
    }
    let mut resolved = declaring.to_path_buf();
    let mut descended = false;
    for segment in value.split('/') {
        match segment {
            ".." => {
                if descended {
                    return Err(PackagePreparationError::LocalDirectoryGraph(format!(
                        "local dependency path `{value}` has a parent segment after a named directory"
                    )));
                }
                if resolved.as_os_str().is_empty() || resolved.ends_with("..") {
                    resolved.push("..");
                } else {
                    resolved.pop();
                }
            }
            "" | "." => {
                return Err(PackagePreparationError::LocalDirectoryGraph(format!(
                    "invalid local dependency path `{value}`"
                )));
            }
            segment => {
                NormalizedRelativePath::parse(segment)?;
                descended = true;
                resolved.push(segment);
            }
        }
    }
    if resolved.components().count() > 64 || resolved.as_os_str().len() > 4096 {
        return Err(PackagePreparationError::LocalDirectoryGraph(
            "local dependency path exceeds its depth or length limit".to_owned(),
        ));
    }
    Ok(resolved)
}

fn open_project_root(path: &Path) -> Result<Dir, PackagePreparationError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .maybe_dir(true)
        .follow(FollowSymlinks::No)
        .nonblock(true);
    let root = File::open_ambient_with(path, &options, ambient_authority()).map_err(|error| {
        PackagePreparationError::LocalDirectoryGraph(format!(
            "cannot open local package project {}: {error}",
            path.display()
        ))
    })?;
    if !root
        .metadata()
        .map_err(|error| {
            PackagePreparationError::LocalDirectoryGraph(format!(
                "cannot inspect local package project {}: {error}",
                path.display()
            ))
        })?
        .is_dir()
    {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "local package project {} is not a directory",
            path.display()
        )));
    }
    Ok(Dir::from_std_file(root.into_std()))
}

fn read_project_manifest(project: &Dir) -> Result<LocalProjectManifest, PackagePreparationError> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No).nonblock(true);
    let file = project
        .open_with(PROJECT_MANIFEST, &options)
        .map_err(|error| {
            PackagePreparationError::LocalDirectoryGraph(format!(
                "cannot open {PROJECT_MANIFEST}: {error}"
            ))
        })?;
    let metadata = file.metadata().map_err(|error| {
        PackagePreparationError::LocalDirectoryGraph(format!(
            "cannot inspect {PROJECT_MANIFEST}: {error}"
        ))
    })?;
    if !metadata.is_file() {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "{PROJECT_MANIFEST} is not a regular file"
        )));
    }
    if metadata.len() > MAX_PROJECT_MANIFEST_BYTES as u64 {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "{PROJECT_MANIFEST} exceeds the {MAX_PROJECT_MANIFEST_BYTES} byte limit"
        )));
    }
    let mut bytes = Vec::new();
    file.take((MAX_PROJECT_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            PackagePreparationError::LocalDirectoryGraph(format!(
                "cannot read {PROJECT_MANIFEST}: {error}"
            ))
        })?;
    if bytes.len() > MAX_PROJECT_MANIFEST_BYTES {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "{PROJECT_MANIFEST} exceeds the {MAX_PROJECT_MANIFEST_BYTES} byte limit"
        )));
    }
    decode_project_manifest(&bytes)
}

fn decode_project_manifest(bytes: &[u8]) -> Result<LocalProjectManifest, PackagePreparationError> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        PackagePreparationError::LocalDirectoryGraph(format!(
            "{PROJECT_MANIFEST} is not UTF-8: {error}"
        ))
    })?;
    toml::from_str(text).map_err(|_| {
        PackagePreparationError::LocalDirectoryGraph(format!("cannot decode {PROJECT_MANIFEST}"))
    })
}

fn read_project_lock(project: &Dir) -> Result<lock::ProjectLock, PackagePreparationError> {
    let _guard = transaction::read_guard(project)
        .map_err(|error| PackagePreparationError::LocalDirectoryGraph(error.to_string()))?;
    let bytes = transaction::accepted_lock(project)
        .map_err(|error| PackagePreparationError::LocalDirectoryGraph(error.to_string()))?;
    lock::ProjectLock::decode(&bytes)
}

fn prepare_local_package(
    key: &PackageKey,
    packages: &BTreeMap<PackageKey, LocalPackageSource>,
    visiting: &mut BTreeSet<PackageKey>,
    prepared: &mut BTreeMap<PackageKey, PreparedLocalPackage>,
) -> Result<PreparedLocalPackage, PackagePreparationError> {
    if let Some(package) = prepared.get(key) {
        return Ok(package.clone());
    }
    let package = packages
        .get(key)
        .expect("only indexed root and dependency keys are prepared");
    if !visiting.insert(key.clone()) {
        return Err(PackagePreparationError::LocalDirectoryGraph(format!(
            "local package dependency cycle reaches `{}`",
            key.0
        )));
    }

    let mut dependencies = BTreeMap::new();
    let mut requirements = Vec::with_capacity(package.dependencies.len());
    for target_key in package.dependencies.values() {
        if !packages.contains_key(target_key) && !prepared.contains_key(target_key) {
            return Err(PackagePreparationError::LocalDirectoryGraph(format!(
                "local package `{}` is missing dependency `{}@{}`",
                key.0, target_key.0, target_key.1
            )));
        }
        let child = prepare_local_package(target_key, packages, visiting, prepared)?;
        let actual = child.release.package_identity()?;
        dependencies.extend(child.dependencies.clone());
        dependencies.insert(actual.clone(), child.release);
        requirements.push(PackageDependencyV1::new(actual));
    }

    let bundle = package
        .files
        .iter()
        .map(|file| BundleEntryV1::new(file.path().clone(), file.role()))
        .collect();
    let manifest = PackageManifestV1::new(
        &package.entry,
        package.name.clone(),
        package.version.clone(),
        requirements,
        bundle,
    )?;
    let sources = PackageSourcesV1::new(manifest, package.files.clone())?;
    let release =
        prepare_package_release_v1(sources, &dependencies.values().cloned().collect::<Vec<_>>())
            .map_err(|source| PackagePreparationError::DirectoryPreparation {
                path: package.path.clone(),
                source: Box::new(source),
            })?;
    visiting.remove(key);
    let package = PreparedLocalPackage {
        release,
        dependencies,
    };
    prepared.insert(key.clone(), package.clone());
    Ok(package)
}

#[cfg(test)]
mod tests;
