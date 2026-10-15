//! Build-time projection of exact dependency releases from Cargo manifests.
//!
//! This crate is a build dependency only. Published Cargo manifests carry their
//! normalized pins; workspace builds resolve the same pins from their workspace.
//! No version registry or runtime dependency inspection is needed.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

type BuildResult<T> = Result<T, Box<dyn Error>>;

/// Emit `EQIORA_DEPENDENCY_VERSION_<NAME>` for one exact-pinned normal dependency.
///
/// Adapters consume this value with `env!` when declaring provider provenance.
/// `<NAME>` is the manifest dependency key, uppercased with hyphens replaced by
/// underscores. Both the member manifest and any inherited workspace manifest
/// are registered as build inputs.
///
/// This proves the selected package version, not its source or binary identity.
/// Optional dependencies retain their declared release even when disabled.
///
/// # Errors
/// Rejects absent dependencies, malformed manifests, missing workspace owners,
/// and version ranges or build metadata that cannot identify one exact release.
pub fn emit_dependency_version(dependency: &str) -> BuildResult<()> {
    if dependency.is_empty()
        || !dependency
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(
            "dependency key must contain only ASCII letters, digits, underscores or hyphens".into(),
        );
    }
    let directory = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").ok_or("Cargo did not provide CARGO_MANIFEST_DIR")?,
    );
    let version = dependency_version(&directory, dependency)?;
    let key = dependency.to_ascii_uppercase().replace('-', "_");
    println!("cargo:rustc-env=EQIORA_DEPENDENCY_VERSION_{key}={version}");
    Ok(())
}

fn dependency_version(directory: &Path, dependency: &str) -> BuildResult<String> {
    let manifest_path = directory.join("Cargo.toml");
    let manifest = read_manifest(&manifest_path)?;
    let selected = dependency_entry(&manifest, dependency)?;
    let version = if selected.get("workspace").and_then(Value::as_bool) == Some(true) {
        let workspace_path = workspace_manifest(directory, &manifest)?;
        let workspace = read_manifest(&workspace_path)?;
        let inherited = workspace
            .get("workspace")
            .and_then(|table| table.get("dependencies"))
            .and_then(|table| table.get(dependency))
            .ok_or_else(|| format!("workspace does not declare dependency {dependency}"))?;
        exact_version(inherited)?
    } else {
        exact_version(selected)?
    };
    Ok(version)
}

fn read_manifest(path: &Path) -> BuildResult<Value> {
    println!("cargo:rerun-if-changed={}", path.display());
    Ok(toml::from_str(&fs::read_to_string(path)?)?)
}

fn dependency_entry<'a>(manifest: &'a Value, dependency: &str) -> BuildResult<&'a Value> {
    manifest
        .get("dependencies")
        .and_then(|table| table.get(dependency))
        .ok_or_else(|| format!("manifest does not declare normal dependency {dependency}").into())
}

fn workspace_manifest(directory: &Path, member: &Value) -> BuildResult<PathBuf> {
    if let Some(path) = member
        .get("package")
        .and_then(|table| table.get("workspace"))
        .and_then(Value::as_str)
    {
        return Ok(directory.join(path).join("Cargo.toml"));
    }
    for ancestor in directory.ancestors() {
        let path = ancestor.join("Cargo.toml");
        if path.is_file() && read_manifest(&path)?.get("workspace").is_some() {
            return Ok(path);
        }
    }
    Err("inherited dependency has no workspace manifest".into())
}

fn exact_version(dependency: &Value) -> BuildResult<String> {
    let requirement = dependency
        .as_str()
        .or_else(|| dependency.get("version").and_then(Value::as_str))
        .ok_or("dependency has no version requirement")?;
    let release = requirement
        .strip_prefix('=')
        .ok_or("provider provenance requires an exact =major.minor.patch dependency pin")?;
    let version = semver::Version::parse(release)?;
    if !version.build.is_empty() {
        return Err("Cargo version requirements do not bind build metadata".into());
    }
    Ok(version.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_and_packaged_manifests_share_one_release_owner() {
        let root =
            std::env::temp_dir().join(format!("eqiora-build-release-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let scratch = Scratch(root);
        let member = scratch.0.join("member");
        fs::create_dir(&member).unwrap();
        fs::write(
            scratch.0.join("Cargo.toml"),
            "[workspace.dependencies]\nbinding = '=2.4.6'\n",
        )
        .unwrap();
        fs::write(
            member.join("Cargo.toml"),
            "[dependencies]\nbinding = { workspace = true }\n",
        )
        .unwrap();
        assert_eq!(dependency_version(&member, "binding").unwrap(), "2.4.6");
        fs::write(
            scratch.0.join("Cargo.toml"),
            "[workspace.dependencies]\nbinding = '=2.4.7'\n",
        )
        .unwrap();
        assert_eq!(dependency_version(&member, "binding").unwrap(), "2.4.7");
        // Cargo's packaged manifest owns the resolved declaration itself.
        fs::write(
            member.join("Cargo.toml"),
            "[dependencies.binding]\nversion = '=2.4.8'\noptional = true\n",
        )
        .unwrap();
        assert_eq!(dependency_version(&member, "binding").unwrap(), "2.4.8");
        // An explicit workspace path is resolved relative to the member.
        fs::write(
            member.join("Cargo.toml"),
            "[package]\nworkspace = '..'\n[dependencies]\nbinding = { workspace = true }\n",
        )
        .unwrap();
        assert_eq!(dependency_version(&member, "binding").unwrap(), "2.4.7");
        fs::write(
            scratch.0.join("Cargo.toml"),
            "[workspace.dependencies]\nother = '=2.4.7'\n",
        )
        .unwrap();
        assert!(dependency_version(&member, "binding").is_err());
    }

    #[test]
    fn exact_release_is_derived_from_string_and_normalized_table_pins() {
        for source in [
            "binding = '=2.4.6'",
            "binding = { version = '=2.4.6', optional = true }",
            "binding = { version = '=2.4.6', package = 'upstream-binding' }",
        ] {
            let manifest: Value = toml::from_str(&format!("[dependencies]\n{source}")).unwrap();
            assert_eq!(
                exact_version(dependency_entry(&manifest, "binding").unwrap()).unwrap(),
                "2.4.6"
            );
        }
        assert_eq!(
            exact_version(&Value::String("=2.4.6-rc.1".into())).unwrap(),
            "2.4.6-rc.1"
        );
    }

    #[test]
    fn ambiguous_or_missing_release_is_rejected() {
        for requirement in [
            "2.4.6",
            "^2.4.6",
            "~2.4.6",
            "=2.4",
            ">=2.4.6",
            "=2.4.6, <3",
            "=2.4.6+unbound",
        ] {
            assert!(
                exact_version(&Value::String(requirement.into())).is_err(),
                "{requirement}"
            );
        }
        let manifest: Value = toml::from_str("[dev-dependencies]\nbinding = '=2.4.6'").unwrap();
        assert!(dependency_entry(&manifest, "binding").is_err());
        let manifest: Value =
            toml::from_str("[dependencies]\nbinding = { workspace = true }").unwrap();
        assert!(exact_version(dependency_entry(&manifest, "binding").unwrap()).is_err());
    }
}
