//! Bounded local-file boundary for canonical Python artifacts.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora::Diagnostic;
use eqiora::api::ModelDocument;
use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope};
use eqiora::diagnostic::codes;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyString};

pub(crate) const MODEL_FILE_EXTENSION: &str = "eqmodel";

#[derive(Clone, Copy)]
pub(crate) struct ArtifactFileSpec {
    pub(crate) artifact_name: &'static str,
    pub(crate) extension: &'static str,
    pub(crate) staging_name: &'static str,
    pub(crate) max_bytes: usize,
}

fn model_file_spec() -> ArtifactFileSpec {
    ArtifactFileSpec {
        artifact_name: "compiled Model",
        extension: MODEL_FILE_EXTENSION,
        staging_name: "model",
        max_bytes: ModelDecoderLimits::default().json.max_bytes,
    }
}

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) enum DecodedModel {
    Document(Box<ModelDocument>),
    Deferred(ModelEnvelope),
}

pub(crate) fn decode_model(bytes: &[u8]) -> Result<DecodedModel, Vec<Diagnostic>> {
    let artifact = ModelEnvelope::from_json(bytes, ModelDecoderLimits::default())
        .map_err(|diagnostic| vec![diagnostic])?;
    let canonical = artifact
        .canonical_json()
        .map_err(|diagnostic| vec![diagnostic])?;
    if canonical != bytes {
        return Err(vec![invalid_artifact(
            "Model artifact bytes are not the exact canonical encoding",
        )]);
    }
    if artifact
        .requires_geometry_admission()
        .map_err(|diagnostic| vec![diagnostic])?
    {
        Ok(DecodedModel::Deferred(artifact))
    } else {
        ModelDocument::replay(bytes).map(|document| DecodedModel::Document(Box::new(document)))
    }
}

pub(crate) fn read_model_bytes(path: &Path) -> Result<Vec<u8>, Diagnostic> {
    read_artifact_bytes(path, model_file_spec())
}

pub(crate) fn write_model_bytes(path: &Path, bytes: &[u8]) -> Result<(), Diagnostic> {
    write_artifact_bytes(path, bytes, model_file_spec())
}

pub(crate) fn unicode_artifact_path(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let path = py.import("os")?.getattr("fspath")?.call1((value,))?;
    let path = path
        .cast::<PyString>()
        .map_err(|_| PyTypeError::new_err("path must resolve to a Unicode filesystem path"))?;
    Ok(PathBuf::from(path.to_str()?))
}

pub(crate) fn read_artifact_bytes(
    path: &Path,
    spec: ArtifactFileSpec,
) -> Result<Vec<u8>, Diagnostic> {
    validate_artifact_path(path, spec)?;
    let link_metadata = fs::symlink_metadata(path)
        .map_err(|error| artifact_io_error("inspect", path, spec, &error))?;
    if !link_metadata.file_type().is_file() {
        return Err(invalid_artifact(format!(
            "{} input {:?} is not a regular file",
            spec.artifact_name,
            path.display()
        )));
    }

    let file = File::open(path).map_err(|error| artifact_io_error("open", path, spec, &error))?;
    let metadata = file
        .metadata()
        .map_err(|error| artifact_io_error("inspect opened", path, spec, &error))?;
    if !metadata.file_type().is_file() {
        return Err(invalid_artifact(format!(
            "{} input {:?} did not remain a regular file",
            spec.artifact_name,
            path.display()
        )));
    }

    let max_bytes = spec.max_bytes;
    if metadata.len() > max_bytes as u64 {
        return Err(invalid_artifact(format!(
            "{} input has {} bytes, exceeding the {max_bytes} byte decoder limit",
            spec.artifact_name,
            metadata.len()
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((max_bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| artifact_io_error("read", path, spec, &error))?;
    if bytes.len() > max_bytes {
        return Err(invalid_artifact(format!(
            "{} input exceeds the {max_bytes} byte decoder limit",
            spec.artifact_name,
        )));
    }
    Ok(bytes)
}

pub(crate) fn write_artifact_bytes(
    path: &Path,
    bytes: &[u8],
    spec: ArtifactFileSpec,
) -> Result<(), Diagnostic> {
    publish_artifact_bytes(path, bytes, spec, false)
}

/// Publish an immutable content-addressed companion without replacing any existing file.
/// A concurrent identical publication succeeds; an existing different file rejects.
pub(crate) fn write_immutable_artifact_bytes(
    path: &Path,
    bytes: &[u8],
    spec: ArtifactFileSpec,
) -> Result<(), Diagnostic> {
    publish_artifact_bytes(path, bytes, spec, true)
}

fn publish_artifact_bytes(
    path: &Path,
    bytes: &[u8],
    spec: ArtifactFileSpec,
    immutable: bool,
) -> Result<(), Diagnostic> {
    validate_artifact_path(path, spec)?;
    if bytes.len() > spec.max_bytes {
        return Err(invalid_artifact(format!(
            "{} output exceeds the {} byte limit",
            spec.artifact_name, spec.max_bytes
        )));
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(invalid_artifact(format!(
                "{} output {:?} is not a regular file",
                spec.artifact_name,
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(artifact_io_error("inspect output", path, spec, &error)),
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let (staging_path, mut staging) = create_staging_file(parent, path, spec)?;
    let mut cleanup = StagingCleanup::new(staging_path.clone());
    staging
        .write_all(bytes)
        .and_then(|()| staging.sync_all())
        .map_err(|error| artifact_io_error("write", &staging_path, spec, &error))?;
    drop(staging);
    if immutable {
        match fs::hard_link(&staging_path, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if read_artifact_bytes(path, spec)? != bytes {
                    return Err(invalid_artifact(format!(
                        "{} immutable output {:?} already contains different bytes",
                        spec.artifact_name,
                        path.display()
                    )));
                }
            }
            Err(error) => return Err(artifact_io_error("publish immutable", path, spec, &error)),
        }
        // The destination owns a separate link. Always remove our private staging link.
    } else {
        fs::rename(&staging_path, path)
            .map_err(|error| artifact_io_error("publish", path, spec, &error))?;
        cleanup.published = true;
    }
    Ok(())
}

fn validate_artifact_path(path: &Path, spec: ArtifactFileSpec) -> Result<(), Diagnostic> {
    if path
        .extension()
        .is_none_or(|suffix| suffix != spec.extension)
    {
        return Err(invalid_artifact(format!(
            "{} paths must use the .{} suffix",
            spec.artifact_name, spec.extension
        )));
    }
    Ok(())
}

fn create_staging_file(
    parent: &Path,
    target: &Path,
    spec: ArtifactFileSpec,
) -> Result<(PathBuf, File), Diagnostic> {
    for _ in 0..128 {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging_path = parent.join(format!(
            ".eqiora-{}-{}-{sequence}.tmp",
            spec.staging_name,
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging_path)
        {
            Ok(file) => return Ok((staging_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(artifact_io_error("stage", target, spec, &error)),
        }
    }
    Err(invalid_artifact(format!(
        "could not allocate a staging file beside {} output {:?}",
        spec.artifact_name,
        target.display()
    )))
}

struct StagingCleanup {
    path: PathBuf,
    published: bool,
}

impl StagingCleanup {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            published: false,
        }
    }
}

impl Drop for StagingCleanup {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn artifact_io_error(
    action: &str,
    path: &Path,
    spec: ArtifactFileSpec,
    error: &std::io::Error,
) -> Diagnostic {
    invalid_artifact(format!(
        "could not {action} {} path {:?}: {error}",
        spec.artifact_name,
        path.display()
    ))
}

fn invalid_artifact(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_publication_preserves_existing_outputs_and_removes_staging_links() {
        let directory = std::env::temp_dir().join(format!(
            "eqiora-artifact-publication-{}-{}",
            std::process::id(),
            STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("value.eqmodel");
        let spec = ArtifactFileSpec {
            max_bytes: 4,
            ..model_file_spec()
        };
        write_artifact_bytes(&path, b"old", spec).unwrap();
        assert!(write_artifact_bytes(&path, b"large", spec).is_err());
        assert!(write_immutable_artifact_bytes(&path, b"new", spec).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
        write_immutable_artifact_bytes(&path, b"old", spec).unwrap();
        let companion = directory.join("other.eqmodel");
        write_immutable_artifact_bytes(&companion, b"new", spec).unwrap();
        assert_eq!(fs::read(&companion).unwrap(), b"new");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        write_artifact_bytes(&path, b"next", spec).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"next");
        fs::remove_dir_all(directory).unwrap();
    }
}
