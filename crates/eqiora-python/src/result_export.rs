//! Local publication of native spatial exports through the shared artifact I/O boundary.
use eqiora::Diagnostic;
use eqiora::api::XdmfHdf5ExportLimits;
use eqiora::artifact::ArtifactDigest;
use eqiora::diagnostic::codes;
use eqiora::io::hdf5::WrittenHdf5FileImage;
use eqiora::io::xml::XdmfExportPlan;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::model_io::{ArtifactFileSpec, write_artifact_bytes, write_immutable_artifact_bytes};

pub(crate) fn write_xdmf(
    path: &Path,
    xdmf: XdmfExportPlan,
    hdf5: WrittenHdf5FileImage,
    limits: XdmfHdf5ExportLimits,
) -> Result<(), Diagnostic> {
    if path.extension().is_none_or(|extension| extension != "xdmf") {
        return Err(Diagnostic::error(
            codes::INVALID_EXTERNAL_DATA_EXPORT,
            "XDMF export paths must use the .xdmf suffix",
        ));
    }
    let digest = ArtifactDigest::from_sha256(Sha256::digest(hdf5.bytes()).into());
    let locator = format!("eqiora-{digest}.h5");
    let metadata = XdmfExportPlan::new(&locator, xdmf.frames().to_vec(), limits.xdmf)?;
    let companion = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&locator);
    write_immutable_artifact_bytes(
        &companion,
        hdf5.bytes(),
        ArtifactFileSpec {
            artifact_name: "XDMF HDF5 companion",
            extension: "h5",
            staging_name: "hdf5",
            max_bytes: limits.hdf5.max_output_bytes,
        },
    )?;
    // Only the metadata name is replaced. Any earlier complete pair stays valid
    // if companion publication or metadata staging fails. A companion is never
    // removed here: another completed export may already reference its content.
    write_artifact_bytes(
        path,
        metadata.metadata_bytes(),
        ArtifactFileSpec {
            artifact_name: "XDMF metadata",
            extension: "xdmf",
            staging_name: "xdmf",
            max_bytes: limits.xdmf.max_metadata_bytes,
        },
    )
}

/// Project typed observations through their native owner, then publish once.
#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _export_observations_csv(
    py: pyo3::Python<'_>,
    path: &pyo3::Bound<'_, pyo3::types::PyAny>,
    observations: &pyo3::Bound<'_, pyo3::types::PyAny>,
    unit: &crate::authoring::PyAstExpression,
    projection: &str,
    component: usize,
    decimal_places: u8,
    locale: &str,
) -> pyo3::PyResult<()> {
    use eqiora_numerics::CommonObservation;
    use pyo3::prelude::*;
    let path = crate::model_io::unicode_artifact_path(py, path)?;
    let mut objects: Vec<Py<crate::result::PyObservation>> = Vec::new();
    for item in observations.try_iter()? {
        if objects.len() >= CommonObservation::CSV_MAX_ROWS {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "CSV exceeds its selected observation budget",
            ));
        }
        objects.push(item?.extract()?);
    }
    let borrowed = objects
        .iter()
        .map(|value| value.borrow(py))
        .collect::<Vec<_>>();
    let bytes = CommonObservation::export_csv(
        borrowed.iter().map(|value| &value.native),
        projection,
        component,
        unit.as_expr(),
        decimal_places,
        locale,
    )
    .map_err(|error| crate::error::validation_error(py, &[error]))?;
    drop(borrowed);
    py.detach(move || {
        write_artifact_bytes(
            &path,
            &bytes,
            ArtifactFileSpec {
                artifact_name: "observation CSV",
                extension: "csv",
                staging_name: "csv",
                max_bytes: CommonObservation::CSV_MAX_BYTES,
            },
        )
    })
    .map_err(|error| crate::error::compatibility_error(py, &[error]))
}
