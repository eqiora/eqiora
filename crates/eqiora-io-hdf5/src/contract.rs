use std::fmt;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

/// Exact Rust binding implementation.
pub const HDF5_BINDING_ID: &str = "hdf5-metno";
/// Exact Rust binding release pinned by the workspace.
pub const HDF5_BINDING_VERSION: &str = "0.14.1";
/// Native storage implementation recorded separately from its runtime release.
pub const HDF5_NATIVE_LIBRARY_ID: &str = "HDF5";

/// Independent native source, graph, declaration, and decoded-value budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hdf5ResolveLimits {
    /// Maximum complete caller-owned HDF5 file-image bytes.
    pub max_source_bytes: usize,
    /// Maximum hard links in the complete reachable graph.
    pub max_links: usize,
    /// Maximum unique groups and datasets, including the root group.
    pub max_objects: usize,
    /// Maximum datasets in the complete reachable graph.
    pub max_datasets: usize,
    /// Maximum UTF-8 bytes in one link name or requested dataset path.
    pub max_name_bytes: usize,
    /// Maximum aggregate UTF-8 bytes across reachable link names.
    pub max_total_name_bytes: usize,
    /// Maximum rank of every dataset, selected or not.
    pub max_rank: usize,
    /// Maximum ordered dataset requests resolved from one file image.
    pub max_requests: usize,
    /// Maximum scalar values declared by one dataset.
    pub max_dataset_values: usize,
    /// Maximum aggregate scalar values declared by all datasets.
    pub max_total_declared_values: usize,
    /// Maximum decoded bytes for one requested dataset.
    pub max_dataset_resolved_bytes: usize,
    /// Maximum aggregate decoded bytes across all requested datasets.
    pub max_total_resolved_bytes: usize,
    /// Maximum explicit Eqiora audit work units.
    pub max_audit_work: usize,
}

impl Default for Hdf5ResolveLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 256 * 1024 * 1024,
            max_links: 100_000,
            max_objects: 100_001,
            max_datasets: 16_384,
            max_name_bytes: 16 * 1024,
            max_total_name_bytes: 8 * 1024 * 1024,
            max_rank: 8,
            max_requests: 16_384,
            max_dataset_values: 8_000_000,
            max_total_declared_values: 64_000_000,
            max_dataset_resolved_bytes: 256 * 1024 * 1024,
            max_total_resolved_bytes: 512 * 1024 * 1024,
            max_audit_work: 64_000_000,
        }
    }
}

impl Hdf5ResolveLimits {
    pub(crate) fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_source_bytes,
            self.max_links,
            self.max_objects,
            self.max_datasets,
            self.max_name_bytes,
            self.max_total_name_bytes,
            self.max_rank,
            self.max_requests,
            self.max_dataset_values,
            self.max_total_declared_values,
            self.max_dataset_resolved_bytes,
            self.max_total_resolved_bytes,
            self.max_audit_work,
        ]
        .contains(&0)
        {
            return Err(invalid_hdf5("HDF5 resolver limits must all be positive"));
        }
        Ok(self)
    }
}

/// Borrowed authority for one complete caller-owned HDF5 source occurrence.
#[derive(Clone, Copy)]
pub struct Hdf5FileImage<'a> {
    bytes: &'a [u8],
}

impl<'a> Hdf5FileImage<'a> {
    /// Bind one complete byte sequence without granting path or URL authority.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Complete source occurrence bytes.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

impl fmt::Debug for Hdf5FileImage<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Hdf5FileImage")
            .field("byte_len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

/// Closed scalar grammar admitted by the first native HDF5 slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hdf5ScalarType {
    /// Unsigned 64-bit integer.
    U64,
    /// IEEE-compatible 64-bit floating point.
    F64,
}

/// Format-neutral native-HDF5 dataset request owned by this adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdf5DatasetRequest {
    dataset_path: String,
    scalar: Hdf5ScalarType,
    shape: Vec<u64>,
}

impl Hdf5DatasetRequest {
    /// Construct one canonical absolute dataset request.
    ///
    /// # Errors
    /// Returns `EQ0810` for a noncanonical path or nonpositive shape.
    pub fn new(
        dataset_path: impl Into<String>,
        scalar: Hdf5ScalarType,
        shape: Vec<u64>,
    ) -> Result<Self, Diagnostic> {
        let dataset_path = dataset_path.into();
        validate_dataset_path(&dataset_path)?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(invalid_hdf5(
                "HDF5 dataset requests require a positive non-scalar shape",
            ));
        }
        Ok(Self {
            dataset_path,
            scalar,
            shape,
        })
    }

    /// Canonical absolute path inside the supplied file image.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }

    /// Required scalar type.
    #[must_use]
    pub const fn scalar(&self) -> Hdf5ScalarType {
        self.scalar
    }

    /// Required exact positive shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// Exact native implementation identity observed during resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdf5RuntimeIdentity {
    native_library_version: String,
}

impl Hdf5RuntimeIdentity {
    pub(crate) fn observed() -> Result<Self, Diagnostic> {
        let (major, minor, patch) = hdf5_metno::library_version();
        if (major, minor, patch) == (0, 0, 0) {
            return Err(invalid_hdf5(
                "native HDF5 runtime version could not be observed",
            ));
        }
        Ok(Self {
            native_library_version: format!("{major}.{minor}.{patch}"),
        })
    }

    /// Exact Rust binding implementation.
    #[must_use]
    pub const fn binding_id(&self) -> &'static str {
        HDF5_BINDING_ID
    }

    /// Exact Rust binding release.
    #[must_use]
    pub const fn binding_version(&self) -> &'static str {
        HDF5_BINDING_VERSION
    }

    /// Native storage implementation.
    #[must_use]
    pub const fn native_library_id(&self) -> &'static str {
        HDF5_NATIVE_LIBRARY_ID
    }

    /// Resolved native HDF5 runtime release.
    #[must_use]
    pub fn native_library_version(&self) -> &str {
        &self.native_library_version
    }
}

/// Normalized values returned only after complete source audit.
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5ResolvedValues {
    /// Unsigned connectivity values.
    U64(Vec<u64>),
    /// Coordinate or field values.
    F64(Vec<f64>),
}

/// Independent declaration, value, hierarchy, and output-image budgets for
/// deterministic in-memory HDF5 writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hdf5WriteLimits {
    /// Maximum datasets in one generated file image.
    pub max_datasets: usize,
    /// Maximum groups, excluding the root group.
    pub max_groups: usize,
    /// Maximum UTF-8 bytes in one dataset path or group path.
    pub max_name_bytes: usize,
    /// Maximum aggregate UTF-8 bytes across dataset paths.
    pub max_total_name_bytes: usize,
    /// Maximum rank of one dataset.
    pub max_rank: usize,
    /// Maximum scalar values in one dataset.
    pub max_dataset_values: usize,
    /// Maximum aggregate scalar values in one file image.
    pub max_total_values: usize,
    /// Maximum complete generated HDF5 file-image bytes.
    pub max_output_bytes: usize,
}

impl Default for Hdf5WriteLimits {
    fn default() -> Self {
        Self {
            max_datasets: 16_384,
            max_groups: 16_384,
            max_name_bytes: 16 * 1024,
            max_total_name_bytes: 8 * 1024 * 1024,
            max_rank: 8,
            max_dataset_values: 8_000_000,
            max_total_values: 64_000_000,
            max_output_bytes: 512 * 1024 * 1024,
        }
    }
}

impl Hdf5WriteLimits {
    pub(crate) fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_datasets,
            self.max_groups,
            self.max_name_bytes,
            self.max_total_name_bytes,
            self.max_rank,
            self.max_dataset_values,
            self.max_total_values,
            self.max_output_bytes,
        ]
        .contains(&0)
        {
            return Err(invalid_hdf5_export(
                "HDF5 writer limits must all be positive",
            ));
        }
        Ok(self)
    }
}

/// Borrowed, closed scalar payload for one generated HDF5 dataset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hdf5WriteValues<'a> {
    /// Unsigned 64-bit values.
    U64(&'a [u64]),
    /// Finite canonical binary64 values.
    F64(&'a [f64]),
}

/// One canonical absolute dataset declaration and its exact borrowed values.
#[derive(Debug, Clone, PartialEq)]
pub struct Hdf5DatasetWrite<'a> {
    dataset_path: String,
    shape: Vec<u64>,
    values: Hdf5WriteValues<'a>,
}

impl<'a> Hdf5DatasetWrite<'a> {
    /// Construct one unsigned-integer dataset write.
    ///
    /// # Errors
    /// Returns `EQ0811` for a noncanonical path, invalid shape, or value-count
    /// mismatch.
    pub fn u64(
        dataset_path: impl Into<String>,
        shape: Vec<u64>,
        values: &'a [u64],
    ) -> Result<Self, Diagnostic> {
        Self::new(dataset_path, shape, Hdf5WriteValues::U64(values))
    }

    /// Construct one binary64 dataset write.
    ///
    /// # Errors
    /// Returns `EQ0811` for a noncanonical path, invalid shape, value-count
    /// mismatch, non-finite value, or negative zero.
    pub fn f64(
        dataset_path: impl Into<String>,
        shape: Vec<u64>,
        values: &'a [f64],
    ) -> Result<Self, Diagnostic> {
        Self::new(dataset_path, shape, Hdf5WriteValues::F64(values))
    }

    fn new(
        dataset_path: impl Into<String>,
        shape: Vec<u64>,
        values: Hdf5WriteValues<'a>,
    ) -> Result<Self, Diagnostic> {
        let dataset_path = dataset_path.into();
        validate_dataset_path(&dataset_path).map_err(|error| {
            invalid_hdf5_export(format!(
                "invalid HDF5 export dataset path: {}",
                error.message()
            ))
        })?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(invalid_hdf5_export(
                "HDF5 dataset writes require a positive non-scalar shape",
            ));
        }
        let required = shape.iter().try_fold(1_u64, |total, extent| {
            total.checked_mul(*extent).ok_or_else(|| {
                invalid_hdf5_export("HDF5 dataset write shape product overflows u64")
            })
        })?;
        let actual = match values {
            Hdf5WriteValues::U64(values) => values.len(),
            Hdf5WriteValues::F64(values) => {
                if values
                    .iter()
                    .any(|value| !value.is_finite() || (*value == 0.0 && value.is_sign_negative()))
                {
                    return Err(invalid_hdf5_export(
                        "HDF5 binary64 export values must be finite and use positive zero",
                    ));
                }
                values.len()
            }
        };
        if u64::try_from(actual).ok() != Some(required) {
            return Err(invalid_hdf5_export(
                "HDF5 dataset write value count differs from its shape",
            ));
        }
        Ok(Self {
            dataset_path,
            shape,
            values,
        })
    }

    /// Canonical absolute dataset path.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }

    /// Exact positive shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Exact borrowed scalar values.
    #[must_use]
    pub const fn values(&self) -> Hdf5WriteValues<'a> {
        self.values
    }
}

fn validate_dataset_path(path: &str) -> Result<(), Diagnostic> {
    if path == "/" || !path.starts_with('/') || path.ends_with('/') {
        return Err(invalid_hdf5(
            "HDF5 dataset path must be a non-root canonical absolute path",
        ));
    }
    if path.chars().any(char::is_control)
        || path
            .split('/')
            .skip(1)
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(invalid_hdf5(
            "HDF5 dataset path contains a forbidden segment or control character",
        ));
    }
    Ok(())
}

pub(crate) fn invalid_hdf5(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_IMPORT, message)
}

pub(crate) fn invalid_hdf5_export(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_EXPORT, message)
}
