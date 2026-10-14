mod manifest;
mod types;
pub(crate) mod wire;

pub use manifest::ExternalImportManifestV1;
pub use types::{
    ExternalAdapterIdentityV1, ExternalImportObservationV1, ExternalImportSelectionV1,
    ExternalImportSourceV1, ExternalRuntimeComponentV1, ExternalRuntimeRoleV1, RawSourceSha256,
    ResolvedImportArrayV1, SelectedSourceEntityV1, StructuralSelectorV1,
};

/// Semantic work budgets for external-import manifest artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalImportDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Work admitted while reconstructing embedded resolved-array references.
    pub resolved_array: crate::ResolvedArrayLimits,
    /// Maximum dynamic UTF-8 text bytes summed across one manifest.
    pub max_import_manifest_text_bytes: usize,
    /// Maximum native runtime components in one manifest.
    pub max_import_runtime_entries: usize,
    /// Maximum selected attributes in one manifest.
    pub max_import_selection_attributes: usize,
    /// Maximum source occurrences in one manifest.
    pub max_import_sources: usize,
    /// Maximum normalized array references in one manifest.
    pub max_import_resolved_arrays: usize,
    /// Maximum accepted artifact references in one manifest.
    pub max_import_accepted_artifacts: usize,
}

impl Default for ExternalImportDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            resolved_array: crate::ResolvedArrayLimits::default(),
            max_import_manifest_text_bytes: 1024 * 1024,
            max_import_runtime_entries: 32,
            max_import_selection_attributes: 100_000,
            max_import_sources: 100_000,
            max_import_resolved_arrays: 100_002,
            max_import_accepted_artifacts: 100_001,
        }
    }
}
