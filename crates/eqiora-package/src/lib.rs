//! Typed model-package requests, exact identities, and offline resolution.
//!
//! This crate owns package contracts and request matching. Project acquisition,
//! candidate selection, and compiler admission belong to the API layer.

mod canonical;
mod digest;
#[cfg(feature = "filesystem")]
mod directory_installation;
#[cfg(feature = "filesystem")]
mod directory_io;
mod execution_binding;
#[cfg(feature = "filesystem")]
mod filesystem;
mod identity;
#[cfg(feature = "filesystem")]
mod package_directory;
mod package_manifest;
mod path;
#[cfg(feature = "filesystem")]
mod project_directory;
mod release;
mod resolution;
mod run_binding;
mod semantic;
mod source;
mod store;
#[cfg(test)]
mod test_support;
mod version_request;

pub use digest::{
    CanonicalModelDigest, CanonicalRealizationDigest, CanonicalRunDigest, PackageCompilationDigest,
    PackageExecutionBindingDigest, PackageRunBindingDigest, PackageSemanticDigest,
    ResolutionDigest, SourceBundleDigest,
};
#[cfg(feature = "filesystem")]
pub use directory_installation::{
    DirectoryPackageInstaller, PackageInstallDisposition, PackageInstallError,
    PackageInstallIoPhase, PackageInstallReceipt, PackageStageCleanup,
};
pub use execution_binding::{
    BoundExecutionRunSchemaV1, BoundRealizationSchemaV1, PackageExecutionBindingV1,
};
#[cfg(feature = "filesystem")]
pub use filesystem::DirectoryPackageStore;
pub use identity::{ExactVersion, ModelPackageIdentityV1, QualifiedName};
#[cfg(feature = "filesystem")]
pub use package_directory::{PackageDirectory, PackageDirectoryError, PackageDirectoryResource};
pub use package_manifest::{BundleEntryV1, BundleRoleV1, PackageDependencyV1, PackageManifestV1};
pub use path::NormalizedRelativePath;
pub use release::{
    CompilationPackageV1, CompilationToolchainV2, PackageCompilationRecordV2, PackageReleaseV1,
};
pub use resolution::{
    ExactResolver, ResolutionEdgeV1, ResolutionError, ResolutionNodeV1, ResolutionRecordV1,
    ResolvedPackageGraph,
};
pub use run_binding::{BoundRunManifestSchemaV1, PackageRunBindingV1};
pub use semantic::{
    CanonicalDeclaration, DeclarationKindV1, SemanticContentV1, SemanticDeclarationV1, VisibilityV1,
};
pub use source::{PackageSourcesV1, SourceBundleIdentityV1, SourceBundleV1, SourceFileV1};
pub use store::{InMemoryPackageStore, PackageStore, StoreError};
pub use version_request::VersionRequest;

/// Errors produced while constructing or decoding closed package contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractError {
    message: String,
}

impl ContractError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ContractError {}
