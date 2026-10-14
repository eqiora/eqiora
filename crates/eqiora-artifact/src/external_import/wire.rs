//! Shared identity leaves for external import and storage provenance.

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use super::{ExternalAdapterIdentityV1, ExternalRuntimeComponentV1, ExternalRuntimeRoleV1};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireAdapter {
    pub(crate) id: String,
    pub(crate) version: String,
}

impl From<&ExternalAdapterIdentityV1> for WireAdapter {
    fn from(value: &ExternalAdapterIdentityV1) -> Self {
        Self {
            id: value.id().to_owned(),
            version: value.version().to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireRuntimeComponent {
    pub(crate) role: WireRuntimeRole,
    pub(crate) implementation: String,
    pub(crate) version: String,
}

impl From<&ExternalRuntimeComponentV1> for WireRuntimeComponent {
    fn from(value: &ExternalRuntimeComponentV1) -> Self {
        Self {
            role: value.role().into(),
            implementation: value.implementation().to_owned(),
            version: value.version().to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireRuntimeRole {
    RustBinding,
    NativeStorageLibrary,
}

impl From<ExternalRuntimeRoleV1> for WireRuntimeRole {
    fn from(value: ExternalRuntimeRoleV1) -> Self {
        match value {
            ExternalRuntimeRoleV1::RustBinding => Self::RustBinding,
            ExternalRuntimeRoleV1::NativeStorageLibrary => Self::NativeStorageLibrary,
        }
    }
}

impl From<WireRuntimeRole> for ExternalRuntimeRoleV1 {
    fn from(value: WireRuntimeRole) -> Self {
        match value {
            WireRuntimeRole::RustBinding => Self::RustBinding,
            WireRuntimeRole::NativeStorageLibrary => Self::NativeStorageLibrary,
        }
    }
}

impl WireAdapter {
    pub(crate) fn decode(&self) -> Result<ExternalAdapterIdentityV1, Diagnostic> {
        ExternalAdapterIdentityV1::new(&self.id, &self.version)
    }
}

impl WireRuntimeComponent {
    pub(crate) fn decode(&self) -> Result<ExternalRuntimeComponentV1, Diagnostic> {
        ExternalRuntimeComponentV1::new(self.role.into(), &self.implementation, &self.version)
    }
}
