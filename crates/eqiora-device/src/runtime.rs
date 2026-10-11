use std::collections::BTreeSet;
use std::fmt::Debug;
use std::num::NonZeroU64;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{DeviceId, RuntimeId};

/// Independently negotiable device capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceCapability {
    /// IEEE-754 binary32 storage and arithmetic.
    Float32,
    /// IEEE-754 binary64 storage and arithmetic.
    Float64,
    /// Compressed-sparse-row matrix-vector action.
    CsrMatrixVectorProduct,
    /// Level-1 dense vector primitives required by Krylov methods.
    DenseVectorLevel1,
    /// Ordered asynchronous transfers and submissions.
    AsynchronousQueue,
}

/// Floating-point ordering selected for a sparse operator action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SparseActionPolicy {
    /// A backend algorithm documented to be bitwise deterministic for repeated
    /// executions with the same inputs on the same admitted runtime.
    Deterministic,
    /// The backend's native performance-oriented algorithm and reduction
    /// order.
    BackendNative,
}

/// Explicit absolute/relative comparison contract for a sparse action oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseActionTolerance {
    absolute: f64,
    relative: f64,
}

impl SparseActionTolerance {
    /// Construct a finite, non-negative tolerance with at least one positive
    /// component.
    ///
    /// # Errors
    /// Returns `EQ0807` for an invalid or jointly zero tolerance.
    pub fn new(absolute: f64, relative: f64) -> Result<Self, Diagnostic> {
        if !absolute.is_finite()
            || !relative.is_finite()
            || absolute < 0.0
            || relative < 0.0
            || (absolute == 0.0 && relative == 0.0)
        {
            return Err(invalid_device(
                "sparse-action tolerances must be finite and non-negative, with at least one positive",
            ));
        }
        Ok(Self { absolute, relative })
    }

    /// Absolute tolerance.
    #[must_use]
    pub const fn absolute(self) -> f64 {
        self.absolute
    }

    /// Relative tolerance.
    #[must_use]
    pub const fn relative(self) -> f64 {
        self.relative
    }

    /// Accepted error for one finite reference value.
    #[must_use]
    pub fn threshold(self, reference: f64) -> f64 {
        self.absolute + self.relative * reference.abs()
    }
}

/// Eqiora-owned snapshot of a device discovered by a runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    id: DeviceId,
    name: String,
    total_memory_bytes: NonZeroU64,
    capabilities: BTreeSet<DeviceCapability>,
}

impl DeviceDescriptor {
    /// Construct a nonempty capability snapshot.
    ///
    /// # Errors
    /// Returns `EQ0807` for an empty name or capability set.
    pub fn new(
        id: DeviceId,
        name: impl Into<String>,
        total_memory_bytes: NonZeroU64,
        capabilities: impl IntoIterator<Item = DeviceCapability>,
    ) -> Result<Self, Diagnostic> {
        let name = name.into();
        let capabilities = capabilities.into_iter().collect::<BTreeSet<_>>();
        if name.trim().is_empty() {
            return Err(invalid_device("device name must not be empty"));
        }
        if capabilities.is_empty() {
            return Err(invalid_device(
                "device capability evidence must contain at least one feature",
            ));
        }
        Ok(Self {
            id,
            name,
            total_memory_bytes,
            capabilities,
        })
    }

    /// Runtime-scoped device identity.
    #[must_use]
    pub const fn id(&self) -> DeviceId {
        self.id
    }

    /// Human-readable device name supplied by the runtime.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total device memory visible to the runtime.
    #[must_use]
    pub const fn total_memory_bytes(&self) -> NonZeroU64 {
        self.total_memory_bytes
    }

    /// Exact admitted capability set.
    #[must_use]
    pub const fn capabilities(&self) -> &BTreeSet<DeviceCapability> {
        &self.capabilities
    }

    /// Require every capability needed by an execution before allocation.
    ///
    /// # Errors
    /// Returns `EQ0807` naming the first absent capability.
    pub fn require(
        &self,
        required: impl IntoIterator<Item = DeviceCapability>,
    ) -> Result<(), Diagnostic> {
        for capability in required {
            if !self.capabilities.contains(&capability) {
                return Err(invalid_device(format!(
                    "device {} lacks required capability {capability:?}",
                    self.id.ordinal()
                )));
            }
        }
        Ok(())
    }
}

/// Discovery boundary implemented by a concrete runtime adapter.
pub trait DeviceRuntime: Debug + Send + Sync {
    /// Stable adapter identity.
    fn id(&self) -> RuntimeId;

    /// Snapshot every visible device and its admitted capabilities.
    ///
    /// # Errors
    /// Returns a stable diagnostic when runtime loading or discovery fails.
    fn discover(&self) -> Result<Vec<DeviceDescriptor>, Diagnostic>;
}

fn invalid_device(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
