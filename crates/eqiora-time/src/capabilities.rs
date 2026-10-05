//! Adapter-declared value domains and coordinate precision, separate from identity.
use crate::TimeBackendIdentity;
use crate::diagnostic::invalid_plan;
use eqiora_core::{Diagnostic, ScalarDomain, ScalarType};

/// Numerical representations an adapter accepts through the real-coordinate seam.
///
/// Complex support means that the adapter accepts the complete paired real and
/// imaginary coordinates and their real error scales; it does not imply native
/// complex arithmetic, complex ordering, or support for every time method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBackendCapabilities {
    identity: TimeBackendIdentity,
    domains: &'static [ScalarDomain],
    coordinate_types: &'static [ScalarType],
}

impl TimeBackendCapabilities {
    /// Declare supported mathematical domains and storage of each real coordinate.
    #[must_use]
    pub const fn new(
        identity: TimeBackendIdentity,
        domains: &'static [ScalarDomain],
        coordinate_types: &'static [ScalarType],
    ) -> Self {
        Self {
            identity,
            domains,
            coordinate_types,
        }
    }

    /// The implementation whose capabilities are being negotiated.
    #[must_use]
    pub const fn identity(self) -> TimeBackendIdentity {
        self.identity
    }

    /// Check the exact requested domain and coordinate precision without fallback.
    ///
    /// # Errors
    /// Rejects discrete time payloads or any representation not advertised here.
    pub fn admit_real_coordinates(
        self,
        domain: ScalarDomain,
        scalar_type: ScalarType,
    ) -> Result<(), Diagnostic> {
        if !matches!(domain, ScalarDomain::Real | ScalarDomain::Complex)
            || !self.domains.contains(&domain)
            || !self.coordinate_types.contains(&scalar_type)
        {
            return Err(invalid_plan(format!(
                "time backend {} does not support {domain:?} values in {scalar_type:?} real coordinates",
                self.identity.id()
            )));
        }
        Ok(())
    }
}
