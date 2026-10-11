//! Query projections retained outside canonical model meaning.
//!
//! Hierarchy normalization may remove an ownerless public physical Port from
//! the Kernel graph. The Port is not an alias for a retained endpoint. Its
//! observable meaning is instead one exact cut through the final conserving
//! Connection: the common across/trace quantity and the net outward
//! through/flux quantity of the retained endpoints inside that occurrence.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;

use crate::identity::{FullElaborationIdentity, ProjectedId};

/// Closed nominal contract referenced by one eliminated exposure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalExposureContract {
    /// Scalar acausal across/through connector.
    ScalarPhysical {
        /// Exact nominal scalar-physical Domain.
        connector: ProjectedId<kinds::Domain>,
    },
    /// Field-valued trace/outward-flux connector on one exact boundary.
    FieldBoundary {
        /// Exact nominal boundary-physical Connector Domain.
        connector: ProjectedId<kinds::Domain>,
        /// Exact boundary Domain bound by this exposure occurrence.
        boundary: ProjectedId<kinds::Domain>,
    },
}

impl PhysicalExposureContract {
    /// Exact nominal Connector Domain shared by both contract families.
    #[must_use]
    pub const fn connector(self) -> ProjectedId<kinds::Domain> {
        match self {
            Self::ScalarPhysical { connector } | Self::FieldBoundary { connector, .. } => connector,
        }
    }

    /// Exact bound boundary Domain for a field-valued exposure.
    #[must_use]
    pub const fn boundary(self) -> Option<ProjectedId<kinds::Domain>> {
        match self {
            Self::ScalarPhysical { .. } => None,
            Self::FieldBoundary { boundary, .. } => Some(boundary),
        }
    }
}

/// One eliminated public physical exposure and its canonical observation cut.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalExposureProjection {
    selector: String,
    exposure: FullElaborationIdentity,
    connection: ProjectedId<kinds::Connection>,
    interior: Box<[ProjectedId<kinds::Port>]>,
    contract: PhysicalExposureContract,
}

impl PhysicalExposureProjection {
    pub(crate) fn new(
        selector: String,
        exposure: FullElaborationIdentity,
        connection: ProjectedId<kinds::Connection>,
        interior: Vec<ProjectedId<kinds::Port>>,
        contract: PhysicalExposureContract,
    ) -> Self {
        debug_assert!(!selector.is_empty());
        debug_assert!(!interior.is_empty());
        Self {
            selector,
            exposure,
            connection,
            interior: interior.into_boxed_slice(),
            contract,
        }
    }

    /// Occurrence-qualified source selector, such as `assembly.left.port`.
    ///
    /// The selector is presentation/query data and does not replace the full
    /// exposure identity used for durable projection identity.
    #[must_use]
    pub fn selector(&self) -> &str {
        &self.selector
    }

    /// Full collision-resistant identity of the eliminated Port occurrence.
    #[must_use]
    pub const fn exposure(&self) -> FullElaborationIdentity {
        self.exposure
    }

    /// Exact final maximal conserving Connection.
    #[must_use]
    pub const fn connection(&self) -> ProjectedId<kinds::Connection> {
        self.connection
    }

    /// Retained physical endpoints inside the exposure occurrence.
    ///
    /// Their across/trace value is the common projected quantity. Summing
    /// their through/outward-flux values gives the projected net-outward
    /// quantity. The slice is nonempty, sorted by full identity, and is a
    /// proper subset of the final Connection.
    #[must_use]
    pub const fn interior(&self) -> &[ProjectedId<kinds::Port>] {
        &self.interior
    }

    /// Closed nominal connector/support reference.
    #[must_use]
    pub const fn contract(&self) -> PhysicalExposureContract {
        self.contract
    }
}

/// Immutable projection catalog in occurrence-selector order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PhysicalExposureProjectionMap {
    entries: Box<[PhysicalExposureProjection]>,
    identity_index: Box<[usize]>,
}

impl PhysicalExposureProjectionMap {
    pub(crate) fn from_sorted(
        entries: Vec<PhysicalExposureProjection>,
    ) -> Result<Self, Diagnostic> {
        if !entries
            .windows(2)
            .all(|pair| pair[0].selector() < pair[1].selector())
        {
            return Err(projection_error(
                "physical exposure projections are not uniquely sorted by selector",
            ));
        }
        let mut identity_index = Vec::new();
        identity_index
            .try_reserve_exact(entries.len())
            .map_err(|_| projection_error("cannot reserve physical exposure identity index"))?;
        identity_index.extend(0..entries.len());
        identity_index.sort_unstable_by_key(|index| entries[*index].exposure());
        if !identity_index
            .windows(2)
            .all(|pair| entries[pair[0]].exposure() < entries[pair[1]].exposure())
        {
            return Err(projection_error(
                "physical exposure projection identities are not unique",
            ));
        }
        Ok(Self {
            entries: entries.into_boxed_slice(),
            identity_index: identity_index.into_boxed_slice(),
        })
    }

    /// Number of eliminated public physical exposures.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether hierarchy normalization retained no exposure projections.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve one occurrence-qualified selector exactly.
    #[must_use]
    pub fn get(&self, selector: &str) -> Option<&PhysicalExposureProjection> {
        self.entries
            .binary_search_by(|entry| entry.selector().cmp(selector))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Resolve one full exposure identity exactly.
    #[must_use]
    pub fn get_by_identity(
        &self,
        identity: FullElaborationIdentity,
    ) -> Option<&PhysicalExposureProjection> {
        self.identity_index
            .binary_search_by_key(&identity, |index| self.entries[*index].exposure())
            .ok()
            .map(|index| &self.entries[self.identity_index[index]])
    }

    /// Iterate in deterministic occurrence-selector order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &PhysicalExposureProjection> {
        self.entries.iter()
    }
}

fn projection_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::LANGUAGE_LOWERING_ERROR, message)
}
