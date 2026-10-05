use eqiora_core::Id;
use eqiora_core::entity::kinds;
use eqiora_schema::kernel::BoundarySide;

/// One explicitly selected real coordinate of a differentiation analysis.
///
/// The coordinate identifies model meaning; whether it is selected or frozen
/// remains an analysis choice. Geometry coordinates name continuous Domain
/// bounds rather than realization-local mesh vertices. A realization supplies
/// the corresponding mesh-motion map while topology remains fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpatialDesignCoordinate {
    /// One row-major component and scalar part of a canonical Model Parameter.
    ModelParameter {
        /// Revision-local Parameter identity.
        parameter: Id<kinds::Parameter>,
        /// Flattened row-major component index; zero for a scalar.
        component: usize,
        /// True selects the imaginary part of a complex component.
        imaginary: bool,
    },
    /// One coherent-SI coordinate bound of a Cartesian Domain axis.
    CartesianBound {
        /// Canonical continuous Domain whose shape is varied.
        domain: Id<kinds::Domain>,
        /// Zero-based physical coordinate axis.
        axis: usize,
        /// Lower or upper side of the axis interval.
        side: BoundarySide,
    },
}

impl From<Id<kinds::Parameter>> for SpatialDesignCoordinate {
    fn from(parameter: Id<kinds::Parameter>) -> Self {
        Self::ModelParameter {
            parameter,
            component: 0,
            imaginary: false,
        }
    }
}
