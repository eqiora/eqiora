//! Numerical contracts shared by more than one scientific family.
//!
//! This namespace is intentionally narrow: it owns meshes, discrete spaces
//! and shared Field representations, boundary classification, local operators,
//! assembled linearization, design coordinates, scalar spatial expressions,
//! and validated step counts. A contract that belongs to one scientific
//! family stays with that family.

pub use crate::assembled_linearization::AssembledLinearizedRelation;
pub use crate::canonical_boundary::{
    CartesianBoundaryEntry, CartesianBoundaryInventory, PhysicalBoundaryDisposition,
    PhysicalBoundaryQuantity, PrescribedBoundaryLaw,
};
pub use crate::discrete_space::{
    CellConstantSpace, DiscreteSpace, HypercubeQ1Space, LocalDof, SimplexP1BubbleSpace,
    SimplexP1Space,
};
pub use crate::operator::LocalOperator;
pub use crate::simplicial_elliptic::SimplicialP1Field;
pub use crate::spatial_design::SpatialDesignCoordinate;
pub use crate::spatial_expression::ScalarSpatialExpression;
pub use crate::step_count::NonZeroStepCount;
