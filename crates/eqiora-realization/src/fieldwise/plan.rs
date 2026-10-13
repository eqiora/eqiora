use eqiora_core::Diagnostic;
use eqiora_solver::{LinearOperatorProperties, SolverPlan};

use super::{
    AlgebraicBlock, FieldwiseSpatialDiscretization, SymmetricCongruenceScaling, block_order,
};
use crate::{
    DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy, SpaceFamily, Target,
    invalid_realization,
};

/// Complete field-wise realization selection, independent from model meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldwiseRealizationPlan {
    spatial: FieldwiseSpatialDiscretization,
    scaling: SymmetricCongruenceScaling,
    operator_properties: LinearOperatorProperties,
    solver: SolverPlan,
    target: Target,
    schedule: ExecutionSchedule,
}

impl FieldwiseRealizationPlan {
    /// Construct and cross-validate one complete field-wise realization.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the spatial contract is one admitted complete
    /// family -- continuous Galerkin on imported simplices or cell-centered
    /// finite volume on generated Cartesian cells -- and scaling covers every
    /// Field and constraint-multiplier block exactly once.
    pub fn new(
        spatial: FieldwiseSpatialDiscretization,
        scaling: SymmetricCongruenceScaling,
        operator_properties: LinearOperatorProperties,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let plan = Self {
            spatial,
            scaling,
            operator_properties,
            solver,
            target,
            schedule,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Field-wise spatial selection.
    #[must_use]
    pub const fn spatial(&self) -> &FieldwiseSpatialDiscretization {
        &self.spatial
    }

    /// Explicit symmetric congruence scaling.
    #[must_use]
    pub const fn scaling(&self) -> &SymmetricCongruenceScaling {
        &self.scaling
    }

    /// Mathematical property asserted for the realized linear operator.
    #[must_use]
    pub const fn operator_properties(&self) -> LinearOperatorProperties {
        self.operator_properties
    }

    /// Sole linear solver plan.
    #[must_use]
    pub const fn solver(&self) -> SolverPlan {
        self.solver
    }

    /// Deployment target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// Deployment schedule.
    #[must_use]
    pub const fn schedule(&self) -> ExecutionSchedule {
        self.schedule
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        let discretization = self.spatial.discretization;
        let admitted = match (
            discretization.method(),
            discretization.mesh(),
            discretization.quadrature(),
        ) {
            (
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial { .. },
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension, ..
                },
            ) if spatial_dimension.get() == 2 => self.spatial.field_spaces.iter().all(|binding| {
                matches!(
                    binding.space.family(),
                    SpaceFamily::ContinuousLagrange { .. } | SpaceFamily::SimplexP1Bubble
                )
            }),
            (
                DiscretizationMethod::CellCenteredFiniteVolume,
                MeshPolicy::GeneratedUniform { .. } | MeshPolicy::SuppliedCartesian { .. },
                QuadraturePolicy::CellCentroid,
            ) => self
                .spatial
                .field_spaces
                .iter()
                .all(|binding| binding.space.family() == SpaceFamily::CellConstant),
            _ => false,
        };
        if !admitted {
            return Err(invalid_realization(
                "field-wise realization requires either continuous Galerkin with an imported affine-simplex mesh, Duffy triangle quadrature, and continuous spaces, or cell-centered finite volume with a generated or supplied Cartesian mesh, cell-centroid quadrature, and cell-constant spaces",
            ));
        }
        if matches!(self.schedule, ExecutionSchedule::RealTime { .. })
            && matches!(self.target, Target::CudaGpu { .. })
        {
            return Err(invalid_realization(
                "the field-wise CUDA target has no declared real-time scheduling contract",
            ));
        }
        let mut expected = self
            .spatial
            .field_spaces
            .iter()
            .map(|binding| AlgebraicBlock::Field(binding.field))
            .chain(self.spatial.constraints.iter().map(|constraint| {
                AlgebraicBlock::ConstraintMultiplier {
                    field: constraint.field(),
                }
            }))
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| block_order(*left, *right));
        let actual = self
            .scaling
            .block_scales
            .iter()
            .map(|entry| entry.block)
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(invalid_realization(
                "congruence scaling must cover every Field and constraint-multiplier block exactly once",
            ));
        }
        Ok(())
    }
}
