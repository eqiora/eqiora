//! Closed numerical policy for the bounded 2D ALE FSI remesh transition.
//!
//! Remeshing is neither Semantic Model meaning nor a model-time step.  This
//! module records only the integration and common-solver choices used to move
//! an already accepted physical state between two mesh-bound Realizations.

use std::num::NonZeroUsize;

use eqiora_core::{Diagnostic, DimExponents, DynQuantity};
use eqiora_solver::{LinearOperatorProperties, PreconditionerPolicy, SolverPlan};

use crate::{PositivePhysicalScale, QuadraturePolicy, invalid_realization};

/// Closed coherent-SI normalization profile for a 2D ALE FSI remesh.
///
/// The same characteristic length `L`, velocity `U`, and pressure `P` that
/// normalize the source and target ALE Realizations also normalize every
/// physical transfer obligation. Keeping these typed quantities in the
/// transfer plan prevents a caller from silently choosing unrelated replay
/// units after the projection has run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleFsiRemeshScaleProfile2d {
    length: PositivePhysicalScale,
    velocity: PositivePhysicalScale,
    pressure: PositivePhysicalScale,
}

impl AleFsiRemeshScaleProfile2d {
    /// Validate finite positive `L`, `U`, and `P` quantities.
    ///
    /// # Errors
    /// Returns `EQ0807` when a quantity has the wrong physical dimension or
    /// is not finite and strictly positive.
    pub fn new(
        length: DynQuantity,
        velocity: DynQuantity,
        pressure: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        require_dimension(length, length_dimension(), "length L")?;
        require_dimension(velocity, velocity_dimension(), "velocity U")?;
        require_dimension(pressure, pressure_dimension(), "pressure P")?;
        Ok(Self {
            length: PositivePhysicalScale::new(length)?,
            velocity: PositivePhysicalScale::new(velocity)?,
            pressure: PositivePhysicalScale::new(pressure)?,
        })
    }

    /// Characteristic length `L` in coherent SI units.
    #[must_use]
    pub const fn length(self) -> DynQuantity {
        self.length.quantity()
    }

    /// Characteristic fluid/solid velocity `U` in coherent SI units.
    #[must_use]
    pub const fn velocity(self) -> DynQuantity {
        self.velocity.quantity()
    }

    /// Characteristic absolute pressure `P` in coherent SI units.
    #[must_use]
    pub const fn pressure(self) -> DynQuantity {
        self.pressure.quantity()
    }
}

/// Complete policy for one conservative common-refinement ALE FSI transfer.
///
/// The type names the sole admitted algorithm: absolute displacement is
/// projected first in the material chart, target harmonic geometry is then
/// derived, and coupled velocity plus absolute pressure are projected in their
/// declared charts with the constraints defined by RFC 0065.  Source/target
/// identities remain orchestration inputs and are intentionally not copied
/// into this reusable numerical policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleFsiRemeshTransferPlan2d {
    quadrature: QuadraturePolicy,
    scales: AleFsiRemeshScaleProfile2d,
    solver: SolverPlan,
}

impl AleFsiRemeshTransferPlan2d {
    /// Admit the bounded degree-eight common-refinement integration and solve.
    ///
    /// # Errors
    /// Returns `EQ0807` unless quadrature is the exact five-by-five triangle
    /// Duffy rule and the common solver satisfies [`Self::require_solver_policy`].
    pub fn new(
        quadrature: QuadraturePolicy,
        scales: AleFsiRemeshScaleProfile2d,
        solver: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        if quadrature
            != (QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                points_per_axis: NonZeroUsize::new(5).expect("five is non-zero"),
            })
        {
            return Err(invalid_realization(
                "ALE FSI remesh transfer requires the exact degree-eight triangle Duffy quadrature policy",
            ));
        }
        Self::require_solver_policy(solver)?;
        Ok(Self {
            quadrature,
            scales,
            solver,
        })
    }

    /// Admit one policy for both mass and constrained velocity projections.
    ///
    /// This contract uses SPD mass matrices and a symmetric-indefinite KKT
    /// system. Its multiplier block has an exact zero diagonal, so Jacobi
    /// preconditioning is unavailable for this projection even when a provider
    /// implements Jacobi for other SPD or symmetric-indefinite matrices.
    /// Exact provider availability is checked separately by execution.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the algorithm admits both operator classes and
    /// identity preconditioning is selected for the zero-diagonal KKT system.
    pub fn require_solver_policy(solver: SolverPlan) -> Result<(), Diagnostic> {
        if !solver
            .algorithm()
            .accepts(LinearOperatorProperties::SymmetricPositiveDefinite)
            || !solver
                .algorithm()
                .accepts(LinearOperatorProperties::SymmetricIndefinite)
        {
            return Err(invalid_realization(
                "ALE FSI remesh transfer requires one solver compatible with SPD mass and symmetric-indefinite constrained projections",
            ));
        }
        if solver.preconditioner() != PreconditionerPolicy::Identity {
            return Err(invalid_realization(
                "ALE FSI remesh transfer requires identity preconditioning because its KKT multiplier diagonal is zero",
            ));
        }
        Ok(())
    }

    /// Integration policy used on every common-refinement fragment.
    #[must_use]
    pub const fn quadrature(self) -> QuadraturePolicy {
        self.quadrature
    }

    /// Typed physical normalization shared by both ALE Realizations.
    #[must_use]
    pub const fn scales(self) -> AleFsiRemeshScaleProfile2d {
        self.scales
    }

    /// Sole common solver policy for constrained and unconstrained projections.
    #[must_use]
    pub const fn solver(self) -> SolverPlan {
        self.solver
    }
}

fn require_dimension(
    quantity: DynQuantity,
    expected: DimExponents,
    name: &'static str,
) -> Result<(), Diagnostic> {
    if quantity.dim() != expected {
        return Err(invalid_realization(format!(
            "ALE FSI remesh normalization {name} has dimension [{}], expected [{expected}]",
            quantity.dim()
        )));
    }
    Ok(())
}

const fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn velocity_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn pressure_dimension() -> DimExponents {
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension")
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::DynQuantity;
    use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy};

    fn plan(algorithm: LinearSolver) -> SolverPlan {
        SolverPlan::new(algorithm, 1.0e-12, 1.0e-14, NonZeroUsize::new(500).unwrap())
            .unwrap()
            .with_reduction(ReductionPolicy::Reproducible)
    }

    fn scales() -> AleFsiRemeshScaleProfile2d {
        AleFsiRemeshScaleProfile2d::new(
            DynQuantity::new(2.0, length_dimension()),
            DynQuantity::new(0.5, velocity_dimension()),
            DynQuantity::new(3.0, pressure_dimension()),
        )
        .unwrap()
    }

    #[test]
    fn exact_reference_policy_is_closed() {
        let quadrature = QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
            points_per_axis: NonZeroUsize::new(5).unwrap(),
        };
        for algorithm in [
            LinearSolver::MinimumResidual,
            LinearSolver::BiConjugateGradientStabilized,
            LinearSolver::SparseLu,
        ] {
            let policy = plan(algorithm);
            let accepted = AleFsiRemeshTransferPlan2d::new(quadrature, scales(), policy).unwrap();
            assert_eq!(accepted.quadrature(), quadrature);
            assert_eq!(accepted.scales(), scales());
            assert_eq!(accepted.solver(), policy);
            let error = AleFsiRemeshTransferPlan2d::require_solver_policy(
                policy.with_preconditioner(PreconditionerPolicy::Jacobi),
            )
            .unwrap_err();
            assert!(error.message().contains("multiplier diagonal is zero"));
        }

        assert!(
            AleFsiRemeshTransferPlan2d::new(
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                    points_per_axis: NonZeroUsize::new(4).unwrap(),
                },
                scales(),
                plan(LinearSolver::MinimumResidual),
            )
            .is_err()
        );
        assert!(
            AleFsiRemeshTransferPlan2d::new(
                quadrature,
                scales(),
                plan(LinearSolver::ConjugateGradient),
            )
            .is_err()
        );
        assert!(
            AleFsiRemeshTransferPlan2d::new(
                quadrature,
                scales(),
                plan(LinearSolver::MinimumResidual)
                    .with_preconditioner(PreconditionerPolicy::Jacobi),
            )
            .is_err()
        );
    }

    #[test]
    fn normalization_dimensions_are_closed() {
        let length = DynQuantity::new(1.0, length_dimension());
        let velocity = DynQuantity::new(1.0, velocity_dimension());
        let pressure = DynQuantity::new(1.0, pressure_dimension());
        assert!(AleFsiRemeshScaleProfile2d::new(length, velocity, pressure).is_ok());
        assert!(AleFsiRemeshScaleProfile2d::new(velocity, length, pressure).is_err());
    }
}
