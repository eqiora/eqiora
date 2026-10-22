//! Rate elimination for a regular rate-affine residual, retaining original equations.
use super::*;
use crate::region_assembly::nonlinear::RegionAssemblyLinearization;
use eqiora_solver::{FixedOrderInnerProduct, SERIAL_LINEAR_EXECUTION, SolveReport};
use eqiora_time::ConstantDerivativeMatrixProof;

struct RatePoint {
    rate: Vec<f64>,
    report: SolveReport,
    mass: RegionAssemblyLinearization,
}

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    /// Solve F(t,y,v)=0 for v only when its current rate Jacobian is invertible.
    pub(in crate::numerical_admission) fn explicit_rate(
        &self,
        time: f64,
        state: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<(Vec<f64>, SolveReport), Diagnostic> {
        let point = self.rate_point(time, state, backend)?;
        Ok((point.rate, point.report))
    }

    fn rate_point(
        &self,
        time: f64,
        state: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<RatePoint, Diagnostic> {
        let mass = self.regular_mass(time, state)?;
        let (rate, report) = self.solve_rate_action(&mass, backend)?;
        let mut residual = vec![0.; self.dimension];
        self.residual(time, state, &rate, &mut residual)?;
        let norm = FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &residual)?;
        if norm > report.residual_target() {
            return Err(invalid(
                "explicit spatial rate violates the original implicit residual",
            ));
        }
        Ok(RatePoint { rate, report, mass })
    }

    /// Implicit differentiation includes the State derivative of the mass action.
    pub(in crate::numerical_admission) fn explicit_rate_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate(direction)?;
        let RatePoint { rate, mut mass, .. } = self.rate_point(time, state, backend)?;
        // F_y is evaluated at the solved rate, so M_y[direction]*rate is retained.
        self.residual_jvp(
            time,
            state,
            &rate,
            direction,
            &vec![0.; self.dimension],
            &mut mass.residuals[0],
        )?;
        let (tangent, report) = self.solve_rate_action(&mass, backend)?;
        let mut residual = vec![0.; self.dimension];
        self.residual_jvp(time, state, &rate, direction, &tangent, &mut residual)?;
        if FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &residual)?
            > report.residual_target()
        {
            return Err(invalid(
                "explicit spatial rate tangent violates the original implicit action",
            ));
        }
        Ok(tangent)
    }

    fn regular_mass(
        &self,
        time: f64,
        state: &[f64],
    ) -> Result<RegionAssemblyLinearization, Diagnostic> {
        self.validate(state)?;
        let (mapping, input) = self.plan.scalar_assembly_at(self.equations, time)?;
        // This traversal proves affinity in rates even for State-dependent storage.
        let temporal = input.temporal_structure()?;
        let algebraic = mapping.temporal_coordinate_partition(&temporal)?.0;
        let mass = self.linearize(
            time,
            state,
            &vec![0.; self.dimension],
            FieldPullback::Uniform {
                current: None,
                rate: Some(1.),
            },
        )?;
        if !algebraic.is_empty() || !temporal.positive_own_capacity {
            let coordinates = (0..self.dimension).collect::<Vec<_>>();
            let proof = ConstantDerivativeMatrixProof::new(
                self.dimension,
                crate::region_assembly::mapping::square_projection(
                    mass.systems[0].matrix(),
                    &coordinates,
                )?,
            )?;
            if proof.exact_rank() != self.dimension {
                return Err(invalid(
                    "explicit spatial time integration requires an invertible rate Jacobian; this point retains algebraic constraints or singular storage",
                ));
            }
        }
        Ok(mass)
    }

    fn solve_rate_action(
        &self,
        action: &RegionAssemblyLinearization,
        backend: &dyn LinearSolverBackend,
    ) -> Result<(Vec<f64>, SolveReport), Diagnostic> {
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        action.solve_correction(
            0,
            LinearSolveRequest::new(&checked, self.plan.linear()),
            self.plan.admission.linear.workers,
        )
    }
}
