//! Original stationary Newton acceptance and its retained Jacobian feed implicit actions.
use super::*;
use crate::region_assembly::{
    mapping::{
        RecoveredRegionField, RegionAlgebraicCoordinates, RegionRealCoordinates, RegionSolveInput,
    },
    nonlinear::RegionAssemblyLinearization,
};

type NonlinearInputs<S> = (
    RegionDofMap<S>,
    RegionSolveInput<S>,
    BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>,
);

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    pub(super) fn solve_strategy(
        self,
        initial: Option<&CommonState>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        if self.plan.admission.nonlinear.is_some() {
            self.solve_nonlinear(initial, backend)
        } else if initial.is_none() {
            self.solve_affine(backend)
        } else {
            Err(invalid(
                "affine spatial differentiation does not consume an initial State",
            ))
        }
    }

    pub(super) fn accept_strategy(
        self,
        initial: Option<&CommonState>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        if self.plan.admission.nonlinear.is_some() {
            self.accept_nonlinear(initial, candidate)
        } else if initial.is_none() {
            self.accept_affine(candidate)
        } else {
            Err(invalid(
                "affine spatial differentiation does not consume an initial State",
            ))
        }
    }

    fn nonlinear_inputs(&self, initial: &CommonState) -> Result<NonlinearInputs<S>, Diagnostic> {
        self.plan.static_initial_fields(initial)?;
        let admission = &self.plan.admission;
        let (mapping, input) = self.equations.static_assembly(
            admission.resources(),
            &admission.discretizations,
            &admission.system_scaling,
            admission.operator_properties,
        )?;
        let seed = self.plan.physical_fields(
            &mapping,
            initial.spatial_values().ok_or_else(|| {
                invalid("Newton differentiation requires physical spatial seed Fields")
            })?,
        )?;
        Ok((mapping, input, seed))
    }

    pub(super) fn solve_nonlinear(
        &self,
        initial: Option<&CommonState>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let initial = initial.ok_or_else(|| {
            invalid("nonlinear spatial differentiation requires its exact initial State")
        })?;
        let (mapping, input, seed) = self.nonlinear_inputs(initial)?;
        let structure = self.equations.algebraic_structure(None)?;
        let linear = &self.plan.admission.linear;
        let checked = linear.checked_backend(backend, Some(&structure))?;
        let solved = self
            .plan
            .solve_region_newton(
                &mapping,
                &input,
                &seed,
                LinearSolveRequest::new(&checked, linear.solver),
            )?
            .solve;
        let initial_point = crate::nonlinear_execution::NewtonInitialPoint {
            residual_norm: solved.evidence.initial_residual_norm,
            at_seed: solved.values
                == RegionRealCoordinates::new(&mapping)?
                    .encode(&mapping.physical_predictor(&seed)?)?,
        };
        let linearization = self.nonlinear_linearization(
            &mapping,
            &input,
            solved.values,
            solved.assembly,
            solved.evidence.residual_target,
            initial_point,
        )?;
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            Primal::Nonlinear {
                initial: initial.clone().into(),
                iterations: solved.evidence.iterations,
                linear_solves: solved.evidence.linear_solves,
            },
        ))
    }

    pub(super) fn accept_nonlinear(
        &self,
        initial: Option<&CommonState>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let initial = initial.ok_or_else(|| {
            invalid("nonlinear spatial differentiation requires its exact initial State")
        })?;
        let (mapping, input, seed) = self.nonlinear_inputs(initial)?;
        let fields = RegionRealCoordinates::new(&mapping)?.recover(candidate)?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.plan.admission.resources() else {
            return Err(invalid(
                "spatial Parameter differentiation requires exact Cartesian resources",
            ));
        };
        let accepted = mapping.accept_nonlinear_coefficients(
            mesh.mesh(),
            &input,
            &fields,
            &seed,
            self.plan.admission.nonlinear.expect("Newton Plan"),
        )?;
        self.nonlinear_linearization(
            &mapping,
            &input,
            accepted.values,
            accepted.assembly,
            accepted.residual_target,
            accepted.initial,
        )
    }

    fn nonlinear_linearization(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        values: Vec<f64>,
        assembly: RegionAssemblyLinearization,
        target: f64,
        initial: crate::nonlinear_execution::NewtonInitialPoint,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let [reduced, full]: [_; 2] = assembly.systems.try_into().map_err(|_| {
            invalid("nonlinear sensitivity requires exact reduced and full Jacobian targets")
        })?;
        let [original, _]: [_; 2] = assembly.residuals.try_into().map_err(|_| {
            invalid("nonlinear sensitivity requires exact reduced and full residual targets")
        })?;
        let (partial, output) = self.output_partials(mapping, &values, |coordinate| {
            self.mapped_parameter_partial(mapping, &full, input, coordinate, &values)
        })?;
        let relation = AssembledLinearizedRelation::from_point(
            reduced.matrix().clone(),
            values,
            original,
            self.design.clone(),
            self.values.clone(),
            partial,
            LinearOperatorProperties::General,
        )?;
        Ok(CommonDifferentiationLinearization::from_spatial(
            relation,
            eqiora_core::ScalarDomain::Real,
            output,
            target,
            Some(initial),
        ))
    }
}
