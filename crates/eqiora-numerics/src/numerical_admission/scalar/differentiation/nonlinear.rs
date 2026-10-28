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
                None,
            )?
            .solve;
        let initial_point = crate::nonlinear_execution::NewtonInitialPoint {
            residual_norm: solved.evidence.initial_residual_norm,
            at_seed: solved.values
                == RegionRealCoordinates::new(&mapping)?
                    .encode(&mapping.physical_predictor(&seed)?)?,
        };
        let linearization = self.assembled_linearization(
            &mapping,
            &input,
            solved.values,
            solved.assembly,
            solved.evidence.residual_target()?,
            Some(initial_point),
        )?;
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            Primal::Nonlinear {
                initial: Some(initial.clone().into()),
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
        self.assembled_linearization(
            &mapping,
            &input,
            accepted.values,
            accepted.assembly,
            accepted.residual_target,
            Some(accepted.initial),
        )
    }

    pub(super) fn assembled_linearization(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        values: Vec<f64>,
        assembly: RegionAssemblyLinearization,
        target: f64,
        initial: Option<crate::nonlinear_execution::NewtonInitialPoint>,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let [reduced, full]: [_; 2] = assembly.systems.try_into().map_err(|_| {
            invalid("spatial sensitivity requires exact reduced and full Jacobian targets")
        })?;
        let [original, _]: [_; 2] = assembly.residuals.try_into().map_err(|_| {
            invalid("spatial sensitivity requires exact reduced and full residual targets")
        })?;
        let layout = RegionRealCoordinates::new(mapping)?;
        let columns = self.coordinates.len();
        let mut history_direct = self
            .step_coordinates
            .as_ref()
            .map(|_| {
                layout
                    .full_count()
                    .checked_mul(columns)
                    .map(|size| vec![0.; size])
                    .ok_or_else(|| invalid("state output Parameter shape overflows"))
            })
            .transpose()?;
        let mut column = 0;
        let (partial, output) =
            self.output_partials(mapping, &values, values.len(), |coordinate| {
                let (residual, lift) =
                    self.mapped_parameter_partial(mapping, &full, input, coordinate, &values)?;
                if let Some(direct) = &mut history_direct {
                    for (row, value) in lift.iter().enumerate() {
                        direct[row * columns + column] = *value;
                    }
                }
                column += 1;
                Ok((residual, lift))
            })?;
        let state_output = history_direct
            .map(|direct| {
                FieldCoefficientLinearization::new(
                    layout.encode_full(&mapping.lift(&layout.decode(&values)?, false)?)?,
                    values.len(),
                    columns,
                    (0..layout.full_count())
                        .map(|global| layout.free_coordinate(global))
                        .collect(),
                    direct,
                )
            })
            .transpose()?;
        // For affine forms the accepted State Jacobian is the original operator,
        // so retain its mathematical certificate (including mixed MINRES pairs).
        // A nonlinear Jacobian has no symmetry promise from primal row pairing.
        let properties = if self
            .equations
            .regions
            .iter()
            .any(|region| region.form.is_nonlinear())
        {
            LinearOperatorProperties::General
        } else {
            input.operator_properties
        };
        let relation = AssembledLinearizedRelation::from_point(
            reduced.matrix().clone(),
            values,
            original,
            self.design.clone(),
            self.values.clone(),
            partial,
            properties,
        )?;
        let point = CommonDifferentiationLinearization::from_spatial(
            relation,
            eqiora_core::ScalarDomain::Real,
            output,
            eqiora_differentiation::PrimalAcceptance::Residual(target),
            initial,
        );
        match state_output {
            Some(output) => self.attach_step_history(point, mapping, input, output),
            None => Ok(point),
        }
    }
}
