//! Parameter products of an accepted step hold its physical predecessor fixed.
use super::*;
use crate::region_assembly::{
    RegionPoint,
    mapping::{RegionAlgebraicCoordinates, RegionSolveInput},
};
mod evaluate;
mod replay;

impl CommonSpatialPlan {
    /// Linearize one accepted backward-Euler step at the canonical Model point.
    ///
    /// Selected Parameter changes hold the preceding physical State and both
    /// endpoint times fixed. This is not a derivative through earlier steps.
    /// Outputs are complete Fields, in the requested order. The original step
    /// is reaccepted before its residual and Parameter actions are constructed.
    /// The returned relation/output pair composes with the existing forward and
    /// adjoint sensitivity APIs using its canonical sparse State Jacobian.
    ///
    /// # Errors
    /// Rejects foreign checkpoints, noncanonical or duplicate Parameter selections,
    /// unavailable input derivatives, and unsupported output/realization bindings.
    /// Currently requires dimensional equations on a fixed affine Mesh without
    /// eliminated kinematics. Temporal Observables and derivatives through prior steps remain
    /// separate, unavailable compositions.
    pub fn linearize_step(
        &self,
        step: &CommonAcceptedSpatialStep,
        selected: &[Id<kinds::Parameter>],
        outputs: &[eqiora_core::RawId],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let accepted = self.accept_step_product(step, outputs)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, None, outputs, &HashMap::new())?
                .linearize_step(equations, &accepted),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, None, outputs, &HashMap::new())?
                .linearize_step(equations, &accepted),
            _ => Err(invalid(
                "step linearization requires exact spatial equations",
            )),
        }
    }
    fn accept_step_product(
        &self,
        step: &CommonAcceptedSpatialStep,
        outputs: &[eqiora_core::RawId],
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic> {
        self.validate_step_outputs(outputs)?;
        let mut accepted = step.clone();
        accepted.reaccept_spatial(self)?;
        Ok(accepted)
    }

    /// Validate the exact step request and output roles without solving.
    /// This binds static request meaning, not candidate residual acceptance.
    ///
    /// # Errors
    /// Rejects foreign Plans, unsupported temporal policies, schedules and outputs.
    pub fn validate_step_differentiation_request(
        &self,
        request: &CommonTransientRunRequest,
        outputs: &[eqiora_core::RawId],
    ) -> Result<(), Diagnostic> {
        self.validate_step_request(request, outputs).map(|_| ())
    }

    fn validate_step_request(
        &self,
        request: &CommonTransientRunRequest,
        outputs: &[eqiora_core::RawId],
    ) -> Result<TimeGrid, Diagnostic> {
        self.validate_step_outputs(outputs)?;
        if request.plan().identity() != self.identity() {
            return Err(invalid("step Parameter request belongs to a foreign Plan"));
        }
        if request.accepted_steps()?.get() != 1 || request.output_steps()? != [1] {
            return Err(invalid(
                "fixed-history step products require exactly one requested step and its endpoint",
            ));
        }
        request.state().time_grid()?.advance(1)
    }

    fn validate_step_outputs(&self, outputs: &[eqiora_core::RawId]) -> Result<(), Diagnostic> {
        self.admission
            .temporal
            .as_ref()
            .ok_or_else(|| invalid("step linearization requires a temporal Plan"))?
            .require_backward_euler()?;
        if outputs
            .iter()
            .any(|id| !self.fields().any(|(field, _)| field.erase() == *id))
        {
            return Err(invalid(
                "step linearization currently requires complete Field outputs",
            ));
        }
        Ok(())
    }
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn prepare_step(
        &mut self,
        template: &ExecutableSpatialEquations<S>,
        previous: &CommonState,
        grid: TimeGrid,
    ) -> Result<(RegionDofMap<S>, RegionSolveInput<S>), Diagnostic> {
        if self
            .equations
            .regions
            .iter()
            .any(|region| !region.form.kinematics().is_empty())
        {
            return Err(invalid(
                "step products through eliminated kinematics require their output pullback",
            ));
        }
        if self
            .equations
            .regions
            .iter()
            .any(|region| region.form.motion().is_some())
        {
            return Err(invalid(
                "step products through moving geometry require its retained geometry partials",
            ));
        }
        let previous_time = previous.require_time_s()?;
        let current = grid.time(
            self.plan
                .admission
                .temporal
                .as_ref()
                .expect("validated temporal Plan")
                .require_backward_euler()?
                .value(),
        )?;
        self.time_interval = Some([previous_time, current]);
        // Authenticate history at its original Parameter point. Its physical
        // values stay fixed even when the new point changes essential traces.
        let (canonical, mut input) = self.plan.scalar_step_assembly(template, previous, grid)?;
        let history = input.previous.take().expect("validated physical history");
        let (mapping, mut input) = self.plan.scalar_assembly_with_history(
            &self.equations,
            current,
            Some(previous_time),
            Some(&history),
            &[],
        )?;
        if !canonical.has_same_coordinate_layout(&mapping) {
            return Err(invalid(
                "step Parameter point changed the exact Field constraint layout",
            ));
        }
        input.previous = Some(history);
        Ok((mapping, input))
    }

    fn linearize_step(
        mut self,
        template: &ExecutableSpatialEquations<S>,
        step: &CommonAcceptedSpatialStep,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let (mapping, input) =
            self.prepare_step(template, step.previous_state(), step.state().time_grid()?)?;
        let fields = self.plan.validate_state_fields(
            &mapping,
            &input,
            step.state()
                .spatial_values()
                .ok_or_else(|| invalid("step lacks spatial Fields"))?,
        )?;
        let coordinates = RegionRealCoordinates::new(&mapping)?;
        let values = coordinates.encode(&mapping.physical_predictor(&fields)?)?;
        self.linearize_step_values(
            &mapping,
            &input,
            values,
            step.residual.expect("reaccepted step").1,
            None,
        )
    }

    fn linearize_step_values(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        values: Vec<f64>,
        target: f64,
        initial: Option<crate::nonlinear_execution::NewtonInitialPoint>,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let coordinates = RegionRealCoordinates::new(mapping)?;
        let prepared = match self.plan.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                coordinates.prepare_assembly(mesh.mesh(), input)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                coordinates.prepare_assembly(mesh.mesh(), input)?
            }
            _ => {
                return Err(invalid(
                    "step Parameter products require the exact admitted affine Mesh",
                ));
            }
        };
        let assembly = prepared.work.linearize(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Bound(&values),
            &eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND,
        )?;
        self.assembled_linearization(mapping, input, values, assembly, target, initial)
    }
}
