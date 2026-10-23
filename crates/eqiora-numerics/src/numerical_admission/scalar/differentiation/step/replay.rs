//! Candidate acceptance reuses ordinary step equations without solving again.
use super::*;

impl CommonSpatialPlan {
    /// Validate candidate free coordinates at a fixed-history Parameter point.
    ///
    /// Reaccepts the original canonical step, then checks the supplied candidate
    /// against the changed current equations with the original physical history.
    /// This executes no solve and authenticates no producer history. Returned
    /// partials and Newton seed facts use the same owners as live evaluation.
    ///
    /// # Errors
    /// Rejects foreign steps, malformed inputs or candidates, unsupported bindings,
    /// and candidates that fail the original residual or constraint checks.
    pub fn validate_differentiated_step_values(
        &self,
        step: &CommonAcceptedSpatialStep,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let accepted = self.accept_step_product(step, outputs)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .accept_step_point(equations, &accepted, candidate),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .accept_step_point(equations, &accepted, candidate),
            _ => Err(invalid("step acceptance requires exact spatial equations")),
        }
    }
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn accept_step_point(
        mut self,
        template: &ExecutableSpatialEquations<S>,
        step: &CommonAcceptedSpatialStep,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let (mapping, input) = self.prepare_step(template, step)?;
        let coordinates = RegionRealCoordinates::new(&mapping)?;
        let fields = coordinates.recover(candidate)?;
        let (_, target, _, _, seed_residual) = self
            .plan
            .accept_step_fields(&mapping, &input, &fields, None)?;
        let initial = seed_residual
            .map(|residual_norm| {
                let seed = input.previous.as_ref().expect("validated step history");
                Ok::<_, Diagnostic>(crate::nonlinear_execution::NewtonInitialPoint {
                    residual_norm,
                    at_seed: candidate == coordinates.encode(&mapping.physical_predictor(seed)?)?,
                })
            })
            .transpose()?;
        self.linearize_step_values(&mapping, &input, candidate.to_vec(), target, initial)
    }
}
