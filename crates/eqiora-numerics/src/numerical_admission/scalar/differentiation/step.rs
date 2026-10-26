//! Parameter products of an accepted step hold its physical predecessor fixed.
use super::*;
use crate::region_assembly::{
    RegionPoint,
    mapping::{RegionAlgebraicCoordinates, RegionSolveInput},
};
mod evaluate;
mod replay;
mod trajectory;

impl CommonSpatialPlan {
    /// Linearize one accepted backward-Euler step at the canonical Model point.
    ///
    /// Selected Parameter changes hold the preceding physical State and both
    /// endpoint times fixed. This is not a derivative through earlier steps.
    /// Outputs may mix complete Fields and reduced Observables, in requested order.
    /// Spatial reductions use the supplied exact-Domain quadratures at the step endpoint. The original step
    /// is reaccepted before its residual and Parameter actions are constructed.
    /// The returned relation/output pair composes with the existing forward and
    /// adjoint sensitivity APIs using its canonical sparse State Jacobian.
    ///
    /// # Errors
    /// Rejects foreign checkpoints, noncanonical or duplicate Parameter selections,
    /// unavailable input derivatives, and unsupported output/realization bindings.
    /// Currently requires dimensional equations on a fixed affine Mesh without
    /// eliminated kinematics. Derivatives through prior steps use the separate trajectory operation.
    pub fn linearize_step(
        &self,
        step: &CommonAcceptedSpatialStep,
        selected: &[Id<kinds::Parameter>],
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let accepted = self.accept_step_product(step, outputs)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, None, outputs, quadratures)?
                .linearize_step(equations, &accepted),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, None, outputs, quadratures)?
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
        self.validate_trajectory_differentiation_request(request, outputs)?;
        if request.accepted_steps()?.get() != 1 || request.output_steps()? != [1] {
            return Err(invalid(
                "fixed-history step products require exactly one requested step and its endpoint",
            ));
        }
        request.state().time_grid()?.advance(1)
    }

    /// Validate the exact Run and Field/reduced Observable roles for trajectory products.
    ///
    /// # Errors
    /// Rejects foreign Plans and unsupported temporal or output bindings.
    pub fn validate_trajectory_differentiation_request(
        &self,
        request: &CommonTransientRunRequest,
        outputs: &[eqiora_core::RawId],
    ) -> Result<(), Diagnostic> {
        self.validate_step_outputs(outputs)?;
        if request.plan().identity() != self.identity() {
            return Err(invalid("step Parameter request belongs to a foreign Plan"));
        }
        request.accepted_steps()?;
        request.output_steps()?;
        Ok(())
    }

    fn validate_step_outputs(&self, outputs: &[eqiora_core::RawId]) -> Result<(), Diagnostic> {
        self.admission
            .temporal
            .as_ref()
            .ok_or_else(|| invalid("step linearization requires a temporal Plan"))?
            .require_backward_euler()?;
        self.differentiation_output_fields(outputs)?;
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
        // Only external canonical history is authenticated against the template.
        let (canonical, mut input) = self.plan.scalar_step_assembly(template, previous, grid)?;
        let history = input.previous.take().expect("validated physical history");
        let (old_mapping, _) = self.plan.scalar_assembly_at(template, previous_time)?;
        let (mapping, input) =
            self.prepare_step_history(&old_mapping, &history, previous_time, grid)?;
        if !canonical.has_same_coordinate_layout(&mapping) {
            return Err(invalid(
                "step Parameter point changed the exact Field constraint layout",
            ));
        }
        Ok((mapping, input))
    }

    fn prepare_step_history(
        &mut self,
        old_mapping: &RegionDofMap<S>,
        history: &BTreeMap<
            eqiora_core::RawId,
            crate::region_assembly::mapping::RecoveredRegionField<S>,
        >,
        previous_time: f64,
        grid: TimeGrid,
    ) -> Result<(RegionDofMap<S>, RegionSolveInput<S>), Diagnostic> {
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
        let (mapping, mut input) = self.plan.scalar_assembly_with_history(
            &self.equations,
            current,
            Some(previous_time),
            Some(history),
            &[],
        )?;
        self.step_coordinates = if old_mapping.has_same_coordinate_layout(&mapping) {
            let identity: std::sync::Arc<str> = self.plan.identity().into();
            Some(
                RegionRealCoordinates::new(&mapping)?
                    .full_keys()
                    .into_iter()
                    .map(
                        |(key, imaginary)| crate::numerical_admission::SpatialStepCoordinate {
                            plan: identity.clone(),
                            key,
                            imaginary,
                        },
                    )
                    .collect(),
            )
        } else {
            None
        };
        input.previous = Some(history.clone());
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
        assembly: Option<crate::region_assembly::nonlinear::RegionAssemblyLinearization>,
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
        let assembly = match assembly {
            Some(assembly) => assembly,
            None => prepared.work.linearize(
                &prepared.plan,
                prepared.plan.target_id(0).expect("reduced target"),
                RegionPoint::Bound(&values),
                &eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND,
            )?,
        };
        self.assembled_linearization(mapping, input, values, assembly, target, initial)
    }

    pub(super) fn attach_step_history(
        &self,
        point: CommonDifferentiationLinearization,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        output: FieldCoefficientLinearization,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let keys = self
            .step_coordinates
            .as_ref()
            .expect("prepared history coordinates");
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
                    "step history requires the exact admitted affine Mesh",
                ));
            }
        };
        let previous = prepared.work.linearize(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Previous(point.relation().accepted_unknowns()),
            &eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND,
        )?;
        // Restrict residual rows only. Prescribed predecessor columns remain
        // independent physical history inputs; their derivatives are supplied
        // by the preceding step's complete-state output projection.
        let full = previous.systems[1].matrix();
        let mut rows =
            vec![std::collections::BTreeMap::<usize, f64>::new(); coordinates.free_count()];
        for global in 0..full.rows() {
            if let Some(row) = coordinates.free_coordinate(global) {
                for entry in full.row_offsets()[global]..full.row_offsets()[global + 1] {
                    *rows[row].entry(full.column_indices()[entry]).or_default() +=
                        full.values()[entry];
                }
            }
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut entries = Vec::new();
        for row in rows {
            for (column, value) in row {
                columns.push(column);
                entries.push(value);
            }
            offsets.push(entries.len());
        }
        let jacobian = eqiora_assembly::CsrMatrix::from_sorted_csr(
            coordinates.free_count(),
            coordinates.full_count(),
            offsets,
            columns,
            entries,
        )?;
        let preceding = coordinates.encode_full(&mapping.normalized_coefficients(
            input.previous.as_ref().expect("prepared physical history"),
        )?)?;
        point.with_step_history(
            jacobian,
            keys.clone(),
            preceding,
            self.time_interval.expect("prepared step interval"),
            output,
        )
    }
}
