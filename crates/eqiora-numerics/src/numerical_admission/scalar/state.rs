use super::*;
use crate::region_assembly::mapping::{
    RecoveredRegionField, RegionAlgebraicCoordinates, RegionDofMap, RegionSolveInput,
};
use crate::spatial_expression::Coefficient;

impl CommonSpatialPlan {
    /// Identity of the exact stationary numerical request, including its seed.
    pub fn run_identity(&self, initial: Option<&CommonState>) -> Result<String, Diagnostic> {
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "stationary request identity requires a stationary Plan",
            ));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, self.identity().as_bytes());
        match initial {
            Some(state) => {
                self.require_static_state(state)?;
                bytes.push(1);
                push_framed(&mut bytes, state.identity().as_bytes());
            }
            None => bytes.push(0),
        }
        Ok(domain_separated_identity(
            b"eqiora.common-static-run-request/v1\0",
            &bytes,
        ))
    }

    fn require_static_state(&self, state: &CommonState) -> Result<(), Diagnostic> {
        if self.admission.temporal.is_some()
            || state.time_s().is_some()
            || state.state_space_identity() != self.identity()
        {
            return Err(invalid(
                "stationary execution requires a timeless State owned by its exact Plan",
            ));
        }
        Ok(())
    }

    pub(crate) fn static_initial_fields(
        &self,
        state: &CommonState,
    ) -> Result<BTreeMap<eqiora_core::RawId, Vec<f64>>, Diagnostic> {
        self.require_static_state(state)?;
        self.state_fields(state)?
            .map(|(id, _, _, values)| Ok((id.erase(), values.to_vec())))
            .collect()
    }

    pub(super) fn state_assembly<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        time_s: Option<f64>,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        match (self.admission.temporal, time_s) {
            (Some(_), Some(time)) if time.is_finite() && time >= 0.0 => {
                self.scalar_assembly_at(equations, time)
            }
            (None, None) => self.stationary_assembly(equations),
            _ => Err(invalid(
                "State model Time must match the Plan: absent for stationary, finite and nonnegative for transient",
            )),
        }
    }

    pub(super) fn stationary_assembly<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
    ) -> Result<(RegionDofMap<S>, RegionSolveInput<S>), Diagnostic> {
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "stationary spatial State requires a stationary Plan",
            ));
        }
        equations.static_assembly(
            self.admission.resources(),
            &self.admission.discretizations,
            &self.admission.system_scaling,
            self.admission.operator_properties,
        )
    }

    pub(super) fn validate_state_fields<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        values: &[f64],
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>, Diagnostic> {
        let history = self.physical_fields(mapping, values)?;
        mapping.validate_prescribed_fields(&history)?;
        for (key, expected) in &input.prescribed_states {
            if history
                .get(&key.field)
                .and_then(|field| field.coefficients.get(key))
                != Some(expected)
            {
                return Err(invalid(
                    "spatial State contradicts prescribed displacement boundary values",
                ));
            }
        }
        Ok(history)
    }

    fn validate_typed_state<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        time_s: Option<f64>,
        values: &[f64],
        previous: Option<&SpatialHistory>,
    ) -> Result<(), Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let (mapping, input) = match previous {
            Some(previous) => self.scalar_history_assembly(
                equations,
                previous,
                time_s.ok_or_else(|| invalid("step State requires model Time"))?,
            )?,
            None => self.state_assembly(equations, time_s)?,
        };
        let history = self.validate_state_fields(&mapping, &input, values)?;
        if previous.is_some() {
            self.accept_step_fields(&mapping, &input, &history, None)?;
        }
        let mut temporal = crate::form_compiler::region::TemporalStructure {
            differential: Default::default(),
            positive_own_capacity: true,
        };
        for (form, _) in &input.forms {
            let region = form.temporal_structure()?;
            temporal.differential.extend(region.differential);
            temporal.positive_own_capacity &= region.positive_own_capacity;
        }
        if let Some(time_s) = time_s {
            let initial_partial = || {
                let time = eqiora_ir::ScalarSymbolCoordinate::for_value(
                    eqiora_schema::kernel::SymbolRef::Time,
                    &eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, TIME)
                        .expect("real scalar Time"),
                )?
                .remove(0);
                self.scalar_assembly_with_history(equations, time_s, None, None, Some(&time))
            };
            match self.admission.resources() {
                NativeMeshResources::Cartesian { mesh, .. } => mapping.validate_temporal_state(
                    mesh.mesh(),
                    input,
                    &history,
                    &temporal,
                    self.linear(),
                    initial_partial,
                )?,
                NativeMeshResources::GmshSimplicial { mesh, .. }
                | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping
                    .validate_temporal_state(
                        mesh.mesh(),
                        input,
                        &history,
                        &temporal,
                        self.linear(),
                        initial_partial,
                    )?,
                _ => return Err(invalid("temporal State requires its exact nodal Mesh")),
            }
        }
        Ok(())
    }

    pub(crate) fn scalar_state(
        &self,
        time_s: Option<f64>,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        self.spatial_state(time_s, values, None)
    }

    pub(in crate::numerical_admission) fn spatial_state(
        &self,
        time_s: Option<f64>,
        values: Vec<f64>,
        previous: Option<SpatialHistory>,
    ) -> Result<CommonState, Diagnostic> {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                self.validate_typed_state(equations, time_s, &values, previous.as_ref())?
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                self.validate_typed_state(equations, time_s, &values, previous.as_ref())?
            }
            _ => return Err(invalid("missing spatial equations")),
        }
        CommonState::new(
            self.identity().to_owned(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            CommonStateKind::Spatial {
                values: values.into_boxed_slice(),
                previous,
            },
        )
    }
}
