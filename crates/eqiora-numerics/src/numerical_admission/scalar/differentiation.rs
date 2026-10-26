//! Execution and differentiated replay consume one exact assembled spatial system.
use super::super::differentiation::{Output, Primal};
use super::regions::{ExecutableSpatialEquations, StaticRegionSystem, StaticSpatialSystem};
use super::*;
use crate::finalized_spatial::FinalizedLinearCore;
use crate::spatial_expression::Coefficient;
use eqiora_core::{Id, entity::kinds};
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;
mod affine;
mod nonlinear;
mod outputs;
mod partials;
mod scalar;
mod step;
use crate::region_assembly::mapping::{RegionDofMap, RegionRealCoordinates};
use scalar::{DifferentiationScalar, coordinates};

impl CommonSpatialPlan {
    /// Accept a complete typed Parameter point through this Plan's Mesh and policies.
    /// Omitted values select the canonical Model point; unselected Parameters stay fixed.
    /// Ordered outputs may mix complete Fields and reduced Observables. Spatial
    /// integrals require explicit rules for their exact integration Domains.
    /// Stationary Newton Plans require their exact initial State; affine Plans
    /// take no seed. Parameter changes retain free seed coordinates and apply
    /// the new point's essential boundary values.
    pub fn differentiate(
        &self,
        initial: Option<&CommonState>,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .solve(initial, backend),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .solve(initial, backend),
            _ => Err(invalid(
                "spatial differentiation requires its exact typed equations",
            )),
        }
    }

    /// Reaccept supplied algebraic coordinates without solving or creating a producer report.
    /// Uses the same Parameter point, original residual, boundary lift and partial actions.
    pub fn validate_differentiated_values(
        &self,
        initial: Option<&CommonState>,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .accept_candidate(initial, candidate),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .accept_candidate(initial, candidate),
            _ => Err(invalid(
                "spatial differentiation requires its exact typed equations",
            )),
        }
    }

    pub(in crate::numerical_admission) fn prepare_differentiation<S: DifferentiationScalar>(
        &self,
        template: &ExecutableSpatialEquations<S>,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<PreparedPoint<'_, S>, Diagnostic> {
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "stationary Parameter differentiation requires a static Plan",
            ));
        }
        if !matches!(
            self.admission.resources(),
            NativeMeshResources::Cartesian { .. }
        ) {
            return Err(invalid(
                "spatial Parameter differentiation requires exact Cartesian resources",
            ));
        }
        self.prepare_parameter_point(template, selected, values, outputs, quadratures)
    }

    fn differentiation_output_fields(
        &self,
        outputs: &[eqiora_core::RawId],
    ) -> Result<Vec<eqiora_core::RawId>, Diagnostic> {
        let unique = outputs
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if outputs.is_empty() || unique.len() != outputs.len() {
            return Err(invalid(
                "spatial outputs require nonempty distinct selections",
            ));
        }
        let mut selected_fields = Vec::new();
        for &output in outputs {
            if self.fields().any(|(field, _)| field.erase() == output) {
                selected_fields.push(output);
            } else if let Some(observable) = output.downcast::<kinds::Observable>() {
                if self
                    .kernel()
                    .observable_output_support(observable)?
                    .is_some()
                {
                    return Err(invalid(
                        "spatial products require reduced Observable outputs",
                    ));
                }
            } else {
                return Err(invalid(
                    "spatial output is not an exact Plan Field or Model Observable",
                ));
            }
        }
        Ok(selected_fields)
    }

    fn prepare_parameter_point<S: DifferentiationScalar>(
        &self,
        template: &ExecutableSpatialEquations<S>,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<PreparedPoint<'_, S>, Diagnostic> {
        if self.harmonic.is_some() {
            return Err(invalid(
                "harmonic solved-response derivatives require the retained reduction chain and are not admitted",
            ));
        }
        if !matches!(
            self.admission.system_scaling,
            eqiora_realization::SystemScaling::Dimensional
        ) {
            return Err(invalid(
                "selected-Parameter differentiation requires dimensional system scaling",
            ));
        }
        if self.admission.spatial.scalar_constraint().is_some() {
            return Err(invalid(
                "constrained TPFA differentiation has no admitted linearization",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let (design, canonical) = crate::parameter_point::canonical_point(self.kernel(), selected)?;
        let values = values.unwrap_or(&canonical).to_vec();
        let point = crate::parameter_point::ParameterPoint::bind(self.kernel(), selected, &values)?;
        let mut equations = template.clone();
        for region in &mut equations.regions {
            region.form = region.form.bind_inputs(&point)?;
        }
        let selected_fields = self.differentiation_output_fields(outputs)?;
        Ok(PreparedPoint {
            plan: self,
            time_interval: None,
            step_coordinates: None,
            equations,
            selected_fields,
            point,
            outputs: outputs.to_vec(),
            quadratures: quadratures.clone(),
            design,
            values,
            coordinates: crate::parameter_point::coordinates(self.kernel(), selected)?,
        })
    }
}

pub(in crate::numerical_admission) struct PreparedPoint<'a, S: DifferentiationScalar> {
    plan: &'a CommonSpatialPlan,
    time_interval: Option<[f64; 2]>,
    step_coordinates: Option<Vec<crate::numerical_admission::SpatialStepCoordinate>>,
    equations: ExecutableSpatialEquations<S>,
    coordinates: Vec<ScalarSymbolCoordinate>,
    design: Vec<SpatialDesignCoordinate>,
    values: Vec<f64>,
    selected_fields: Vec<eqiora_core::RawId>,
    point: crate::parameter_point::ParameterPoint,
    outputs: Vec<eqiora_core::RawId>,
    quadratures: HashMap<Id<kinds::Domain>, QuadratureRule>,
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    fn solve(
        self,
        initial: Option<&CommonState>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        S::solve_point(self, initial, backend)
    }

    fn accept_candidate(
        self,
        initial: Option<&CommonState>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        S::accept_point(self, initial, candidate)
    }

    /// Output selection and boundary-chain products have one owner for affine
    /// and nonlinear accepted points; the residual algorithm remains separate.
    fn output_partials(
        &self,
        mapping: &RegionDofMap<S>,
        state_coordinates: &[f64],
        mut partial: impl FnMut(&ScalarSymbolCoordinate) -> Result<(Vec<f64>, Vec<f64>), Diagnostic>,
    ) -> Result<(Vec<f64>, Output), Diagnostic> {
        let layout = RegionRealCoordinates::new(mapping)?;
        let physical = layout.decode(state_coordinates)?;
        let full = mapping.lift(&physical, false)?;
        let keys = if self.selected_fields.is_empty() {
            Vec::new()
        } else {
            mapping.selected_field_keys(&self.selected_fields)?
        };
        let count = self.coordinates.len();
        let mut observations =
            self.observation_partials(mapping, &full, state_coordinates.len())?;
        let output_coordinates = keys
            .iter()
            .flat_map(|key| layout.global_coordinates(*key))
            .collect::<Vec<_>>();
        let output_count = output_coordinates.len();
        let mut residual = vec![
            0.;
            state_coordinates.len().checked_mul(count).ok_or_else(
                || invalid("spatial Parameter Jacobian shape overflow")
            )?
        ];
        let mut direct = vec![
            0.;
            output_count.checked_mul(count).ok_or_else(|| invalid(
                "Field Parameter Jacobian shape overflow"
            ))?
        ];
        for (column, coordinate) in self.coordinates.iter().enumerate() {
            let (values, lift) = partial(coordinate)?;
            if values.len() != state_coordinates.len() || lift.len() != layout.full_count() {
                return Err(invalid(
                    "Parameter partial differs from the exact Field coordinate map",
                ));
            }
            for (row, value) in values.into_iter().enumerate() {
                residual[row * count + column] = value;
            }
            for (row, &global) in output_coordinates.iter().enumerate() {
                direct[row * count + column] = lift[global];
            }
            for observation in &mut observations {
                observation.add_lift(column, count, &lift)?;
            }
        }
        let output = self.selected_output(
            mapping,
            &keys,
            &full,
            state_coordinates.len(),
            direct,
            observations,
        )?;
        Ok((residual, output))
    }
}
