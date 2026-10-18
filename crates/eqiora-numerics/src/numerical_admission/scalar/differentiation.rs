//! Execution and differentiated replay consume one exact assembled spatial system.
use super::super::differentiation::{Output, Primal};
use super::regions::{ExecutableSpatialEquations, StaticRegionSystem};
use super::*;
use crate::finalized_spatial::FinalizedLinearCore;
use crate::spatial_expression::Coefficient;
use eqiora_core::{Id, entity::kinds};
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;
mod outputs;
mod scalar;
use scalar::{DifferentiationScalar, coordinates};

impl CommonSpatialPlan {
    /// Accept a complete typed Parameter point through this Plan's Mesh and policies.
    /// Omitted values select the canonical Model point; unselected Parameters stay fixed.
    /// Ordered outputs may mix complete Fields and reduced Observables. Spatial
    /// integrals require explicit rules for their exact integration Domains.
    pub fn differentiate(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .solve(backend),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .solve(backend),
            _ => Err(invalid(
                "spatial differentiation requires its exact typed equations",
            )),
        }
    }

    /// Reaccept supplied algebraic coordinates without solving or creating a producer report.
    /// Uses the same Parameter point, original residual, boundary lift and partial actions.
    pub fn validate_differentiated_values(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .accept_candidate(candidate),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_differentiation(equations, selected, values, outputs, quadratures)?
                .accept_candidate(candidate),
            _ => Err(invalid(
                "spatial differentiation requires its exact typed equations",
            )),
        }
    }

    fn prepare_differentiation<S: DifferentiationScalar>(
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
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "stationary Parameter differentiation requires a static Plan",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        if !matches!(
            self.admission.resources(),
            NativeMeshResources::Cartesian { .. }
        ) {
            return Err(invalid(
                "spatial Parameter differentiation requires exact Cartesian resources",
            ));
        }
        let (design, canonical) = crate::parameter_point::canonical_point(self.kernel(), selected)?;
        let values = values.unwrap_or(&canonical).to_vec();
        let point = crate::parameter_point::ParameterPoint::bind(self.kernel(), selected, &values)?;
        let mut equations = template.clone();
        for region in &mut equations.regions {
            region.form = region.form.bind_parameter_point(&point)?;
        }
        let system = equations.finalize_static(
            self.admission.linear.workers,
            self.admission.linear.solver,
            self.admission.resources(),
            &self.admission.discretizations,
            &self.admission.system_scaling,
            self.admission.operator_properties,
        )?;
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
            } else if !matches!(
                self.kernel().node(output),
                Some(eqiora_schema::kernel::KernelNode::Observable(_))
            ) {
                return Err(invalid(
                    "spatial output is not an exact Plan Field or Model Observable",
                ));
            }
        }
        let keys = if selected_fields.is_empty() {
            Vec::new()
        } else {
            system.mapping.selected_field_keys(&selected_fields)?
        };
        Ok(PreparedPoint {
            plan: self,
            equations,
            keys,
            point,
            outputs: outputs.to_vec(),
            quadratures: quadratures.clone(),
            system,
            design,
            values,
            coordinates: crate::parameter_point::coordinates(self.kernel(), selected)?,
        })
    }
}

struct PreparedPoint<'a, S: DifferentiationScalar> {
    plan: &'a CommonSpatialPlan,
    equations: ExecutableSpatialEquations<S>,
    system: StaticRegionSystem<S>,
    coordinates: Vec<ScalarSymbolCoordinate>,
    design: Vec<SpatialDesignCoordinate>,
    values: Vec<f64>,
    keys: Vec<crate::region_assembly::mapping::FieldDof>,
    point: crate::parameter_point::ParameterPoint,
    outputs: Vec<eqiora_core::RawId>,
    quadratures: HashMap<Id<kinds::Domain>, QuadratureRule>,
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S> {
    fn solve(
        self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let (values, primal) = S::solve(self.plan, &self.equations, &self.system.core, backend)?;
        let target = match &primal {
            Primal::Linear(receipt) => receipt.report().residual_target(),
            Primal::Affine(report) => report.residual_target(),
            Primal::Nonlinear { .. } => unreachable!("static linear solve"),
        };
        let linearization = self.linearize(values, target)?;
        if let Primal::Linear(receipt) = &primal
            && receipt.operator()
                != linearization
                    .relation()
                    .state_jacobian()
                    .agreement_fingerprint()
        {
            return Err(invalid(
                "common Plan solve receipt differs from its differentiated state system",
            ));
        }
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            primal,
        ))
    }

    fn accept_candidate(
        self,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let values = S::from_coordinates(candidate)?;
        let (_, target) = self.system.core.validate_coefficients(&values)?;
        self.linearize(values, target)
    }

    fn linearize(
        self,
        values: Vec<S>,
        target: f64,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let mapping = &self.system.mapping;
        let full = mapping.lift(&values, false)?;
        let keys = &self.keys;
        let count = self.coordinates.len();
        let state_coordinates = coordinates(values.iter().copied()).collect::<Vec<_>>();
        let mut observations = self.observation_partials(&full, state_coordinates.len())?;
        let output_count = keys
            .len()
            .checked_mul(S::WIDTH)
            .ok_or_else(|| invalid("Field coordinate count overflow"))?;
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
            let (partial, lift) = self.system.parameter_partial(
                &self.equations,
                coordinate,
                self.plan.admission.resources(),
                &self.plan.admission.discretizations,
                &values,
            )?;
            for (row, value) in coordinates(partial).enumerate() {
                residual[row * count + column] = value;
            }
            let lift = coordinates(lift).collect::<Vec<_>>();
            for (row, key) in keys.iter().enumerate() {
                let global = mapping.global_dof(*key).expect("mapped Field DOF") * S::WIDTH;
                for part in 0..S::WIDTH {
                    direct[(row * S::WIDTH + part) * count + column] = lift[global + part];
                }
            }
            for observation in &mut observations {
                observation.add_lift(column, count, &lift)?;
            }
        }
        let output = self.selected_output(&full, state_coordinates.len(), direct, observations)?;
        let relation = AssembledLinearizedRelation::from_shared_canonical(
            S::real_system(&self.system.core)?,
            state_coordinates,
            self.design,
            self.values,
            residual,
        )?;
        Ok(CommonDifferentiationLinearization::from_spatial(
            relation, output, target,
        ))
    }
}
