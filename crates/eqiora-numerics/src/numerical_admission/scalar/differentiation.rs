//! Execution and differentiated replay consume one exact assembled spatial system.
use super::regions::{ExecutableSpatialEquations, StaticRegionSystem};
use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_ir::ScalarSymbolCoordinate;

impl CommonSpatialPlan {
    /// Accept a complete typed Parameter point through this Plan's Mesh and policies.
    /// Omitted values select the canonical Model point; unselected Parameters stay fixed.
    pub fn differentiate(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let prepared = self.prepare_differentiation(selected, values)?;
        let core = &prepared.system.core;
        let executor = HostExecutorDescriptor::new(
            self.admission.linear.provider,
            self.admission.linear.execution,
            self.admission.linear.workers,
            self.admission.linear.capabilities.clone(),
        );
        let binding = DeploymentBinding::bind_host(&self.portable, executor)?;
        let admitted = AdmittedExecution::admit_host_linear(
            &self.portable,
            core.canonical_csr_system_view(),
            binding,
        )?;
        let produced =
            REFERENCE_LINEAR_SOLVER.solve(&core.linear_problem()?, core.solver_plan())?;
        let (solution, receipt) = admitted.accept(produced)?.into_parts();
        core.validate_solution(&solution)?;
        let (values, _) = solution.into_parts();
        let linearization = prepared.linearize(values, receipt.report().residual_target())?;
        if linearization
            .relation()
            .state_jacobian()
            .agreement_fingerprint()
            != receipt.operator()
        {
            return Err(invalid(
                "common Plan solve receipt differs from its differentiated state system",
            ));
        }
        Ok(CommonDifferentiationPoint::from_linear(
            linearization,
            receipt,
        ))
    }

    /// Reaccept supplied algebraic coordinates without solving or creating a producer report.
    /// Uses the same Parameter point, original residual, boundary lift and partial actions.
    pub fn validate_differentiated_values(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let prepared = self.prepare_differentiation(selected, values)?;
        let (_, target) = prepared.system.core.validate_coefficients(candidate)?;
        prepared.linearize(candidate.to_vec(), target)
    }

    fn prepare_differentiation(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
    ) -> Result<PreparedPoint<'_>, Diagnostic> {
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
        if self.solver_provider() != REFERENCE_LINEAR_SOLVER.provider() {
            return Err(invalid(
                "selected-Parameter differentiation requires the reference solver provider",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Spatial(template) = self.admission.recognized_model() else {
            return Err(invalid(
                "common spatial Plan lost its recognized mathematics",
            ));
        };
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
        let mut equations = template.as_ref().clone();
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
        Ok(PreparedPoint {
            plan: self,
            equations,
            system,
            design,
            values,
            coordinates: crate::parameter_point::coordinates(self.kernel(), selected)?,
        })
    }
}

struct PreparedPoint<'a> {
    plan: &'a CommonSpatialPlan,
    equations: ExecutableSpatialEquations<f64>,
    system: StaticRegionSystem<f64>,
    coordinates: Vec<ScalarSymbolCoordinate>,
    design: Vec<SpatialDesignCoordinate>,
    values: Vec<f64>,
}

impl PreparedPoint<'_> {
    fn linearize(
        self,
        values: Vec<f64>,
        target: f64,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let mapping = &self.system.mapping;
        let full = mapping.lift(&values, false)?;
        let keys = mapping.keys().collect::<Vec<_>>();
        let count = self.coordinates.len();
        let mut residual = vec![
            0.;
            values.len().checked_mul(count).ok_or_else(|| invalid(
                "spatial Parameter Jacobian shape overflow"
            ))?
        ];
        let mut direct = vec![
            0.;
            keys.len().checked_mul(count).ok_or_else(|| invalid(
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
            for (row, value) in partial.into_iter().enumerate() {
                residual[row * count + column] = value;
            }
            for (row, key) in keys.iter().enumerate() {
                direct[row * count + column] =
                    lift[mapping.global_dof(*key).expect("mapped Field DOF")];
            }
        }
        let output = FieldCoefficientLinearization::new(
            keys.iter()
                .map(|key| full[mapping.global_dof(*key).expect("mapped Field DOF")])
                .collect(),
            values.len(),
            count,
            keys.iter()
                .map(|key| mapping.free_dof(*key).map(|dof| dof.index()))
                .collect(),
            direct,
        )?;
        let relation = AssembledLinearizedRelation::from_shared_canonical(
            self.system.core.shared_canonical_system(),
            values,
            self.design,
            self.values,
            residual,
        )?;
        Ok(CommonDifferentiationLinearization::from_spatial(
            relation, output, target,
        ))
    }
}
