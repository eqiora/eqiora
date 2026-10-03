//! Common Plan dispatch and immutable evaluation acceptance.
use super::*;

impl DifferentiableProgram {
    /// Compile an exact Plan, ordered Parameter selection, and scalar output.
    /// Finite nonlinear Plans require their complete initial State and an
    /// Observable; spatial scalar Plans select a Field and require no seed.
    ///
    /// # Errors
    /// Rejects foreign identities, unsupported roles, and unaccepted primals.
    pub fn compile<E: Entity>(
        plan: ResolvedCommonPlan,
        inputs: &[ModelEntityRef<kinds::Parameter>],
        output: &ModelEntityRef<E>,
        initial: Option<CommonAlgebraicState>,
        backend: &'static dyn LinearSolverBackend,
    ) -> Result<Self, Vec<Diagnostic>> {
        if inputs.is_empty() {
            return Err(single(invalid(
                "differentiable program requires selected Parameters",
            )));
        }
        let model = plan.model_artifact().artifact_reference().map_err(single)?;
        if output.model != model || inputs.iter().any(|input| input.model != model) {
            return Err(single(invalid(
                "differentiable references must belong to the exact Model artifact",
            )));
        }
        if inputs
            .iter()
            .enumerate()
            .any(|(index, input)| inputs[..index].iter().any(|seen| seen.id == input.id))
        {
            return Err(single(invalid(
                "differentiable inputs contain a duplicate Parameter",
            )));
        }
        if plan.linear_solver_provider() != Some(backend.provider()) {
            return Err(single(invalid(
                "differentiable backend must match the exact Plan provider",
            )));
        }
        let selected = inputs.iter().map(|input| input.id).collect::<Vec<_>>();
        let native = accept_plan_point(
            &plan,
            initial.as_ref(),
            &selected,
            None,
            output.id.erase(),
            backend,
        )
        .map_err(single)?;
        let primal_residual_norm = accept_linearization(&native).map_err(single)?;
        let identity = DifferentiableProgramIdentity {
            model,
            plan_identity: plan.identity().to_owned(),
            inputs: selected,
            output: output.id.erase(),
            initial_state_identity: initial.as_ref().map(|state| state.identity().to_owned()),
            input_dimension: native.relation().parameter_dimension(),
            output_dimension: native.output_dimension(),
            scalar_type: DifferentiableScalarType::F64,
            device: DifferentiableDevice::HostCpu,
            derivative: DerivativeContract::ImplicitFirstOrder,
            solver: plan
                .effective_solver()
                .ok_or_else(|| single(invalid("differentiable Plan requires a linear solver")))?,
        };
        let default = DifferentiableEvaluation {
            point: DifferentiableParameterPoint {
                inputs: identity.inputs.clone(),
                values: native.relation().design_values().to_vec(),
            },
            identity: identity.clone(),
            residual_tolerance: native.residual_target(),
            primal_residual_norm,
            native,
            backend,
        };
        Ok(Self {
            identity,
            plan,
            initial,
            backend,
            default,
        })
    }

    /// Complete exact program identity.
    #[must_use]
    pub const fn identity(&self) -> &DifferentiableProgramIdentity {
        &self.identity
    }

    /// Canonical Model values in selected input order.
    #[must_use]
    pub const fn default_point(&self) -> &DifferentiableParameterPoint {
        &self.default.point
    }

    /// Accept a complete finite Parameter point without mutating the Program.
    ///
    /// # Errors
    /// Rejects invalid shapes, values, unsupported points, or failed acceptance.
    pub fn evaluate(
        &self,
        parameters: &[f64],
    ) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        if parameters.len() != self.identity.input_dimension
            || parameters.iter().any(|value| !value.is_finite())
        {
            return Err(single(invalid(
                "differentiable Parameter point requires complete finite values",
            )));
        }
        if parameters
            .iter()
            .zip(self.default.point.values())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(self.default.clone());
        }
        let native = accept_plan_point(
            &self.plan,
            self.initial.as_ref(),
            &self.identity.inputs,
            Some(parameters),
            self.identity.output,
            self.backend,
        )
        .map_err(single)?;
        if native.relation().design_values().len() != parameters.len()
            || parameters
                .iter()
                .zip(native.relation().design_values())
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(single(invalid(
                "accepted relation differs from the requested Parameter point",
            )));
        }
        if native.output_dimension() != self.identity.output_dimension {
            return Err(single(invalid(
                "accepted output shape differs from the Program",
            )));
        }
        Ok(DifferentiableEvaluation {
            identity: self.identity.clone(),
            point: DifferentiableParameterPoint {
                inputs: self.identity.inputs.clone(),
                values: parameters.to_vec(),
            },
            primal_residual_norm: accept_linearization(&native).map_err(single)?,
            residual_tolerance: native.residual_target(),
            native,
            backend: self.backend,
        })
    }

    /// Return the accepted default output.
    #[must_use]
    pub fn primal(&self) -> DifferentiablePrimal {
        self.default.primal()
    }

    /// Apply the default-point total output JVP.
    pub fn jvp(&self, tangent: &[f64]) -> Result<DifferentiableJvp, Diagnostic> {
        self.default.jvp(tangent)
    }

    /// Apply the default-point total output VJP.
    pub fn vjp(&self, cotangent: &[f64]) -> Result<DifferentiableVjp, Diagnostic> {
        self.default.vjp(cotangent)
    }
}

fn accept_plan_point(
    plan: &ResolvedCommonPlan,
    initial: Option<&CommonAlgebraicState>,
    selected: &[Id<kinds::Parameter>],
    values: Option<&[f64]>,
    output: RawId,
    backend: &dyn LinearSolverBackend,
) -> Result<CommonScalarDifferentiationPoint, Diagnostic> {
    match plan {
        ResolvedCommonPlan::Scalar(plan) => {
            if initial.is_some() || !plan.fields().any(|(field, _)| field.erase() == output) {
                return Err(invalid(
                    "spatial differentiation requires a Plan Field and no finite seed",
                ));
            }
            plan.differentiate(selected, values)
        }
        ResolvedCommonPlan::Algebraic(plan) => plan.differentiate(
            initial.ok_or_else(|| {
                invalid("finite differentiation requires its exact initial State")
            })?,
            selected,
            values,
            output
                .downcast()
                .ok_or_else(|| invalid("finite differentiation requires an Observable"))?,
            backend,
        ),
        _ => Err(invalid(
            "this Plan does not admit implicit output differentiation",
        )),
    }
}

fn accept_linearization(native: &CommonScalarDifferentiationPoint) -> Result<f64, Diagnostic> {
    let relation = native.relation();
    if native.receipt().is_some_and(|receipt| {
        receipt.operator() != relation.state_jacobian().agreement_fingerprint()
    }) {
        return Err(invalid(
            "execution receipt differs from the paired state Jacobian",
        ));
    }
    let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        relation,
        native,
        relation.state_jacobian(),
        native.residual_target(),
    )?;
    Ok(accepted.relation().primal_residual_norm())
}
