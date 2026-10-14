//! Common Plan dispatch and immutable evaluation acceptance.
use super::*;

impl DifferentiableProgram {
    /// Compile an exact Plan, ordered Parameters, and ordered selected outputs.
    /// Finite nonlinear Plans require their complete initial State and an
    /// output; affine finite Plans need no seed. Finite outputs are Fields or
    /// instantaneous real/complex Observables, in component real/imaginary order.
    /// Spatial scalar Plans select a Field and require no seed.
    ///
    /// # Errors
    /// Rejects foreign identities, unsupported roles, and unaccepted primals.
    pub fn compile(
        plan: ResolvedCommonPlan,
        inputs: &[ModelEntityRef<Id<kinds::Parameter>>],
        outputs: &[ModelEntityRef],
        initial: Option<CommonAlgebraicState>,
        backend: &'static dyn LinearSolverBackend,
    ) -> Result<Self, Vec<Diagnostic>> {
        if inputs.is_empty() {
            return Err(single(invalid(
                "differentiable program requires selected Parameters",
            )));
        }
        let model = plan.model_artifact().artifact_reference().map_err(single)?;
        if outputs.iter().any(|output| output.model != model)
            || inputs.iter().any(|input| input.model != model)
        {
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
        if outputs.is_empty()
            || outputs
                .iter()
                .enumerate()
                .any(|(index, output)| outputs[..index].iter().any(|seen| seen.id == output.id))
        {
            return Err(single(invalid(
                "differentiable outputs require nonempty distinct selections",
            )));
        }
        let outputs = outputs.iter().map(|output| output.id).collect::<Vec<_>>();
        let selected = inputs.iter().map(|input| input.id).collect::<Vec<_>>();
        let (input_signatures, output_signatures) =
            signature::build(&plan, &model, &selected, &outputs).map_err(single)?;
        let native = accept_plan_point(&plan, initial.as_ref(), &selected, None, &outputs, backend)
            .map_err(single)?;
        let identity = DifferentiableProgramIdentity {
            model,
            plan_identity: plan.identity().to_owned(),
            inputs: input_signatures.into(),
            outputs: output_signatures.into(),
            initial_state_identity: initial.as_ref().map(|state| state.identity().to_owned()),
            scalar_type: ScalarType::F64,
            solver: plan
                .effective_solver()
                .ok_or_else(|| single(invalid("differentiable Plan requires a linear solver")))?,
        };
        let default =
            DifferentiableEvaluation::accept(identity.clone(), native, backend).map_err(single)?;
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
    pub const fn default_point(&self) -> &DifferentiableValues {
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
        if parameters.len() != self.identity.input_dimension()
            || parameters.iter().any(|value| !value.is_finite())
        {
            return Err(single(invalid(
                "differentiable Parameter point requires complete finite values",
            )));
        }
        if parameters
            .iter()
            .zip(self.default.point.coordinates())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(self.default.clone());
        }
        let native = accept_plan_point(
            &self.plan,
            self.initial.as_ref(),
            &self
                .identity
                .inputs
                .iter()
                .map(DifferentiableValueSignature::parameter)
                .collect::<Vec<_>>(),
            Some(parameters),
            &self
                .identity
                .outputs
                .iter()
                .map(DifferentiableValueSignature::entity)
                .collect::<Vec<_>>(),
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
        DifferentiableEvaluation::accept(self.identity.clone(), native, self.backend)
            .map_err(single)
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

impl DifferentiableEvaluation {
    fn accept(
        identity: DifferentiableProgramIdentity,
        native: CommonDifferentiationPoint,
        backend: &'static dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        if identity.input_dimension() != native.relation().parameter_dimension()
            || identity.output_dimension() != native.output_dimension()
        {
            return Err(invalid(
                "accepted linearization differs from the Model/Plan value signatures",
            ));
        }
        let primal_residual_norm = accept_linearization(&native)?;
        Ok(Self {
            point: DifferentiableValues::from_coordinates(
                identity.inputs.clone(),
                native.relation().design_values().to_vec(),
            )?,
            identity,
            accepted_state: AcceptedOutputFingerprintV1::from_values(
                native.relation().accepted_unknowns(),
            )?,
            accepted_output: AcceptedOutputFingerprintV1::from_values(native.output_values())?,
            residual_tolerance: native.residual_target(),
            primal_residual_norm,
            native,
            backend,
        })
    }
}

fn accept_plan_point(
    plan: &ResolvedCommonPlan,
    initial: Option<&CommonAlgebraicState>,
    selected: &[Id<kinds::Parameter>],
    values: Option<&[f64]>,
    outputs: &[RawId],
    backend: &dyn LinearSolverBackend,
) -> Result<CommonDifferentiationPoint, Diagnostic> {
    match plan {
        ResolvedCommonPlan::Spatial(plan) => {
            if initial.is_some()
                || outputs.len() != 1
                || !plan.fields().any(|(field, _)| field.erase() == outputs[0])
            {
                return Err(invalid(
                    "spatial differentiation currently requires one Plan Field and no finite seed",
                ));
            }
            plan.differentiate(selected, values)
        }
        ResolvedCommonPlan::Algebraic(plan) => {
            let canonical;
            let initial = match initial {
                Some(initial) => initial,
                None if plan.nonlinear().is_none() => {
                    canonical = plan.initial_state(&[])?;
                    &canonical
                }
                None => {
                    return Err(invalid(
                        "finite nonlinear differentiation requires its exact initial State",
                    ));
                }
            };
            plan.differentiate(initial, selected, values, outputs, backend)
        }
        _ => Err(invalid(
            "this Plan does not admit implicit output differentiation",
        )),
    }
}

fn accept_linearization(native: &CommonDifferentiationPoint) -> Result<f64, Diagnostic> {
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
