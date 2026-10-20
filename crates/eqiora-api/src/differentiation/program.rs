//! Common Plan dispatch and immutable evaluation acceptance.
use super::*;
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;

impl DifferentiableProgram {
    pub(crate) const fn plan(&self) -> &ResolvedCommonPlan {
        &self.plan
    }
    pub(crate) const fn initial_state(&self) -> Option<&CommonInitialState<'static>> {
        self.initial.as_ref()
    }

    pub(crate) fn kernel_program(&self) -> Result<&eqiora_sem::KernelProgram, Diagnostic> {
        signature::kernel(&self.plan)
    }
    /// Compile an exact Plan, ordered Parameters, and ordered selected outputs.
    /// Nonlinear Plans require their complete initial State and an
    /// output; affine finite Plans need no seed. Finite outputs are Fields or
    /// instantaneous real/complex Observables, in component real/imaginary order.
    /// Static spatial Plans select complete Fields and reduced Observables, with
    /// explicit quadrature for spatial integrals. Stationary Newton uses a
    /// spatial seed; affine spatial execution consumes no seed.
    ///
    /// # Errors
    /// Rejects foreign identities, unsupported roles, and unaccepted primals.
    pub fn compile(
        plan: ResolvedCommonPlan,
        inputs: &[ModelEntityRef<Id<kinds::Parameter>>],
        outputs: &[ModelEntityRef],
        quadratures: HashMap<Id<kinds::Domain>, QuadratureRule>,
        initial: Option<CommonInitialState<'static>>,
        backend: &'static dyn LinearSolverBackend,
    ) -> Result<Self, Vec<Diagnostic>> {
        let identity = DifferentiableProgramIdentity::from_plan(
            &plan,
            inputs,
            outputs,
            &quadratures,
            initial.as_ref(),
        )?;
        if plan.linear_solver_provider() != Some(backend.provider()) {
            return Err(single(invalid(
                "differentiable backend must match the exact Plan provider",
            )));
        }
        let selected = identity
            .inputs
            .iter()
            .map(|value| value.parameter())
            .collect::<Vec<_>>();
        let outputs = identity
            .outputs
            .iter()
            .map(|value| value.entity())
            .collect::<Vec<_>>();
        let native = accept_plan_point(
            &plan,
            initial.as_ref(),
            &selected,
            None,
            &outputs,
            identity.observation_quadratures(),
            backend,
        )
        .map_err(single)?;
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

    pub(crate) fn validate_point(
        &self,
        parameters: &DifferentiableValues,
    ) -> Result<(), Diagnostic> {
        parameters.validate_signatures(self.identity.inputs())
    }

    /// Accept a complete finite Parameter point without mutating the Program.
    ///
    /// # Errors
    /// Rejects foreign Models, roles, types or input order before numerical execution.
    /// Unsupported points and failed acceptance retain their native diagnostics.
    pub fn evaluate(
        &self,
        parameters: &DifferentiableValues,
    ) -> Result<DifferentiableEvaluation, Vec<Diagnostic>> {
        self.validate_point(parameters).map_err(single)?;
        let parameters = parameters.coordinates();
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
            self.identity.observation_quadratures(),
            self.backend,
        )
        .map_err(single)?;
        if native.linearization().relation().design_values().len() != parameters.len()
            || parameters
                .iter()
                .zip(native.linearization().relation().design_values())
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
    pub fn primal(&self) -> DifferentiableResult {
        self.default.primal()
    }

    /// Apply the default-point total output JVP.
    pub fn jvp(
        &self,
        tangent: &DifferentiableTangent,
    ) -> Result<DifferentiableResult<DifferentiableTangent>, Diagnostic> {
        self.default.jvp(tangent)
    }

    /// Apply the default-point total output VJP.
    pub fn vjp(
        &self,
        cotangent: &DifferentiableCotangent,
    ) -> Result<DifferentiableResult<DifferentiableCotangent>, Diagnostic> {
        self.default.vjp(cotangent)
    }
}

impl DifferentiableEvaluation {
    fn accept(
        identity: DifferentiableProgramIdentity,
        native: CommonDifferentiationPoint,
        backend: &'static dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        if native.receipt().is_some_and(|receipt| {
            receipt.operator()
                != native
                    .linearization()
                    .relation()
                    .state_jacobian()
                    .agreement_fingerprint()
        }) {
            return Err(invalid(
                "execution receipt differs from the paired state Jacobian",
            ));
        }
        let acceptance = evidence::Acceptance::new(&identity, native.linearization())?;
        Ok(Self {
            point: DifferentiableValues::from_coordinates(
                identity.inputs.clone(),
                native.linearization().relation().design_values().to_vec(),
            )?,
            identity,
            acceptance,
            native,
            backend,
        })
    }
}

fn accept_plan_point(
    plan: &ResolvedCommonPlan,
    initial: Option<&CommonInitialState<'_>>,
    selected: &[Id<kinds::Parameter>],
    values: Option<&[f64]>,
    outputs: &[RawId],
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    backend: &dyn LinearSolverBackend,
) -> Result<CommonDifferentiationPoint, Diagnostic> {
    match plan {
        ResolvedCommonPlan::Spatial(plan) => plan.differentiate(
            spatial_initial(initial)?,
            selected,
            values,
            outputs,
            quadratures,
            backend,
        ),
        ResolvedCommonPlan::Algebraic(plan) => {
            let canonical;
            let initial = match algebraic_initial(initial)? {
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

/// Typed projections of the same seed owner; wrong families never become absence.
pub(super) fn spatial_initial<'a>(
    initial: Option<&'a CommonInitialState<'_>>,
) -> Result<Option<&'a eqiora_numerics::CommonState>, Diagnostic> {
    initial
        .map(|state| match state {
            CommonInitialState::Spatial(state) => Ok(state.as_ref()),
            _ => Err(invalid(
                "spatial differentiation requires a spatial initial State",
            )),
        })
        .transpose()
}

pub(super) fn algebraic_initial<'a>(
    initial: Option<&'a CommonInitialState<'_>>,
) -> Result<Option<&'a eqiora_numerics::CommonAlgebraicState>, Diagnostic> {
    initial
        .map(|state| match state {
            CommonInitialState::Algebraic(state) => Ok(state.as_ref()),
            _ => Err(invalid(
                "finite differentiation requires an algebraic initial State",
            )),
        })
        .transpose()
}
