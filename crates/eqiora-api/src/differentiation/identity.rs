//! Static meaning of a differentiable Program, independent of primal execution.

use super::*;

mod artifact;

/// Complete typed identity of one differentiable application program.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableProgramIdentity {
    pub(super) model: ModelArtifactReference,
    pub(super) plan_identity: String,
    pub(super) inputs: Arc<[DifferentiableValueSignature]>,
    pub(super) outputs: Arc<[DifferentiableValueSignature]>,
    pub(super) initial_state_identity: Option<String>,
    pub(super) scalar_type: ScalarType,
    pub(super) solver: SolverPlan,
}

impl DifferentiableProgramIdentity {
    /// Exact canonical Model artifact.
    #[must_use]
    pub const fn model(&self) -> &ModelArtifactReference {
        &self.model
    }

    /// Exact common Plan identity.
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    /// Ordered canonical Parameter inputs.
    #[must_use]
    pub fn inputs(&self) -> &[DifferentiableValueSignature] {
        &self.inputs
    }

    /// Ordered selected Fields and Observables with their exact types and layouts.
    #[must_use]
    pub fn outputs(&self) -> &[DifferentiableValueSignature] {
        &self.outputs
    }

    /// Exact finite seed bound by this Program; absent for spatial linear execution.
    #[must_use]
    pub fn initial_state_identity(&self) -> Option<&str> {
        self.initial_state_identity.as_deref()
    }

    /// Flat tangent/gradient input dimension.
    #[must_use]
    pub fn input_dimension(&self) -> usize {
        signature::dimension(&self.inputs).expect("admitted input signature")
    }

    /// Total real-coordinate dimension of the ordered selected outputs.
    #[must_use]
    pub fn output_dimension(&self) -> usize {
        signature::dimension(&self.outputs).expect("admitted output signature")
    }

    /// Shared storage representation; current program admission requires [`ScalarType::F64`].
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }

    /// Solver policy shared by normal and transposed actions.
    #[must_use]
    pub const fn solver(&self) -> SolverPlan {
        self.solver
    }
}

impl DifferentiableProgramIdentity {
    /// Bind static meaning without accepting a default primal or retaining a provider.
    /// Exact collection reopening can use this same admission boundary without solving.
    pub(crate) fn from_plan(
        plan: &ResolvedCommonPlan,
        inputs: &[ModelEntityRef<Id<kinds::Parameter>>],
        outputs: &[ModelEntityRef],
        initial: Option<&CommonAlgebraicState>,
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
        let mut selected_inputs = std::collections::BTreeSet::new();
        if inputs
            .iter()
            .any(|input| !selected_inputs.insert(input.id.erase()))
        {
            return Err(single(invalid(
                "differentiable inputs contain a duplicate Parameter",
            )));
        }
        let mut selected_outputs = std::collections::BTreeSet::new();
        if outputs.is_empty()
            || outputs
                .iter()
                .any(|output| !selected_outputs.insert(output.id))
        {
            return Err(single(invalid(
                "differentiable outputs require nonempty distinct selections",
            )));
        }
        let outputs = outputs.iter().map(|output| output.id).collect::<Vec<_>>();
        let selected = inputs.iter().map(|input| input.id).collect::<Vec<_>>();
        match plan {
            ResolvedCommonPlan::Spatial(plan)
                if initial.is_some()
                    || outputs.iter().any(|output| {
                        !plan.fields().any(|(field, _)| field.erase() == *output)
                    }) =>
            {
                return Err(single(invalid(
                    "spatial differentiation requires exact Plan Fields and no finite seed",
                )));
            }
            ResolvedCommonPlan::Algebraic(plan) => {
                if initial.is_some_and(|state| state.plan_identity() != plan.identity()) {
                    return Err(single(invalid(
                        "differentiable initial State belongs to a foreign Plan",
                    )));
                }
                if initial.is_none() && plan.nonlinear().is_some() {
                    return Err(single(invalid(
                        "finite nonlinear differentiation requires its exact initial State",
                    )));
                }
            }
            _ => {}
        }
        let (input_signatures, output_signatures) =
            signature::build(plan, &model, &selected, &outputs).map_err(single)?;
        Ok(Self {
            model,
            plan_identity: plan.identity().to_owned(),
            inputs: input_signatures.into(),
            outputs: output_signatures.into(),
            initial_state_identity: initial.map(|state| state.identity().to_owned()),
            scalar_type: ScalarType::F64,
            solver: plan
                .effective_solver()
                .ok_or_else(|| single(invalid("differentiable Plan requires a linear solver")))?,
        })
    }
}
