//! Original-equation nonlinear acceptance followed by implicit partial actions.
use super::*;
use crate::common::{AssembledLinearizedRelation, SpatialDesignCoordinate};
use eqiora_core::{Id, ScalarDomain, entity::kinds};
use eqiora_ir::ScalarObjectiveLinearization;
use eqiora_schema::kernel::ObservableReduction;

impl CommonAlgebraicPlan {
    /// Accept one exact Parameter point from this Plan's finite seed and Observable.
    /// Unselected Parameters retain canonical Model values. Newton establishes the
    /// primal; Operator IR then supplies partials at the independently accepted point.
    pub fn differentiate(
        &self,
        initial: &CommonAlgebraicState,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        observable: Id<kinds::Observable>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonScalarDifferentiationPoint, Diagnostic> {
        if initial != &self.state_from_values(initial.values().to_vec())?
            || backend.provider() != self.linear.provider
        {
            return Err(invalid(
                "finite differentiation requires its exact seed Plan and solver provider",
            ));
        }
        let nonlinear = self
            .nonlinear
            .ok_or_else(|| invalid("finite differentiation requires a Newton Plan"))?;
        let selected_values = selected
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                    return Err(invalid(
                        "differentiable Parameter is outside the exact finite Model",
                    ));
                };
                match values {
                    Some(values) => values
                        .get(index)
                        .copied()
                        .ok_or_else(|| invalid("finite Parameter point has the wrong arity")),
                    None => parameter
                        .value()
                        .real_scalar_value()
                        .map(|value| value.value())
                        .ok_or_else(|| {
                            invalid("finite differentiation requires real scalar Parameters")
                        }),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if selected.is_empty() || values.is_some_and(|values| values.len() != selected.len()) {
            return Err(invalid(
                "finite differentiation requires a complete nonempty Parameter point",
            ));
        }
        let typed = self.kernel.typed_observable(observable).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("Observable typing diagnostic")
        })?;
        let Some(KernelNode::Observable(output)) = self.kernel.node(observable.erase()) else {
            return Err(invalid(
                "finite differentiation requires an exact Observable",
            ));
        };
        if output.reduction() != ObservableReduction::Value
            || output.value_type().scalar_domain() != ScalarDomain::Real
            || !output.value_type().shape().is_scalar()
        {
            return Err(invalid(
                "finite differentiation requires a real scalar instantaneous Observable",
            ));
        }
        let problem = self
            .problem
            .nonlinear()?
            .at_parameters(selected, &selected_values)?;
        let checked = self.linear.checked_backend(backend, None)?;
        let solution = problem.solve_at_point(
            initial.values(),
            nonlinear,
            LinearSolveRequest::new(&checked, self.linear.solver),
        )?;
        let (initial_norm, assessment) =
            problem.validate_nonlinear_values(initial.values(), &solution.values, nonlinear)?;
        if initial_norm.to_bits() != solution.initial_residual_norm.to_bits()
            || assessment != solution.assessment
        {
            return Err(invalid(
                "finite differentiated primal differs from original acceptance",
            ));
        }
        let (actions, jacobian) = problem.equality_jacobian(&solution.values, selected)?;
        let original = problem.original_residual(&solution.values)?;
        if actions.values != original {
            return Err(invalid(
                "finite derivative primal differs from original Relation operands",
            ));
        }
        let relation = AssembledLinearizedRelation::from_point(
            jacobian,
            solution.values.clone(),
            original,
            selected
                .iter()
                .copied()
                .map(SpatialDesignCoordinate::ModelParameter)
                .collect(),
            selected_values,
            actions.parameter_jacobian,
            LinearOperatorProperties::General,
        )?;
        let output =
            problem.linearize_expression(typed.expression(), &solution.values, selected)?;
        let [value] = output.values.as_slice() else {
            return Err(invalid(
                "finite Observable linearization has the wrong output dimension",
            ));
        };
        let output = ScalarObjectiveLinearization::new(
            *value,
            output.unknown_jacobian,
            output.parameter_jacobian,
        )?;
        Ok(CommonScalarDifferentiationPoint::from_nonlinear(
            relation,
            output,
            initial.clone(),
            solution,
        ))
    }
}
