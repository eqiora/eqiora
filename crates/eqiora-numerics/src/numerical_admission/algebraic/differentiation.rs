//! Original-equation finite acceptance followed by implicit partial actions.
use super::super::differentiation::Primal;
use super::*;
use crate::common::AssembledLinearizedRelation;
use eqiora_core::{Id, ScalarDomain, entity::kinds};
use eqiora_ir::ScalarObjectiveLinearization;
use eqiora_schema::kernel::ObservableReduction;

impl CommonAlgebraicPlan {
    /// Accept one exact Parameter point from this Plan's finite seed and Observable.
    /// Unselected Parameters retain canonical Model values. The admitted affine
    /// or Newton solve establishes the primal; component IR supplies partials at
    /// the independently accepted regular point.
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
        if self.gauge.is_some() {
            return Err(invalid(
                "finite differentiation requires an unconstrained regular branch; gauge derivatives are not admitted",
            ));
        }
        let AlgebraicProblem::Constrained(base) = &self.problem else {
            return Err(invalid("finite differentiation requires typed Fields"));
        };
        if base.enforcement().is_some() && !base.is_strict_interior() {
            return Err(invalid(
                "finite differentiation does not admit changing active sets",
            ));
        }
        let (design_coordinates, defaults) = base.parameter_point(selected)?;
        if selected.is_empty() || values.is_some_and(|values| values.len() != defaults.len()) {
            return Err(invalid(
                "finite differentiation requires a complete nonempty Parameter point",
            ));
        }
        let selected_values = values.map_or(defaults, <[f64]>::to_vec);
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
        let problem = base.at_parameters(selected, &selected_values)?;
        let checked = self.linear.checked_backend(backend, None)?;
        let (point, primal) = if let Some(nonlinear) = self.nonlinear {
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
            (
                solution.values.clone(),
                Primal::Nonlinear {
                    initial: initial.clone(),
                    solution,
                },
            )
        } else {
            let (point, report) = if self.complex_system.is_some() {
                let system = problem.complex_linear_system()?.ok_or_else(|| {
                    invalid("finite Parameter point changed its admitted complex execution profile")
                })?;
                let solution =
                    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, self.linear.solver)
                        .solve(&system.linear_problem()?)?;
                (
                    solution
                        .values()
                        .iter()
                        .flat_map(|z| [z.re, z.im])
                        .collect::<Vec<_>>(),
                    solution.report().clone(),
                )
            } else {
                let solution = crate::finite_constraints::solve_finite_constraints(
                    &problem,
                    LinearSolveRequest::new(&checked, self.linear.solver),
                )?;
                (solution.values().to_vec(), solution.report().clone())
            };
            let assessment = problem.validate_values(&point, self.linear.solver, 0)?;
            if report.residual_target().to_bits() != assessment.residual_target().to_bits() {
                return Err(invalid(
                    "finite derivative primal target differs from original acceptance",
                ));
            }
            (point, Primal::Affine { report, assessment })
        };
        let (actions, jacobian) = problem.equality_jacobian(&point, selected)?;
        problem.require_regular(actions.unknown_jacobian.clone())?;
        let original = problem.original_residual(&point)?;
        if actions.values != original {
            return Err(invalid(
                "finite derivative primal differs from original Relation operands",
            ));
        }
        let relation = AssembledLinearizedRelation::from_point(
            jacobian,
            point.clone(),
            original,
            design_coordinates,
            selected_values,
            actions.parameter_jacobian,
            LinearOperatorProperties::General,
        )?;
        let output = problem.linearize_expression(typed.expression(), &point, selected)?;
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
        Ok(CommonScalarDifferentiationPoint::from_finite(
            relation, output, primal,
        ))
    }
}
