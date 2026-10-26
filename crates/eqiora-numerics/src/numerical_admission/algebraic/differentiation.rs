//! Original-equation finite acceptance followed by implicit partial actions.
use super::super::differentiation::Primal;
use super::*;
use crate::common::AssembledLinearizedRelation;
use crate::finite_constraints::FiniteConstraintProblem;
use eqiora_core::{Id, RawId, ScalarDomain, entity::kinds};
use eqiora_ir::DenseOutputLinearization;
use eqiora_schema::kernel::{ExprDag, ExprDagBuilder, ObservableReduction};
use eqiora_solver::CanonicalCsrSystemView;

impl CommonAlgebraicPlan {
    /// Accept one exact Parameter point from this Plan's finite seed and selected outputs.
    /// Unselected Parameters retain canonical Model values. The admitted affine
    /// or Newton solve establishes the primal; component IR supplies partials at
    /// the independently accepted regular point.
    pub fn differentiate(
        &self,
        initial: &CommonAlgebraicState,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[RawId],
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        if backend.provider() != self.linear.provider {
            return Err(invalid(
                "finite differentiation requires its exact solver provider",
            ));
        }
        let prepared = self.prepare_differentiation(initial, selected, values, outputs)?;
        let problem = &prepared.problem;
        let linear = &prepared.linear;
        let (point, primal, reported_acceptance) = if let Some(nonlinear) = self.nonlinear {
            let checked = self.linear.checked_backend(backend, None)?;
            let solution = problem.solve_at_point(
                initial.values(),
                nonlinear,
                LinearSolveRequest::new(&checked, self.linear.solver),
            )?;
            (
                solution.values,
                Primal::Nonlinear {
                    initial: Some(initial.clone().into()),
                    iterations: solution.iterations,
                    linear_solves: solution.linear_solves,
                },
                Some((solution.initial_residual_norm, solution.assessment)),
            )
        } else {
            let (point, report) = match prepared.system.as_ref().expect("affine point system") {
                PointSystem::Complex(system) => {
                    let checked = linear.checked_complex_backend(backend, None)?;
                    let solution = LinearSolveRequest::new(&checked, linear.solver)
                        .solve(&system.linear_problem()?)?;
                    (
                        solution
                            .values()
                            .iter()
                            .flat_map(|z| [z.re, z.im])
                            .collect(),
                        solution.report().clone(),
                    )
                }
                PointSystem::Real(system) => {
                    let checked = linear.checked_backend(backend, None)?;
                    LinearSolveRequest::new(&checked, linear.solver)
                        .solve(&system.linear_problem()?)?
                        .into_parts()
                }
            };
            (point, Primal::Affine(report), None)
        };
        let (linearization, assessment) = prepared.accept(self, initial, selected, point)?;
        if let Some((initial_norm, reported)) = reported_acceptance {
            if linearization.initial_residual_norm().map(f64::to_bits)
                != Some(initial_norm.to_bits())
                || assessment != reported
            {
                return Err(invalid(
                    "finite differentiated primal differs from original acceptance",
                ));
            }
        } else if let Primal::Affine(report) = &primal
            && report.residual_target().to_bits() != assessment.residual_target().to_bits()
        {
            return Err(invalid(
                "finite derivative primal target differs from original acceptance",
            ));
        }
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            primal,
        ))
    }

    /// Revalidate a supplied finite candidate and derive its exact partial actions.
    ///
    /// Uses the same original-equation acceptance as ordinary differentiation,
    /// including strict-interior and regularity checks. This executes no solve
    /// and makes no claim about producer identity or historical iterations.
    pub fn validate_differentiated_values(
        &self,
        initial: &CommonAlgebraicState,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[RawId],
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        self.prepare_differentiation(initial, selected, values, outputs)?
            .accept(self, initial, selected, candidate.to_vec())
            .map(|(linearization, _)| linearization)
    }

    fn prepare_differentiation(
        &self,
        initial: &CommonAlgebraicState,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[RawId],
    ) -> Result<PreparedPoint, Diagnostic> {
        if self.harmonic.is_some() {
            return Err(invalid(
                "harmonic solved-response derivatives require the retained reduction chain and are not admitted",
            ));
        }
        if initial != &self.state_from_values(initial.values().to_vec())? {
            return Err(invalid(
                "finite differentiation requires its exact seed Plan",
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
        if outputs.is_empty() {
            return Err(invalid("finite differentiation requires selected outputs"));
        }
        let mut expressions = Vec::with_capacity(outputs.len());
        for (index, &id) in outputs.iter().enumerate() {
            if outputs[..index].contains(&id) {
                return Err(invalid(
                    "finite differentiated outputs repeat an exact identity",
                ));
            }
            let expression = match self.kernel.node(id) {
                Some(KernelNode::Field(field)) => {
                    if !base.symbols().contains(&SymbolRef::Field(field.id())) {
                        return Err(invalid("finite output Field is outside the admitted state"));
                    }
                    let mut builder = ExprDagBuilder::new();
                    let root = builder.symbol(SymbolRef::Field(field.id()))?;
                    builder.finish([root])?
                }
                Some(KernelNode::Observable(output)) => {
                    if output.reduction() != ObservableReduction::Value
                        || !matches!(
                            output.value_type().scalar_domain(),
                            ScalarDomain::Real | ScalarDomain::Complex
                        )
                    {
                        return Err(invalid(
                            "finite differentiated Observable requires an instantaneous real or complex value",
                        ));
                    }
                    self.kernel
                        .typed_observable(output.id())
                        .map_err(|errors| {
                            errors
                                .into_iter()
                                .next()
                                .expect("Observable typing diagnostic")
                        })?;
                    output.expression().clone()
                }
                _ => {
                    return Err(invalid(
                        "finite differentiation requires exact Fields or Observables",
                    ));
                }
            };
            expressions.push(expression);
        }
        let problem = base.at_parameters(selected, &selected_values)?;
        let system = if self.nonlinear.is_some() {
            None
        } else if let Some(system) = problem.complex_linear_system()? {
            Some(PointSystem::Complex(system))
        } else {
            Some(PointSystem::Real(problem.linear_system()?))
        };
        let linear = match &system {
            Some(PointSystem::Real(system)) => self.linear.at_parameter_problem(system, None)?,
            Some(PointSystem::Complex(system)) => self.linear.at_parameter_problem(system, None)?,
            None => self.linear.clone(),
        };
        Ok(PreparedPoint {
            problem,
            system,
            linear,
            design_coordinates,
            selected_values,
            expressions,
        })
    }
}

enum PointSystem {
    Real(CanonicalCsrSystemView),
    Complex(CanonicalCsrSystemView<num_complex::Complex64>),
}

struct PreparedPoint {
    problem: FiniteConstraintProblem,
    system: Option<PointSystem>,
    linear: NativeLinearPolicy,
    design_coordinates: Vec<SpatialDesignCoordinate>,
    selected_values: Vec<f64>,
    expressions: Vec<ExprDag>,
}

impl PreparedPoint {
    fn accept(
        self,
        plan: &CommonAlgebraicPlan,
        initial: &CommonAlgebraicState,
        selected: &[Id<kinds::Parameter>],
        point: Vec<f64>,
    ) -> Result<(CommonDifferentiationLinearization, ConstraintAssessment), Diagnostic> {
        let problem = &self.problem;
        let (initial_residual_norm, assessment) = match plan.nonlinear {
            Some(nonlinear) => {
                let (norm, assessment) =
                    problem.validate_nonlinear_values(initial.values(), &point, nonlinear)?;
                (Some(norm), assessment)
            }
            None => (
                None,
                problem.validate_values(&point, plan.linear.solver, 0)?,
            ),
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
            self.design_coordinates,
            self.selected_values,
            actions.parameter_jacobian,
            LinearOperatorProperties::General,
        )?;
        let output = linearize_outputs(problem, &self.expressions, &point, selected)?;
        Ok((
            CommonDifferentiationLinearization::from_finite(
                relation,
                if matches!(self.system, Some(PointSystem::Complex(_))) {
                    ScalarDomain::Complex
                } else {
                    ScalarDomain::Real
                },
                output,
                eqiora_differentiation::PrimalAcceptance::Residual(assessment.residual_target()),
                initial_residual_norm.map(|residual_norm| {
                    crate::nonlinear_execution::NewtonInitialPoint {
                        residual_norm,
                        at_seed: point == initial.values(),
                    }
                }),
            ),
            assessment,
        ))
    }
}

fn linearize_outputs(
    problem: &crate::finite_constraints::FiniteConstraintProblem,
    expressions: &[ExprDag],
    point: &[f64],
    selected: &[Id<kinds::Parameter>],
) -> Result<DenseOutputLinearization, Diagnostic> {
    let mut values = Vec::new();
    let mut unknown = Vec::new();
    let mut parameter = Vec::new();
    for expression in expressions {
        let output = problem.linearize_expression(expression, point, selected)?;
        values.extend(output.values);
        unknown.extend(output.unknown_jacobian);
        parameter.extend(output.parameter_jacobian);
    }
    DenseOutputLinearization::new(values, unknown, parameter)
}
