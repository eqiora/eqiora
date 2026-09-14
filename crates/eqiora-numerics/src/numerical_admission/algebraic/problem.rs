//! Admitted finite mathematical problems behind one Plan/State/Result lifecycle.
use super::*;
use crate::finite_constraints::{
    ConstraintAssessment, FiniteConstraintEnforcement, FiniteConstraintProblem,
    lower_finite_constraints, solve_finite_constraints,
};
use eqiora_solver::SolveReport;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum AlgebraicProblem {
    Conserving(ScalarPhysicalAffineProblem),
    Constrained(FiniteConstraintProblem),
}

pub(super) struct AlgebraicSolution {
    pub values: Vec<f64>,
    pub report: SolveReport,
    pub active_set_mask: Option<u32>,
}

impl AlgebraicProblem {
    pub(super) fn admit(
        kernel: &KernelProgram,
        enforcement: Option<FiniteConstraintEnforcement>,
    ) -> Result<Self, Diagnostic> {
        if let Some(enforcement) = enforcement {
            return lower_finite_constraints(kernel, &enforcement).map(Self::Constrained);
        }
        if kernel.nodes().any(|node| {
            matches!(node,
            KernelNode::Relation(relation) if relation.has_constraints())
        }) {
            return Err(invalid(
                "mathematical constraints require explicit finite enforcement",
            ));
        }
        if kernel.nodes().any(|node| {
            matches!(node, KernelNode::Field(_) | KernelNode::ClockDomain(_))
                || matches!(node, KernelNode::Port(port) if port.signal_contract().is_some())
        }) {
            return Err(invalid(
                "conserving affine admission does not admit unresolved Fields or clocked/signal execution",
            ));
        }
        let connection = kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Connection(value) => Some(value.id()),
                _ => None,
            })
            .min_by_key(|id| id.ulid())
            .ok_or_else(|| invalid("conserving affine admission requires a Connection"))?;
        let problem = lower_scalar_physical_affine(kernel, connection, None)?;
        let composed = problem.composed_system();
        let physical_ports = kernel
            .nodes()
            .filter(
                |node| matches!(node, KernelNode::Port(port) if port.physical_domain().is_some()),
            )
            .count();
        let relation_count = kernel
            .nodes()
            .filter(|node| matches!(node, KernelNode::Relation(_)))
            .count();
        if physical_ports * 2 != composed.unknowns().len()
            || relation_count != composed.relations().len()
        {
            return Err(invalid(
                "finite Plan requires one complete conserving closure with no omitted Relations",
            ));
        }
        Ok(Self::Conserving(problem))
    }

    pub(super) fn symbols(&self) -> Vec<SymbolRef> {
        match self {
            Self::Conserving(problem) => problem
                .composed_system()
                .unknowns()
                .iter()
                .map(|unknown| match unknown {
                    PhysicalUnknown::Across(port) => SymbolRef::Across(*port),
                    PhysicalUnknown::Through(port) => SymbolRef::Through(*port),
                })
                .collect(),
            Self::Constrained(problem) => problem.symbols().to_vec(),
        }
    }

    pub(super) fn dimensions(&self) -> Vec<DimExponents> {
        match self {
            Self::Conserving(problem) => problem
                .composed_system()
                .unknown_types()
                .iter()
                .map(|value| value.dimension())
                .collect(),
            Self::Constrained(problem) => problem.dimensions().to_vec(),
        }
    }

    pub(super) fn enforcement(&self) -> Option<&FiniteConstraintEnforcement> {
        match self {
            Self::Conserving(_) => None,
            Self::Constrained(problem) => Some(problem.enforcement()),
        }
    }

    pub(super) fn solve(
        &self,
        initial: &[f64],
        solver: LinearSolveRequest<'_>,
    ) -> Result<AlgebraicSolution, Diagnostic> {
        match self {
            Self::Conserving(problem) => {
                let solution = solve_scalar_physical_affine(problem, initial, solver)?;
                Ok(AlgebraicSolution {
                    values: solution.values().to_vec(),
                    report: solution.report().clone(),
                    active_set_mask: None,
                })
            }
            Self::Constrained(problem) => {
                let solution = solve_finite_constraints(problem, solver)?;
                Ok(AlgebraicSolution {
                    values: solution.values().to_vec(),
                    report: solution.report().clone(),
                    active_set_mask: Some(solution.assessment().active_set_mask()),
                })
            }
        }
    }

    pub(super) fn validate_values(
        &self,
        values: &[f64],
        plan: SolverPlan,
        target: f64,
        mask: Option<u32>,
    ) -> Result<(f64, Option<ConstraintAssessment>), Diagnostic> {
        match self {
            Self::Constrained(problem) => {
                let mask = mask.ok_or_else(|| {
                    invalid("constrained Result requires its selected active-set mask")
                })?;
                let assessment = problem.validate_values(values, plan, mask)?;
                if target.to_bits() != assessment.residual_target().to_bits() {
                    return Err(invalid(
                        "finite Result target differs from exact selected branch",
                    ));
                }
                Ok((assessment.equality_residual_norm(), Some(assessment)))
            }
            Self::Conserving(problem) => {
                if mask.is_some() {
                    return Err(invalid(
                        "unconstrained conserving Result cannot carry an active-set mask",
                    ));
                }
                let rhs = problem.canonical_system().right_hand_side();
                let rhs_norm = SERIAL_LINEAR_EXECUTION
                    .inner_product(FixedOrderInnerProduct::new(rhs, rhs)?)?
                    .sqrt();
                if plan.residual_target(rhs_norm)?.to_bits() != target.to_bits() {
                    return Err(invalid(
                        "finite Result target differs from exact original right-hand side",
                    ));
                }
                let residuals = problem.reference_residuals(values)?;
                let norm = SERIAL_LINEAR_EXECUTION
                    .inner_product(FixedOrderInnerProduct::new(&residuals, &residuals)?)?
                    .sqrt();
                if !norm.is_finite() || norm > target {
                    return Err(invalid(
                        "finite Result original semantic residual exceeds acceptance target",
                    ));
                }
                Ok((norm, None))
            }
        }
    }
}
