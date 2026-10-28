//! Damped Newton execution for the bounded fixed-topology ALE FSI slice.

use crate::simplicial_fsi::layout::FsiLayout;
use std::ops::ControlFlow;

use eqiora_assembly::AssemblyBackend;
#[cfg(test)]
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_meshing::{QuadratureRule, SimplicialMesh};
use eqiora_solver::{LinearOperatorProperties, LinearSolverBackend};

use super::acceptance::accept_step_prepared;
use super::api::{AleFsiStepEvidence, AleFsiTrajectory};
use super::assembly::{
    PreparedAleFsiAction, PreparedAleFsiStructure, assemble_step_linearization_with_structure,
    prepare_ale_fsi_structure,
};
use super::boundary_step::{PreparedAleFsiBoundaryRun, PreparedAleFsiBoundaryStep};
use super::contract::{AleFsiState, AleFsiStepPlan};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::nonlinear_execution::{NewtonTrial, solve_damped_newton};
use crate::prepared_execution::advance_prepared_actions;
use crate::simplicial_fsi::{FixedReferenceFsiBoundary, FixedReferenceFsiPartition};
use crate::step_count::NonZeroStepCount;

/// Advance one fixed-topology reference through accepted monolithic ALE steps.
///
/// # Errors
/// Fails closed on stale reference/state data, invalid quadrature, unsupported
/// solver policy, analytic linearization failure, Krylov nonconvergence,
/// unsuccessful globalization, or any acceptance falsifier. A failed step does
/// not mutate the last accepted trajectory state.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn advance_simplicial_ale_fsi_2d(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    boundary: &FixedReferenceFsiBoundary<2>,
    motion: &P1HarmonicMeshMotionAction<2>,
    initial: AleFsiState<2>,
    step_count: NonZeroStepCount,
    plan: AleFsiStepPlan<2>,
    quadrature: &QuadratureRule,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<2>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<AleFsiTrajectory<2>, Diagnostic> {
    advance_simplicial_ale_fsi_2d_with_assembly(
        reference,
        partition,
        boundary,
        motion,
        initial,
        step_count,
        plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
        base_layout,
        graph,
    )
}

/// Advance through an explicit assembly adapter and common linear backend.
///
/// # Errors
/// Preserves all ALE, assembly, Newton, Krylov, and independent acceptance
/// diagnostics without fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_ale_fsi_2d_with_assembly(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    boundary: &FixedReferenceFsiBoundary<2>,
    motion: &P1HarmonicMeshMotionAction<2>,
    initial: AleFsiState<2>,
    step_count: NonZeroStepCount,
    plan: AleFsiStepPlan<2>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<2>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<AleFsiTrajectory<2>, Diagnostic> {
    advance_simplicial_ale_fsi_with_assembly::<2>(
        reference,
        partition,
        boundary,
        motion,
        initial,
        step_count,
        &plan,
        quadrature,
        assembly,
        solver,
        base_layout,
        graph,
    )
}

/// Advance one tetrahedral fixed-topology reference through accepted ALE FSI steps.
///
/// # Errors
/// Preserves the same fail-closed geometry, nonlinear, linearization, solver,
/// and independent-acceptance boundary as the established two-dimensional
/// entry point.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn advance_simplicial_ale_fsi_3d(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
    boundary: &FixedReferenceFsiBoundary<3>,
    motion: &P1HarmonicMeshMotionAction<3>,
    initial: AleFsiState<3>,
    step_count: NonZeroStepCount,
    plan: AleFsiStepPlan<3>,
    quadrature: &QuadratureRule,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<3>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<AleFsiTrajectory<3>, Diagnostic> {
    advance_simplicial_ale_fsi_3d_with_assembly(
        reference,
        partition,
        boundary,
        motion,
        initial,
        step_count,
        plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
        base_layout,
        graph,
    )
}

/// Advance tetrahedral ALE FSI through an explicit assembly adapter.
///
/// # Errors
/// Preserves all typed tetrahedral, degree-eleven quadrature, serial-host,
/// assembly, Newton, Krylov, and independent acceptance diagnostics without
/// fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_simplicial_ale_fsi_3d_with_assembly(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
    boundary: &FixedReferenceFsiBoundary<3>,
    motion: &P1HarmonicMeshMotionAction<3>,
    initial: AleFsiState<3>,
    step_count: NonZeroStepCount,
    plan: AleFsiStepPlan<3>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<3>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<AleFsiTrajectory<3>, Diagnostic> {
    advance_simplicial_ale_fsi_with_assembly::<3>(
        reference,
        partition,
        boundary,
        motion,
        initial,
        step_count,
        &plan,
        quadrature,
        assembly,
        solver,
        base_layout,
        graph,
    )
}

struct PreparedAleFsiRun<'a, const D: usize> {
    reference: &'a SimplicialMesh,
    partition: &'a FixedReferenceFsiPartition<D>,
    boundary: PreparedAleFsiBoundaryRun<D>,
    structure: PreparedAleFsiStructure<D>,
    motion: &'a P1HarmonicMeshMotionAction<D>,
    plan: AleFsiStepPlan<D>,
    quadrature: &'a QuadratureRule,
    assembly: &'a dyn AssemblyBackend<f64>,
    solver: &'a dyn LinearSolverBackend,
    prepared_linear: Option<Box<dyn eqiora_solver::PreparedLinearSolver>>,
    #[cfg(test)]
    phases: AleFsiRunPhaseCounts,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AleFsiRunPhaseCounts {
    authentication: usize,
    normalization: usize,
    boundary: usize,
    layout: usize,
    maps: usize,
    quadrature: usize,
    sparsity: usize,
}

impl<'a, const D: usize> PreparedAleFsiRun<'a, D> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        reference: &'a SimplicialMesh,
        partition: &'a FixedReferenceFsiPartition<D>,
        boundary: &'a FixedReferenceFsiBoundary<D>,
        motion: &'a P1HarmonicMeshMotionAction<D>,
        initial: &AleFsiState<D>,
        plan: &AleFsiStepPlan<D>,
        quadrature: &'a QuadratureRule,
        assembly: &'a dyn AssemblyBackend<f64>,
        solver: &'a dyn LinearSolverBackend,
        base_layout: &FsiLayout<D>,
        graph: &eqiora_realization::PortableRealizationGraph,
    ) -> Result<Self, Diagnostic> {
        solver.capabilities().require_problem(
            plan.linear_solver(),
            eqiora_core::ScalarDomain::Real,
            ScalarType::F64,
            LinearOperatorProperties::General,
        )?;
        let boundary = PreparedAleFsiBoundaryRun::new(reference, boundary, initial, plan)?;
        let structure = prepare_ale_fsi_structure(
            reference,
            partition,
            boundary.template(),
            motion,
            initial,
            plan,
            quadrature,
            base_layout,
            graph,
        )?;
        #[cfg(test)]
        let phases = {
            let structural = structure.phase_counts();
            AleFsiRunPhaseCounts {
                authentication: structural.authentication,
                normalization: boundary.normalization_count(),
                boundary: 1,
                layout: structural.layout,
                maps: structural.maps,
                quadrature: structural.quadrature,
                sparsity: structural.sparsity,
            }
        };
        Ok(Self {
            reference,
            partition,
            boundary,
            structure,
            motion,
            plan: plan.clone(),
            quadrature,
            assembly,
            solver,
            prepared_linear: solver.prepare_linear(plan.linear_solver())?,
            #[cfg(test)]
            phases,
        })
    }

    fn advance(
        &mut self,
        previous: &AleFsiState<D>,
    ) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
        if let Some(linear) = &mut self.prepared_linear {
            linear.begin_action()?;
        }
        let result = self.advance_candidate(previous);
        if result.is_err() {
            if let Some(linear) = &mut self.prepared_linear {
                let _ = linear.discard_action();
            }
        }
        result
    }

    fn advance_candidate(
        &mut self,
        previous: &AleFsiState<D>,
    ) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
        let boundary = self.boundary.action(previous, &self.plan)?;
        let action = self.structure.prepare_action(
            self.reference,
            self.partition,
            boundary,
            previous,
            &self.plan,
        )?;
        solve_one_step_prepared(
            self.reference,
            self.partition,
            &self.structure,
            &action,
            self.motion,
            previous,
            &self.plan,
            self.quadrature,
            self.assembly,
            self.solver,
            &mut self.prepared_linear,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn advance_simplicial_ale_fsi_with_assembly<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    initial: AleFsiState<D>,
    step_count: NonZeroStepCount,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<D>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<AleFsiTrajectory<D>, Diagnostic> {
    match advance_prepared_actions(
        AleFsiTrajectory::<D>::new(initial),
        step_count.get(),
        |trajectory| {
            PreparedAleFsiRun::new(
                reference,
                partition,
                boundary,
                motion,
                trajectory
                    .states()
                    .last()
                    .expect("ALE FSI trajectory owns its initial state"),
                plan,
                quadrature,
                assembly,
                solver,
                base_layout,
                graph,
            )
        },
        |step, trajectory| {
            let current = trajectory
                .states()
                .last()
                .expect("ALE FSI trajectory owns its initial state");
            eqiora_execution::telemetry_span!(time_step(
                step,
                current.time() + plan.time_step(),
                plan.time_step()
            ))
        },
        |prepared, trajectory| {
            prepared.advance(
                trajectory
                    .states()
                    .last()
                    .expect("ALE FSI trajectory owns its initial state"),
            )
        },
        |prepared, trajectory, _, (next, evidence)| {
            if let Err(error) = trajectory.push(next, evidence) {
                if let Some(linear) = &mut prepared.prepared_linear {
                    let _ = linear.discard_action();
                }
                return Err(error);
            }
            if let Some(linear) = &mut prepared.prepared_linear {
                if let Err(error) = linear.commit_action() {
                    let _ = linear.discard_action();
                    return Err(error);
                }
            }
            Ok(())
        },
        |_, _| None::<()>,
    )? {
        ControlFlow::Continue(trajectory) => Ok(trajectory),
        ControlFlow::Break(()) => unreachable!("ALE FSI run has no early boundary"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_one_step<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    base_layout: &FsiLayout<D>,
    graph: &eqiora_realization::PortableRealizationGraph,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    let prepared = match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
        Some(prepared) => prepared,
        None => PreparedAleFsiBoundaryStep::homogeneous(
            reference,
            boundary,
            previous.time(),
            previous.time() + plan.time_step(),
            plan.scale().velocity(),
        )?,
    };
    let structure = prepare_ale_fsi_structure(
        reference,
        partition,
        &prepared,
        motion,
        previous,
        plan,
        quadrature,
        base_layout,
        graph,
    )?;
    let action = structure.prepare_action(reference, partition, prepared, previous, plan)?;
    let mut prepared_linear = solver.prepare_linear(plan.linear_solver())?;
    if let Some(linear) = &mut prepared_linear {
        linear.begin_action()?;
    }
    let result = solve_one_step_prepared(
        reference,
        partition,
        &structure,
        &action,
        motion,
        previous,
        plan,
        quadrature,
        assembly_backend,
        solver,
        &mut prepared_linear,
    );
    match result {
        Ok(accepted) => {
            if let Some(linear) = &mut prepared_linear {
                if let Err(error) = linear.commit_action() {
                    let _ = linear.discard_action();
                    return Err(error);
                }
            }
            Ok(accepted)
        }
        Err(error) => {
            if let Some(linear) = &mut prepared_linear {
                let _ = linear.discard_action();
            }
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_one_step_prepared<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    structure: &PreparedAleFsiStructure<D>,
    action: &PreparedAleFsiAction<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    assembly_backend: &dyn AssemblyBackend<f64>,
    solver: &dyn LinearSolverBackend,
    prepared_linear: &mut Option<Box<dyn eqiora_solver::PreparedLinearSolver>>,
) -> Result<(AleFsiState<D>, AleFsiStepEvidence<D>), Diagnostic> {
    let point = structure.initial_point(action, previous, plan)?;
    let linearize = |point: &[f64]| {
        assemble_step_linearization_with_structure(
            reference,
            partition,
            structure,
            action,
            motion,
            previous,
            point,
            plan,
            quadrature,
            assembly_backend,
        )
    };
    let (current, evidence) = solve_damped_newton(
        point,
        plan.nonlinear().into(),
        &linearize,
        |current| current.residual_norm(),
        |point, current| {
            let linear_problem = current
                .relation
                .state_jacobian()
                .linear_problem()?
                .with_initial_guess(point)?;
            let solution =
                if let Some(prepared_linear) = prepared_linear.as_deref_mut() {
                    let identity = current.assembly_report.structure_identity().ok_or_else(|| {
                    invalid("prepared ALE FSI solve requires an exact assembly structure identity")
                })?;
                    prepared_linear.solve(identity, &linear_problem)?
                } else {
                    solver.solve(&linear_problem, plan.linear_solver())?
                };
            let (proposed, report) = solution.into_parts();
            if proposed.len() != point.len() {
                return Err(invalid(
                    "ALE linear solution differs from its exact point coordinates",
                ));
            }
            Ok((
                proposed
                    .iter()
                    .zip(point)
                    .map(|(proposed, point)| proposed - point)
                    .collect(),
                report,
            ))
        },
        |candidate, _| match linearize(candidate) {
            Ok(assembled) => Ok(Some(NewtonTrial {
                residual_norm: assembled.residual_norm()?,
                linearization: Some(assembled),
            })),
            // Only trial-dependent mesh admissibility permits backtracking.
            // Contract and execution diagnostics propagate through the common loop.
            Err(diagnostic) if diagnostic.code() == codes::INVALID_MESH => Ok(None),
            Err(diagnostic) => Err(diagnostic),
        },
    )?;
    accept_step_prepared::<D>(
        reference,
        partition,
        structure,
        action,
        motion,
        previous,
        plan,
        quadrature,
        assembly_backend,
        current,
        evidence,
    )
}

#[cfg(test)]
fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests;
