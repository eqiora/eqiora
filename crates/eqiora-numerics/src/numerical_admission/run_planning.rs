//! Schedule-owned forecasts for bounded ordinary scalar-storage Runs.
use super::*;
use serde::{Deserialize, Serialize};

const PROOF: &str = "scalar-q1-autonomous-positive-storage/v1";

/// Persisted inputs and independently rederived projections, never provider cache state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireRunPlanning {
    pub(super) steps: usize,
    pub(super) output_steps: Vec<usize>,
    proof: String,
    assembly_passes: usize,
    local_quadrature_work: Option<u128>,
    common_work: Option<u128>,
    retained_numeric_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunPlanningContext {
    pub(super) request: CommonLinearRequest,
    pub(super) wire: WireRunPlanning,
}

impl RunPlanningContext {
    pub(super) fn require_schedule(
        &self,
        steps: usize,
        outputs: &[usize],
    ) -> Result<(), Diagnostic> {
        state::validate_step_schedule(steps, outputs)?;
        if self.wire.steps != steps || self.wire.output_steps != outputs {
            return Err(invalid(
                "Run workload differs from its Plan; explicitly call Plan.for_run for the new schedule",
            ));
        }
        Ok(())
    }
}

impl ResolvedCommonPlan {
    /// Resolve a new immutable Plan for an explicit relative Run schedule.
    ///
    /// The bounded profile is fixed Cartesian Q1, real scalar Fields, autonomous
    /// linear equations and positive own storage in every free equation. Work
    /// counts are deterministic proxies, not measured elapsed-time predictions.
    /// The original Plan and exact manual solver controls remain unchanged.
    pub fn for_run(
        &self,
        steps: usize,
        output_steps: Vec<usize>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        state::validate_step_schedule(steps, &output_steps)?;
        let Self::Spatial(original) = self else {
            return Err(invalid("Run cost planning requires scalar spatial storage"));
        };
        require_bounded_profile(original)?;
        let RecognizedNativeModel::Spatial(equations) = original.admission.recognized_model()
        else {
            return Err(invalid("Run cost planning requires real spatial equations"));
        };
        // Bind through the existing owner, without constructing a State or
        // evaluating a numerical assembly. Source Time was checked before this.
        let (mapping, input) = original.scalar_assembly_at(equations, 0.)?;
        let temporal = input.temporal_structure()?;
        let (algebraic, _) = mapping.temporal_coordinate_partition(&temporal)?;
        if !temporal.positive_own_capacity || !algebraic.is_empty() {
            return Err(invalid(
                "Run cost planning requires positive own capacity and an all-differential free partition",
            ));
        }
        let size = eqiora_solver::SolverPlanningSize::dense_bound(mapping.free_count())
            .ok_or_else(|| invalid("Run planning system-size bound overflow"))?;
        let wire = forecast(&mapping, &input, steps, output_steps)?;
        let request = match self.canonical_solve_request() {
            Some(CommonSolvePolicy::Linear(request)) => request,
            _ => return Err(invalid("Run planning requires an existing linear request")),
        };
        let horizon = NonZeroUsize::new(steps).expect("validated horizon");
        let caller_cost = request.cost().unwrap_or_default();
        let mut cost = eqiora_solver::SolverCostRequest::new(horizon, horizon, horizon)?
            .with_iteration_forecasts(
                caller_cost.identity_iterations(),
                caller_cost.jacobi_iterations(),
            );
        if let Some(limit) = caller_cost.numeric_payload_limit() {
            cost = cost.with_numeric_payload_limit(limit);
        }
        let profile = original
            .admission
            .linear
            .planning_profile
            .clone()
            .ok_or_else(|| invalid("Run planning requires the original admitted solver profile"))?
            .with_size(size)
            .with_cost_request(cost)
            .with_run_forecast(
                wire.common_work,
                wire.retained_numeric_bytes,
                Some(backend.provider()),
            );
        // The existing selector remains the sole exact/manual admission and ranking owner.
        let mut linear = solver_planning::resolve_typed_linear(
            request.with_cost(cost),
            input.operator_properties,
            profile,
            &REFERENCE_LINEAR_SOLVER,
            Some(backend),
        )?;
        if linear.planning_objective.is_some() {
            linear.planning_reasons.extend([
                ("run", "run.schedule-replaces-caller-horizon"),
                ("run", "run.autonomous-positive-storage-proof"),
                ("run", "run.faer-lu-only-retained-preparation-forecast"),
                ("run", "run.two-step-assemblies-plus-output-replay"),
                ("run", "run.unmeasured-binding-metadata-and-provider-memory"),
                ("run", "run.retained-coordinate-payload-separate-from-cap"),
            ]);
            if wire.common_work.is_none() {
                linear
                    .planning_reasons
                    .push(("run", "run.common-work-unknown"));
            }
        }
        let admission = original.admission.with_linear_policy(linear)?;
        let mut planned = CommonSpatialPlan::from_admission(
            original.admission.model(),
            admission,
            original.formulation.as_ref().map(|form| form.requested()),
            original.authored_formulation.as_ref(),
        )?;
        let mut identity = Vec::new();
        push_framed(&mut identity, planned.identity().as_bytes());
        push_framed(
            &mut identity,
            &plan_artifact::linear_request_bytes(request)?,
        );
        let encoded = serde_json::to_vec(&wire)
            .map_err(|error| invalid(format!("cannot encode Run planning context: {error}")))?;
        push_framed(&mut identity, &encoded);
        planned.lineage.identity =
            domain_separated_identity(b"eqiora.common-run-plan/v1\0", &identity);
        planned.run_planning = Some(RunPlanningContext { request, wire });
        Ok(Self::Spatial(Box::new(planned)))
    }
}

fn require_bounded_profile(plan: &CommonSpatialPlan) -> Result<(), Diagnostic> {
    if plan.harmonic.is_some()
        || plan.admission.nonlinear.is_some()
        || plan
            .admission
            .temporal
            .as_ref()
            .and_then(CommonTemporalPolicy::as_backward_euler)
            .is_none()
        || !matches!(
            plan.admission.resources(),
            NativeMeshResources::Cartesian { .. }
        )
        || plan.admission.discretizations.iter().any(|domain| {
            domain
                .field_spaces()
                .iter()
                .any(|field| field.space() != Space::continuous_lagrange(std::num::NonZeroU16::MIN))
        })
    {
        return Err(invalid(
            "Run planning requires fixed Cartesian Q1 linear BackwardEuler",
        ));
    }
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        return Err(invalid("Run planning requires real scalar storage"));
    };
    if equations.regions.iter().any(|region| {
        let form = &region.form;
        form.is_nonlinear()
            || form.motion().is_some()
            || !form.kinematics().is_empty()
            || form.constraint_pairs().next().is_some()
            || form.depends_on_time(plan.admission.program())
            || form.represented_fields().iter().any(|(_, ty)| {
                ty.scalar_domain() != eqiora_core::ScalarDomain::Real
                    || ty.shape() != &eqiora_core::ValueShape::scalar()
            })
    }) {
        return Err(invalid(
            "Run preparation forecast requires autonomous scalar source dependencies without motion, kinematics or multipliers",
        ));
    }
    Ok(())
}

fn forecast(
    mapping: &crate::region_assembly::mapping::RegionDofMap<f64>,
    input: &crate::region_assembly::mapping::RegionSolveInput<f64>,
    steps: usize,
    output_steps: Vec<usize>,
) -> Result<WireRunPlanning, Diagnostic> {
    let assembly_passes = steps
        .checked_mul(2)
        .and_then(|n| n.checked_add(output_steps.len()))
        .ok_or_else(|| invalid("Run assembly count overflow"))?;
    // Each complete ordinary Run performs one solve assembly and one live
    // State replay per step, then one reacceptance per retained Trajectory output.
    // Count dense local matrix/load quadrature samples through the exact maps.
    // Coefficient expression evaluation, binding and metadata are unmeasured;
    // these units are not FLOPs or a claim about elapsed time.
    let mut local = Some(0_u128);
    for (cell, domain) in mapping.cell_domains().iter().enumerate() {
        let (_, quadrature) = input
            .forms
            .iter()
            .find(|(form, _)| form.domain() == *domain)
            .ok_or_else(|| invalid("Run planning cell has no bound Region form"))?;
        let width = mapping.cell_map(cell, false)?.unknowns().len() as u128;
        local = local.and_then(|sum| {
            sum.checked_add(
                width
                    .checked_mul(width)?
                    .checked_add(width)?
                    .checked_mul(quadrature.points().len() as u128)?,
            )
        });
    }
    if !input.natural.is_empty() || input.conservative.is_some() {
        local = None;
    }
    let n = mapping.free_count() as u128;
    let full = mapping.full_count();
    let common_work = local.and_then(|local| {
        // Existing dense CSR envelope, explicit finalized-core residual replay,
        // and a coordinate recovery/copy envelope including retained outputs.
        let replay = n
            .checked_mul(n)?
            .checked_mul(2)?
            .checked_add(n.checked_mul(5)?)?;
        let copies = (steps as u128)
            .checked_add(output_steps.len() as u128)?
            .checked_mul(4)?
            .checked_add(4)?
            .checked_mul(full as u128)?;
        local
            .checked_add(replay)?
            .checked_mul(assembly_passes as u128)?
            .checked_add(copies)
    });
    // At most two snapshots in each State; each retained step has previous and
    // current States. Include request/current State snapshots separately. This
    // is a coordinate-payload envelope, never a whole-Run memory bound.
    let retained_numeric_bytes = output_steps
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_mul(full))
        .and_then(|n| n.checked_mul(size_of::<f64>()));
    Ok(WireRunPlanning {
        steps,
        output_steps,
        proof: PROOF.to_owned(),
        assembly_passes,
        local_quadrature_work: local,
        common_work,
        retained_numeric_bytes,
    })
}
