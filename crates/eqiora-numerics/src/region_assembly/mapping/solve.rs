use super::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::{InterfaceReactions, RecoveredInterfaceReactions};
use eqiora_assembly::{
    AssemblyBackend, AssemblyReport, LocalContribution, REFERENCE_ASSEMBLY_BACKEND,
};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule,
};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{LinearSolveRequest, SolveReport};

pub(crate) struct RegionSolveOutput<S: Coefficient> {
    pub(crate) fields: BTreeMap<RawId, RecoveredRegionField<S>>,
    pub(crate) reactions: RecoveredInterfaceReactions<S>,
    pub(crate) solve_report: SolveReport,
    pub(crate) assembly_report: AssemblyReport,
}

/// Exact local forms, exterior loads, and physical history consumed by one solve.
pub(crate) struct RegionSolveInput<S: Coefficient> {
    /// Derived from admitted mathematics, never inferred from the selected solver.
    pub(crate) operator_properties: eqiora_solver::LinearOperatorProperties,
    pub(crate) forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
    pub(crate) natural: Vec<(usize, LocalContribution<S>)>,
    pub(crate) previous: Option<BTreeMap<RawId, RecoveredRegionField<S>>>,
    /// Exact strong boundary data for eliminated physical state coordinates.
    pub(crate) prescribed_states: BTreeMap<FieldDof, S>,
    /// Sealed 2D geometry history; fluxes must already have their ALE meaning.
    pub(crate) geometry_action: Option<FixedTopologyGeometryAction<2>>,
}

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> RegionDofMap<S> {
    /// Execute an exact mapped region system. The caller authenticates
    /// Mesh/Geometry lineage; the common owner checks bound Field layouts and
    /// retains physical coefficient interpretation in the recovered inventory.
    pub(crate) fn solve<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: RegionSolveInput<S>,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        complete: impl FnOnce(
            &InterfaceReactions<S>,
            &[S],
        ) -> Result<RecoveredInterfaceReactions<S>, Diagnostic>,
    ) -> Result<RegionSolveOutput<S>, Diagnostic> {
        let RegionSolveInput {
            operator_properties,
            forms,
            natural,
            previous,
            geometry_action,
            prescribed_states,
        } = input;
        if let Some(action) = &geometry_action {
            let current = action.current_mesh();
            if mesh.topological_dimension() != 2
                || mesh.geometric_dimension() != 2
                || (0..=2).any(|d| mesh.entity_count(d) != current.entity_count(d))
                || forms
                    .iter()
                    .any(|(form, _)| form.time_step() != Some(action.time_step()))
            {
                return Err(invalid(
                    "region geometry action differs from its exact mesh or time step",
                ));
            }
            for index in 0..current.entity_count(2).expect("accepted cells") {
                let cell = MeshEntity::new(2, index);
                if mesh.incidence(cell, 0) != current.incidence(cell, 0)
                    || mesh.geometry_map(cell) != current.geometry_map(cell)
                {
                    return Err(invalid(
                        "region geometry action differs from current cell topology or coordinates",
                    ));
                }
            }
        }
        let step = kinematic_step(&forms)?;
        if step.is_none() && !prescribed_states.is_empty() {
            return Err(invalid(
                "state boundary data requires an exact kinematic step",
            ));
        }
        if previous.is_some()
            != (step.is_some()
                || forms
                    .iter()
                    .any(|(form, _)| !form.previous_fields().is_empty()))
        {
            return Err(invalid(
                "region solve requires history exactly when its forms consume previous Fields",
            ));
        }
        if let Some(previous) = &previous {
            if let Some(step) = &step {
                self.validate_step_history(previous, step)?;
                self.validate_state_constraints(previous, step, &prescribed_states)?;
            } else if previous.keys().copied().collect::<BTreeSet<_>>()
                != self.fields.keys().copied().collect()
            {
                return Err(invalid(
                    "region history differs from the complete physical Field inventory",
                ));
            }
            self.validate_physical(previous)?;
        }
        let super::assembly::MappedRegionAssembly {
            plan,
            work,
            packet_domains,
        } = self.prepare_assembly(
            mesh,
            forms,
            natural,
            previous.as_ref(),
            geometry_action.as_ref(),
        )?;
        let reactions = crate::region_assembly::InterfaceReactions::prepare(
            &work,
            plan.target_id(1).expect("full target"),
            self,
            &packet_domains,
        )?;
        let (systems, assembly_report) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &work)?
            .into_parts();
        let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &systems[0],
            operator_properties,
        )?);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            request.plan(),
            VectorLayoutKind::Replicated,
            Target::HostCpu { threads: workers },
            canonical,
        );
        let solution = request.solve(&core.linear_problem()?)?;
        core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        let reactions = complete(&reactions, &self.lift(&values, false)?)?;
        let fields = if let Some(step) = &step {
            self.recover_step(
                &values,
                previous.as_ref().expect("validated step history"),
                step,
                &prescribed_states,
            )?
        } else {
            self.recover(&values, &self.fields.keys().copied().collect::<Vec<_>>())?
        };
        Ok(RegionSolveOutput {
            reactions,
            fields,
            solve_report,
            assembly_report,
        })
    }
}

/// Consume the explicit temporal bindings already authenticated by each region.
pub(super) fn kinematic_step<S: Coefficient>(
    forms: &[(BoundRegionForm<S>, QuadratureRule)],
) -> Result<Option<eqiora_realization::BackwardEulerStep>, Diagnostic> {
    let states = forms
        .iter()
        .filter_map(|(form, _)| form.time_binding())
        .flat_map(|time| time.states.iter().copied())
        .collect::<Vec<_>>();
    if states.is_empty() {
        return Ok(None);
    }
    let duration = forms
        .iter()
        .find_map(|(form, _)| form.time_binding())
        .expect("state inventory has an explicit time binding")
        .step;
    if forms
        .iter()
        .any(|(form, _)| form.time_binding().is_none_or(|time| time.step != duration))
    {
        return Err(invalid(
            "coupled kinematic forms require one exact Backward Euler step",
        ));
    }
    eqiora_realization::BackwardEulerStep::new(duration, states).map(Some)
}
