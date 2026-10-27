use super::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::form_compiler::region::BoundRegionForm;
use eqiora_assembly::{AssemblyBackend, AssemblyReport, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule, SimplicialMesh,
};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{LinearOperatorProperties, LinearSolveRequest, SolveReport};

pub(crate) struct RegionSolveOutput<S: Coefficient> {
    pub(crate) fields: BTreeMap<RawId, RecoveredRegionField<S>>,
    pub(crate) solve_report: SolveReport,
    pub(crate) assembly_report: AssemblyReport,
    pub(crate) residual_target: f64,
}

/// Dimension-independent view of the existing authenticated geometry action.
/// Cell measures and path validity remain owned by `FixedTopologyGeometryAction`.
pub(crate) trait RegionGeometryHistory: Send + Sync {
    fn current_mesh(&self) -> &SimplicialMesh;
    fn time_step(&self) -> f64;
    fn previous_map(&self, cell: usize) -> Option<&AffineGeometryMap>;
}

impl<const D: usize> RegionGeometryHistory for FixedTopologyGeometryAction<D> {
    fn current_mesh(&self) -> &SimplicialMesh {
        self.current_mesh()
    }

    fn time_step(&self) -> f64 {
        self.time_step()
    }

    fn previous_map(&self, cell: usize) -> Option<&AffineGeometryMap> {
        self.cell(cell).map(|cell| cell.previous_map())
    }
}

pub(crate) type RegionCandidateCheck<'a, S> =
    dyn Fn(&BTreeMap<RawId, RecoveredRegionField<S>>) -> Result<bool, Diagnostic> + Sync + 'a;

/// Exact local forms, exterior loads, and physical history consumed by one solve.
pub(crate) struct RegionSolveInput<S: Coefficient> {
    /// Derived from admitted mathematics, never inferred from the selected solver.
    pub(crate) operator_properties: eqiora_solver::LinearOperatorProperties,
    pub(crate) forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
    pub(crate) conservative: Option<Box<crate::region_assembly::ConservativeFacets<S>>>,
    pub(crate) natural: Vec<(usize, crate::form_compiler::region::PreparedRegionFacet<S>)>,
    pub(crate) previous: Option<BTreeMap<RawId, RecoveredRegionField<S>>>,
    /// Exact strong boundary data for eliminated physical state coordinates.
    pub(crate) prescribed_states: BTreeMap<FieldDof, S>,
    /// Authenticated geometry history; fluxes must already have their ALE meaning.
    pub(crate) geometry_action: Option<Box<dyn RegionGeometryHistory>>,
}

impl<S: Coefficient> RegionSolveInput<S> {
    /// Input derivatives retain the primal integration and nonlinear face operation.
    pub(crate) fn retain_primal_integration(&mut self, primal: &Self) -> Result<(), Diagnostic> {
        if self.forms.len() != primal.forms.len() {
            return Err(invalid("input partial changed the Region inventory"));
        }
        for ((form, rule), (original, original_rule)) in self.forms.iter_mut().zip(&primal.forms) {
            if form.domain() != original.domain() || form.fields() != original.fields() {
                return Err(invalid("input partial changed the exact Field layout"));
            }
            *rule = original_rule.clone();
        }
        match (&mut self.conservative, &primal.conservative) {
            (None, None) => Ok(()),
            (Some(partial), Some(faces)) if primal.forms.len() == 1 => {
                partial.retain_primal(&primal.forms[0].0, faces)
            }
            _ => Err(invalid(
                "input partial changed the conservative face binding",
            )),
        }
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    pub(super) fn validate_solve_input<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
    ) -> Result<Option<eqiora_realization::BackwardEulerStep>, Diagnostic> {
        let RegionSolveInput {
            conservative,
            forms,
            previous,
            geometry_action,
            prescribed_states,
            ..
        } = input;
        if forms.iter().any(|(form, _)| form.has_conservative_faces()) != conservative.is_some() {
            return Err(invalid(
                "conservative cell actions require their complete face binding",
            ));
        }
        if let Some(faces) = conservative {
            let dimension = faces.mesh.topological_dimension();
            if geometry_action.is_some()
                || !input.natural.is_empty()
                || mesh.topological_dimension() != dimension
                || mesh.entity_count(dimension) != faces.mesh.entity_count(dimension)
                || (0..faces.mesh.entity_count(dimension).expect("Cartesian cells")).any(|index| {
                    let cell = MeshEntity::new(dimension, index);
                    mesh.geometry_map(cell) != faces.mesh.geometry_map(cell)
                        || mesh.incidence(cell, 0) != faces.mesh.incidence(cell, 0)
                })
            {
                return Err(invalid(
                    "conservative face geometry differs from its exact static cell geometry",
                ));
            }
        }
        if let Some(action) = geometry_action {
            let current = action.current_mesh();
            let dimension = current.topological_dimension();
            if mesh.topological_dimension() != dimension
                || mesh.geometric_dimension() != current.geometric_dimension()
                || (0..=dimension).any(|d| mesh.entity_count(d) != current.entity_count(d))
                || forms
                    .iter()
                    .any(|(form, _)| form.time_step() != Some(action.time_step()))
            {
                return Err(invalid(
                    "region geometry action differs from its exact mesh or time step",
                ));
            }
            for index in 0..current.entity_count(dimension).expect("accepted cells") {
                let cell = MeshEntity::new(dimension, index);
                if mesh.incidence(cell, 0) != current.incidence(cell, 0)
                    || mesh.geometry_map(cell) != current.geometry_map(cell)
                {
                    return Err(invalid(
                        "region geometry action differs from current cell topology or coordinates",
                    ));
                }
            }
        }
        let step = kinematic_step(forms)?;
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
        if let Some(previous) = previous {
            if let Some(step) = &step {
                self.validate_step_history(previous, step)?;
                self.validate_state_constraints(previous, step, prescribed_states)?;
            } else if previous.keys().copied().collect::<BTreeSet<_>>()
                != self.fields.keys().copied().collect()
            {
                return Err(invalid(
                    "region history differs from the complete physical Field inventory",
                ));
            }
            self.validate_physical(previous)?;
        }
        Ok(step)
    }

    /// One finalized system for execution and coefficient reacceptance.
    pub(crate) fn finalize_linear<'mesh, A: Coefficient + Send + Sync>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
    ) -> Result<
        (
            crate::finalized_spatial::FinalizedLinearCore<A>,
            AssemblyReport,
            eqiora_assembly::LinearSystem<A>,
        ),
        Diagnostic,
    >
    where
        Self: RegionAlgebraicCoordinates<A, S>,
    {
        let _phase =
            eqiora_execution::telemetry_span!(assembly("scalar_region_finalization")).entered();
        self.validate_solve_input(mesh, input)?;
        let multipliers = input
            .forms
            .iter()
            .flat_map(|(form, _)| form.constraint_multipliers())
            .collect::<BTreeSet<_>>();
        let super::assembly::MappedRegionAssembly {
            plan,
            work,
            packet_domains: _,
        } = self.prepare_numeric_assembly(mesh, input)?;
        // Seal the actual packet maps through the existing assembly owner.
        // Reading maps does not evaluate any numeric local contribution.
        let plan = plan.prepare(
            work.packets
                .iter()
                .map(|packet| packet.mappings.to_vec())
                .collect(),
        )?;
        let (mut systems, assembly_report) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &work)?
            .into_parts();
        self.require_numeric_multiplier_closure(systems[0].matrix(), &multipliers)?;
        if input.operator_properties == LinearOperatorProperties::SymmetricPositiveDefinite
            && input.conservative.is_none()
            && input.geometry_action.is_none()
            && input.previous.is_none()
            && input
                .forms
                .iter()
                .all(|(form, quadrature)| form.has_original_q1_diffusion_energy(quadrature))
        {
            let fields = input
                .forms
                .iter()
                .flat_map(|(form, _)| form.fields().iter().map(|field| field.field))
                .collect();
            if !self.nodal_components_are_anchored(&fields)?
                || !work.preserves_q1_diffusion_energy(&plan, systems[0].matrix())?
            {
                return Err(invalid(
                    "bound Q1 diffusion SPD representability margin is unproved",
                ));
            }
        }

        let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &systems[0],
            systems[0].rhs(),
            input.operator_properties,
        )?);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            solver,
            VectorLayoutKind::Replicated,
            Target::HostCpu { threads: workers },
            canonical,
        );
        let full_system = systems.pop().expect("full assembly target");
        Ok((core, assembly_report, full_system))
    }

    /// Execute an exact mapped region system. The caller authenticates
    /// Mesh/Geometry lineage; the common owner checks bound Field layouts and
    /// retains physical coefficient interpretation in the recovered inventory.
    #[cfg(test)] // Cold test reference; production callers supply their check explicitly.
    pub(crate) fn solve<'mesh, A: Coefficient + Send + Sync>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, A>,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver<A>>,
    ) -> Result<RegionSolveOutput<S>, Diagnostic>
    where
        Self: RegionAlgebraicCoordinates<A, S>,
    {
        self.solve_with_candidate_check(mesh, input, workers, request, prepared, None)
    }

    /// Primal physical checks accompany the original canonical solve; derivative
    /// right-hand sides use `solve` without a physical-State criterion.
    pub(crate) fn solve_with_candidate_check<'mesh, A: Coefficient + Send + Sync>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, A>,
        prepared: Option<&mut dyn eqiora_solver::PreparedLinearSolver<A>>,
        candidate_check: Option<&RegionCandidateCheck<'_, S>>,
    ) -> Result<RegionSolveOutput<S>, Diagnostic>
    where
        Self: RegionAlgebraicCoordinates<A, S>,
    {
        let (core, assembly_report, _) =
            self.finalize_linear::<A>(mesh, input, workers, request.plan())?;
        let step = kinematic_step(&input.forms)?;
        let recover = |values: &[A]| {
            let values = self.decode_coordinates(values)?;
            if let Some(step) = &step {
                self.recover_step(
                    &values,
                    input.previous.as_ref().expect("validated step history"),
                    step,
                    &input.prescribed_states,
                )
            } else {
                self.recover(&values, &self.fields.keys().copied().collect::<Vec<_>>())
            }
        };
        let check = |values: &[A]| {
            candidate_check.expect("attached physical candidate check")(&recover(values)?)
        };
        let mut problem = core.linear_problem()?;
        if candidate_check.is_some() {
            problem = problem.with_candidate_check(&check);
        }
        let residual_target = core.residual_target()?;
        let solution = match prepared {
            Some(prepared) => prepared.solve(
                assembly_report.structure_identity().ok_or_else(|| {
                    invalid("prepared Region solve requires its assembly structure identity")
                })?,
                &problem,
            )?,
            None => request.solve(&problem)?,
        };
        core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        let fields = recover(&values)?;
        Ok(RegionSolveOutput {
            fields,
            solve_report,
            assembly_report,
            residual_target,
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
