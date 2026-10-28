use super::*;

/// Ephemeral invariant structure for one exact fixed-mesh MINI Run.
pub(crate) struct PreparedResolvedTransientMiniRun2d<'a> {
    pub(in crate::canonical_stokes) model: TransientIncompressibleNavierStokesModel<2>,
    pub(in crate::canonical_stokes) resolved: &'a ResolvedTransientFieldwiseRealization,
    pub(in crate::canonical_stokes) mesh_artifact: MeshArtifactReference,
    pub(in crate::canonical_stokes) physical_mesh: &'a SimplicialMesh,
    pub(in crate::canonical_stokes) normalized: SimplicialMesh,
    pub(in crate::canonical_stokes) scales: IncompressibleFlowScaleProfile2d,
    pub(in crate::canonical_stokes) numerical_plan: MiniNavierStokesStepPlan2d,
    pub(in crate::canonical_stokes) realization_graph: PortableRealizationGraph,
    pub(in crate::canonical_stokes) block_system: DiscreteBlockSystem,
    pub(in crate::canonical_stokes) step_structure: PreparedStepStructure,
    pub(in crate::canonical_stokes) assembly: &'a dyn AssemblyBackend<f64>,
    pub(in crate::canonical_stokes) cell_quadrature: QuadratureRule,
    pub(in crate::canonical_stokes) facet_quadrature: QuadratureRule,
    pub(in crate::canonical_stokes) with_gauge: bool,
}

impl PreparedResolvedTransientMiniRun2d<'_> {
    pub(crate) fn advance(
        &self,
        initial: TransientNavierStokesInitialState2d,
        run: TransientNavierStokesRun2d,
        solver: &dyn LinearSolverBackend,
    ) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
        let numerical_initial = self.normalize_state(&initial)?;
        let checked_assembly = self.block_system.checked_backend(self.assembly);
        let numerical = advance_simplicial_mini_navier_stokes_2d_with_prepared_structure(
            &self.normalized,
            &self.step_structure,
            numerical_initial,
            run.step_count,
            self.numerical_plan.clone(),
            &self.cell_quadrature,
            &self.facet_quadrature,
            &checked_assembly,
            solver,
        )?;
        self.reconstruct(numerical, solver)
    }

    pub(crate) fn advance_with_linear(
        &self,
        initial: TransientNavierStokesInitialState2d,
        run: TransientNavierStokesRun2d,
        solver: &dyn LinearSolverBackend,
        prepared_linear: Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
    ) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
        let numerical_initial = self.normalize_state(&initial)?;
        let checked_assembly = self.block_system.checked_backend(self.assembly);
        let numerical =
            advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear(
                &self.normalized,
                &self.step_structure,
                numerical_initial,
                run.step_count,
                self.numerical_plan.clone(),
                &self.cell_quadrature,
                &self.facet_quadrature,
                &checked_assembly,
                solver,
                prepared_linear,
            )?;
        self.reconstruct(numerical, solver)
    }

    fn reconstruct(
        &self,
        numerical: crate::simplicial_navier_stokes::SimplicialMiniNavierStokesTrajectory2d,
        solver: &dyn LinearSolverBackend,
    ) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
        let states = numerical
            .states()
            .iter()
            .map(|state| reconstruct_state(state, self.physical_mesh, &self.model, self.scales))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(ResolvedTransientNavierStokesTrajectory2d {
            model: self.model.clone(),
            realization: self.resolved.clone(),
            realization_graph: self.realization_graph.clone(),
            solver_backend: solver.id(),
            mesh_artifact: self.mesh_artifact,
            scales: self.scales,
            states,
            steps: numerical.steps().to_vec(),
        })
    }
    fn normalize_state(
        &self,
        initial: &TransientNavierStokesInitialState2d,
    ) -> Result<SimplicialMiniNavierStokesState2d, Diagnostic> {
        if initial.mesh_artifact != self.mesh_artifact
            || initial.velocity_field != velocity_id(&self.model)
            || initial.pressure_field != pressure_id(&self.model)
        {
            return Err(invalid_realization(
                "transient initial state identity differs from the resolved Model or mesh revision",
            ));
        }
        if initial.velocity.mesh() != self.physical_mesh
            || initial.pressure.mesh() != self.physical_mesh
        {
            return Err(invalid_realization(
                "transient Navier--Stokes initial fields are stale for the selected mesh artifact",
            ));
        }
        normalize_state(initial, &self.normalized, self.scales, self.with_gauge)
    }

    pub(crate) fn boundary_force(
        &self,
        previous: &TransientNavierStokesInitialState2d,
        current: &TransientNavierStokesInitialState2d,
        name: &str,
    ) -> Result<[f64; 2], Diagnostic> {
        let previous = self.normalize_state(previous)?;
        let current = self.normalize_state(current)?;
        let force = crate::simplicial_navier_stokes::boundary_reaction_prepared(
            &self.normalized,
            &self.step_structure,
            &previous,
            &current,
            &self.numerical_plan,
            name,
        )?;
        let scale = self.scales.pressure_value() * self.scales.length_value();
        let force = force.map(|value| value * scale);
        if force.iter().any(|value| !value.is_finite()) {
            return Err(invalid_realization("physical boundary force is non-finite"));
        }
        Ok(force)
    }
}

pub(crate) fn prepare_resolved_transient_navier_stokes_mini_run_2d<'a>(
    program: &KernelProgram,
    resolved: &'a ResolvedTransientFieldwiseRealization,
    mesh: &'a SimplicialMeshEnvelopeV1,
) -> Result<PreparedResolvedTransientMiniRun2d<'a>, Diagnostic> {
    prepare_resolved_transient_navier_stokes_mini_run_2d_with_assembly(
        program,
        resolved,
        mesh,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

pub(super) fn prepare_resolved_transient_navier_stokes_mini_run_2d_with_assembly<'a>(
    program: &KernelProgram,
    resolved: &'a ResolvedTransientFieldwiseRealization,
    mesh: &'a SimplicialMeshEnvelopeV1,
    assembly: &'a dyn AssemblyBackend<f64>,
) -> Result<PreparedResolvedTransientMiniRun2d<'a>, Diagnostic> {
    let mesh_artifact = mesh.artifact_reference()?;
    if program.model() != resolved.model()
        || program.revision().0 != resolved.semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved transient realization does not reference this exact Semantic Model revision",
        ));
    }
    let model = lower_transient_incompressible_navier_stokes_cartesian::<2>(program)?;
    let with_gauge = boundary::pressure_uses_gauge(&model)?;
    let realization_graph = resolved.portable_graph()?;
    let (scales, numerical_plan) =
        require_exact_transient_plan(program, &model, resolved, &realization_graph, mesh_artifact)?;
    let normalized = normalize_cartesian_mesh(
        model.bounds(),
        mesh.mesh(),
        scales.length_value(),
        "Navier--Stokes",
    )?;
    let boundary = boundary::numerical_boundary(&model, &normalized)?;
    if with_gauge {
        boundary::require_compatible_complete_trace(&model, &normalized, scales)?;
    }
    let block_system = super::super::block::transient_navier_stokes_block_system(
        program,
        &model,
        mesh_artifact,
        &normalized.mesh,
        &boundary,
        resolved,
        scales,
    )?;
    let essential_velocity =
        |coordinate_hat| boundary::essential_velocity(&model, scales, coordinate_hat);
    let cell_quadrature = simplex_duffy_gauss_legendre(2, DUFFY_POINTS_PER_AXIS)?;
    let facet_quadrature = simplex_duffy_gauss_legendre(DIMENSION - 1, 2)?;
    let step_structure = prepare_step_structure(
        &normalized.mesh,
        &boundary,
        &essential_velocity,
        &cell_quadrature,
        &facet_quadrature,
        &numerical_plan,
    )?
    .bind(&|_| Ok([0.0; 2]))?;
    Ok(PreparedResolvedTransientMiniRun2d {
        model,
        resolved,
        mesh_artifact,
        physical_mesh: mesh.mesh(),
        normalized: normalized.mesh,
        scales,
        numerical_plan,
        realization_graph,
        block_system,
        step_structure,
        assembly,
        cell_quadrature,
        facet_quadrature,
        with_gauge,
    })
}
