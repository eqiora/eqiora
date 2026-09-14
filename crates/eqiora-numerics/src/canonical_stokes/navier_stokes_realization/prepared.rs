use super::*;

/// Ephemeral invariant structure for one exact fixed-mesh MINI Run.
pub(crate) struct PreparedResolvedTransientMiniRun2d<'a> {
    model: TransientIncompressibleNavierStokesCartesianModel2d,
    common: TransientIncompressibleNavierStokesModel2d,
    resolved: &'a ResolvedTransientFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    physical_mesh: &'a SimplicialMesh,
    normalized: NormalizedCartesianSimplicialMesh2d,
    scales: IncompressibleFlowScaleProfile2d,
    numerical_plan: MiniNavierStokesStepPlan2d,
    realization_graph: PortableRealizationGraph,
    block_system: DiscreteBlockSystem,
    step_structure: PreparedStepStructure,
    assembly: &'a dyn AssemblyBackend,
    cell_quadrature: QuadratureRule,
    facet_quadrature: QuadratureRule,
    with_gauge: bool,
}

impl PreparedResolvedTransientMiniRun2d<'_> {
    pub(crate) fn advance(
        &self,
        initial: TransientNavierStokesInitialState2d,
        run: TransientNavierStokesRun2d,
        solver: &dyn LinearSolverBackend,
    ) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
        let mut prepared_linear = solver.prepare_linear(self.numerical_plan.linear_solver())?;
        self.advance_with_linear(initial, run, solver, prepared_linear.as_deref_mut())
    }

    pub(crate) fn advance_with_linear(
        &self,
        initial: TransientNavierStokesInitialState2d,
        run: TransientNavierStokesRun2d,
        solver: &dyn LinearSolverBackend,
        prepared_linear: Option<&mut (dyn eqiora_solver::PreparedLinearSolver + '_)>,
    ) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
        if initial.mesh_artifact != self.mesh_artifact
            || initial.velocity_field != velocity_id(&self.common)
            || initial.pressure_field != pressure_id(&self.common)
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
        let numerical_initial = normalize_state(
            &initial,
            &self.normalized.mesh,
            self.scales,
            self.with_gauge,
        )?;
        let checked_assembly = self.block_system.checked_backend(self.assembly);
        let model = &self.model;
        let common = &self.common;
        let scales = self.scales;
        let lower = [model.bounds()[0][0], model.bounds()[1][0]];
        let length = scales.length_value();
        let pressure = scales.pressure_value();
        let body_force = |coordinate_hat: [f64; DIMENSION]| {
            let coordinate = [
                lower[0] + length * coordinate_hat[0],
                lower[1] + length * coordinate_hat[1],
            ];
            let force = common.conservative_body_force(&coordinate)?;
            Ok([length * force[0] / pressure, length * force[1] / pressure])
        };
        let numerical =
            advance_simplicial_mini_navier_stokes_2d_with_prepared_structure_and_linear(
                &self.normalized.mesh,
                &self.step_structure,
                &body_force,
                numerical_initial,
                run.step_count,
                self.numerical_plan,
                &self.cell_quadrature,
                &self.facet_quadrature,
                &checked_assembly,
                solver,
                prepared_linear,
            )?;
        let validated_block_materializations = checked_assembly.validated_materialization_count();
        if validated_block_materializations == 0 {
            return Err(invalid_realization(
                "transient execution returned without a validated block materialization",
            ));
        }
        let states = numerical
            .states()
            .iter()
            .enumerate()
            .map(|(index, state)| {
                reconstruct_state(
                    state,
                    self.physical_mesh,
                    &self.common,
                    self.scales,
                    index.checked_sub(1).map(|step| &numerical.steps()[step]),
                )
            })
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
            validated_block_materializations,
        })
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
    assembly: &'a dyn AssemblyBackend,
) -> Result<PreparedResolvedTransientMiniRun2d<'a>, Diagnostic> {
    let mesh_artifact = mesh.artifact_reference()?;
    if program.model() != resolved.model()
        || program.revision().0 != resolved.semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved transient realization does not reference this exact Semantic Model revision",
        ));
    }
    let model = lower_transient_incompressible_navier_stokes_cartesian_2d(program)?;
    let common = model.common_projection();
    let with_gauge = boundary::pressure_uses_gauge(&common)?;
    let realization_graph = resolved.portable_graph()?;
    let (scales, numerical_plan) =
        require_exact_transient_plan(&common, resolved, &realization_graph, mesh_artifact)?;
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
        &common,
        mesh_artifact,
        &normalized.mesh,
        &boundary,
        resolved,
        scales,
    )?;
    let essential_velocity =
        |coordinate_hat| boundary::essential_velocity(&model, scales, coordinate_hat);
    let cell_quadrature = triangle_duffy_gauss_legendre(DUFFY_POINTS_PER_AXIS)?;
    let facet_quadrature = simplex_duffy_gauss_legendre(DIMENSION - 1, 2)?;
    let step_structure = prepare_step_structure(
        &normalized.mesh,
        &boundary,
        &essential_velocity,
        &cell_quadrature,
        &facet_quadrature,
    )?;
    Ok(PreparedResolvedTransientMiniRun2d {
        model,
        common,
        resolved,
        mesh_artifact,
        physical_mesh: mesh.mesh(),
        normalized,
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
