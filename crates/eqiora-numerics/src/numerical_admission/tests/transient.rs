use super::*;

fn cartesian_rectangle_resources(
    geometry: &CanonicalGeometryV1,
    cells: [usize; 2],
) -> AuthenticatedCommonMesh {
    let policy = CartesianMeshCellsV2::new(cells.to_vec()).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(geometry, cells)
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &policy,
        geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh,
        correspondence,
        production,
    )
    .unwrap()
}

fn newton_policy(linear: CommonLinearRequest, nonlinear: NonlinearSolvePlan) -> CommonSolvePolicy {
    CommonSolvePolicy::Newton { nonlinear, linear }
}

#[test]
fn scalar_q1_storage_initializes_spatial_data_at_mesh_vertices_and_advances_steadily() {
    let geometry = rectangle();
    let source = r#"
public component AffineStorage(
  support body: volume(ambient_dimension = 2),
  support left: boundary(parent = body),
  support right: boundary(parent = body),
  support bottom: boundary(parent = body),
  support top: boundary(parent = body),
  parameter diffusivity: m ^ 2 / s
) {
  coordinate x: m on body from body[0];
  state u: m on body in h1;
  initial { u = x; }
  law balance on body {
    storage u;
    flux -diffusivity * grad(u);
    source 0 [m / s];
  }
  relation left_value on left { trace(u) = 0 [m]; }
  relation right_value on right { trace(u) = 1 [m]; }
  relation bottom_value on bottom { trace(u) = coordinate(0); }
  relation top_value on top { trace(u) = coordinate(0); }
}
"#;
    let supports = [
        ("body", "region", None),
        ("left", "left", Some("body")),
        ("right", "right", Some("body")),
        ("bottom", "bottom", Some("body")),
        ("top", "top", Some("body")),
    ]
    .map(|(name, set, parent)| {
        let selection = geometry.entity_set(set).unwrap();
        (
            name,
            selection,
            parent.map(|parent| (parent, geometry.entity_set("region").unwrap())),
        )
    });
    let diffusion_dimension = DimExponents::from_integers([0, 2, -1, 0, 0, 0, 0]).unwrap();
    let model = compile_model(
        "affine-storage.eqi",
        source,
        &geometry,
        "AffineStorage",
        &supports,
        &[("diffusivity", DynQuantity::new(1.0, diffusion_dimension))],
    );
    let resources = cartesian_rectangle_resources(&geometry, [2, 2]);
    let mesh = resources.cartesian_mesh().unwrap().mesh();
    let vertex_points = (0..mesh.entity_count(0).unwrap())
        .map(|index| {
            mesh.vertex_coordinates(eqiora_meshing::MeshEntity::new(0, index))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let temporal = CommonBackwardEuler::from_seconds(0.125).unwrap();
    let linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-11,
        1e-13,
        NonZeroUsize::new(1000).unwrap(),
    );
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        resources,
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(linear),
        None,
        Some(temporal),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let plan = resolved.as_linear().unwrap();
    let initial = plan.initial_state().unwrap();
    let initial_values = initial.scalar_values().unwrap();
    assert_eq!(initial_values.len(), vertex_points.len());
    for (point, value) in vertex_points.iter().zip(initial_values) {
        let expected = point[0];
        assert!((value - expected).abs() < 1e-12, "{value} != {expected}");
    }

    let run = CommonTransientRunRequest::from_steps(resolved, initial.clone(), 1, vec![1]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("one accepted scalar step should complete the Run");
    };
    let accepted = &outputs[0].1;
    assert_eq!(accepted.time_s().to_bits(), 0.125_f64.to_bits());
    for (actual, expected) in accepted.scalar_values().unwrap().iter().zip(initial_values) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    let mismatched_source = source.replace("trace(u) = 1 [m]", "trace(u) = 0 [m]");
    let mismatched = compile_model(
        "affine-storage-boundary-mismatch.eqi",
        &mismatched_source,
        &geometry,
        "AffineStorage",
        &supports,
        &[("diffusivity", DynQuantity::new(1.0, diffusion_dimension))],
    );
    let mismatched_plan = ResolvedCommonPlan::resolve(
        &mismatched,
        cartesian_rectangle_resources(&geometry, [2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(linear),
        None,
        Some(temporal),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let mismatch = mismatched_plan
        .as_linear()
        .unwrap()
        .initial_state()
        .unwrap_err();
    assert!(
        mismatch
            .message()
            .contains("scalar State contradicts prescribed boundary values")
    );

    let singular_source = source.replace("u = x", "u = 1 [m ^ 2] / (x - 1 [m])");
    let singular = compile_model(
        "affine-storage-nonfinite-node.eqi",
        &singular_source,
        &geometry,
        "AffineStorage",
        &supports,
        &[("diffusivity", DynQuantity::new(1.0, diffusion_dimension))],
    );
    let singular_plan = ResolvedCommonPlan::resolve(
        &singular,
        cartesian_rectangle_resources(&geometry, [2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-11,
            1e-13,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        Some(temporal),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let nonfinite = singular_plan
        .as_linear()
        .unwrap()
        .initial_state()
        .unwrap_err();
    assert!(
        nonfinite
            .message()
            .contains("non-finite linear coefficient data")
    );
}

fn assert_solver_structure(
    plan: &CommonTransientFlowPlan,
    backend: &dyn LinearSolverBackend,
    with_gauge: bool,
) {
    use eqiora_solver::{AlgebraicConstraint, AlgebraicStructure, HostSerialSolverProfile};
    let (velocity, pressure) = match plan.admission.recognized_model() {
        RecognizedNativeModel::Transient(model) => (model.velocity(), model.pressure()),
        RecognizedNativeModel::TransientGeometry(binding) => {
            (binding.velocity(), binding.pressure())
        }
        _ => panic!("transient fixture"),
    };
    let velocity = velocity.downcast().unwrap();
    let pressure = pressure.downcast().unwrap();
    let gauge = AlgebraicConstraint::ZeroIntegral { field: pressure };
    let constraints: Vec<_> = with_gauge.then_some(gauge).into_iter().collect();
    let expected = AlgebraicStructure::new([velocity, pressure], constraints.clone()).unwrap();
    plan.admission
        .linear
        .planning_profile
        .as_ref()
        .unwrap()
        .require_structure(Some(&expected))
        .unwrap();
    let foreign = eqiora_core::Id::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap());
    let state = if with_gauge {
        plan.zero_state(0.0).unwrap()
    } else {
        let NativeMeshResources::AffineTriangleSimplicial { mesh, .. } = plan.admission.resources()
        else {
            panic!("traction fixture owns affine triangles");
        };
        let vertices = mesh.mesh().vertices().len();
        let cells = mesh.mesh().entity_count(2).unwrap();
        let digest = plan.admission.model().digest().unwrap();
        plan.initial_state(
            0.0,
            vec![
                CommonInitialField::new(
                    digest.clone(),
                    velocity,
                    Some(CommonInitialValues::Vector2(
                        vec![[0.0; 2]; vertices].into_boxed_slice(),
                    )),
                    Some(CommonInitialValues::Vector2(
                        vec![[0.0; 2]; cells].into_boxed_slice(),
                    )),
                )
                .unwrap(),
                CommonInitialField::new(
                    digest,
                    pressure,
                    Some(CommonInitialValues::Scalar(
                        vec![0.0; vertices].into_boxed_slice(),
                    )),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    };
    for structure in [
        None,
        Some(AlgebraicStructure::new([velocity], []).unwrap()),
        Some(AlgebraicStructure::new([pressure], constraints.clone()).unwrap()),
        Some(AlgebraicStructure::new([foreign, pressure], constraints.clone()).unwrap()),
        Some(AlgebraicStructure::new([velocity, pressure, foreign], constraints).unwrap()),
        Some(
            AlgebraicStructure::new(
                [velocity, pressure],
                [AlgebraicConstraint::ZeroIntegral { field: velocity }],
            )
            .unwrap(),
        ),
        Some(
            AlgebraicStructure::new([velocity, pressure], (!with_gauge).then_some(gauge)).unwrap(),
        ),
    ] {
        let mut changed = plan.clone();
        let profile =
            HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
        changed.admission.linear.planning_profile = Some(match structure {
            Some(structure) => profile.with_structure(structure).unwrap(),
            None => profile,
        });
        let error = changed
            .prepare_execution(&state, backend)
            .err()
            .expect("stale profile reached preparation");
        assert!(error.message().contains("structure"), "{error:?}");
    }
    let mut omitted = plan.clone();
    omitted.admission.linear.planning_profile = None;
    assert!(omitted.prepare_execution(&state, backend).is_err());
}

#[test]
fn prepared_transient_methods_keep_authoritative_common_grid_time_bits() {
    let geometry = rectangle();
    let model = transient_model();
    let step_s = 0.0001_f64;
    let temporal = CommonBackwardEuler::from_seconds(step_s).unwrap();
    let scaling =
        IncompressibleScalingRequest2d::from_si(Some(0.41), Some(0.3), Some(0.09)).unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap();
    let resolve = |owner, spatial| {
        let linear = if spatial == CommonSpatialPolicy::MiniP1 {
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-10,
                    1e-12,
                    NonZeroUsize::new(2_000).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Identity)
                .with_reduction(ReductionPolicy::Fast),
                ResolveOnlyBackend.provider(),
            )
            .unwrap()
        } else {
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            )
        };
        ResolvedCommonPlan::resolve(
            &model,
            owner,
            spatial,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .unwrap()
        .as_transient_flow()
        .cloned()
        .expect("fixture retains its admitted transient_flow Plan")
    };

    let mini = resolve(affine_resources(&geometry), CommonSpatialPolicy::MiniP1);
    let cell_resources = {
        let cells = CartesianMeshCellsV2::new([3, 4]).unwrap();
        let (mesh, correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
                &geometry,
                cells.cells().try_into().unwrap(),
            )
            .unwrap();
        let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
            &cells,
            &geometry,
            &mesh,
            &correspondence,
        )
        .unwrap();
        AuthenticatedCommonMesh::structured_cartesian(
            geometry.clone(),
            mesh,
            correspondence,
            production,
        )
        .unwrap()
    };
    let cell_centered = resolve(cell_resources, CommonSpatialPolicy::CellCentered);
    for (plan, backend) in [
        (&mini, &ResolveOnlyBackend as &dyn LinearSolverBackend),
        (&cell_centered, &REFERENCE_LINEAR_SOLVER),
    ] {
        let previous = plan.zero_state(0.0).unwrap();
        let accepted = plan.advance_one(&previous, backend).unwrap();
        assert_eq!(accepted.time_s().to_bits(), step_s.to_bits());
        assert!(Arc::ptr_eq(&accepted.model, &previous.model));
        assert!(Arc::ptr_eq(&accepted.resources, &previous.resources));
    }
}

#[test]
pub(super) fn transient_common_plan_resolves_exact_mini_and_supplied_cartesian_resources() {
    let geometry = rectangle();
    let model = transient_model();
    let replayed = ModelEnvelope::from_json(
        &model.canonical_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-10,
            1e-12,
            NonZeroUsize::new(2_000).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast),
        ResolveOnlyBackend.provider(),
    )
    .unwrap();
    let temporal = CommonBackwardEuler::from_seconds(0.01).unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap();
    let scaling = IncompressibleScalingRequest2d::from_si(Some(1.0), Some(2.0), Some(3.0)).unwrap();
    let resolve = |model: &ModelEnvelope, owner, spatial, formulation| {
        let linear = if spatial == CommonSpatialPolicy::MiniP1 {
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-10,
                    1e-12,
                    NonZeroUsize::new(2_000).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Identity)
                .with_reduction(ReductionPolicy::Fast),
                ResolveOnlyBackend.provider(),
            )
            .unwrap()
        } else {
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            )
        };
        let method = match formulation {
            None => CommonMethodRequest::Uniform(spatial),
            Some(formulation) => CommonMethodRequest::Exact {
                spatial,
                formulation,
            },
        };
        let resolved = ResolvedCommonPlan::resolve(
            model,
            owner,
            method,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .unwrap();
        replay_plan(resolved, &ResolveOnlyBackend)
            .as_transient_flow()
            .cloned()
            .expect("fixture retains its admitted transient_flow Plan")
    };
    let mini = resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    let mini_replay = resolve(
        &replayed,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    let fvm = resolve(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        None,
    );
    let resolve_program_controlled = |objective| {
        let linear = CommonLinearRequest::program_controlled(
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(2_000).unwrap(),
            objective,
        )
        .unwrap();
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::CellCentered,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &PlanningFaerBackend,
            None,
        )
        .unwrap()
        .as_transient_flow()
        .cloned()
        .expect("fixture retains its admitted transient_flow Plan")
    };
    let robust = resolve_program_controlled(SolverPlanningObjective::Robust);
    let fast = resolve_program_controlled(SolverPlanningObjective::Fast);
    let low_memory = resolve_program_controlled(SolverPlanningObjective::LowMemory);
    assert_solver_structure(&mini, &ResolveOnlyBackend, true);
    assert_solver_structure(&fvm, &REFERENCE_LINEAR_SOLVER, true);
    for planned in [&robust, &fast, &low_memory] {
        assert_solver_structure(planned, &PlanningFaerBackend, true);
    }
    // Replacing one essential boundary with zero traction removes the pressure
    // nullspace mathematically; the solver profile must lose exactly that gauge.
    let source = TRANSIENT_SOURCE
        .replace("state velocity: vector<m / s, 2> on body in h1;", "state velocity: vector<m / s, 2> on body in smooth;")
        .replace("variable pressure: kg / (m * s ^ 2) on body;", "variable pressure: kg / (m * s ^ 2) on body in h1;")
        .replace(
        "relation y_upper_value on y_upper { trace(velocity) = 0; }",
        "relation y_upper_value on y_upper { normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) = 0; }",
    );
    let compiled = eqiora_compiler::compile("mixed-transient.eqi", &source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model_id, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let traction_model = ModelEnvelope::from_program(&program).unwrap();
    let traction = resolve(
        &traction_model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    assert_solver_structure(&traction, &ResolveOnlyBackend, false);

    let mini_exact = resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        Some(FormulationKind::MixedGalerkin),
    );
    let fvm_exact = resolve(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        Some(FormulationKind::IntegralConservative),
    );
    let custom_nonlinear =
        NonlinearSolvePlan::new(2.0e-9, 3.0e-11, NonZeroUsize::new(19).unwrap(), 7).unwrap();
    let custom = ResolvedCommonPlan::resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        newton_policy(linear, custom_nonlinear),
        Some(scaling),
        Some(temporal),
        &ResolveOnlyBackend,
        None,
    )
    .unwrap()
    .as_transient_flow()
    .cloned()
    .expect("fixture retains its admitted transient_flow Plan");
    let alternate_scaling =
        IncompressibleScalingRequest2d::from_si(Some(4.0), Some(5.0), Some(6.0)).unwrap();
    let fvm_alternate_scaling = ResolvedCommonPlan::resolve(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        newton_policy(
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            ),
            nonlinear,
        ),
        Some(alternate_scaling),
        Some(temporal),
        &ResolveOnlyBackend,
        None,
    )
    .unwrap()
    .as_transient_flow()
    .cloned()
    .expect("fixture retains its admitted transient_flow Plan");

    assert_eq!(mini.identity(), mini_replay.identity());
    assert_eq!(mini.realization_digest(), mini_replay.realization_digest());
    assert_ne!(mini.realization_digest(), fvm.realization_digest());
    assert_eq!(
        mini.realization_digest(),
        hex_bytes(&mini.portable_realization().digest().unwrap())
    );
    assert_ne!(mini.identity(), mini_exact.identity());
    assert_ne!(fvm.identity(), fvm_exact.identity());
    assert_ne!(robust.identity(), fast.identity());
    assert_ne!(fast.identity(), low_memory.identity());
    assert_eq!(
        robust.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        fast.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        low_memory.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    for plan in [&robust, &fast, &low_memory] {
        assert_eq!(
            plan.solver_planning_policy_id(),
            Some("eqiora.host-serial-solver-planning/v2")
        );
        assert_eq!(plan.solver_planning_reasons().len(), 4);
        assert!(plan.selected_solver_evidence_case().is_some());
    }
    let planned_mini = ResolvedCommonPlan::resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        newton_policy(
            CommonLinearRequest::program_controlled(
                1.0e-10,
                1.0e-12,
                NonZeroUsize::new(2_000).unwrap(),
                SolverPlanningObjective::Robust,
            )
            .unwrap(),
            nonlinear,
        ),
        Some(scaling),
        Some(temporal),
        &PlanningFaerBackend,
        None,
    )
    .unwrap();
    assert_eq!(
        planned_mini.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        mini.formulation().effective(),
        mini_exact.formulation().effective()
    );
    assert_eq!(
        fvm.formulation().effective(),
        fvm_exact.formulation().effective()
    );
    assert_eq!(
        mini_exact.formulation().requested(),
        FormulationSelectionMode::Exact
    );
    assert_eq!(
        fvm_exact.formulation().requested(),
        FormulationSelectionMode::Exact
    );
    assert_eq!(
        mini.state_space_identity(),
        mini_exact.state_space_identity()
    );
    assert_eq!(fvm.state_space_identity(), fvm_exact.state_space_identity());
    assert_eq!(mini.realization_digest(), mini_exact.realization_digest());
    assert_eq!(fvm.realization_digest(), fvm_exact.realization_digest());
    assert_ne!(
        fvm.realization_digest(),
        fvm_alternate_scaling.realization_digest()
    );
    assert_eq!(
        mini_exact.formulation().selection_reason_codes(),
        &["eqiora.formulation.exact.mixed-galerkin-admitted/v1"]
    );
    assert_eq!(
        fvm_exact.formulation().selection_reason_codes(),
        &["eqiora.formulation.exact.integral-conservative-admitted/v1"]
    );
    assert_ne!(mini.identity(), fvm.identity());
    assert_ne!(mini.identity(), custom.identity());
    assert_eq!(custom.nonlinear(), custom_nonlinear);
    assert_eq!(mini.model_digest(), model.digest().unwrap().to_string());
    assert_eq!(mini.velocity_field_id(), fvm.velocity_field_id());
    assert_eq!(mini.pressure_field_id(), fvm.pressure_field_id());
    assert_eq!(mini.velocity_space().family(), SpaceFamily::SimplexP1Bubble);
    assert!(
        matches!(mini.pressure_space().family(), SpaceFamily::ContinuousLagrange { order } if order.get() == 1)
    );
    assert_eq!(fvm.velocity_space().family(), SpaceFamily::CellConstant);
    assert_eq!(fvm.pressure_space().family(), SpaceFamily::CellConstant);
    assert_eq!(mini.temporal().step().value().to_bits(), 0.01_f64.to_bits());
    assert_eq!(mini.scales().length().value().to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        mini.scales().velocity().value().to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(
        mini.scales().pressure().value().to_bits(),
        3.0_f64.to_bits()
    );
    assert_eq!(mini.linear().algorithm(), LinearSolver::SparseLu);
    assert_eq!(mini.linear().reduction(), ReductionPolicy::Fast);
    assert_eq!(fvm.linear().reduction(), ReductionPolicy::Reproducible);
    let mini_formulation = mini.formulation();
    assert_eq!(
        mini_formulation.requested(),
        FormulationSelectionMode::Automatic
    );
    assert_eq!(mini_formulation.effective(), FormulationKind::MixedGalerkin);
    assert_eq!(
        mini_formulation.boundary_treatment(),
        "explicit-trace-flux-laws"
    );
    assert_eq!(mini_formulation.rule_ids().len(), 6);
    assert_eq!(mini_formulation.selection_reason_codes().len(), 1);
    let fvm_formulation = fvm.formulation();
    assert_eq!(
        fvm_formulation.effective(),
        FormulationKind::IntegralConservative
    );
    assert_eq!(fvm_formulation.rule_ids().len(), 7);
    assert_ne!(mini_formulation, fvm_formulation);

    let mini_zero = mini.zero_state(0.0).unwrap();
    let fvm_zero = fvm.zero_state(0.0).unwrap();
    let mini_bytes = mini_zero.to_bytes().unwrap();
    assert_eq!(
        CommonState::from_bytes(
            &mini_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .unwrap(),
        mini_zero
    );
    let fvm_bytes = fvm_zero.to_bytes().unwrap();
    assert_eq!(
        CommonState::from_bytes(
            &fvm_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(fvm.clone())),
        )
        .unwrap(),
        fvm_zero
    );
    let mut noncanonical = mini_bytes;
    noncanonical.push(b'\n');
    assert!(
        CommonState::from_bytes(
            &noncanonical,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .is_err()
    );
    assert!(
        CommonState::from_bytes(
            &fvm_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .is_err()
    );
    assert_eq!(mini_zero.velocity_vertex_values().unwrap().len(), 12);
    assert_eq!(mini_zero.velocity_cell_values().unwrap().len(), 12);
    assert_eq!(mini_zero.pressure_vertex_values().unwrap().len(), 12);
    assert!(mini_zero.method_history_values().is_empty());
    assert_eq!(fvm_zero.velocity_cell_values().unwrap().len(), 6);
    assert_eq!(fvm_zero.pressure_cell_values().unwrap().len(), 6);
    assert!(!fvm_zero.method_history_values().is_empty());
    let curl = mini.cell_average_velocity_curl_2d(&mini_zero).unwrap();
    assert_eq!(curl.as_ref(), &[0.0; 12]);
    assert_eq!(
        curl,
        custom.cell_average_velocity_curl_2d(&mini_zero).unwrap(),
        "derived values exclude solve policy when the exact State and field are unchanged",
    );
    assert!(fvm.cell_average_velocity_curl_2d(&fvm_zero).is_err());
    assert_eq!(custom.state_space_identity(), mini.state_space_identity());
    assert_eq!(
        fvm_alternate_scaling.state_space_identity(),
        fvm.state_space_identity(),
        "coherent-SI State compatibility excludes numerical scaling",
    );
    assert!(
        CommonTransientRunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(custom.clone())),
            mini_zero.clone(),
            2,
            vec![1, 2],
        )
        .is_ok()
    );
    assert!(
        CommonTransientRunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
            fvm_zero,
            1,
            vec![1],
        )
        .is_err()
    );
    let by_steps = CommonTransientRunRequest::from_steps(
        ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        mini_zero.clone(),
        2,
        vec![1, 2],
    )
    .unwrap();
    let by_times = CommonTransientRunRequest::from_times(
        ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        mini_zero,
        0.02,
        vec![0.01, 0.02],
    )
    .unwrap();
    assert_eq!(by_steps.identity(), by_times.identity());
    assert!(
        CommonTransientRunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(mini)),
            by_times.state().clone(),
            2,
            vec![2, 1],
        )
        .is_err()
    );

    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            CommonSolvePolicy::Linear(linear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(IncompressibleScalingRequest2d::from_si(Some(1.0), None, Some(3.0)).unwrap()),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(scaling),
            None,
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            affine_resources(&geometry),
            CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::MiniP1,
                formulation: FormulationKind::IntegralConservative,
            },
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::CellCentered,
                formulation: FormulationKind::MixedGalerkin,
            },
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
}
