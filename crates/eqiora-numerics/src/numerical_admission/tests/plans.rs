mod complex;
mod differentiation;
mod harmonic;
mod interval;
mod nonlinear_interfaces;
mod observables;
mod planar;
mod result_replay;
mod solver_diagonal;
mod stationary_state;
mod storage;
mod structured;
mod temporal_system;
mod tetrahedral;
mod transfer;
mod vector;
mod vector_interfaces;
mod weak_evidence;
use super::*;

const POISSON_INTERVAL: &str = r#"
public component PoissonInterval(
  support body: volume(ambient_dimension = 1),
  support left: boundary(parent = body),
  support right: boundary(parent = body),
  parameter source_scale: 1 / m ^ 2
) {
  variable potential: 1 on body in h1;
  relation balance on body {
    -div(grad(potential)) - source_scale = 0;
  }
  relation left_value on left { trace(potential) = 0; }
  relation right_value on right { trace(potential) = 0; }
}
"#;

const POISSON_BOX: &str = r#"
public component PoissonBox(
  support body: volume(ambient_dimension = 3),
  support x_lower: boundary(parent = body),
  support x_upper: boundary(parent = body),
  support y_lower: boundary(parent = body),
  support y_upper: boundary(parent = body),
  support z_lower: boundary(parent = body),
  support z_upper: boundary(parent = body),
  parameter source_scale: 1 / m ^ 2
) {
  variable potential: 1 on body in h1;
  relation balance on body {
    -div(grad(potential)) - source_scale = 0;
  }
  relation x_lower_value on x_lower { trace(potential) = 0; }
  relation x_upper_value on x_upper { trace(potential) = 0; }
  relation y_lower_value on y_lower { trace(potential) = 0; }
  relation y_upper_value on y_upper { trace(potential) = 0; }
  relation z_lower_value on z_lower { trace(potential) = 0; }
  relation z_upper_value on z_upper { trace(potential) = 0; }
}
"#;

fn cartesian_box_3d() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"x_lower","dimension":2,"members":[0]},{"name":"x_upper","dimension":2,"members":[1]},{"name":"y_lower","dimension":2,"members":[2]},{"name":"y_upper","dimension":2,"members":[3]},{"name":"z_lower","dimension":2,"members":[4]},{"name":"z_upper","dimension":2,"members":[5]},{"name":"body","dimension":3,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    )
    .unwrap()
}

fn cartesian_interval() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 1.0]).unwrap();
    let [left, right]: [_; 2] = interval.boundaries().try_into().unwrap();
    graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![left.into()]),
                ("right".to_owned(), vec![right.into()]),
            ]),
        )
        .unwrap()
}

fn scalar_box_model(
    geometry: &CanonicalGeometryV1,
    source: &str,
    component: &str,
    boundaries: &[&str],
) -> ModelEnvelope {
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(boundaries.iter().map(|&name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    compile_model(
        "poisson-box.eqi",
        source,
        geometry,
        component,
        &supports,
        &[(
            "source_scale",
            DynQuantity::new(
                1.0,
                DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        )],
    )
}

fn cartesian_box_resources(
    geometry: &CanonicalGeometryV1,
    cells: &[usize],
) -> AuthenticatedCommonMesh {
    let policy = CartesianMeshCellsV2::new(cells.to_vec()).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(geometry, policy.cells())
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

fn zero_real_coefficients(plan: &CommonSpatialPlan) -> BTreeMap<eqiora_core::RawId, Vec<f64>> {
    plan.fields()
        .map(|(field, ty)| {
            assert_eq!(ty.scalar_domain(), eqiora_core::ScalarDomain::Real);
            let count = plan.field_coefficient_entities(field).unwrap().len()
                * ty.shape().component_count().unwrap();
            (field.erase(), vec![0.; count])
        })
        .collect()
}

fn resolve_scalar_box(
    model: &ModelEnvelope,
    resources: AuthenticatedCommonMesh,
    spatial: CommonSpatialPolicy,
) -> CommonSpatialPlan {
    let linear = exact_reference_linear(
        if spatial == CommonSpatialPolicy::CellCenteredTpfa {
            LinearSolver::ConjugateGradient
        } else {
            LinearSolver::BiConjugateGradientStabilized
        },
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(10_000).unwrap(),
    );
    ResolvedCommonPlan::resolve(
        model,
        resources,
        spatial,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
    )
    .unwrap()
    .as_spatial()
    .cloned()
    .expect("fixture retains its admitted scalar Plan")
}

#[test]
fn common_scalar_plan_executes_exact_one_and_three_dimensional_meshes() {
    let interval = cartesian_interval();
    let interval_model = scalar_box_model(
        &interval,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    exercise_scalar_box(&interval_model, &interval, &[3]);

    let box_3d = cartesian_box_3d();
    let box_model = scalar_box_model(
        &box_3d,
        POISSON_BOX,
        "PoissonBox",
        &[
            "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
        ],
    );
    exercise_scalar_box(&box_model, &box_3d, &[2, 2, 2]);
}

fn exercise_scalar_box(model: &ModelEnvelope, geometry: &CanonicalGeometryV1, cells: &[usize]) {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let plan = resolve_scalar_box(model, cartesian_box_resources(geometry, cells), spatial);
        assert_eq!(plan.cartesian_cells().unwrap(), cells);
        match (
            cells.len(),
            plan.portable_realization().domains()[0]
                .discretization()
                .mesh(),
        ) {
            (1, MeshPolicy::SuppliedCartesian1d { .. })
            | (2, MeshPolicy::SuppliedCartesian { .. })
            | (3, MeshPolicy::SuppliedCartesian3d { .. }) => {}
            _ => panic!("common linear Plan lost its exact Cartesian dimension"),
        }
        let replayed = replay_plan(
            ResolvedCommonPlan::Spatial(Box::new(plan.clone())),
            &ResolveOnlyBackend,
        )
        .as_spatial()
        .cloned()
        .expect("fixture retains its admitted scalar Plan");
        assert_eq!(replayed.cartesian_cells().unwrap(), cells);
        let result = replayed.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let expected_shape = match spatial {
            CommonSpatialPolicy::Q1 => cells.iter().map(|count| count + 1).collect::<Vec<_>>(),
            CommonSpatialPolicy::CellCenteredTpfa => cells.to_vec(),
            _ => unreachable!("exercise admits scalar policies only"),
        };
        assert_eq!(result.field_block(0, 0).unwrap().2, expected_shape);
        let expected_space = if spatial == CommonSpatialPolicy::Q1 {
            Space::continuous_lagrange(std::num::NonZeroU16::MIN)
        } else {
            Space::cell_constant()
        };
        assert_eq!(
            result.field(0).map(|(_, _, space)| space),
            Some(expected_space)
        );
        let field = replayed.fields().next().unwrap().0;
        let entities = replayed.field_coefficient_entities(field).unwrap();
        assert_eq!(entities.len(), expected_shape.iter().product::<usize>());
        assert!(entities.iter().all(|entity| entity.dimension()
            == if spatial == CommonSpatialPolicy::Q1 {
                0
            } else {
                cells.len()
            }));

        // Both authored potentials are dimensionless. Point/constant coefficients
        // retain those units independently of Cartesian cell sizes.
        assert_eq!(
            result
                .field(0)
                .and_then(|(_, ty, space)| space.coefficient_dimension(ty.dimension())),
            Some(DimExponents::DIMENSIONLESS)
        );
        let decoded =
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap();
        assert_eq!(
            decoded.field(0).map(|(_, _, space)| space),
            Some(expected_space)
        );
        assert_eq!(
            decoded
                .field(0)
                .and_then(|(_, ty, space)| space.coefficient_dimension(ty.dimension())),
            Some(DimExponents::DIMENSIONLESS)
        );
    }
}

#[test]
pub(super) fn scalar_q1_and_tpfa_consume_one_exact_anisotropic_common_mesh() {
    let geometry = rectangle();
    let model = model(&geometry);
    let exact_owner = resources(&geometry);
    let caller_resources = exact_owner.resources.clone();
    let admit = |owner, spatial, policy: NativeLinearPolicy, backend: &dyn LinearSolverBackend| {
        ResolvedCommonPlan::resolve(
            &model,
            owner,
            spatial,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(policy.solver, policy.provider).unwrap(),
            ),
            None,
            None,
            backend,
            None,
            None,
        )
        .unwrap()
        .as_spatial()
        .unwrap()
        .admission
        .clone()
    };
    let q1 = admit(
        exact_owner.clone(),
        CommonSpatialPolicy::Q1,
        general_linear(),
        &REFERENCE_LINEAR_SOLVER,
    );
    let q1_repeat = admit(
        resources(&geometry),
        CommonSpatialPolicy::Q1,
        general_linear(),
        &REFERENCE_LINEAR_SOLVER,
    );
    let tpfa = admit(
        exact_owner,
        CommonSpatialPolicy::CellCenteredTpfa,
        linear(),
        &REFERENCE_LINEAR_SOLVER,
    );
    let alternate_provider = admit(
        resources(&geometry),
        CommonSpatialPolicy::Q1,
        NativeLinearPolicy::exact::<f64>(
            SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1.0e-10,
                1.0e-13,
                NonZeroUsize::new(1000).unwrap(),
            )
            .unwrap(),
            &AlternateScalarBackend,
        )
        .unwrap(),
        &AlternateScalarBackend,
    );

    assert_eq!(q1.model(), &model);
    assert_eq!(q1.model_digest(), model.digest().unwrap().to_string());
    assert_ne!(q1.policy_identity(), tpfa.policy_identity());
    assert_eq!(q1.policy_identity(), q1_repeat.policy_identity());
    assert_ne!(q1.policy_identity(), alternate_provider.policy_identity());
    assert_eq!(q1.resources(), &caller_resources);
    assert_eq!(tpfa.resources(), &caller_resources);
    assert_eq!(q1.resources(), q1_repeat.resources());
    assert!(q1.execute_linear(&AlternateScalarBackend, None).is_err());
    assert_eq!(
        q1.execute_linear(&REFERENCE_LINEAR_SOLVER, None)
            .unwrap()
            .fields[0]
            .2
            .len(),
        12
    );
    assert_eq!(
        tpfa.execute_linear(&REFERENCE_LINEAR_SOLVER, None)
            .unwrap()
            .fields[0]
            .2
            .len(),
        6
    );
}

#[test]
pub(super) fn common_scalar_plan_owns_exact_lineage_and_executes_without_repeated_inputs() {
    let geometry = rectangle();
    let model = model(&geometry);
    let linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(10_000).unwrap(),
    );
    let resolve_scalar = |method, solve| {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            method,
            CommonSolvePolicy::Linear(solve),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
        )
        .unwrap();
        replay_plan(resolved, &ResolveOnlyBackend)
            .as_spatial()
            .cloned()
            .expect("fixture retains its admitted scalar Plan")
    };
    let q1 = resolve_scalar(
        CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1),
        linear,
    );
    let repeat = resolve_scalar(
        CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1),
        linear,
    );
    let exact = resolve_scalar(
        CommonMethodRequest::Exact {
            spatial: CommonSpatialPolicy::Q1,
            formulation: FormulationKind::PrimalGalerkin,
        },
        linear,
    );
    let tpfa = resolve_scalar(
        CommonMethodRequest::Uniform(CommonSpatialPolicy::CellCenteredTpfa),
        exact_reference_linear(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(10_000).unwrap(),
        ),
    );
    let alternate_tolerance = resolve_scalar(
        CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1),
        exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-9,
            1.0e-12,
            NonZeroUsize::new(10_000).unwrap(),
        ),
    );

    assert_eq!(q1.identity(), repeat.identity());
    assert_ne!(q1.identity(), tpfa.identity());
    assert_ne!(q1.identity(), alternate_tolerance.identity());
    assert_ne!(q1.identity(), exact.identity());
    assert_eq!(q1.realization_digest(), exact.realization_digest());
    let automatic_form = q1.formulation().unwrap();
    let exact_form = exact.formulation().unwrap();
    assert_eq!(automatic_form.effective(), FormulationKind::PrimalGalerkin);
    assert_eq!(
        automatic_form.requested(),
        FormulationSelectionMode::Automatic
    );
    assert_eq!(exact_form.requested(), FormulationSelectionMode::Exact);
    assert_eq!(automatic_form.boundary_treatment(), "complete-essential");
    assert_eq!(automatic_form.rule_ids().len(), 4);
    assert_eq!(
        automatic_form.selection_reason_codes(),
        ["eqiora.formulation.auto.primal-galerkin-for-q1/v1"]
    );
    assert!(tpfa.formulation().is_none());
    assert_eq!(q1.realization_digest(), repeat.realization_digest());
    assert_ne!(q1.realization_digest(), tpfa.realization_digest());
    assert_eq!(
        q1.realization_digest(),
        hex_bytes(&q1.portable_realization().digest().unwrap())
    );
    assert_eq!(q1.model_digest(), model.digest().unwrap().to_string());
    assert_eq!(q1.cartesian_cells().unwrap(), [2, 3]);
    let mut crossed_realization = q1.clone();
    crossed_realization.portable = tpfa.portable_realization().clone();
    assert!(
        crossed_realization
            .run(None, &REFERENCE_LINEAR_SOLVER)
            .is_err()
    );
    assert_eq!(
        q1.run(None, &REFERENCE_LINEAR_SOLVER).unwrap().fields[0]
            .2
            .len(),
        12
    );
    assert_eq!(
        tpfa.run(None, &REFERENCE_LINEAR_SOLVER).unwrap().fields[0]
            .2
            .len(),
        6
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            CommonSolvePolicy::Linear(linear),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
        )
        .is_err()
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::Q1,
                formulation: FormulationKind::MixedGalerkin,
            },
            CommonSolvePolicy::Linear(linear),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
        )
        .is_err()
    );
}

#[test]
pub(super) fn common_elasticity_plan_consumes_exact_mesh_and_model_meaning() {
    let geometry = rectangle();
    let model = elasticity_model(&geometry, 3.0);
    let alternate_material = elasticity_model(&geometry, 4.0);
    let solve = exact_reference_linear(
        LinearSolver::ConjugateGradient,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(10_000).unwrap(),
    );
    let resolve_elasticity = |model: &ModelEnvelope| {
        let resolved = ResolvedCommonPlan::resolve(
            model,
            resources(&geometry),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(solve),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
        )
        .unwrap();
        replay_plan(resolved, &ResolveOnlyBackend)
            .as_spatial()
            .cloned()
            .expect("fixture retains its common linear Plan")
    };
    let plan = resolve_elasticity(&model);
    let repeat = resolve_elasticity(&model);
    let alternate = resolve_elasticity(&alternate_material);
    assert_eq!(plan.identity(), repeat.identity());
    assert_ne!(plan.identity(), alternate.identity());
    assert_eq!(plan.realization_digest(), repeat.realization_digest());
    assert_eq!(
        plan.realization_digest(),
        alternate.realization_digest(),
        "material coefficients belong to Model identity, not numerical realization identity"
    );
    assert_eq!(
        plan.realization_digest(),
        hex_bytes(&plan.portable_realization().digest().unwrap())
    );
    assert_eq!(plan.model_digest(), model.digest().unwrap().to_string());
    assert_eq!(plan.cartesian_cells().unwrap(), [2, 3]);
    crate::region_assembly::InterfaceReactions::<f64>::reset_evaluations();
    let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    let before = result.to_bytes().unwrap();
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        0
    );
    let observed = result.elasticity_observation().unwrap().unwrap();
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        1
    );
    assert_eq!(result.to_bytes().unwrap(), before);
    let replayed = crate::CommonResult::from_bytes(
        &before,
        &ResolvedCommonPlan::Spatial(Box::new(plan.clone())),
    )
    .unwrap();
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        1
    );
    assert_eq!(
        replayed.elasticity_observation().unwrap().unwrap(),
        observed
    );
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        2
    );

    let NativeMeshResources::Cartesian { mesh, .. } = plan.admission.resources() else {
        panic!("Cartesian mesh")
    };
    assert_eq!(mesh.mesh().axis_cell_count(0), Some(2));
    assert_eq!(mesh.mesh().axis_cell_count(1), Some(3));
    let (_, coefficients, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(coefficients.len(), 24);
    assert_eq!(shape, &[3, 4, 2]);
    // For lambda=0 and q=2*mu*x, -2*mu*u_xx=2*mu with
    // u(0)=0 and u_x(1)=0 gives u=(x-x*x/2, 0). Q1 nodal
    // interpolation is exact for this separable constant-load problem.
    let (values, remainder) = coefficients.as_chunks::<2>();
    assert!(remainder.is_empty());
    for (vertex, values) in values.iter().enumerate() {
        let x = mesh
            .mesh()
            .vertex_coordinates(eqiora_meshing::MeshEntity::new(0, vertex))
            .unwrap()[0];
        assert!((values[0] - (x - x * x / 2.0)).abs() < 1e-9);
        assert!(values[1].abs() < 1e-9);
    }
    use eqiora_solver::{AlgebraicStructure, HostSerialSolverProfile};
    let continuum = plan
        .admission
        .elasticity_proof()
        .expect("constitutive proof");
    let displacement = continuum.displacement().downcast().unwrap();
    let load_potential = continuum.load_potential().downcast().unwrap();
    let expected = AlgebraicStructure::new([displacement], []).unwrap();
    plan.admission
        .linear
        .planning_profile
        .as_ref()
        .unwrap()
        .require_structure(Some(&expected))
        .unwrap();
    let foreign = eqiora_core::Id::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap());
    for stale in [
        None,
        Some(AlgebraicStructure::new([load_potential], []).unwrap()),
        Some(AlgebraicStructure::new([displacement, load_potential], []).unwrap()),
        Some(AlgebraicStructure::new([foreign], []).unwrap()),
    ] {
        let mut changed = plan.clone();
        let profile = HostSerialSolverProfile::canonical_csr(
            LinearOperatorProperties::SymmetricPositiveDefinite,
            None,
            None,
        );
        changed.admission.linear.planning_profile = Some(match stale {
            Some(structure) => profile.with_structure(structure).unwrap(),
            None => profile,
        });
        let error = changed
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap_err();
        assert!(error.message().contains("structure"), "{error:?}");
    }
    let mut omitted = plan.clone();
    omitted.admission.linear.planning_profile = None;
    assert!(omitted.run_result(None, &REFERENCE_LINEAR_SOLVER).is_err());

    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::CellCenteredTpfa,
            CommonSolvePolicy::Linear(solve),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
        )
        .is_err()
    );
}

#[test]
pub(super) fn admission_rejects_policy_and_resource_cross_wires() {
    let geometry = rectangle();
    let model = model(&geometry);
    assert!(
        NativeLinearPolicy::exact::<f64>(
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                -0.0,
                1.0e-13,
                NonZeroUsize::new(1000).unwrap(),
            )
            .unwrap(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err()
    );
    for solver in [
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            1.0e-13,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Jacobi),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            1.0e-13,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            1.0e-13,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap(),
    ] {
        assert!(
            NativeNumericalAdmission::admit(
                &model,
                resources(&geometry),
                NativeSpatialPolicy::FiniteElement(CommonSpatialRequest::Uniform(
                    CommonSpatialPolicy::Q1
                )),
                NativeLinearPolicy::exact::<f64>(solver, &REFERENCE_LINEAR_SOLVER).unwrap(),
            )
            .is_err()
        );
    }
    assert!(
        NativeNumericalAdmission::admit(
            &model,
            resources(&geometry),
            NativeSpatialPolicy::StokesMiniP1(
                IncompressibleFlowScaleProfile2d::new(
                    DynQuantity::new(
                        1.0,
                        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                            .expect("bounded dimension")
                    ),
                    DynQuantity::new(
                        1.0,
                        DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0])
                            .expect("bounded dimension")
                    ),
                    DynQuantity::new(
                        1.0,
                        DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0])
                            .expect("bounded dimension")
                    ),
                )
                .unwrap(),
            ),
            linear(),
        )
        .is_err()
    );

    let owner = resources(&geometry);
    let NativeMeshResources::Cartesian {
        geometry: exact_geometry,
        mesh,
        correspondence,
        production,
    } = owner.resources
    else {
        unreachable!()
    };
    let substituted_mesh = CartesianMeshEnvelopeV1::from_mesh(
        &CartesianMesh::from_axes(vec![vec![0.0, 0.25, 1.0], vec![0.0, 0.2, 0.8, 1.0]]).unwrap(),
    )
    .unwrap();
    assert!(
        AuthenticatedCommonMesh::structured_cartesian(
            exact_geometry.clone(),
            substituted_mesh,
            correspondence.clone(),
            production.clone(),
        )
        .is_err()
    );
    let mut correspondence_value: serde_json::Value =
        serde_json::from_slice(&correspondence.canonical_json().unwrap()).unwrap();
    let frontiers = correspondence_value["frontiers"].as_array_mut().unwrap();
    let left = frontiers[0]["facet_indices"].clone();
    let right = frontiers[1]["facet_indices"].clone();
    frontiers[0]["facet_indices"] = right;
    frontiers[1]["facet_indices"] = left;
    let relabelled = GeometryMeshCorrespondenceEnvelopeV1::from_json(
        &serde_json::to_vec(&correspondence_value).unwrap(),
        GeometryDecoderLimits::default(),
    )
    .unwrap();
    assert!(
        AuthenticatedCommonMesh::structured_cartesian(
            exact_geometry.clone(),
            mesh.clone(),
            relabelled,
            production.clone(),
        )
        .is_err()
    );
    let production_json = String::from_utf8(production.canonical_json().unwrap()).unwrap();
    let provider_mutation = production_json.replace(
        "\"identity\":\"eqiora.structured-cartesian\",\"version\":\"2\"",
        "\"identity\":\"eqiora.gmsh-cli\",\"version\":\"4.15.2\"",
    );
    assert!(MeshProductionLineageEnvelopeV1::from_json(provider_mutation.as_bytes()).is_err());
    let foreign_production_json = production_json.replace(
        &correspondence.digest().unwrap().to_string(),
        &"00".repeat(32),
    );
    let foreign_production =
        MeshProductionLineageEnvelopeV1::from_json(foreign_production_json.as_bytes()).unwrap();
    assert!(
        AuthenticatedCommonMesh::structured_cartesian(
            exact_geometry,
            mesh,
            correspondence,
            foreign_production,
        )
        .is_err()
    );

    let reaction_source = COMPONENT.replace(
            "-div(grad(potential))\n      - source_scale * math.sin(wave_number * coordinate(0))\n        * math.sin(wave_number * coordinate(1)) = 0;",
            "potential - 1 = 0;",
        );
    let reaction = scalar_model_from_source(&geometry, &reaction_source);
    let reaction_program = replay_program(&reaction, &geometry, &[]).unwrap();
    let reaction_owner = resources(&geometry);
    let reaction_scalar = lower_scalar_candidate(
        &reaction_program,
        &reaction_owner.resources,
        None,
        &BTreeMap::new(),
    );
    let reaction_transient =
        lower_transient_incompressible_navier_stokes_cartesian::<2>(&reaction_program);
    assert!(
        select_exact_model(
            &reaction_program,
            &reaction_owner.resources,
            &CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1),
            reaction_scalar,
            reaction_transient,
            Err(invalid("not Geometry transient")),
        )
        .is_err()
    );

    let stokes_geometry = stokes_geometry();
    let non_stokes_source =
        STOKES_COMPONENT.replace("div(velocity) = 0;", "pressure - zero_pressure = 0;");
    let non_stokes = stokes_model_from_source(&stokes_geometry, &non_stokes_source);
    let non_stokes_program = replay_program(&non_stokes, &stokes_geometry, &[]).unwrap();
    let non_stokes_transient =
        lower_transient_incompressible_navier_stokes_cartesian::<2>(&non_stokes_program);
    assert!(
        select_exact_model(
            &non_stokes_program,
            &reaction_owner.resources,
            &CommonSpatialRequest::Uniform(CommonSpatialPolicy::MiniP1),
            Err(invalid("not scalar conservation form")),
            non_stokes_transient,
            Err(invalid("not Geometry transient")),
        )
        .is_err()
    );

    let foreign = rectangle();
    let mut foreign_resources = resources(&foreign);
    if let NativeMeshResources::Cartesian { geometry, .. } = &mut foreign_resources.resources {
        *geometry = {
            let graph = GeometryGraph::new();
            let rectangle = graph.rectangle([0.0, 2.0], [0.0, 1.0]).unwrap();
            let edges = rectangle.boundaries();
            graph
                .build(
                    &rectangle,
                    &BTreeMap::from([
                        ("region".to_owned(), vec![rectangle.region().into()]),
                        ("left".to_owned(), vec![edges[0].into()]),
                        ("right".to_owned(), vec![edges[1].into()]),
                        ("bottom".to_owned(), vec![edges[2].into()]),
                        ("top".to_owned(), vec![edges[3].into()]),
                    ]),
                )
                .unwrap()
        };
    }
    assert!(
        NativeNumericalAdmission::admit(
            &model,
            foreign_resources,
            NativeSpatialPolicy::FiniteElement(CommonSpatialRequest::Uniform(
                CommonSpatialPolicy::Q1
            )),
            linear(),
        )
        .is_err()
    );
}

fn general_linear() -> NativeLinearPolicy {
    NativeLinearPolicy::exact::<f64>(
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-10,
            1.0e-13,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap()
}

#[test]
fn scalar_parameter_points_share_the_run_primal_and_preserve_operator_properties() {
    let geometry = rectangle();
    let model = model(&geometry);
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let plan = resolve_scalar_box(&model, resources(&geometry), spatial);
        let parameter = plan
            .admission
            .program()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Parameter(value)
                    if value.value_type().dimension()
                        == DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap() =>
                {
                    Some(value.id())
                }
                _ => None,
            })
            .unwrap();
        let outputs = plan
            .fields()
            .map(|(field, _)| field.erase())
            .collect::<Vec<_>>();
        let output = plan.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let expected_space = match spatial {
            CommonSpatialPolicy::Q1 => {
                eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN)
            }
            CommonSpatialPolicy::CellCenteredTpfa => eqiora_realization::Space::cell_constant(),
            _ => unreachable!(),
        };
        assert_eq!(output.fields[0].3, expected_space);
        crate::CommonResult::accept_linear(plan.clone(), None, 0.0, output.clone()).unwrap();
        // Preserve every value, type and ID while changing only its coefficient
        // interpretation. Neither equal shape nor a finite solve authenticates it.
        for substituted in [
            eqiora_realization::Space::tetrahedral_edge(),
            eqiora_realization::Space::tetrahedral_face(),
        ] {
            let mut corrupted = output.clone();
            corrupted.fields[0].3 = substituted;
            assert!(
                crate::CommonResult::accept_linear(plan.clone(), None, 0.0, corrupted)
                    .unwrap_err()
                    .message()
                    .contains("coefficient Space")
            );
        }
        if spatial == CommonSpatialPolicy::Q1 {
            let field = plan.fields().next().unwrap().0;
            let scales = eqiora_realization::SymmetricCongruenceScaling::new(
                [eqiora_realization::AlgebraicBlockScale::new(
                    eqiora_solver::AlgebraicBlock::Field(field),
                    eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                        2.0,
                        DimExponents::DIMENSIONLESS,
                    ))
                    .unwrap(),
                )],
                eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                    3.0,
                    DimExponents::DIMENSIONLESS,
                ))
                .unwrap(),
            )
            .unwrap();
            let request = ResolvedCommonPlan::Spatial(Box::new(plan.clone()))
                .canonical_solve_request()
                .unwrap();
            let scaled = ResolvedCommonPlan::resolve(
                &model,
                resources(&geometry),
                spatial,
                request,
                Some(CommonScalingRequest::SymmetricCongruence(scales)),
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let scaled = replay_plan(scaled, &REFERENCE_LINEAR_SOLVER);
            let scaled = scaled.as_spatial().unwrap();
            // The scaled Plan executes normally; differentiation must not issue
            // a receipt for this Plan while silently assembling an SI operator.
            let scaled_output = scaled.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            for (actual, expected) in scaled_output.fields[0].2.iter().zip(&output.fields[0].2) {
                assert!((actual - expected).abs() < 1e-10);
            }
            assert!(
                scaled
                    .differentiate(
                        None,
                        &[parameter],
                        None,
                        &outputs,
                        &std::collections::HashMap::new(),
                        &REFERENCE_LINEAR_SOLVER
                    )
                    .unwrap_err()
                    .message()
                    .contains("requires dimensional system scaling")
            );
        }
        let field = plan
            .differentiate(
                None,
                &[parameter],
                None,
                &outputs,
                &std::collections::HashMap::new(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        let reopened = plan
            .validate_differentiated_values(
                None,
                &[parameter],
                None,
                &outputs,
                &std::collections::HashMap::new(),
                field.linearization().relation().accepted_unknowns(),
            )
            .unwrap();
        assert_eq!(&reopened, field.linearization());
        let relation = field.linearization().relation();
        assert_eq!(
            relation.state_jacobian().properties(),
            match spatial {
                CommonSpatialPolicy::Q1 => LinearOperatorProperties::General,
                CommonSpatialPolicy::CellCenteredTpfa =>
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                _ => unreachable!(),
            }
        );
        for (actual, expected) in field
            .linearization()
            .output_values()
            .iter()
            .zip(&output.fields[0].2)
        {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        let doubled = plan
            .differentiate(
                None,
                &[parameter],
                Some(&[4.0 * std::f64::consts::PI.powi(2)]),
                &outputs,
                &std::collections::HashMap::new(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        for (actual, expected) in doubled
            .linearization()
            .output_values()
            .iter()
            .zip(&output.fields[0].2)
        {
            assert!((actual - 2.0 * expected).abs() < 1.0e-10);
        }
    }
}

#[test]
fn scalar_interval_parameter_point_uses_point_boundary_facets() {
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[3]),
        CommonSpatialPolicy::Q1,
    );
    let parameter = plan
        .admission
        .program()
        .nodes()
        .find_map(|node| match node {
            eqiora_schema::kernel::KernelNode::Parameter(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let outputs = plan
        .fields()
        .map(|(field, _)| field.erase())
        .collect::<Vec<_>>();
    let output = plan.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    let field = plan
        .differentiate(
            None,
            &[parameter],
            None,
            &outputs,
            &std::collections::HashMap::new(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    for (actual, expected) in field
        .linearization()
        .output_values()
        .iter()
        .zip(&output.fields[0].2)
    {
        assert!((actual - expected).abs() < 1.0e-10);
    }
}

#[test]
pub(super) fn scalar_linear_blocks_execute_and_replay_complete_one_two_three_field_results() {
    for count in [1_i32, 2, 3] {
        let mut source = String::from(
            "public component Coupled(
  support body: volume(ambient_dimension = 1),
  support left: boundary(parent = body),
  support right: boundary(parent = body),
  parameter source_scale: 1 / m ^ 2
) {
  \n",
        );
        for row in 0..count {
            source += &format!("variable f{row}: 1 on body in h1;\n");
        }
        for row in 0..count {
            let coefficients = (0..count)
                .map(|column| if row == column { 6 } else { 3 * (row - column) })
                .collect::<Vec<_>>();
            // Two h=1/2 elements: the interior diffusion entry is 4,
            // reaction mass entry is C/3, and a constant load integrates to f/2.
            // Choosing f=8+2*sum(C)/3 therefore gives every interior value 1.
            let load = 8 + 2 * coefficients.iter().sum::<i32>() / 3;
            source += &format!("relation row{row} on body {{ -div(grad(f{row}))");
            for (column, coefficient) in coefficients.iter().enumerate() {
                source += &format!(" + ({coefficient}) * source_scale * f{column}");
            }
            source += &format!(
                " - ({load}) * source_scale = 0; }}\nrelation left{row} on left {{ trace(f{row}) = 0; }}\nrelation right{row} on right {{ trace(f{row}) = 0; }}\n"
            );
        }
        source += "}";
        let geometry = cartesian_interval();
        let model = scalar_box_model(&geometry, &source, "Coupled", &["left", "right"]);
        let plan = resolve_scalar_box(
            &model,
            cartesian_box_resources(&geometry, &[2]),
            CommonSpatialPolicy::Q1,
        );
        assert_eq!(plan.fields().len(), usize::try_from(count).unwrap());
        let resolved = ResolvedCommonPlan::Spatial(Box::new(plan));
        assert_eq!(
            resolved.operator_properties(),
            Some(LinearOperatorProperties::General)
        );
        let replayed = replay_plan(resolved, &ResolveOnlyBackend);
        let ResolvedCommonPlan::Spatial(plan) = &replayed else {
            unreachable!()
        };
        let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        assert_eq!(result.field_count(), usize::try_from(count).unwrap());
        for index in 0..result.field_count() {
            let (_, values, shape) = result.field_block(index, 0).unwrap();
            assert_eq!(shape, &[3]);
            for (actual, expected) in values.iter().zip([0.0, 1.0, 0.0]) {
                assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
            }
        }
        let bytes = result.to_bytes().unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&bytes, &replayed)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let old = String::from_utf8(bytes)
            .unwrap()
            .replace("eqiora.common-result/v27", "eqiora.common-result/v2");
        assert!(crate::CommonResult::from_bytes(old.as_bytes(), &replayed).is_err());
    }
}

#[test]
pub(super) fn mathematical_resolution_ignores_names_and_rejects_changed_operator_or_boundary_law() {
    let geometry = cartesian_interval();
    let original_model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let renamed_source = POISSON_INTERVAL
        .replace("PoissonInterval", "RenamedInterval")
        .replace("potential", "temperature")
        .replace("balance", "thermal_equilibrium")
        .replace("left_value", "cold_end")
        .replace("right_value", "warm_end");
    let renamed_model = scalar_box_model(
        &geometry,
        &renamed_source,
        "RenamedInterval",
        &["left", "right"],
    );
    let original = resolve_scalar_box(
        &original_model,
        cartesian_box_resources(&geometry, &[3]),
        CommonSpatialPolicy::Q1,
    );
    let renamed = resolve_scalar_box(
        &renamed_model,
        cartesian_box_resources(&geometry, &[3]),
        CommonSpatialPolicy::Q1,
    );
    assert_ne!(original.identity(), renamed.identity());
    assert_eq!(original.formulation(), renamed.formulation());
    assert_eq!(original.linear(), renamed.linear());
    assert_eq!(
        original
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap()
            .field_block(0, 0)
            .unwrap()
            .1,
        renamed
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap()
            .field_block(0, 0)
            .unwrap()
            .1,
    );

    let nonlinear_flux = POISSON_INTERVAL.replace(
        "-div(grad(potential))",
        "-div(math.sin(potential) * grad(potential))",
    );
    let nonlinear_model = scalar_box_model(
        &geometry,
        &nonlinear_flux,
        "PoissonInterval",
        &["left", "right"],
    );
    let error = ResolvedCommonPlan::resolve(
        &nonlinear_model,
        cartesian_box_resources(&geometry, &[3]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(10_000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
    )
    .unwrap_err();
    assert!(
        error.message().contains("explicit Newton policy"),
        "{error:?}"
    );
    assert!(!error.message().contains("native capability"));

    let flux_boundary = POISSON_INTERVAL
        .replace(
            "potential: 1 on body in h1;",
            "potential: 1 on body in smooth;",
        )
        .replacen(
            "relation right_value on right { trace(potential) = 0; }",
            "relation right_value on right { normal(grad(potential)) = 0; }",
            1,
        );
    let flux_model = scalar_box_model(
        &geometry,
        &flux_boundary,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = ResolvedCommonPlan::resolve(
        &flux_model,
        cartesian_box_resources(&geometry, &[3]),
        CommonMethodRequest::Exact {
            spatial: CommonSpatialPolicy::Q1,
            formulation: FormulationKind::PrimalGalerkin,
        },
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(10_000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        plan.formulation().unwrap().boundary_treatment(),
        "explicit-trace-flux-laws"
    );
}

#[test]
fn planned_common_execution_reauthenticates_before_backend_or_operator_work() {
    #[derive(Debug)]
    struct Unexecuted;
    impl eqiora_solver::LinearOperator for Unexecuted {
        type Scalar = f64;

        fn rows(&self) -> usize {
            1
        }
        fn columns(&self) -> usize {
            1
        }
        fn apply(&self, _: &[f64], _: &mut [f64]) -> Result<(), Diagnostic> {
            unreachable!("profile rejection must precede numerical operator work")
        }
    }
    let request = CommonLinearRequest::program_controlled(
        1e-10,
        1e-12,
        NonZeroUsize::new(100).unwrap(),
        SolverPlanningObjective::Robust,
    )
    .unwrap();
    let policy = super::super::solver_planning::resolve_linear(
        request,
        LinearOperatorProperties::General,
        None,
        None,
        None,
        None,
        &PlanningFaerBackend,
    )
    .unwrap();
    assert_eq!(
        policy.solver.algorithm(),
        LinearSolver::BiConjugateGradientStabilized
    );
    let checked = policy
        .checked_backend(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    let problem =
        LinearProblem::new(&Unexecuted, &[1.], LinearOperatorProperties::General).unwrap();
    assert!(
        checked
            .solve(&problem, policy.solver)
            .unwrap_err()
            .message()
            .contains("canonical-csr-mismatch")
    );
    // PlanningFaerBackend's solve also panics: the test cannot pass by reaching
    // a provider and relabeling its failure as profile rejection.
}

mod physical_interface;
mod plural_regions;

mod q1_diffusion_spd;
