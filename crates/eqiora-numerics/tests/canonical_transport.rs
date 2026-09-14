use eqiora_solver::AlgebraicBlock;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora_assembly::{
    AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, IndexedAssemblyWork,
    REFERENCE_ASSEMBLY_BACKEND,
};
use eqiora_compiler::compile;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, ScalarType};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{MeshEntity, MeshTopology};
use eqiora_numerics::{
    scalar::ScalarTransportCartesianBoundary, scalar::ScalarTransportCellState2d,
    scalar::finalize_resolved_scalar_transport_fvm_step_2d,
    scalar::finalize_resolved_scalar_transport_fvm_step_2d_with_assembly,
    scalar::initialize_resolved_scalar_transport_fvm_2d,
    scalar::lower_scalar_transport_cartesian_2d,
    scalar::solve_resolved_scalar_transport_fvm_step_2d,
};
use eqiora_realization::{
    AlgebraicBlockScale, BackwardEulerRelationStep, CellCenteredConvection,
    CellCenteredConvectionScheme, Discretization, DiscretizationMethod, ExecutionSchedule,
    FieldSpaceBinding, FieldwiseRealizationPlan, FieldwiseRealizationRequirements,
    FieldwiseSpatialDiscretization, MeshPolicy, OrthogonalTwoPointDiffusion, PositivePhysicalScale,
    QuadraturePolicy, RealizationCapabilities, RealizationRequirements, RealizationRevision,
    SemanticRevision, Space, SymmetricCongruenceScaling, Target,
    TransientCellCenteredTransportCapabilities, TransientCellCenteredTransportRealizationPlan,
    TransientCellCenteredTransportRealizationRequest,
    TransientCellCenteredTransportRealizationRequirements, VectorLayoutKind,
    resolve_transient_cell_centered_transport,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    DiagonalAvailability, ExecutionId, ExecutionProvider, ExecutionReport, FixedOrderInnerProduct,
    LinearOperator, LinearOperatorOrientation, LinearOperatorProperties, LinearProblem,
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverPlan,
};

const SOURCE: &str =
    include_str!("../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/direct.eqi");
const CONSTANT_SOURCE: &str =
    include_str!("../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/constant.eqi");
const ZERO_ADVECTION_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/zero-advection.eqi"
);
const MIRRORED_SOURCE: &str =
    include_str!("../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/mirrored.eqi");
const MISSING_INFLOW_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/invalid/missing-inflow-relation.eqi"
);
const OUTFLOW_TRACE_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/invalid/outflow-trace.eqi"
);
const MIRRORED_UNSWAPPED_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/invalid/mirrored-unswapped.eqi"
);
const NONAFFINE_POTENTIAL_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/invalid/nonaffine-potential.eqi"
);
const VARYING_BOUNDARY_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-advection-diffusion-fvm-2d/models/invalid/varying-boundary.eqi"
);
const PERIODIC_TRANSVERSE_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-periodic-transport-fvm-2d/models/transverse-inflow.eqi"
);
const PERIODIC_SEAM_ADVECTION_SOURCE: &str = include_str!(
    "../../../verify/fluid/cartesian-periodic-transport-fvm-2d/models/seam-advection.eqi"
);

#[test]
fn fresh_transport_initial_equation_is_distinct_from_inflow_and_support_shape() {
    for (condition, expected) in [
        ("concentration = 0", 0.0),
        ("concentration = 0.25[K]", 0.25),
        ("0.25[K] = concentration", 0.25),
        ("concentration = inflow / 2", 0.5),
    ] {
        let source = SOURCE.replace("concentration = 0;", &format!("{condition};"));
        let program = compile_program(&source);
        let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
        let resolved = resolve_transport(&program, &model, 4, 0.025);
        let (_, initial) =
            initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
        // These are cell averages of a scalar constant function, not shaped
        // initial data or the distinct 1 K inflow boundary condition.
        assert!(initial.values().iter().all(|value| *value == expected));
        assert_eq!(
            model
                .boundary(0, BoundarySide::Lower)
                .unwrap()
                .value()
                .unwrap()
                .constant_value(),
            Some(1.0)
        );
    }
}

#[test]
fn transport_initializer_rejects_missing_extra_and_inapplicable_equations() {
    for condition in [
        "",
        "initial { concentration = 0; concentration = 1[K]; }",
        "initial { concentration = 0; } initial { concentration = 0; }",
        "initial { flow_potential = 0; }",
        "initial { derivative(concentration) = 0; }",
        "initial { concentration - concentration = 0; }",
        "initial { concentration = inflow / 0; }",
    ] {
        let source = SOURCE.replace("initial { concentration = 0; }", condition);
        let program = compile_program(&source);
        let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
        let resolved = resolve_transport(&program, &model, 4, 0.025);
        let error = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap_err();
        assert_eq!(
            error.code(),
            eqiora_core::diagnostic::codes::INVALID_SPATIAL_LOWERING,
            "{condition}: {error:?}"
        );
    }
    let wrong_support = SOURCE.replace("state concentration: K on body;",
        "domain other = box(0,2,0,1); state foreign: K on other; state concentration: K on body;")
        .replace("initial { concentration = 0; }", "initial { concentration = foreign; }");
    assert!(compile("wrong-support-initial.eqi", &wrong_support).is_err());
    for shaped in ["array<K, 2>", "vector<K, 2>"] {
        let source = SOURCE
            .replace(
                "state concentration: K",
                &format!("state concentration: {shaped}"),
            )
            .replace(
                "initial { concentration = 0; }",
                "initial { concentration = 1[K]; }",
            );
        assert!(
            compile("shaped-initial.eqi", &source).is_err(),
            "scalar initial value must not broadcast to {shaped}"
        );
    }
}

#[test]
fn canonical_transport_retains_meaning_without_numerical_policy() {
    let program = compile_program(SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();

    assert_eq!(model.bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(model.diffusivity(), 0.2);
    assert_eq!(model.advecting_velocity(&[0.4, 0.7]).unwrap(), [1.0, 0.0]);
    assert!(matches!(
        model.boundary(0, BoundarySide::Lower),
        Some(ScalarTransportCartesianBoundary::PrescribedTrace(value))
            if value.evaluate(&[0.0, 0.5]).unwrap() == 1.0
    ));
    for side in [
        (0, BoundarySide::Upper),
        (1, BoundarySide::Lower),
        (1, BoundarySide::Upper),
    ] {
        assert!(matches!(
            model.boundary(side.0, side.1),
            Some(ScalarTransportCartesianBoundary::PrescribedDiffusiveFlux(value))
                if value.evaluate(&[0.5, 0.5]).unwrap() == 0.0
        ));
    }
}

#[test]
fn canonical_transport_retains_exact_spatial_periodic_pairing() {
    let program = compile_program(PERIODIC_TRANSVERSE_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let lower = model
        .boundary(0, BoundarySide::Lower)
        .and_then(ScalarTransportCartesianBoundary::spatial_periodic_binding)
        .expect("lower side keeps its exact periodic binding");
    let upper = model
        .boundary(0, BoundarySide::Upper)
        .and_then(ScalarTransportCartesianBoundary::spatial_periodic_binding)
        .expect("upper side keeps its exact periodic binding");
    assert_eq!(lower.0, upper.0);
    assert_ne!(lower.1, upper.1);
    assert!(matches!(
        model.boundary(1, BoundarySide::Lower),
        Some(ScalarTransportCartesianBoundary::PrescribedTrace(_))
    ));
}

#[test]
fn periodic_fvm_step_uses_one_conservative_packet_per_seam_face() {
    let program = compile_program(PERIODIC_SEAM_ADVECTION_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 8, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let (_, step) = solve_resolved_scalar_transport_fvm_step_2d(
        &program,
        &resolved,
        &initial,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();

    assert!(
        step.state()
            .values()
            .iter()
            .all(|value| (value - 1.0).abs() < 2.0e-12)
    );
    let evidence = step.evidence();
    assert_eq!(evidence.interior_face_count(), 112);
    assert_eq!(evidence.periodic_face_count(), 8);
    assert_eq!(evidence.boundary_face_count(), 16);
    assert_eq!(evidence.inflow_face_count(), 0);
    assert_eq!(evidence.outflow_face_count(), 0);
    assert_eq!(evidence.wall_face_count(), 16);
    assert_eq!(evidence.maximum_interior_cancellation_defect(), 0.0);
    assert!(evidence.outward_boundary_flux().abs() <= evidence.conservation_tolerance());
    assert!(evidence.conservation_defect().abs() <= evidence.conservation_tolerance());
}

#[test]
fn spatial_periodic_transport_matches_the_transverse_oracle_and_seam_action() {
    let transverse = compile_program(PERIODIC_TRANSVERSE_SOURCE);
    let oracle = InflowStepOracle::new(64);
    let final_time = 1.0 / 16.0;
    let mut errors = Vec::new();
    for cells in [4, 8, 16] {
        let step = 1.0 / (2.0 * (cells * cells) as f64);
        let run = run_transport(&transverse, cells, step, final_time);
        assert!(run.maximum_relative_balance < 5.0e-11);
        assert!(run.accumulated_relative_balance < 5.0e-11);
        assert_transverse_rows_are_periodic(&run.state, 3.0e-12);
        errors.push(l2_error(&run.state, |point| {
            oracle.value(&[point[1], point[0]], final_time)
        }));
    }
    let orders = observed_orders(&errors);
    assert!(
        orders.iter().all(|order| *order > 0.8),
        "transverse-periodic errors/orders: {errors:?}/{orders:?}"
    );

    let positive = periodic_seam_basis_action(PERIODIC_SEAM_ADVECTION_SOURCE, 4, [3, 0], [0, 0]);
    let reversed_source = PERIODIC_SEAM_ADVECTION_SOURCE.replacen(
        "parameter speed: m / s = 1;",
        "parameter speed: m / s = -1;",
        1,
    );
    let negative = periodic_seam_basis_action(&reversed_source, 4, [3, 0], [0, 0]);
    let mispaired = periodic_seam_basis_action(PERIODIC_SEAM_ADVECTION_SOURCE, 4, [3, 1], [0, 0]);
    assert!((positive + 0.45).abs() < 64.0 * f64::EPSILON);
    assert!((negative + 0.2).abs() < 64.0 * f64::EPSILON);
    assert_eq!(mispaired, 0.0);

    let model = lower_scalar_transport_cartesian_2d(&transverse).unwrap();
    let no_periodic_capability = TransientCellCenteredTransportCapabilities::new(
        RealizationCapabilities::cell_centered_transport_2d_reference(),
        [CellCenteredConvectionScheme::ImplicitFirstOrderUpwind],
    )
    .unwrap();
    let request = TransientCellCenteredTransportRealizationRequest::explicit(
        transverse.model(),
        SemanticRevision::new(transverse.revision().0),
        next_realization_revision(),
        transport_plan(&model, 4, 0.025, TransportScales::UNIT),
    );
    let error = resolve_transient_cell_centered_transport(
        &request,
        transport_requirements(&model),
        &no_periodic_capability,
    )
    .unwrap_err();
    assert!(error.message().contains("does not support"));

    let minmod_request = TransientCellCenteredTransportRealizationRequest::explicit(
        transverse.model(),
        SemanticRevision::new(transverse.revision().0),
        next_realization_revision(),
        transport_plan_with_scheme(
            &model,
            4,
            0.025,
            TransportScales::UNIT,
            CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod,
        ),
    );
    let error = resolve_transient_cell_centered_transport(
        &minmod_request,
        transport_requirements(&model),
        &transport_capabilities(),
    )
    .unwrap_err();
    assert!(error.message().contains("first-order upwind"));
}

#[test]
fn resolved_fvm_step_preserves_a_constant_and_proves_global_conservation() {
    let program = compile_program(CONSTANT_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 8, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let (_, step) = solve_resolved_scalar_transport_fvm_step_2d(
        &program,
        &resolved,
        &initial,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();

    assert_eq!(step.realization(), &resolved);
    assert!((step.state().time().value() - 0.025).abs() < f64::EPSILON);
    assert!(
        step.state()
            .values()
            .iter()
            .all(|value| (value - 1.0).abs() < 2.0e-12)
    );
    let evidence = step.evidence();
    assert!(evidence.conservation_defect().abs() <= evidence.conservation_tolerance());
    assert_eq!(evidence.interior_face_count(), 112);
    assert_eq!(evidence.boundary_face_count(), 32);
    assert_eq!(evidence.inflow_face_count(), 8);
    assert_eq!(evidence.outflow_face_count(), 8);
    assert_eq!(evidence.wall_face_count(), 16);
    assert_eq!(evidence.maximum_interior_cancellation_defect(), 0.0);
    assert!(evidence.maximum_assembly_replay_defect() <= evidence.assembly_replay_tolerance());
    assert!(evidence.maximum_operator_replay_defect() <= evidence.operator_replay_tolerance());
    assert!(evidence.replayed_residual_norm() <= evidence.solve_report().residual_target());
}

#[test]
fn minmod_step_preserves_a_constant_and_its_face_hull() {
    let program = compile_program(CONSTANT_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let scheme = CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod;
    let resolved = resolve_transport_with_scales_and_scheme(
        &program,
        &model,
        8,
        0.025,
        TransportScales::UNIT,
        scheme,
    );
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let (_, step) = solve_resolved_scalar_transport_fvm_step_2d(
        &program,
        &resolved,
        &initial,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();

    assert!(
        step.state()
            .values()
            .iter()
            .all(|value| (value - 1.0).abs() < 2.0e-12)
    );
    let evidence = step.evidence();
    assert_eq!(evidence.convection_scheme(), scheme);
    assert_eq!(evidence.advective_face_bound_defect(), 0.0);
    assert_eq!(evidence.advective_face_value_range(), Some([1.0, 1.0]));
    assert!(evidence.advective_face_bound_tolerance().is_some());
    assert!(evidence.conservation_defect().abs() <= evidence.conservation_tolerance());
}

#[test]
fn zero_advection_preserves_optional_face_evidence_for_both_schemes() {
    let program = compile_program(ZERO_ADVECTION_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    for scheme in [
        CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
        CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod,
    ] {
        let resolved = resolve_transport_with_scales_and_scheme(
            &program,
            &model,
            2,
            0.1,
            TransportScales::UNIT,
            scheme,
        );
        let (_, initial) =
            initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
        let (_, step) = solve_resolved_scalar_transport_fvm_step_2d(
            &program,
            &resolved,
            &initial,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        assert!(
            step.state()
                .values()
                .iter()
                .all(|value| (value - 1.0).abs() < 2.0e-12)
        );
        assert_eq!(step.evidence().convection_scheme(), scheme);
        assert_eq!(step.evidence().maximum_courant_number(), 0.0);
        assert_eq!(step.evidence().advective_face_value_range(), None);
        assert_eq!(step.evidence().advective_face_bound_defect(), 0.0);
        assert_eq!(
            step.evidence().advective_face_bound_tolerance().is_some(),
            scheme == CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod
        );
    }
}

#[test]
fn finalized_step_rejects_solver_orientation_and_execution_substitution() {
    let program = compile_program(CONSTANT_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 4, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let (_, finalized) =
        finalize_resolved_scalar_transport_fvm_step_2d(&program, &resolved, &initial).unwrap();

    let substituted_plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(2_001).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Reproducible);
    let substituted_solution = REFERENCE_LINEAR_SOLVER
        .solve(&finalized.linear_problem().unwrap(), substituted_plan)
        .unwrap();
    let error = finalized.clone().finish(substituted_solution).unwrap_err();
    assert!(error.message().contains("different SolverPlan"));

    let transposed_solution = {
        let normal = finalized.linear_problem().unwrap();
        let transposed = ReportedTranspose(normal.operator());
        let transposed_problem =
            LinearProblem::new(&transposed, normal.right_hand_side(), normal.properties()).unwrap();
        REFERENCE_LINEAR_SOLVER
            .solve(&transposed_problem, finalized.solver_plan())
            .unwrap()
    };
    let error = finalized.clone().finish(transposed_solution).unwrap_err();
    assert!(error.message().contains("normal-orientation"));

    let foreign_execution_solution = REFERENCE_LINEAR_SOLVER
        .solve_with_execution(
            &finalized.linear_problem().unwrap(),
            finalized.solver_plan(),
            &TWO_WORKER_EXECUTION,
        )
        .unwrap();
    let error = finalized.finish(foreign_execution_solution).unwrap_err();
    assert!(error.message().contains("host producer"));
}

#[test]
fn physical_replay_rejects_a_duplicated_face_packet() {
    let program = compile_program(SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 4, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let error = finalize_resolved_scalar_transport_fvm_step_2d_with_assembly(
        &program,
        &resolved,
        &initial,
        &DuplicateFirstFaceAssembly { cell_count: 16 },
    )
    .unwrap_err();
    assert!(error.message().contains("independent operator replay"));
}

#[test]
fn transport_rejects_forged_assembly_receipts_before_system_exposure() {
    let program = compile_program(CONSTANT_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 4, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();

    for substitution in [
        ReceiptSubstitution::PacketCount,
        ReceiptSubstitution::Execution,
    ] {
        let error = finalize_resolved_scalar_transport_fvm_step_2d_with_assembly(
            &program,
            &resolved,
            &initial,
            &ForgedAssemblyReceipt(substitution),
        )
        .unwrap_err();
        assert!(error.message().contains("assembly receipt"));
    }
}

#[test]
fn transport_rejects_a_nonfinite_endpoint_time_before_assembly() {
    let program = compile_program(CONSTANT_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, 2, f64::MAX);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let (_, first) = solve_resolved_scalar_transport_fvm_step_2d(
        &program,
        &resolved,
        &initial,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(first.state().time().value(), f64::MAX);

    let error = finalize_resolved_scalar_transport_fvm_step_2d(&program, &resolved, first.state())
        .unwrap_err();
    assert!(error.message().contains("endpoint time must be finite"));
}

#[test]
fn cartesian_transport_meets_space_time_reversal_and_falsifier_contracts() {
    let direct = compile_program(SOURCE);
    let mirrored = compile_program(MIRRORED_SOURCE);
    let oracle = InflowStepOracle::new(64);
    let final_time = 1.0 / 16.0;
    let mut spatial_errors = Vec::new();
    let direct_model = lower_scalar_transport_cartesian_2d(&direct).unwrap();
    let coarse_realization = resolve_transport(&direct, &direct_model, 8, 1.0 / 128.0);
    let fine_realization = resolve_transport(&direct, &direct_model, 32, 1.0 / 512.0);
    assert_eq!(coarse_realization.model(), fine_realization.model());
    assert_eq!(
        coarse_realization.semantic_revision(),
        fine_realization.semantic_revision()
    );
    assert_ne!(
        coarse_realization.realization_revision(),
        fine_realization.realization_revision()
    );
    for cells in [4, 8, 16] {
        let step = 1.0 / (2.0 * (cells * cells) as f64);
        let run = run_transport(&direct, cells, step, final_time);
        assert!(run.maximum_relative_balance < 5.0e-11);
        assert!(run.accumulated_relative_balance < 5.0e-11);
        assert!(run.minimum >= run.initial_minimum - 5.0e-12);
        assert!(run.maximum <= 1.0 + 5.0e-12);
        spatial_errors.push(l2_error(&run.state, |point| {
            oracle.value(point, final_time)
        }));
    }
    let spatial_orders = observed_orders(&spatial_errors);
    assert!(
        spatial_orders.iter().all(|order| *order > 0.8),
        "spatial errors/orders: {spatial_errors:?}/{spatial_orders:?}"
    );

    let temporal = [0.02, 0.01, 0.005].map(|step| run_transport(&direct, 32, step, 0.08).state);
    let coarse_difference = l2_difference(&temporal[0], &temporal[1]);
    let fine_difference = l2_difference(&temporal[1], &temporal[2]);
    let temporal_order = (coarse_difference / fine_difference).log2();
    assert!(
        temporal_order > 0.8,
        "temporal differences/order: {coarse_difference:e}/{fine_difference:e}/{temporal_order}"
    );

    let direct_run = run_transport(&direct, 16, 0.01, 0.04);
    let mirrored_run = run_transport(&mirrored, 16, 0.01, 0.04);
    assert_reflected_equal(&direct_run.state, &mirrored_run.state, 3.0e-12);

    let unit_scaled = run_transport(&direct, 16, 0.005, 0.02);
    let nonunit_scaled = run_transport_with_scales(
        &direct,
        16,
        0.005,
        0.02,
        TransportScales {
            coordinate: 2.5,
            state: 7.0,
            weak_functional: 11.0,
        },
    );
    assert_state_values_equal(&unit_scaled.state, &nonunit_scaled.state, 2.0e-11);
    assert!(nonunit_scaled.maximum_relative_balance < 5.0e-11);
    assert!(nonunit_scaled.accumulated_relative_balance < 5.0e-11);

    let missing = compile_program(MISSING_INFLOW_SOURCE);
    assert!(lower_scalar_transport_cartesian_2d(&missing).is_err());
    for source in [NONAFFINE_POTENTIAL_SOURCE, VARYING_BOUNDARY_SOURCE] {
        let program = compile_program(source);
        assert!(lower_scalar_transport_cartesian_2d(&program).is_err());
    }
    for source in [OUTFLOW_TRACE_SOURCE, MIRRORED_UNSWAPPED_SOURCE] {
        let program = compile_program(source);
        let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
        let resolved = resolve_transport(&program, &model, 8, 0.01);
        let (_, initial) =
            initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
        assert!(
            finalize_resolved_scalar_transport_fvm_step_2d(&program, &resolved, &initial).is_err()
        );
    }

    let plan = transport_plan(&direct_model, 8, 0.01, TransportScales::UNIT);
    let foreign_relation = direct_model.potential_definition();
    let relation_drift = TransientCellCenteredTransportRealizationPlan::new(
        plan.fieldwise().clone(),
        BackwardEulerRelationStep::new(
            foreign_relation,
            direct_model.state(),
            DynQuantity::new(0.01, TIME),
        )
        .unwrap(),
        CellCenteredConvection::new(
            foreign_relation,
            direct_model.state(),
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
        ),
        OrthogonalTwoPointDiffusion::new(foreign_relation, direct_model.state()),
    )
    .unwrap();
    let request = TransientCellCenteredTransportRealizationRequest::explicit(
        direct.model(),
        SemanticRevision::new(direct.revision().0),
        next_realization_revision(),
        relation_drift,
    );
    assert!(
        resolve_transient_cell_centered_transport(
            &request,
            transport_requirements(&direct_model),
            &transport_capabilities(),
        )
        .is_err()
    );

    let wrong_quadrature = FieldwiseSpatialDiscretization::new(
        direct_model.domain(),
        scale(1.0, LENGTH),
        [FieldSpaceBinding::new(
            direct_model.state(),
            Space::cell_constant(),
        )],
        [],
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(8).unwrap(),
            },
            QuadraturePolicy::SimplexCentroid,
        ),
    )
    .unwrap();
    assert!(
        FieldwiseRealizationPlan::new(
            wrong_quadrature,
            plan.fieldwise().scaling().clone(),
            plan.fieldwise().operator_properties(),
            plan.fieldwise().solver(),
            plan.fieldwise().target(),
            plan.fieldwise().schedule(),
        )
        .is_err()
    );
}

#[test]
fn minmod_realization_has_superlinear_convergence_and_bounded_face_fluxes() {
    let direct = compile_program(SOURCE);
    let mirrored = compile_program(MIRRORED_SOURCE);
    let model = lower_scalar_transport_cartesian_2d(&direct).unwrap();
    let oracle = InflowStepOracle::new(64);
    let final_time = 1.0 / 16.0;
    let scheme = CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod;

    let first_order = resolve_transport(&direct, &model, 8, 1.0 / 128.0);
    let second_order = resolve_transport_with_scales_and_scheme(
        &direct,
        &model,
        8,
        1.0 / 128.0,
        TransportScales::UNIT,
        scheme,
    );
    assert_eq!(first_order.model(), second_order.model());
    assert_eq!(
        first_order.semantic_revision(),
        second_order.semantic_revision()
    );
    assert_ne!(
        first_order.realization_revision(),
        second_order.realization_revision()
    );

    let mut errors = Vec::new();
    let mut saw_active_limiter = false;
    for cells in [4, 8, 16] {
        let step = 1.0 / (2.0 * (cells * cells) as f64);
        let run = run_transport_with_scheme(&direct, cells, step, final_time, scheme);
        assert!(run.maximum_relative_balance < 5.0e-11);
        assert!(run.accumulated_relative_balance < 5.0e-11);
        assert!(run.minimum >= run.initial_minimum - 5.0e-12);
        assert!(run.maximum <= 1.0 + 5.0e-12);
        errors.push(l2_error(&run.state, |point| {
            oracle.value(point, final_time)
        }));

        let resolved = resolve_transport_with_scales_and_scheme(
            &direct,
            &model,
            cells,
            step,
            TransportScales::UNIT,
            scheme,
        );
        let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&direct, &resolved).unwrap();
        let (_, accepted) = solve_resolved_scalar_transport_fvm_step_2d(
            &direct,
            &resolved,
            &initial,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let evidence = accepted.evidence();
        assert_eq!(evidence.convection_scheme(), scheme);
        assert!(
            evidence.maximum_courant_number() <= scheme.maximum_explicit_courant_number().unwrap()
        );
        assert!(evidence.advective_face_bound_defect() <= 64.0 * f64::EPSILON);
        saw_active_limiter |= evidence.limited_face_count() > 0;
    }
    let orders = observed_orders(&errors);
    assert!(
        orders.iter().all(|order| *order > 1.6),
        "minmod spatial errors/orders: {errors:?}/{orders:?}"
    );
    let first_order_fine = run_transport(&direct, 16, 1.0 / 512.0, final_time);
    let first_order_error = l2_error(&first_order_fine.state, |point| {
        oracle.value(point, final_time)
    });
    assert!(
        errors[2] < first_order_error,
        "selected minmod path did not improve on first-order upwind: {} versus {first_order_error}",
        errors[2]
    );
    assert!(saw_active_limiter);

    let direct_run = run_transport_with_scheme(&direct, 16, 0.005, 0.04, scheme);
    let mirrored_run = run_transport_with_scheme(&mirrored, 16, 0.005, 0.04, scheme);
    assert_reflected_equal(&direct_run.state, &mirrored_run.state, 3.0e-12);

    let maximum_courant = scheme.maximum_explicit_courant_number().unwrap();
    let at_limit = resolve_transport_with_scales_and_scheme(
        &direct,
        &model,
        4,
        maximum_courant / 4.0,
        TransportScales::UNIT,
        scheme,
    );
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&direct, &at_limit).unwrap();
    let (_, accepted) = solve_resolved_scalar_transport_fvm_step_2d(
        &direct,
        &at_limit,
        &initial,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert!((accepted.evidence().maximum_courant_number() - maximum_courant).abs() < f64::EPSILON);

    let too_large_step = resolve_transport_with_scales_and_scheme(
        &direct,
        &model,
        4,
        (maximum_courant + 1.0e-6) / 4.0,
        TransportScales::UNIT,
        scheme,
    );
    let (_, initial) =
        initialize_resolved_scalar_transport_fvm_2d(&direct, &too_large_step).unwrap();
    assert!(
        finalize_resolved_scalar_transport_fvm_step_2d(&direct, &too_large_step, &initial)
            .unwrap_err()
            .message()
            .contains("Courant number")
    );
}

#[derive(Debug)]
struct ReportedTranspose<'a>(&'a dyn LinearOperator);

impl LinearOperator for ReportedTranspose<'_> {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn columns(&self) -> usize {
        self.0.columns()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.0.apply(input, output)
    }

    fn orientation(&self) -> LinearOperatorOrientation {
        LinearOperatorOrientation::Transposed
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        self.0.diagonal(output)
    }
}

#[derive(Debug, Clone, Copy)]
struct TwoWorkerExecution;

const TWO_WORKER_EXECUTION: TwoWorkerExecution = TwoWorkerExecution;
const TWO_WORKER_EXECUTION_PROVIDER: ExecutionProvider = ExecutionProvider::new(
    ExecutionId::new("eqiora.test.host.two-workers"),
    env!("CARGO_PKG_VERSION"),
    &[],
);

impl ReplicatedLinearExecution for TwoWorkerExecution {
    fn provider(&self) -> ExecutionProvider {
        TWO_WORKER_EXECUTION_PROVIDER
    }

    fn report(&self) -> ExecutionReport {
        ExecutionReport::host(
            TWO_WORKER_EXECUTION_PROVIDER.id(),
            NonZeroUsize::new(2).unwrap(),
        )
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.require_reduction(policy)
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.apply(operator, input, output)
    }

    fn inner_product(&self, action: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        SERIAL_LINEAR_EXECUTION.inner_product(action)
    }
}

#[derive(Debug)]
struct DuplicateFirstFaceAssembly {
    cell_count: usize,
}

#[derive(Debug, Clone, Copy)]
enum ReceiptSubstitution {
    PacketCount,
    Execution,
}

#[derive(Debug)]
struct ForgedAssemblyReceipt(ReceiptSubstitution);

impl AssemblyBackend for ForgedAssemblyReceipt {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        original: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let (systems, report) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(plan, original)?
            .into_parts();
        let packet_count = match self.0 {
            ReceiptSubstitution::PacketCount => report.packet_count() - 1,
            ReceiptSubstitution::Execution => report.packet_count(),
        };
        let execution = match self.0 {
            ReceiptSubstitution::PacketCount => report.execution(),
            ReceiptSubstitution::Execution => ExecutionReport::host(
                ExecutionId::new("eqiora.test.assembly.two-workers"),
                NonZeroUsize::new(2).unwrap(),
            ),
        };
        AssemblyResult::from_complete_systems(plan, systems, packet_count, execution)
    }
}

impl AssemblyBackend for DuplicateFirstFaceAssembly {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        original: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let work = IndexedAssemblyWork::for_packet_set(
            original.packet_set_identity(),
            original.packet_count(),
            |packet_index| {
                let source = if packet_index + 1 == original.packet_count() {
                    self.cell_count
                } else {
                    packet_index
                };
                original.evaluate(source)
            },
        );
        REFERENCE_ASSEMBLY_BACKEND.assemble(plan, &work)
    }
}

struct TransportRun {
    state: ScalarTransportCellState2d,
    maximum_relative_balance: f64,
    accumulated_relative_balance: f64,
    initial_minimum: f64,
    minimum: f64,
    maximum: f64,
}

fn run_transport(
    program: &KernelProgram,
    cells: usize,
    step: f64,
    final_time: f64,
) -> TransportRun {
    run_transport_with_scales(program, cells, step, final_time, TransportScales::UNIT)
}

fn run_transport_with_scheme(
    program: &KernelProgram,
    cells: usize,
    step: f64,
    final_time: f64,
    scheme: CellCenteredConvectionScheme,
) -> TransportRun {
    run_transport_with_scales_and_scheme(
        program,
        cells,
        step,
        final_time,
        TransportScales::UNIT,
        scheme,
    )
}

fn run_transport_with_scales(
    program: &KernelProgram,
    cells: usize,
    step: f64,
    final_time: f64,
    scales: TransportScales,
) -> TransportRun {
    run_transport_with_scales_and_scheme(
        program,
        cells,
        step,
        final_time,
        scales,
        CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
    )
}

fn run_transport_with_scales_and_scheme(
    program: &KernelProgram,
    cells: usize,
    step: f64,
    final_time: f64,
    scales: TransportScales,
    scheme: CellCenteredConvectionScheme,
) -> TransportRun {
    let model = lower_scalar_transport_cartesian_2d(program).unwrap();
    let resolved =
        resolve_transport_with_scales_and_scheme(program, &model, cells, step, scales, scheme);
    let (_, mut state) = initialize_resolved_scalar_transport_fvm_2d(program, &resolved).unwrap();
    let initial_minimum = state.values().iter().copied().fold(f64::INFINITY, f64::min);
    let initial_maximum = state
        .values()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let steps = (final_time / step).round() as usize;
    assert!(((steps as f64) * step - final_time).abs() < 64.0 * f64::EPSILON);
    let original_mass = integrated_cell_mass(&state);
    let mut integrated_flux = 0.0;
    let mut maximum_relative_balance: f64 = 0.0;
    let mut minimum = initial_minimum;
    let mut maximum = initial_maximum;
    for _ in 0..steps {
        let (_, accepted) = solve_resolved_scalar_transport_fvm_step_2d(
            program,
            &resolved,
            &state,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let evidence = accepted.evidence();
        let balance_scale = ((evidence.new_mass() - evidence.old_mass()) / step)
            .abs()
            .max(evidence.outward_boundary_flux().abs())
            .max(1.0);
        maximum_relative_balance =
            maximum_relative_balance.max(evidence.conservation_defect().abs() / balance_scale);
        integrated_flux += step * evidence.outward_boundary_flux();
        minimum = minimum.min(evidence.minimum_value());
        maximum = maximum.max(evidence.maximum_value());
        state = accepted.into_state();
    }
    let final_mass = integrated_cell_mass(&state);
    let accumulated_defect = final_mass - original_mass + integrated_flux;
    let accumulated_scale = (final_mass - original_mass)
        .abs()
        .max(integrated_flux.abs())
        .max(1.0);
    TransportRun {
        state,
        maximum_relative_balance,
        accumulated_relative_balance: accumulated_defect.abs() / accumulated_scale,
        initial_minimum,
        minimum,
        maximum,
    }
}

struct InflowStepOracle {
    modes: Vec<(f64, f64)>,
}

impl InflowStepOracle {
    fn new(mode_count: usize) -> Self {
        let beta = 1.0 / (2.0 * 0.2);
        let residual = |mu: f64| mu * mu.cos() + beta * mu.sin();
        let modes = (0..mode_count)
            .map(|index| {
                let mut lower = (index as f64 + 0.5) * std::f64::consts::PI;
                let mut upper = (index as f64 + 1.0) * std::f64::consts::PI;
                let lower_sign = residual(lower).is_sign_positive();
                for _ in 0..80 {
                    let midpoint = 0.5 * (lower + upper);
                    if residual(midpoint).is_sign_positive() == lower_sign {
                        lower = midpoint;
                    } else {
                        upper = midpoint;
                    }
                }
                let mu = 0.5 * (lower + upper);
                let norm = 0.5 - (2.0 * mu).sin() / (4.0 * mu);
                let coefficient = -mu / ((beta * beta + mu * mu) * norm);
                (mu, coefficient)
            })
            .collect();
        Self { modes }
    }

    fn value(&self, point: &[f64; 2], time: f64) -> f64 {
        assert!(
            time > 0.0,
            "the inflow-step oracle is evaluated only after t=0"
        );
        let diffusivity = 0.2;
        let beta = 1.0 / (2.0 * diffusivity);
        let transformed = self
            .modes
            .iter()
            .map(|(mu, coefficient)| {
                let decay = diffusivity * mu * mu + 1.0 / (4.0 * diffusivity);
                coefficient * (-decay * time).exp() * (mu * point[0]).sin()
            })
            .sum::<f64>();
        1.0 + (beta * point[0]).exp() * transformed
    }
}

fn l2_error<F>(state: &ScalarTransportCellState2d, exact: F) -> f64
where
    F: Fn(&[f64; 2]) -> f64,
{
    let count = state.values().len();
    let measure = 1.0 / count as f64;
    state
        .values()
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let point = state
                .mesh()
                .entity_center(MeshEntity::new(2, index))
                .unwrap();
            let point = [point[0], point[1]];
            measure * (value - exact(&point)).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

fn assert_transverse_rows_are_periodic(state: &ScalarTransportCellState2d, tolerance: f64) {
    let cells = state
        .mesh()
        .axis_cell_count(0)
        .expect("2D Cartesian state owns axis zero");
    assert_eq!(state.mesh().axis_cell_count(1), Some(cells));
    for y in 0..cells {
        let anchor = state.values()[cartesian_cell_index(state.mesh(), [0, y])];
        for x in 1..cells {
            let value = state.values()[cartesian_cell_index(state.mesh(), [x, y])];
            assert!((value - anchor).abs() <= tolerance);
        }
    }
}

fn periodic_seam_basis_action(
    source: &str,
    cells: usize,
    input_cell: [usize; 2],
    output_cell: [usize; 2],
) -> f64 {
    let program = compile_program(source);
    let model = lower_scalar_transport_cartesian_2d(&program).unwrap();
    let resolved = resolve_transport(&program, &model, cells, 0.025);
    let (_, initial) = initialize_resolved_scalar_transport_fvm_2d(&program, &resolved).unwrap();
    let input_index = cartesian_cell_index(initial.mesh(), input_cell);
    let output_index = cartesian_cell_index(initial.mesh(), output_cell);
    let (_, finalized) =
        finalize_resolved_scalar_transport_fvm_step_2d(&program, &resolved, &initial).unwrap();
    let problem = finalized.linear_problem().unwrap();
    let mut input = vec![0.0; problem.operator().columns()];
    let mut output = vec![0.0; problem.operator().rows()];
    input[input_index] = 1.0;
    problem.operator().apply(&input, &mut output).unwrap();
    output[output_index]
}

fn cartesian_cell_index(mesh: &eqiora_meshing::CartesianMesh, multi_index: [usize; 2]) -> usize {
    (0..mesh.entity_count(2).expect("2D mesh owns cells"))
        .find(|index| {
            mesh.cell_multi_index(MeshEntity::new(2, *index)) == Some(multi_index.as_slice())
        })
        .expect("requested Cartesian cell exists")
}

fn l2_difference(left: &ScalarTransportCellState2d, right: &ScalarTransportCellState2d) -> f64 {
    assert_eq!(left.mesh(), right.mesh());
    let measure = 1.0 / left.values().len() as f64;
    left.values()
        .iter()
        .zip(right.values())
        .map(|(left, right)| measure * (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn observed_orders(errors: &[f64]) -> Vec<f64> {
    errors
        .windows(2)
        .map(|pair| (pair[0] / pair[1]).log2())
        .collect()
}

fn assert_reflected_equal(
    direct: &ScalarTransportCellState2d,
    mirrored: &ScalarTransportCellState2d,
    tolerance: f64,
) {
    assert_eq!(direct.values().len(), mirrored.values().len());
    let cells = direct.mesh().axis_cell_count(0).unwrap();
    assert_eq!(direct.mesh().axis_cell_count(1), Some(cells));
    for x in 0..cells {
        for y in 0..cells {
            let direct_index = x * cells + y;
            let mirrored_index = (cells - 1 - x) * cells + y;
            assert!(
                (direct.values()[direct_index] - mirrored.values()[mirrored_index]).abs()
                    < tolerance
            );
        }
    }
}

fn assert_state_values_equal(
    left: &ScalarTransportCellState2d,
    right: &ScalarTransportCellState2d,
    tolerance: f64,
) {
    assert_eq!(left.mesh(), right.mesh());
    assert_eq!(left.time(), right.time());
    for (left, right) in left.values().iter().zip(right.values()) {
        assert!(
            (left - right).abs() <= tolerance,
            "physical states differ after scale substitution: {left:e} versus {right:e}"
        );
    }
}

fn integrated_cell_mass(state: &ScalarTransportCellState2d) -> f64 {
    state.values().iter().sum::<f64>() / state.values().len() as f64
}

fn resolve_transport(
    program: &KernelProgram,
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
    cells: usize,
    duration: f64,
) -> eqiora_realization::ResolvedTransientCellCenteredTransportRealization {
    resolve_transport_with_scales(program, model, cells, duration, TransportScales::UNIT)
}

#[derive(Clone, Copy)]
struct TransportScales {
    coordinate: f64,
    state: f64,
    weak_functional: f64,
}

impl TransportScales {
    const UNIT: Self = Self {
        coordinate: 1.0,
        state: 1.0,
        weak_functional: 1.0,
    };
}

fn resolve_transport_with_scales(
    program: &KernelProgram,
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
    cells: usize,
    duration: f64,
    scales: TransportScales,
) -> eqiora_realization::ResolvedTransientCellCenteredTransportRealization {
    resolve_transport_with_scales_and_scheme(
        program,
        model,
        cells,
        duration,
        scales,
        CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
    )
}

fn resolve_transport_with_scales_and_scheme(
    program: &KernelProgram,
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
    cells: usize,
    duration: f64,
    scales: TransportScales,
    scheme: CellCenteredConvectionScheme,
) -> eqiora_realization::ResolvedTransientCellCenteredTransportRealization {
    let plan = transport_plan_with_scheme(model, cells, duration, scales, scheme);
    let request = TransientCellCenteredTransportRealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        next_realization_revision(),
        plan,
    );
    resolve_transient_cell_centered_transport(
        &request,
        transport_requirements(model),
        &transport_capabilities(),
    )
    .unwrap()
}

fn transport_plan(
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
    cells: usize,
    duration: f64,
    scales: TransportScales,
) -> TransientCellCenteredTransportRealizationPlan {
    transport_plan_with_scheme(
        model,
        cells,
        duration,
        scales,
        CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
    )
}

fn transport_plan_with_scheme(
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
    cells: usize,
    duration: f64,
    scales: TransportScales,
    scheme: CellCenteredConvectionScheme,
) -> TransientCellCenteredTransportRealizationPlan {
    let spatial = FieldwiseSpatialDiscretization::new(
        model.domain(),
        scale(scales.coordinate, LENGTH),
        [FieldSpaceBinding::new(
            model.state(),
            Space::cell_constant(),
        )],
        [],
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            QuadraturePolicy::CellCentroid,
        ),
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [AlgebraicBlockScale::new(
            AlgebraicBlock::Field(model.state()),
            scale(scales.state, TEMPERATURE),
        )],
        scale(scales.weak_functional, TRANSPORT_WEAK_FUNCTIONAL),
    )
    .unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Reproducible);
    let fieldwise = FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        LinearOperatorProperties::General,
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let time_step = BackwardEulerRelationStep::new(
        model.transport_relation(),
        model.state(),
        DynQuantity::new(duration, TIME),
    )
    .unwrap();
    TransientCellCenteredTransportRealizationPlan::new(
        fieldwise,
        time_step,
        CellCenteredConvection::new(model.transport_relation(), model.state(), scheme),
        OrthogonalTwoPointDiffusion::new(model.transport_relation(), model.state()),
    )
    .unwrap()
}

fn transport_requirements(
    model: &eqiora_numerics::scalar::ScalarTransportCartesianModel2d,
) -> TransientCellCenteredTransportRealizationRequirements {
    let execution = RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    TransientCellCenteredTransportRealizationRequirements::new(
        FieldwiseRealizationRequirements::new(model.domain(), [model.state()], execution).unwrap(),
        model.transport_relation(),
        model.state(),
    )
    .unwrap()
    .with_spatial_periodic_connections(model.spatial_periodic_connections())
}

fn transport_capabilities() -> TransientCellCenteredTransportCapabilities {
    TransientCellCenteredTransportCapabilities::new(
        RealizationCapabilities::cell_centered_transport_2d_reference(),
        [
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
            CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod,
        ],
    )
    .unwrap()
    .with_spatial_periodic_translation()
}

fn next_realization_revision() -> RealizationRevision {
    RealizationRevision::new(NEXT_REALIZATION_REVISION.fetch_add(1, Ordering::Relaxed))
}

static NEXT_REALIZATION_REVISION: AtomicU64 = AtomicU64::new(1);

fn scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const TEMPERATURE: DimExponents =
    DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).expect("bounded dimension");
const TRANSPORT_WEAK_FUNCTIONAL: DimExponents =
    DimExponents::from_integers([0, 2, -1, 0, 1, 0, 0]).expect("bounded dimension");

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("canonical-transport.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("program validates")
}
