use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_compiler::compile;
use eqiora_core::Id;
use eqiora_core::entity::kinds;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_numerics::{
    common::SpatialDesignCoordinate, scalar::finalize_resolved_scalar_elliptic_cartesian,
    scalar::finalize_scalar_elliptic_parameter_point, scalar::lower_scalar_elliptic_cartesian,
};
use eqiora_realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, Target, VectorLayoutKind,
    resolve,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, SolverPlan};

const SOURCE: &str =
    include_str!("../../../verify/differentiation/spatial-poisson-fem-fvm/models/poisson.eqi");

#[test]
fn selected_parameter_binding_is_immutable_and_reuses_the_exact_finalizer() {
    let program = compile_program("parameter-binding.eqi", SOURCE);
    let base = lower_scalar_elliptic_cartesian(&program).unwrap();
    let resolved = resolve_plan(&program);
    let original_values = base.parameter_values().to_vec();
    let source_index = original_values
        .iter()
        .position(|value| *value > 10.0)
        .expect("fixture has one positive manufactured-source scale");
    let source = base.parameter_fields()[source_index];
    let coefficient = base.parameter_fields()[0];
    let p0_value = original_values[source_index];

    let p0 = base
        .bind_selected_parameters(&[source], &[p0_value])
        .unwrap();
    let p1 = base
        .bind_selected_parameters(&[source], &[0.5 * p0_value])
        .unwrap();
    let p0_again = base
        .bind_selected_parameters(&[source], &[p0_value])
        .unwrap();

    assert_eq!(base.parameter_values(), original_values);
    assert_eq!(p0, base);
    assert_eq!(p0_again, p0);
    assert_ne!(p1, p0);
    let selected_forward = base
        .bind_selected_parameters(&[coefficient, source], &[2.0, 0.5 * p0_value])
        .unwrap();
    let selected_reverse = base
        .bind_selected_parameters(&[source, coefficient], &[0.5 * p0_value, 2.0])
        .unwrap();
    assert_eq!(
        selected_reverse, selected_forward,
        "values follow explicit selected-Parameter order rather than model order"
    );
    for (index, value) in original_values.iter().copied().enumerate() {
        if index == source_index {
            assert_eq!(p1.parameter_values()[index], 0.5 * value);
        } else {
            assert_eq!(
                p1.parameter_values()[index],
                value,
                "unselected Parameters remain frozen"
            );
        }
    }

    let (lowered_by_legacy_entry, legacy) =
        finalize_resolved_scalar_elliptic_cartesian(&program, &resolved).unwrap();
    let finalized_p0 = finalize_scalar_elliptic_parameter_point(p0, &resolved).unwrap();
    let finalized_p1 = finalize_scalar_elliptic_parameter_point(p1, &resolved).unwrap();
    let finalized_p0_again = finalize_scalar_elliptic_parameter_point(p0_again, &resolved).unwrap();

    assert_eq!(lowered_by_legacy_entry, base);
    assert_eq!(
        legacy.canonical_csr_system_view(),
        finalized_p0.canonical_csr_system_view()
    );
    assert_eq!(
        finalized_p0_again.canonical_csr_system_view(),
        finalized_p0.canonical_csr_system_view()
    );
    assert_ne!(
        finalized_p1.canonical_csr_system_view().right_hand_side(),
        finalized_p0.canonical_csr_system_view().right_hand_side()
    );
    assert_eq!(base.parameter_values(), original_values);
}

#[test]
fn selected_parameter_binding_rejects_invalid_points_before_finalization() {
    let program = compile_program("invalid-parameter-binding.eqi", SOURCE);
    let base = lower_scalar_elliptic_cartesian(&program).unwrap();
    let first = base.parameter_fields()[0];
    let foreign = Id::<kinds::Parameter>::new();

    assert!(base.bind_selected_parameters(&[], &[]).is_err());
    assert!(base.bind_selected_parameters(&[first], &[]).is_err());
    assert!(
        base.bind_selected_parameters(&[first, first], &[1.0, 2.0])
            .is_err()
    );
    assert!(base.bind_selected_parameters(&[foreign], &[1.0]).is_err());
    assert!(
        base.bind_selected_parameters(&[first], &[f64::NAN])
            .is_err()
    );
    assert!(
        base.bind_selected_parameters(&[first], &[-1.0]).is_err(),
        "the first deterministic Parameter coordinate is the coefficient"
    );
}

#[test]
fn lowered_model_retains_its_exact_model_revision_for_finalization() {
    let program = compile_program("parameter-binding-owner.eqi", SOURCE);
    let foreign_program = compile_program(
        "foreign-parameter-binding-owner.eqi",
        &SOURCE.replace(
            "model differentiated_poisson_plane",
            "model foreign_differentiated_poisson_plane",
        ),
    );
    let model = lower_scalar_elliptic_cartesian(&program).unwrap();
    let foreign_resolved = resolve_plan(&foreign_program);

    assert!(
        finalize_scalar_elliptic_parameter_point(model, &foreign_resolved).is_err(),
        "a lowered model must not be cross-wired to a foreign Realization"
    );
}

#[test]
fn accepted_point_keeps_equal_primal_systems_with_different_derivatives_distinct() {
    let source = SOURCE.replace(
        "diffusion * grad(potential)",
        "diffusion ^ 2 * grad(potential)",
    );
    let program = compile_program("parameter-binding-linearization.eqi", &source);
    let base = lower_scalar_elliptic_cartesian(&program).unwrap();
    let resolved = resolve_plan(&program);
    let diffusion = base.parameter_fields()[0];
    let negative = base
        .bind_selected_parameters(&[diffusion], &[-1.0])
        .unwrap();
    let positive = finalize_scalar_elliptic_parameter_point(base, &resolved).unwrap();
    let negative = finalize_scalar_elliptic_parameter_point(negative, &resolved).unwrap();
    assert_eq!(
        positive.canonical_csr_system_view(),
        negative.canonical_csr_system_view(),
        "the falsifier requires identical primal systems at +p and -p"
    );

    let positive_solution = REFERENCE_LINEAR_SOLVER
        .solve(&positive.linear_problem().unwrap(), positive.solver_plan())
        .unwrap();
    let negative_solution = REFERENCE_LINEAR_SOLVER
        .solve(&negative.linear_problem().unwrap(), negative.solver_plan())
        .unwrap();
    let positive = positive.finish(positive_solution).unwrap();
    let negative = negative.finish(negative_solution).unwrap();
    let coordinates = [SpatialDesignCoordinate::ModelParameter(diffusion)];
    let (positive, _) = positive.linearize(&coordinates).unwrap();
    let (negative, _) = negative.linearize(&coordinates).unwrap();

    assert_eq!(positive.design_values(), &[1.0]);
    assert_eq!(negative.design_values(), &[-1.0]);
    assert!(
        positive
            .design_jacobian()
            .iter()
            .zip(negative.design_jacobian())
            .any(|(positive, negative)| {
                positive.abs() > 1.0e-12 && (*positive + *negative).abs() < 1.0e-12
            }),
        "the accepted points must retain their opposite derivative actions even though their primal systems collide"
    );
}

#[test]
fn affine_coordinate_coefficient_uses_the_existing_fem_and_fvm_finalizers() {
    let source = SOURCE.replace(
        "diffusion * grad(potential)",
        "diffusion * (1 + wave_number * coordinate(0)) * grad(potential)",
    );
    let program = compile_program("affine-coordinate-coefficient.eqi", &source);
    let model = lower_scalar_elliptic_cartesian(&program).unwrap();
    assert_eq!(
        model
            .coefficient_expression()
            .evaluate(&[0.0, 0.5])
            .unwrap(),
        1.0
    );
    assert!(
        model
            .coefficient_expression()
            .evaluate(&[1.0, 0.5])
            .unwrap()
            > 4.0
    );
    let wave_number = model
        .parameter_fields()
        .iter()
        .copied()
        .zip(model.parameter_values())
        .find_map(|(parameter, value)| {
            (value.to_bits() == std::f64::consts::PI.to_bits()).then_some(parameter)
        })
        .unwrap();
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];
    parameter_tangent[model
        .parameter_fields()
        .iter()
        .position(|parameter| *parameter == wave_number)
        .unwrap()] = 1.0;
    let (coefficient, coefficient_tangent) = model
        .coefficient_jvp(&[0.5, 0.5], &[0.0, 0.0], &parameter_tangent)
        .unwrap();
    assert!(coefficient > 2.0);
    assert_eq!(coefficient_tangent, 0.5);

    for method in [
        DiscretizationMethod::ContinuousGalerkin,
        DiscretizationMethod::CellCenteredFiniteVolume,
    ] {
        let resolved = resolve_plan_with_method(&program, method);
        let finalized = finalize_scalar_elliptic_parameter_point(model.clone(), &resolved).unwrap();
        let solved = REFERENCE_LINEAR_SOLVER
            .solve(
                &finalized.linear_problem().unwrap(),
                finalized.solver_plan(),
            )
            .unwrap();
        let accepted = finalized.finish(solved).unwrap();
        let (relation, _) = accepted
            .linearize(&[SpatialDesignCoordinate::ModelParameter(wave_number)])
            .unwrap();
        assert!(
            relation
                .design_jacobian()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            relation
                .design_jacobian()
                .iter()
                .any(|value| value.abs() > 1.0e-12)
        );
        let finite = match accepted.solution() {
            eqiora_numerics::scalar::ResolvedScalarEllipticCartesianSolution::FiniteElement(
                solution,
            ) => solution
                .field()
                .vertex_values()
                .iter()
                .all(|value| value.is_finite()),
            eqiora_numerics::scalar::ResolvedScalarEllipticCartesianSolution::FiniteVolume(
                solution,
            ) => solution.cell_values().iter().all(|value| value.is_finite()),
        };
        assert!(finite);
    }
}

#[test]
fn coordinate_coefficient_fails_closed_without_affinity_or_domain_positivity() {
    for (file, coefficient) in [
        (
            "nonaffine-coordinate-coefficient.eqi",
            "diffusion * (1 + (wave_number * coordinate(0)) ^ 2)",
        ),
        (
            "nonpositive-coordinate-coefficient.eqi",
            "diffusion * (wave_number * coordinate(0) - 1)",
        ),
    ] {
        let source = SOURCE.replace(
            "diffusion * grad(potential)",
            &format!("{coefficient} * grad(potential)"),
        );
        let program = compile_program(file, &source);
        assert!(lower_scalar_elliptic_cartesian(&program).is_err());
    }
}

#[test]
fn mixed_natural_boundary_reuses_the_volume_coefficient_for_fem_and_fvm() {
    let coefficient = "diffusion * (1 + wave_number * coordinate(0))";
    let source = SOURCE
        .replace(
            "diffusion * grad(potential)",
            &format!("{coefficient} * grad(potential)"),
        )
        .replace(
            "relation y_upper_value on y_upper { trace(potential) - boundary_offset = 0; }",
            &format!(
                "relation y_upper_value on y_upper {{ normal({coefficient} * grad(potential)) = wave_number; }}"
            ),
        );
    let program = compile_program("mixed-natural-affine-coefficient.eqi", &source);
    let model = lower_scalar_elliptic_cartesian(&program).unwrap();
    let wave_number = model
        .parameter_fields()
        .iter()
        .copied()
        .zip(model.parameter_values())
        .find_map(|(parameter, value)| {
            (value.to_bits() == std::f64::consts::PI.to_bits()).then_some(parameter)
        })
        .unwrap();
    for method in [
        DiscretizationMethod::ContinuousGalerkin,
        DiscretizationMethod::CellCenteredFiniteVolume,
    ] {
        let resolved = resolve_plan_with_method(&program, method);
        let finalized = finalize_scalar_elliptic_parameter_point(model.clone(), &resolved).unwrap();
        let solved = REFERENCE_LINEAR_SOLVER
            .solve(
                &finalized.linear_problem().unwrap(),
                finalized.solver_plan(),
            )
            .unwrap();
        let accepted = finalized.finish(solved).unwrap();
        let (relation, _) = accepted
            .linearize(&[SpatialDesignCoordinate::ModelParameter(wave_number)])
            .unwrap();
        assert!(
            relation
                .design_jacobian()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    let drifted = source.replace(
        &format!("normal({coefficient} * grad(potential))"),
        "normal(diffusion * grad(potential))",
    );
    let drifted = compile_program("drifted-natural-coefficient.eqi", &drifted);
    assert!(lower_scalar_elliptic_cartesian(&drifted).is_err());
}

#[test]
fn unrelated_declaration_order_does_not_change_the_effective_coefficient_or_operator() {
    let reordered = SOURCE.replace(
        "  parameter source_scale: 1 / m ^ 2 = 19.739208802178716;\n  parameter boundary_offset: 1 = 0;",
        "  parameter boundary_offset: 1 = 0;\n  parameter source_scale: 1 / m ^ 2 = 19.739208802178716;",
    );
    let original = compile_program("declaration-order.eqi", SOURCE);
    let reordered = compile_program("declaration-order.eqi", &reordered);
    let original_model = lower_scalar_elliptic_cartesian(&original).unwrap();
    let reordered_model = lower_scalar_elliptic_cartesian(&reordered).unwrap();
    for coordinates in [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]] {
        assert_eq!(
            original_model
                .coefficient_expression()
                .evaluate(&coordinates)
                .unwrap(),
            reordered_model
                .coefficient_expression()
                .evaluate(&coordinates)
                .unwrap(),
        );
    }
    let original =
        finalize_scalar_elliptic_parameter_point(original_model, &resolve_plan(&original)).unwrap();
    let reordered =
        finalize_scalar_elliptic_parameter_point(reordered_model, &resolve_plan(&reordered))
            .unwrap();
    assert_eq!(
        original.canonical_csr_system_view(),
        reordered.canonical_csr_system_view()
    );
}

fn compile_program(file: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(file, source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn resolve_plan(program: &KernelProgram) -> ResolvedRealization {
    resolve_plan_with_method(program, DiscretizationMethod::ContinuousGalerkin)
}

fn resolve_plan_with_method(
    program: &KernelProgram,
    method: DiscretizationMethod,
) -> ResolvedRealization {
    let (space, quadrature) = match method {
        DiscretizationMethod::ContinuousGalerkin => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            (Space::cell_constant(), QuadraturePolicy::CellCentroid)
        }
    };
    let plan = RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(3).unwrap(),
            },
            quadrature,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(512).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(0),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap()
}
