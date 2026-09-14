mod support;

use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::api::{DifferentiableProgram, LinearizationState};
use eqiora::differentiation::{AcceptedLinearization, adjoint_gradient, forward_sensitivity};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, Target, VectorLayoutKind,
    resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorOrientation, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use eqiora::{Id, compiler::compile, entity::kinds};
use eqiora_numerics::{
    CommonSpatialPolicy, common::SpatialDesignCoordinate,
    scalar::ResolvedScalarEllipticCartesianSolution,
    scalar::solve_and_linearize_resolved_scalar_elliptic_cartesian,
    scalar::solve_resolved_scalar_elliptic_cartesian,
};
use support::common_scalar_plan::{COMPONENT, document_and_plan, document_and_plan_with_source};

const SOURCE: &str =
    include_str!("../../../verify/differentiation/spatial-poisson-fem-fvm/models/poisson.eqi");
const SOURCE_SCALE_DECLARATION: &str = "parameter source_scale: 1 / m ^ 2 = 19.739208802178716;";
const DIFFUSION_DECLARATION: &str = "parameter diffusion: 1 = 1;";
const BOUNDARY_DECLARATION: &str = "parameter boundary_offset: 1 = 0;";

#[derive(Clone, Copy)]
struct ParameterValues {
    source_scale: f64,
    diffusion: f64,
    boundary_offset: f64,
}

const NOMINAL: ParameterValues = ParameterValues {
    source_scale: 19.739208802178716,
    diffusion: 1.0,
    boundary_offset: 0.0,
};

struct ParameterFields {
    source_scale: Id<kinds::Parameter>,
    diffusion: Id<kinds::Parameter>,
    boundary_offset: Id<kinds::Parameter>,
}

#[test]
fn canonical_parameters_differentiate_through_fem_and_fvm() {
    let (program, parameters) = compile_program("spatial-poisson.eqi", SOURCE);
    for method in [
        DiscretizationMethod::ContinuousGalerkin,
        DiscretizationMethod::CellCenteredFiniteVolume,
    ] {
        verify_method(&program, &parameters, method);
    }
}

#[test]
fn accepted_application_program_publishes_complete_field_jvp_and_vjp() {
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        verify_application_program(method);
    }
}

#[test]
fn application_program_is_not_published_without_an_accepted_primal() {
    let source = COMPONENT.replacen(
        "source_scale * math.sin",
        "1e308 * source_scale * math.sin",
        1,
    );
    assert_ne!(source, COMPONENT, "the overflow falsifier must be applied");
    let (document, plan) = document_and_plan_with_source(CommonSpatialPolicy::Q1, &source);
    let inputs = [document.parameter_ref("source_scale").unwrap()];
    let output = document
        .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
        .unwrap();
    assert!(DifferentiableProgram::compile(plan, &inputs, &output).is_err());
}

#[test]
fn application_program_admission_is_exactly_the_verified_two_dimensional_slice() {
    let (_, plan) = document_and_plan(CommonSpatialPolicy::Q1);
    assert_eq!(plan.cells(), [12, 12]);
}

#[test]
fn equal_primal_systems_do_not_alias_distinct_parameter_derivatives() {
    let source = COMPONENT.replace(
        "diffusion * grad(potential)",
        "diffusion ^ 2 * grad(potential)",
    );
    let (document, plan) = document_and_plan_with_source(CommonSpatialPolicy::Q1, &source);
    let diffusion = document.parameter_ref("diffusion").unwrap();
    let output = document
        .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
        .unwrap();
    let program = DifferentiableProgram::compile(plan, &[diffusion], &output).unwrap();
    let positive = program.evaluate(&[1.0]).unwrap();
    let negative = program.evaluate(&[-1.0]).unwrap();
    let positive_primal = positive.primal();
    let negative_primal = negative.primal();
    assert_eq!(
        positive_primal.evidence().state_system(),
        negative_primal.evidence().state_system(),
        "the falsifier requires a non-injective point-to-primal-system map"
    );
    assert_eq!(positive_primal.output(), negative_primal.output());
    assert_eq!(positive.point().values(), &[1.0]);
    assert_eq!(negative.point().values(), &[-1.0]);

    let positive = positive.jvp(&[1.0]).unwrap();
    let negative = negative.jvp(&[1.0]).unwrap();
    assert!(
        positive
            .tangent()
            .iter()
            .zip(negative.tangent())
            .any(|(positive, negative)| {
                positive.abs() > 1.0e-12 && (*positive + *negative).abs() < 1.0e-12
            }),
        "equal primal systems must retain their opposite accepted derivative actions"
    );
}

fn verify_application_program(method: CommonSpatialPolicy) {
    let (document, plan) = document_and_plan(method);
    let source_scale = document.parameter_ref("source_scale").unwrap();
    let diffusion = document.parameter_ref("diffusion").unwrap();
    let boundary = document.parameter_ref("boundary_offset").unwrap();
    let output = document
        .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
        .unwrap();
    let inputs = [source_scale, diffusion, boundary];
    let program = DifferentiableProgram::compile(plan.clone(), &inputs, &output).unwrap();
    assert_eq!(program.identity().input_dimension(), 3);
    assert_eq!(
        program.identity().inputs(),
        &inputs.iter().map(|input| input.id()).collect::<Vec<_>>()
    );
    let field_value_count = match method {
        CommonSpatialPolicy::Q1 => 13 * 13,
        CommonSpatialPolicy::CellCenteredTpfa => 12 * 12,
        CommonSpatialPolicy::P1
        | CommonSpatialPolicy::MiniP1
        | CommonSpatialPolicy::CellCentered => {
            panic!("this Cartesian fixture does not admit the requested policy")
        }
    };
    assert_eq!(program.identity().output_dimension(), field_value_count);

    let primal = program.primal();
    assert_eq!(primal.output().len(), field_value_count);
    assert_eq!(
        primal.evidence().primal_solve().orientation(),
        LinearOperatorOrientation::Normal
    );
    assert_eq!(
        primal.evidence().receipt().operator(),
        primal.evidence().state_system()
    );
    assert_eq!(
        primal.evidence().linearization_state(),
        LinearizationState::Established
    );
    assert_eq!(
        program.primal().evidence().linearization_state(),
        LinearizationState::Established,
        "publishing the retained primal reports how the evaluation was established, not another solve"
    );
    let direction = [0.7, -0.2, 0.3];
    let jvp = program.jvp(&direction).unwrap();
    assert_eq!(jvp.output(), primal.output());
    assert_eq!(
        jvp.evidence()
            .derivative_solve()
            .expect("JVP has one normal derivative solve")
            .orientation(),
        LinearOperatorOrientation::Normal
    );

    let step = 1.0e-5;
    let plus = perturbed_field_values(method, shifted(NOMINAL, direction, step));
    let minus = perturbed_field_values(method, shifted(NOMINAL, direction, -step));
    let finite_difference = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * step))
        .collect::<Vec<_>>();
    assert_eq!(jvp.tangent().len(), finite_difference.len());
    for (computed, reference) in jvp.tangent().iter().zip(finite_difference) {
        assert_relative_close(*computed, reference, 6.0e-7, 4.0e-10);
    }

    let cotangent = (0..primal.output().len())
        .map(|index| (index + 1) as f64 / primal.output().len() as f64)
        .collect::<Vec<_>>();
    let vjp = program.vjp(&cotangent).unwrap();
    assert_eq!(vjp.output(), primal.output());
    assert_eq!(
        vjp.evidence()
            .derivative_solve()
            .expect("VJP has one transposed derivative solve")
            .orientation(),
        LinearOperatorOrientation::Transposed
    );
    let finite_gradient = std::array::from_fn::<_, 3, _>(|parameter| {
        let mut basis = [0.0; 3];
        basis[parameter] = 1.0;
        let plus = perturbed_field_values(method, shifted(NOMINAL, basis, step));
        let minus = perturbed_field_values(method, shifted(NOMINAL, basis, -step));
        plus.iter()
            .zip(minus)
            .zip(&cotangent)
            .map(|((plus, minus), cotangent)| cotangent * (plus - minus) / (2.0 * step))
            .sum::<f64>()
    });
    for (computed, reference) in vjp.input_cotangent().iter().zip(finite_gradient) {
        assert_relative_close(*computed, reference, 8.0e-7, 8.0e-10);
    }
    let forward_pairing = jvp
        .tangent()
        .iter()
        .zip(&cotangent)
        .map(|(tangent, cotangent)| tangent * cotangent)
        .sum::<f64>();
    let reverse_pairing = direction
        .iter()
        .zip(vjp.input_cotangent())
        .map(|(tangent, cotangent)| tangent * cotangent)
        .sum::<f64>();
    assert_relative_close(forward_pairing, reverse_pairing, 2.0e-10, 2.0e-11);

    let alternate = ParameterValues {
        source_scale: 17.25,
        diffusion: 1.35,
        boundary_offset: 0.08,
    };
    let alternate_values = [
        alternate.source_scale,
        alternate.diffusion,
        alternate.boundary_offset,
    ];
    let alternate_evaluation = program.evaluate(&alternate_values).unwrap();
    assert_eq!(alternate_evaluation.identity(), program.identity());
    assert_eq!(
        alternate_evaluation.point().inputs(),
        program.identity().inputs()
    );
    assert_eq!(alternate_evaluation.point().values(), &alternate_values);
    let alternate_primal = alternate_evaluation.primal();
    let independently_rebuilt = perturbed_field_values(method, alternate);
    assert_eq!(alternate_primal.output().len(), independently_rebuilt.len());
    for (computed, reference) in alternate_primal.output().iter().zip(independently_rebuilt) {
        assert_relative_close(*computed, reference, 2.0e-12, 2.0e-12);
    }

    let alternate_jvp = alternate_evaluation.jvp(&direction).unwrap();
    let plus = perturbed_field_values(method, shifted(alternate, direction, step));
    let minus = perturbed_field_values(method, shifted(alternate, direction, -step));
    for ((computed, plus), minus) in alternate_jvp.tangent().iter().zip(plus).zip(minus) {
        assert_relative_close(*computed, (plus - minus) / (2.0 * step), 8.0e-7, 8.0e-10);
    }
    let alternate_vjp = alternate_evaluation.vjp(&cotangent).unwrap();
    let alternate_forward_pairing = alternate_jvp
        .tangent()
        .iter()
        .zip(&cotangent)
        .map(|(tangent, cotangent)| tangent * cotangent)
        .sum::<f64>();
    let alternate_reverse_pairing = direction
        .iter()
        .zip(alternate_vjp.input_cotangent())
        .map(|(tangent, cotangent)| tangent * cotangent)
        .sum::<f64>();
    assert_relative_close(
        alternate_forward_pairing,
        alternate_reverse_pairing,
        2.0e-10,
        2.0e-11,
    );

    let default_again = program.evaluate(program.default_point().values()).unwrap();
    assert_eq!(default_again.primal().output(), primal.output());
    assert_eq!(
        default_again.primal().evidence().state_system(),
        primal.evidence().state_system()
    );
    assert_eq!(program.identity(), alternate_evaluation.identity());
    assert_eq!(
        program.default_point().values(),
        &[19.739208802178716, 1.0, 0.0]
    );

    let mut invalid_coefficient = alternate_values;
    invalid_coefficient[1] = -1.0;
    assert!(program.evaluate(&invalid_coefficient).is_err());
    assert!(program.evaluate(&alternate_values[..2]).is_err());
    let mut non_finite = alternate_values;
    non_finite[0] = f64::NAN;
    assert!(program.evaluate(&non_finite).is_err());

    std::thread::scope(|scope| {
        let default = scope.spawn(|| {
            program
                .evaluate(program.default_point().values())
                .unwrap()
                .primal()
        });
        let alternate = scope.spawn(|| program.evaluate(&alternate_values).unwrap().primal());
        assert_eq!(default.join().unwrap().output(), primal.output());
        assert_eq!(
            alternate.join().unwrap().output(),
            alternate_primal.output()
        );
    });

    assert!(program.jvp(&[1.0, 2.0, 3.0, 4.0]).is_err());
    assert!(program.vjp(&cotangent[..cotangent.len() - 1]).is_err());
    let recomputed = DifferentiableProgram::compile(plan, &inputs, &output).unwrap();
    assert_eq!(recomputed.identity(), program.identity());
    assert_eq!(recomputed.primal().output(), primal.output());
    assert_eq!(
        recomputed.primal().evidence().state_system(),
        primal.evidence().state_system()
    );

    let (foreign, foreign_plan) = document_and_plan_with_source(
        method,
        &COMPONENT.replacen(
            "diffusion * grad(potential)",
            "2 * diffusion * grad(potential)",
            1,
        ),
    );
    assert!(DifferentiableProgram::compile(foreign_plan.clone(), &inputs, &output).is_err());
    let foreign_inputs = [foreign.parameter_ref("source_scale").unwrap()];
    assert!(DifferentiableProgram::compile(foreign_plan, &foreign_inputs, &output).is_err());
}

fn verify_method(
    program: &KernelProgram,
    parameters: &ParameterFields,
    method: DiscretizationMethod,
) {
    let resolved = resolved(program, method);
    // This analysis order is intentionally different from lowering order.
    let selected_parameters = [
        parameters.source_scale,
        parameters.diffusion,
        parameters.boundary_offset,
    ];
    let selected = selected_parameters.map(SpatialDesignCoordinate::from);
    let (model, solution, linearization) = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        program,
        &resolved,
        &REFERENCE_LINEAR_SOLVER,
        &selected,
    )
    .unwrap();
    assert_eq!(linearization.design_coordinates(), &selected);
    assert_eq!(
        linearization.design_values(),
        &[
            NOMINAL.source_scale,
            NOMINAL.diffusion,
            NOMINAL.boundary_offset,
        ]
    );
    assert_eq!(model.parameter_fields().len(), 4);
    assert_eq!(
        linearization.design_jacobian().len(),
        3 * linearization.accepted_unknowns().len()
    );

    let accepted = AcceptedLinearization::new(&linearization, 2.0e-9).unwrap();
    let solver_plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(4096).unwrap(),
    )
    .unwrap();
    let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver_plan);
    let parameter_tangent = [0.7, -0.2, 0.3];
    let forward = forward_sensitivity(
        &accepted,
        &parameter_tangent,
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver,
    )
    .unwrap();

    let step = 1.0e-5;
    let plus = perturbed_values(method, shifted(NOMINAL, parameter_tangent, step));
    let minus = perturbed_values(method, shifted(NOMINAL, parameter_tangent, -step));
    let finite_difference = plus
        .iter()
        .zip(&minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * step))
        .collect::<Vec<_>>();
    assert_eq!(forward.values().len(), finite_difference.len());
    for (computed, reference) in forward.values().iter().zip(&finite_difference) {
        assert_relative_close(*computed, *reference, 5.0e-7, 3.0e-10);
    }

    let objective_unknown_cotangent =
        vec![1.0 / forward.values().len() as f64; forward.values().len()];
    let adjoint = adjoint_gradient(
        &accepted,
        &objective_unknown_cotangent,
        &[0.0; 3],
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver,
    )
    .unwrap();
    let finite_objective_gradient = std::array::from_fn::<_, 3, _>(|parameter| {
        let mut direction = [0.0; 3];
        direction[parameter] = 1.0;
        let plus = perturbed_values(method, shifted(NOMINAL, direction, step));
        let minus = perturbed_values(method, shifted(NOMINAL, direction, -step));
        (mean(&plus) - mean(&minus)) / (2.0 * step)
    });
    for (computed, reference) in adjoint.gradient().iter().zip(finite_objective_gradient) {
        assert_relative_close(*computed, reference, 5.0e-7, 3.0e-10);
    }
    let adjoint_directional = adjoint
        .gradient()
        .iter()
        .zip(parameter_tangent)
        .map(|(gradient, tangent)| gradient * tangent)
        .sum();
    assert_relative_close(
        adjoint_directional,
        mean(forward.values()),
        2.0e-10,
        2.0e-12,
    );

    match (method, solution) {
        (
            DiscretizationMethod::ContinuousGalerkin,
            ResolvedScalarEllipticCartesianSolution::FiniteElement(_),
        )
        | (
            DiscretizationMethod::CellCenteredFiniteVolume,
            ResolvedScalarEllipticCartesianSolution::FiniteVolume(_),
        ) => {}
        _ => panic!("resolved method and spatial solution differ"),
    }
}

fn perturbed_values(method: DiscretizationMethod, values: ParameterValues) -> Vec<f64> {
    let source_scale = format!(
        "parameter source_scale: 1 / m ^ 2 = {:.17};",
        values.source_scale
    );
    let diffusion = format!("parameter diffusion: 1 = {:.17};", values.diffusion);
    let boundary = format!(
        "parameter boundary_offset: 1 = {:.17};",
        values.boundary_offset
    );
    let source = SOURCE
        .replacen(SOURCE_SCALE_DECLARATION, &source_scale, 1)
        .replacen(DIFFUSION_DECLARATION, &diffusion, 1)
        .replacen(BOUNDARY_DECLARATION, &boundary, 1);
    assert_ne!(
        source, SOURCE,
        "finite-difference declarations were not replaced"
    );
    let (program, _) = compile_program("perturbed-spatial-poisson.eqi", &source);
    let resolved = resolved(&program, method);
    let (_, solution) =
        solve_resolved_scalar_elliptic_cartesian(&program, &resolved, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
    match solution {
        ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) => {
            solution.algebraic_values().to_vec()
        }
        ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution) => {
            solution.cell_values().to_vec()
        }
    }
}

fn perturbed_field_values(method: CommonSpatialPolicy, values: ParameterValues) -> Vec<f64> {
    let source_scale = format!(
        "parameter source_scale: 1 / m ^ 2 = {:.17};",
        values.source_scale
    );
    let diffusion = format!("parameter diffusion: 1 = {:.17};", values.diffusion);
    let boundary = format!(
        "parameter boundary_offset: 1 = {:.17};",
        values.boundary_offset
    );
    let source = SOURCE
        .replacen(SOURCE_SCALE_DECLARATION, &source_scale, 1)
        .replacen(DIFFUSION_DECLARATION, &diffusion, 1)
        .replacen(BOUNDARY_DECLARATION, &boundary, 1);
    let (program, _) = compile_program("perturbed-field-spatial-poisson.eqi", &source);
    let method = match method {
        CommonSpatialPolicy::Q1 => DiscretizationMethod::ContinuousGalerkin,
        CommonSpatialPolicy::CellCenteredTpfa => DiscretizationMethod::CellCenteredFiniteVolume,
        CommonSpatialPolicy::P1
        | CommonSpatialPolicy::MiniP1
        | CommonSpatialPolicy::CellCentered => {
            panic!("this Cartesian fixture does not admit the requested policy")
        }
    };
    let (_, solution) = solve_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved(&program, method),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    match solution {
        ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) => {
            solution.field().vertex_values().to_vec()
        }
        ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution) => {
            solution.cell_values().to_vec()
        }
    }
}

fn compile_program(file: &str, source: &str) -> (KernelProgram, ParameterFields) {
    let mut compiled = compile(file, source).unwrap();
    let parameter = |name| {
        compiled[0]
            .symbols()
            .get(name)
            .unwrap()
            .downcast::<kinds::Parameter>()
            .unwrap()
    };
    let parameters = ParameterFields {
        source_scale: parameter("source_scale"),
        diffusion: parameter("diffusion"),
        boundary_offset: parameter("boundary_offset"),
    };
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        parameters,
    )
}

fn resolved(program: &KernelProgram, method: DiscretizationMethod) -> ResolvedRealization {
    let cells = NonZeroUsize::new(12).unwrap();
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
                cells_per_axis: cells,
            },
            quadrature,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(4096).unwrap(),
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
            RealizationRevision::new(method as u64),
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

fn shifted(values: ParameterValues, direction: [f64; 3], step: f64) -> ParameterValues {
    ParameterValues {
        source_scale: values.source_scale + step * direction[0],
        diffusion: values.diffusion + step * direction[1],
        boundary_offset: values.boundary_offset + step * direction[2],
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn assert_relative_close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    let tolerance = absolute + relative * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}
