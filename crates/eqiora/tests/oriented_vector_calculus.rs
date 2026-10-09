//! Coordinate polynomials are differentiated by the retained spatial graph.
use eqiora::api::ModelDocument;
use eqiora::{DimExponents, DynQuantity};
use eqiora_sem::EvaluationPoint;

const POLYNOMIAL: &str =
    include_str!("../../../verify/language/oriented-vector-calculus/models/cartesian.eqi");

fn length(power: i32) -> DimExponents {
    DimExponents::from_integers([0, power, 0, 0, 0, 0, 0]).unwrap()
}

fn sample(
    document: &ModelDocument,
    program: &eqiora_sem::KernelProgram,
    name: &str,
    domain: &str,
    xyz: [f64; 3],
) -> DynQuantity {
    sample_program(document, program, name, domain, xyz, 1.).unwrap()
}

fn sample_bound(
    document: &ModelDocument,
    name: &str,
    domain: &str,
    xyz: [f64; 3],
    x_coefficient: f64,
) -> Result<DynQuantity, eqiora::Diagnostic> {
    sample_program(
        document,
        document.program(),
        name,
        domain,
        xyz,
        x_coefficient,
    )
}

fn sample_program(
    document: &ModelDocument,
    program: &eqiora_sem::KernelProgram,
    name: &str,
    domain: &str,
    xyz: [f64; 3],
    x_coefficient: f64,
) -> Result<DynQuantity, eqiora::Diagnostic> {
    let body = document.aliases()["body"];
    let point = EvaluationPoint::new(
        program,
        document.aliases()[domain].downcast().unwrap(),
        xyz.into_iter()
            .enumerate()
            .map(|(axis, value)| ((body, axis), DynQuantity::new(value, length(1))))
            .collect(),
        None,
    )?;
    program
        .evaluate_observable_with_points(
            document.aliases()[name].downcast().unwrap(),
            Some(&point),
            &mut |input, _| match input {
                eqiora_sem::EvaluationInput::Value(eqiora::kernel::SymbolRef::Parameter(id)) => {
                    let value = program.typed_value(id.erase()).unwrap();
                    Ok(if id.erase() == document.aliases()["ex"] {
                        eqiora::ValueLiteral::new(
                            value.value_type().clone(),
                            [(x_coefficient, 0.), (0., 0.), (0., 0.)],
                        )
                        .unwrap()
                    } else {
                        value.clone()
                    })
                }
                _ => panic!("explicit coordinate polynomial needs no reconstructed Field"),
            },
        )
        .map(|value| value.real_scalar_value().unwrap())
}

#[test]
fn polynomial_curl_curl_and_oriented_boundary_values_follow_direct_derivatives() {
    let document = ModelDocument::compile("oriented.eqi", POLYNOMIAL).unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
    // Source aliases are intentionally absent from artifact replay; exact IDs survive.
    for program in [document.program(), replay.program()] {
        for [x, y, z] in [[2., 3., 5.], [1., 4., 2.], [6., 2., 3.]] {
            // Differentiate F=(y²z,z²x,x²y) directly, with derivative axis last.
            for (name, expected) in [
                ("c0", x * x - 2. * x * z),
                ("c1", y * y - 2. * x * y),
                ("c2", z * z - 2. * y * z),
            ] {
                assert_eq!(
                    sample(&document, program, name, "body", [x, y, z]),
                    DynQuantity::new(expected, length(2))
                );
            }
            for (name, expected) in [
                ("cc0", -2. * z),
                ("cc1", -2. * x),
                ("cc2", -2. * y),
                ("divergence", 0.),
                ("potential_curl", 0.),
                ("laplace", 2. * z),
            ] {
                assert_eq!(
                    sample(&document, program, name, "body", [x, y, z]),
                    DynQuantity::new(expected, length(1))
                );
            }
        }
        // Same constant vector U=(2,3,5)m³ on the two oppositely oriented x-faces.
        assert_eq!(
            sample(&document, program, "lower_tangent", "lo", [0., 3., 5.]),
            DynQuantity::new(5., length(3))
        );
        assert_eq!(
            sample(&document, program, "upper_tangent", "hi", [8., 3., 5.]),
            DynQuantity::new(-5., length(3))
        );
    }
}

#[test]
fn analytic_derivatives_read_the_actual_bound_parameter_values() {
    let document = ModelDocument::compile("oriented.eqi", POLYNOMIAL).unwrap();
    // Fx=2*y²*z: curl_y=2*y²-2*x*y, curl_z=z²-4*y*z.
    assert_eq!(
        sample_bound(&document, "c1", "body", [2., 3., 5.], 2.).unwrap(),
        DynQuantity::new(6., length(2))
    );
    assert_eq!(
        sample_bound(&document, "c2", "body", [2., 3., 5.], 2.).unwrap(),
        DynQuantity::new(-35., length(2))
    );
}

#[test]
fn unknown_spatial_fields_are_never_differentiated_as_constants() {
    let source = POLYNOMIAL.replace(
        "let F=ex*y*y*z+ey*z*z*x+ez*x*x*y;",
        "variable F:vector<m^3,3> on body;",
    );
    let document = ModelDocument::compile("unknown-field.eqi", &source).unwrap();
    let failure = sample_bound(&document, "c0", "body", [2., 3., 5.], 1.).unwrap_err();
    assert!(
        failure
            .message()
            .contains("unknown Field derivatives require a reconstruction"),
        "{failure:?}"
    );
}

#[test]
fn boundary_points_require_the_exact_parent_axes_and_oriented_face() {
    let document = ModelDocument::compile("boundary.eqi", POLYNOMIAL).unwrap();
    let body = document.aliases()["body"];
    let hi = document.aliases()["hi"].downcast().unwrap();
    let coordinates = |x| {
        vec![
            ((body, 0), DynQuantity::new(x, length(1))),
            ((body, 1), DynQuantity::new(3., length(1))),
            ((body, 2), DynQuantity::new(5., length(1))),
        ]
    };
    EvaluationPoint::new(document.program(), hi, coordinates(8.), None).unwrap();
    let failure = EvaluationPoint::new(document.program(), hi, coordinates(7.), None).unwrap_err();
    assert!(failure.message().contains("exact oriented face"));
    let mut foreign = coordinates(8.);
    foreign[0].0.0 = document.aliases()["lo"];
    assert!(
        EvaluationPoint::new(document.program(), hi, foreign, None)
            .unwrap_err()
            .message()
            .contains("exact support coordinate")
    );
    assert!(
        EvaluationPoint::new(document.program(), hi, coordinates(8.)[..2].to_vec(), None).is_err()
    );
}

#[test]
fn polynomial_curl_is_observed_through_an_ordinary_plan_and_result() {
    use eqiora_artifact::ModelEnvelope;
    use eqiora_backend_faer::FaerLinearSolver;
    use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
    use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
    use std::{collections::HashMap, num::NonZeroUsize};

    let source = POLYNOMIAL.replace(
        "observable c0:m^2 on body=component(curl(F),indices=(0,));",
        "observable c0:m^2=evaluate(component(curl(F),indices=(0,)),at=(x=2[m],y=3[m],z=5[m]));",
    );
    // Physical basis vectors are fixed typed expressions, not finite invariant
    // Parameter channels.
    let source = ["ex", "ey", "ez"].into_iter().fold(source, |source, name| {
        source.replace(
            &format!("parameter {name}:vector<1,3>="),
            &format!("let {name}="),
        )
    });
    let document = ModelDocument::compile("curl-result.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-12,
            NonZeroUsize::new(10).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast),
        FaerLinearSolver.provider(),
    )
    .unwrap();
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(linear),
        None,
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let observed = result
        .observe(
            &model,
            document.aliases()["c0"].downcast().unwrap(),
            &HashMap::new(),
        )
        .unwrap();
    // ∂y Fz − ∂z Fy at (2,3,5) is x² − 2xz = −16 m².
    assert_eq!(
        observed.value().real_scalar_value().unwrap(),
        DynQuantity::new(-16., length(2))
    );
    let spatial_source = source.replace(
        "variable anchor:1;",
        "variable anchor:1; variable spatial:1 on body;",
    );
    let spatial = ModelDocument::compile("spatial-unknown.eqi", &spatial_source).unwrap();
    // Source spatial Fields also synthesize a continuum Representation; finite
    // admission may reject that inventory before reaching the Field support edge.
    assert!(
        spatial
            .program()
            .nodes()
            .any(|node| matches!(node, eqiora::kernel::KernelNode::Representation(_)))
    );
    let spatial_model = ModelEnvelope::from_program(spatial.program()).unwrap();
    let failure = CommonAlgebraicPlan::resolve(
        &spatial_model,
        CommonSolvePolicy::Linear(linear),
        None,
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap_err();
    assert!(failure.message().contains("spatial support"), "{failure:?}");
}
