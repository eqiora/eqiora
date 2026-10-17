#[path = "complex_spatial_differentiation/selection.rs"]
mod selection;
mod support;
use eqiora::api::{
    DerivativeImplementation, DifferentiableCotangent, DifferentiableProgram,
    DifferentiableTangent, DifferentiableValues, ModelDocument, RecordedDifferentiableEvaluation,
};
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::GeometryGraph;
use eqiora::solver::{
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER,
    ReductionPolicy, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonSolvePolicy, CommonSpatialPolicy,
    ResolvedCommonPlan,
};
use num_complex::Complex64 as C;
use std::{collections::BTreeMap, num::NonZeroUsize};
use support::derivative_products::FixtureCoordinates;

const SOURCE: &str = r#"
public component ComplexPoisson(
 support body:volume(ambient_dimension=1),
 support left:boundary(parent=body), support right:boundary(parent=body),
 parameter a:complex<1>,
 parameter p:array<complex<1>,2>,
 parameter q:array<complex<1>,2>
) {
 variable u:array<complex<1>,2> on body in h1;
 relation balance on body { -div(math.conj(a)*grad(u)) = q / 1[m^2]; }
 relation left_value on left {trace(u)=p;}
 relation right_value on right {trace(u)=2*p;}
}
"#;

fn fixture(method: CommonSpatialPolicy) -> (ModelDocument, ResolvedCommonPlan) {
    fixture_source(method, SOURCE)
}

fn fixture_source(
    method: CommonSpatialPolicy,
    source: &str,
) -> (ModelDocument, ResolvedCommonPlan) {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 3.]).unwrap();
    let boundaries = interval.boundaries();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![boundaries[0].into()]),
                ("right".into(), vec![boundaries[1].into()]),
            ]),
        )
        .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let scalar = eqiora::ValueType::scalar(
        eqiora::ScalarDomain::Complex,
        eqiora::DimExponents::DIMENSIONLESS,
    )
    .unwrap();
    let parameters = [
        (
            "a",
            eqiora::ValueLiteral::new(scalar.clone(), [(2., 3.)]).unwrap(),
        ),
        (
            "p",
            eqiora::ValueLiteral::new(scalar.clone().array(2).unwrap(), [(5., 7.), (-1., 4.)])
                .unwrap(),
        ),
        (
            "q",
            eqiora::ValueLiteral::new(scalar.array(2).unwrap(), [(11., -2.), (3., 8.)]).unwrap(),
        ),
    ];
    let mut bindings = vec![
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: body,
                parent: None,
            },
        ),
        (
            "left",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("left").unwrap(),
                parent: Some(body),
            },
        ),
        (
            "right",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("right").unwrap(),
                parent: Some(body),
            },
        ),
    ];
    bindings.extend(
        parameters
            .iter()
            .map(|(name, value)| (*name, StaticBindingValue::Value(value))),
    );
    let document =
        ModelDocument::compile_selected("complex-spatial.eqi", source, "ComplexPoisson", &bindings)
            .unwrap();
    let cells = CartesianMeshCellsV2::new(vec![4]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, cells.cells())
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    // Coupled adjoint products amplify residual error; request enough solve
    // accuracy to retain the fixed absolute analytic-product tolerance below.
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-14,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    let plan = ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.program()).unwrap(),
        owner,
        method,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, REFERENCE_SOLVER_PROVIDER).unwrap(),
        ),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    (document, plan)
}

// Independently solve -conj(a)*u''=q with u(0)=p, u(3)=2p.
// Q1 reproduces quadratic nodal values. TPFA adds q*h²/(8*conj(a)).
fn oracle(method: CommonSpatialPolicy, point: &[C; 5], direction: &[C; 5]) -> (Vec<f64>, Vec<f64>) {
    let cells = 4;
    let nodal = method == CommonSpatialPolicy::Q1;
    let count = cells + usize::from(nodal);
    let length = 3.;
    let h = length / cells as f64;
    let c = point[0].conj();
    let dc = direction[0].conj();
    let mut values = vec![];
    let mut tangents = vec![];
    for entity in 0..count {
        let x = (entity as f64 + if nodal { 0. } else { 0.5 }) * h;
        let f = x * (length - x) / 2. + if nodal { 0. } else { h * h / 8. };
        for channel in 0..2 {
            let p = point[1 + channel];
            let q = point[3 + channel];
            let dp = direction[1 + channel];
            let dq = direction[3 + channel];
            let u = p * (1. + x / length) + q / c * f;
            let du = dp * (1. + x / length) + (dq / c - q * dc / (c * c)) * f;
            values.extend([u.re, u.im]);
            tangents.extend([du.re, du.im]);
        }
    }
    (values, tangents)
}

#[track_caller]
fn near(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-10, "{a} != {b}");
    }
}

#[test]
fn complex_spatial_application_preserves_conjugation_lifts_and_paired_total_derivatives() {
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let (document, plan) = fixture(method);
        let inputs = ["a", "p", "q"].map(|name| document.parameter_ref(name).unwrap());
        let output = document.field_ref("definition.u").unwrap().erase();
        let output_id = output.id();
        let program = DifferentiableProgram::compile(
            plan.clone(),
            &inputs,
            &[output],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let direction = [
            C::new(0.5, -0.25),
            C::new(2., -3.),
            C::new(-4., 1.),
            C::new(5., 6.),
            C::new(-2., 7.),
        ];
        for a in [C::new(2., 3.), C::new(3., -2.)] {
            let point = [
                a,
                C::new(5., 7.),
                C::new(-1., 4.),
                C::new(11., -2.),
                C::new(3., 8.),
            ];
            let parts = point.map(|v| (v.re, v.im));
            let input = DifferentiableValues::from_components(
                program.identity().inputs(),
                &[&parts[..1], &parts[1..3], &parts[3..]],
            )
            .unwrap();
            let evaluation = program.evaluate(&input).unwrap();
            let primal = evaluation.primal();
            assert_eq!(
                primal.evidence().implementation(),
                DerivativeImplementation::AnalyticAssembled
            );
            assert!(primal.evidence().primal_solve().is_some());
            let directions = direction.map(|v| (v.re, v.im));
            let tangent = DifferentiableTangent::from_components(
                program.identity().inputs(),
                &[&directions[..1], &directions[1..3], &directions[3..]],
            )
            .unwrap();
            let jvp = evaluation.jvp(&tangent).unwrap();
            let (values, expected) = oracle(method, &point, &direction);
            near(&primal.output().fixture_coordinates(), &values);
            near(&jvp.derivative().fixture_coordinates(), &expected);
            let weights = (0..values.len() / 2)
                .map(|i| ((i + 1) as f64, -0.5 * (i + 2) as f64))
                .collect::<Vec<_>>();
            let cotangent =
                DifferentiableCotangent::from_components(program.identity().outputs(), &[&weights])
                    .unwrap();
            let actual = evaluation
                .vjp(&cotangent)
                .unwrap()
                .derivative()
                .fixture_coordinates();
            let flat_weights = weights
                .iter()
                .flat_map(|&(a, b)| [a, b])
                .collect::<Vec<_>>();
            let expected_gradient = (0..10)
                .map(|coordinate| {
                    let mut seed = [C::new(0., 0.); 5];
                    seed[coordinate / 2] = if coordinate % 2 == 0 {
                        C::new(1., 0.)
                    } else {
                        C::new(0., 1.)
                    };
                    oracle(method, &point, &seed)
                        .1
                        .iter()
                        .zip(&flat_weights)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            near(&actual, &expected_gradient);
            let record = evaluation.record().unwrap();
            let reopened = RecordedDifferentiableEvaluation::from_bytes(
                &record.to_bytes().unwrap(),
                &plan,
                None,
            )
            .unwrap();
            assert_eq!(reopened, record);
            let encoded: serde_json::Value =
                serde_json::from_slice(&record.to_bytes().unwrap()).unwrap();
            let mut candidate = encoded["state"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            let selected = inputs.each_ref().map(|p| p.id());
            assert!(
                plan.as_spatial()
                    .unwrap()
                    .validate_differentiated_values(
                        &selected,
                        Some(&input.fixture_coordinates()),
                        &[output_id],
                        &candidate[..candidate.len() - 1],
                    )
                    .is_err(),
                "incomplete complex coordinate pairs must reject"
            );
            candidate[0] += 1.;
            assert!(
                plan.as_spatial()
                    .unwrap()
                    .validate_differentiated_values(
                        &selected,
                        Some(&input.fixture_coordinates()),
                        &[output_id],
                        &candidate
                    )
                    .is_err()
            );
        }
    }
}
