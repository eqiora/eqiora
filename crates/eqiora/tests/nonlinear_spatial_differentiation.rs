//! Nonlinear material products through the ordinary Program and immutable records.
mod support;
#[path = "nonlinear_spatial_differentiation/tables.rs"]
mod tables;
use eqiora::api::{
    DifferentiableProgram, EvaluationCollection, EvaluationMapExecutionPolicy, EvaluationMapPlan,
    EvaluationMapRetention, ModelDocument, RecordedDifferentiableEvaluation,
};
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::geometry::GeometryGraph;
use eqiora::realization::{NonlinearSolvePlan, Target};
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonInitialField, CommonInitialState, CommonInitialValues,
    CommonLinearRequest, CommonSolvePolicy, CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};
use support::derivative_products::{FixtureCoordinates, cotangent, tangent};

fn fixture() -> (
    ModelDocument,
    ResolvedCommonPlan,
    CommonInitialState<'static>,
) {
    let document = ModelDocument::compile(
        "nonlinear-spatial.eqi",
        "model Material() {
        domain body=box(0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        parameter p:1=1; parameter r:1=2;
        variable u:1 on body in smooth;
        relation balance on body {-div((1+p*u)*grad(u))=0;}
        relation lower on left {trace(u)=1;}
        relation upper on right {trace(u)=r;}
    }",
    )
    .unwrap();
    fixture_from_document(document, 2.)
}

fn fixture_from_document(
    document: ModelDocument,
    right: f64,
) -> (
    ModelDocument,
    ResolvedCommonPlan,
    CommonInitialState<'static>,
) {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 1.]).unwrap();
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
    let cells = CartesianMeshCellsV2::new(vec![2]).unwrap();
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
    let resources =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let plan = ResolvedCommonPlan::resolve(
        &model,
        resources,
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Newton {
            linear: CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
            nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                .unwrap(),
        },
        None,
        None,
        &FaerLinearSolver,
        None,
        None,
    )
    .unwrap();
    let seed = plan
        .as_spatial()
        .unwrap()
        .initial_state(
            None,
            vec![
                CommonInitialField::new(
                    document.artifact_reference().unwrap().artifact().clone(),
                    plan.as_spatial().unwrap().fields().next().unwrap().0,
                    Some(
                        CommonInitialValues::new(
                            eqiora::ValueShape::scalar(),
                            vec![1., (1. + right) / 2., right],
                        )
                        .unwrap(),
                    ),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
    (document, plan, seed.into())
}

fn program(
    document: &ModelDocument,
    plan: &ResolvedCommonPlan,
    seed: &CommonInitialState<'static>,
) -> DifferentiableProgram {
    DifferentiableProgram::compile(
        plan.clone(),
        &[
            document.parameter_ref("p").unwrap(),
            document.parameter_ref("r").unwrap(),
        ],
        &[document
            .field_ref(
                &plan
                    .as_spatial()
                    .unwrap()
                    .fields()
                    .next()
                    .unwrap()
                    .0
                    .ulid()
                    .to_string(),
            )
            .unwrap()
            .erase()],
        Default::default(),
        Some(seed.clone()),
        &FaerLinearSolver,
    )
    .unwrap()
}

fn expected(p: f64, r: f64) -> (f64, f64, f64) {
    // q=u+p*u²/2 is linear in x. Exact P1 cell fluxes therefore give
    // q(mid)=(q(1)+q(r))/2, R_u=4(1+p*u), R_p=2u²-1-r², R_r=-2(1+p*r).
    let q = (1. + r) / 2. + p * (1. + r * r) / 4.;
    let u = if p == 0. {
        q
    } else {
        ((1. + 2. * p * q).sqrt() - 1.) / p
    };
    let jacobian = 4. * (1. + p * u);
    (
        u,
        -(2. * u * u - 1. - r * r) / jacobian,
        2. * (1. + p * r) / jacobian,
    )
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-10, "{a:e} != {b:e}");
    }
}

#[test]
fn nonlinear_spatial_program_pairs_total_products_and_reauthenticates_records() {
    let (document, plan, seed) = fixture();
    let bytes = seed.to_bytes().unwrap();
    let reopened_seed = CommonInitialState::from_bytes(&bytes, &plan).unwrap();
    assert_eq!(seed, reopened_seed);
    let program = program(&document, &plan, &seed);
    for (p, r) in [(0., 2.), (1., 2.), (1., 3.)] {
        let (u, dp, dr) = expected(p, r);
        let evaluation = program
            .evaluate(&support::real_point(&program, &[p, r]).unwrap())
            .unwrap();
        close(
            &evaluation.primal().output().fixture_coordinates(),
            &[1., u, r],
        );
        for (direction, expected) in [([1., 0.], [0., dp, 0.]), ([0., 1.], [0., dr, 1.])] {
            let result = evaluation
                .jvp(&tangent(program.identity(), &direction).unwrap())
                .unwrap();
            close(&result.derivative().fixture_coordinates(), &expected);
        }
        let result = evaluation
            .vjp(&cotangent(program.identity(), &[2., 3., 5.]).unwrap())
            .unwrap();
        close(
            &result.derivative().fixture_coordinates(),
            &[3. * dp, 3. * dr + 5.],
        );
        let record = evaluation.record().unwrap();
        assert_eq!(
            record.result().evidence().steps()[0]
                .nonlinear_iterations(0)
                .unwrap()
                == 0,
            p == 0.
        );
        let encoded = record.to_bytes().unwrap();
        assert_eq!(
            RecordedDifferentiableEvaluation::from_bytes(&encoded, &plan, Some(&reopened_seed))
                .unwrap(),
            record
        );
        assert!(RecordedDifferentiableEvaluation::from_bytes(&encoded, &plan, None).is_err());
        for key in ["unknown", "output", "point"] {
            let mut wire: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
            let coordinate = if key == "unknown" {
                &mut wire["steps"][0][key][0]
            } else {
                &mut wire[key][0]
            };
            *coordinate = serde_json::json!(coordinate.as_f64().unwrap() + 0.125);
            assert!(
                RecordedDifferentiableEvaluation::from_bytes(
                    &serde_json::to_vec(&wire).unwrap(),
                    &plan,
                    Some(&seed)
                )
                .is_err()
            );
        }
        for iterations in [if p == 0. { 1 } else { 0 }, 17] {
            let mut wire: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
            wire["steps"][0]["primal"]["record"]["iterations"] = serde_json::json!(iterations);
            assert!(
                RecordedDifferentiableEvaluation::from_bytes(
                    &serde_json::to_vec(&wire).unwrap(),
                    &plan,
                    Some(&seed)
                )
                .is_err()
            );
        }
    }
    assert_eq!(seed.to_bytes().unwrap(), bytes);
    assert!(
        DifferentiableProgram::compile(
            plan,
            &[document.parameter_ref("p").unwrap()],
            &[document.field_ref("u").unwrap().erase()],
            Default::default(),
            None,
            &FaerLinearSolver
        )
        .is_err()
    );
}

#[derive(Debug)]
struct NoSolve;
impl LinearSolverBackend for NoSolve {
    fn provider(&self) -> eqiora::solver::SolverProvider {
        FaerLinearSolver.provider()
    }
    fn capabilities(&self) -> eqiora::solver::SolverCapabilities {
        FaerLinearSolver.capabilities()
    }
    fn solve_with_execution(
        &self,
        _: &eqiora::solver::LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn eqiora::solver::ReplicatedLinearExecution,
    ) -> Result<eqiora::solver::LinearSolution, eqiora::Diagnostic> {
        panic!("record reopening must not execute a solve")
    }
}

#[test]
fn spatial_seed_survives_retained_and_recomputed_collection_requests() {
    let (document, plan, seed) = fixture();
    let program = Arc::new(program(&document, &plan, &seed));
    let points = support::real_points(&program, &[&[0., 2.], &[1., 3.], &[1., 2.]]);
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let policy = EvaluationMapExecutionPolicy::new(
            2,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
            retention,
            1 << 28,
        )
        .unwrap();
        let map = EvaluationMapPlan::new(program.clone(), &points, &[], policy).unwrap();
        let collection = map.execute_recorded(|| false).unwrap();
        assert!(collection.groups()[0].is_complete());
        let encoded = collection.to_bytes().unwrap();
        let reopened = EvaluationCollection::from_bytes(
            &encoded,
            &NoSolve,
            &[eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES],
        )
        .unwrap();
        assert_eq!(reopened, collection);
        assert_eq!(reopened.to_bytes().unwrap(), encoded);
        let too_small = EvaluationMapExecutionPolicy::new(
            2,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
            retention,
            map.estimated_storage_bytes() - 1,
        )
        .unwrap();
        assert!(EvaluationMapPlan::new(program.clone(), &points, &[], too_small).is_err());
    }
}
