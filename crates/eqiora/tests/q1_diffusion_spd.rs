//! Positive material energies and anchored quotient components admit the ordinary CG path.
mod support;
use eqiora::api::{DifferentiableProgram, ModelDocument, RecordedDifferentiableEvaluation};
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::geometry::GeometryGraph;
use eqiora::solver::{
    LinearOperatorOrientation, LinearOperatorProperties, LinearProblem, LinearSolution,
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReplicatedLinearExecution, SolverCapabilities, SolverPlan, SolverProvider,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use support::derivative_products::{FixtureCoordinates, cotangent, tangent};

const SOURCE: &str = r#"
public connector ScalarBoundary {
 trace value:1; flux outward_flux:1/m; shape []; frame invariant;
 pairing euclidean_boundary_duality; orientation parent_outward;
}
public component Interface(support body:volume(ambient_dimension=1), support face:boundary(parent=body), variable value:1 on body in smooth, parameter conductivity:1, port edge:ScalarBoundary over face) {
 relation carrier on face {trace(value)-edge.value=0;normal(conductivity*grad(value))-edge.outward_flux=0;}
}
model Materials() {
 parameter kL:1=1; parameter kR:1=2;
 domain left=box(0,0.5); domain right=box(0.5,1);
 domain outerL=boundary(left,axis=0,side=lower); domain contactL=boundary(left,axis=0,side=upper);
 domain contactR=boundary(right,axis=0,side=lower); domain outerR=boundary(right,axis=0,side=upper);
 variable u:1 on left in smooth; variable v:1 on right in smooth;
 relation first on left {-div(kL*grad(u))=0;}
 relation second on right {-div(kR*grad(v))=0;}
 relation fixedL on outerL {trace(u)=0;}
 relation fixedR on outerR {trace(v)=1;}
 instance l:Interface(body=left,face=contactL,value=u,conductivity=kL);
 instance r:Interface(body=right,face=contactR,value=v,conductivity=kR);
 connect l.edge,r.edge;
 observable midpoint:1=integral(trace(u),measure(contactL));
 observable flux:1/m=integral(normal(-kL*grad(u)),measure(contactL));
}
"#;

fn mesh(cells: usize) -> AuthenticatedCommonMesh {
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
    let cells = CartesianMeshCellsV2::new(vec![cells]).unwrap();
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
    AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
        .unwrap()
}
fn solver(algorithm: LinearSolver, preconditioner: PreconditionerPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(100).unwrap())
        .unwrap()
        .with_preconditioner(preconditioner)
}
fn resolve(
    document: &ModelDocument,
    cells: usize,
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
) -> Result<ResolvedCommonPlan, eqiora::Diagnostic> {
    resolve_solver(
        document,
        cells,
        solver(algorithm, preconditioner),
        &REFERENCE_LINEAR_SOLVER,
    )
}
fn resolve_solver(
    document: &ModelDocument,
    cells: usize,
    solver: SolverPlan,
    backend: &dyn LinearSolverBackend,
) -> Result<ResolvedCommonPlan, eqiora::Diagnostic> {
    ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.program()).unwrap(),
        mesh(cells),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(CommonLinearRequest::exact(solver, backend.provider()).unwrap()),
        None,
        None,
        backend,
        None,
        None,
    )
}
fn rules(
    plan: &ResolvedCommonPlan,
    document: &ModelDocument,
    names: &[&str],
    points: usize,
) -> std::collections::HashMap<
    eqiora::Id<eqiora::entity::kinds::Domain>,
    eqiora::meshing::QuadratureRule,
> {
    plan.observation_quadratures(
        &names
            .iter()
            .map(|name| document.observable_ref(name).unwrap().id())
            .collect::<Vec<_>>(),
        points,
    )
    .unwrap()
}
fn near(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-10, "{actual:?} != {expected:?}");
    }
}

#[test]
fn material_q1_cg_identity_and_jacobi_preserve_run_fields_observables_and_products() {
    let document = ModelDocument::compile("material-diffusion.eqi", SOURCE).unwrap();
    for cells in [2, 4] {
        for (algorithm, preconditioner) in [
            (
                LinearSolver::BiConjugateGradientStabilized,
                PreconditionerPolicy::Identity,
            ),
            (
                LinearSolver::ConjugateGradient,
                PreconditionerPolicy::Identity,
            ),
            (
                LinearSolver::ConjugateGradient,
                PreconditionerPolicy::Jacobi,
            ),
        ] {
            let plan = resolve(&document, cells, algorithm, preconditioner).unwrap();
            assert_eq!(
                plan.operator_properties(),
                Some(LinearOperatorProperties::SymmetricPositiveDefinite)
            );
            let plan = ResolvedCommonPlan::from_bytes(
                &plan.to_bytes().unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                &[],
            )
            .unwrap();
            let result = plan
                .as_spatial()
                .unwrap()
                .run_result(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            assert_eq!(
                CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
            for (name, expected) in [("midpoint", 2. / 3.), ("flux", -4. / 3.)] {
                let value = result
                    .observe(
                        plan.model_artifact(),
                        document.observable_ref(name).unwrap().id(),
                        &rules(&plan, &document, &[name], 1),
                        None,
                    )
                    .unwrap();
                assert!(
                    (value.value().real_scalar_value().unwrap().value() - expected).abs() < 2e-10
                );
            }
            let inputs = ["kL", "kR"].map(|name| document.parameter_ref(name).unwrap());
            let outputs = [
                document.field_ref("u").unwrap().erase(),
                document.observable_ref("midpoint").unwrap().erase(),
            ];
            let program = DifferentiableProgram::compile(
                plan.clone(),
                &inputs,
                &outputs,
                rules(&plan, &document, &["midpoint"], 1),
                None,
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
            let point = program.evaluate(program.default_point()).unwrap();
            let expected = if cells == 2 {
                vec![0., 2. / 3., 2. / 3.]
            } else {
                vec![0., 1. / 3., 2. / 3., 2. / 3.]
            };
            near(&point.primal().output().fixture_coordinates(), &expected);
            let left = point
                .jvp(&tangent(point.identity(), &[1., 0.]).unwrap())
                .unwrap();
            let right = point
                .jvp(&tangent(point.identity(), &[0., 1.]).unwrap())
                .unwrap();
            near(
                &left.derivative().fixture_coordinates(),
                &expected.iter().map(|v| -v / 3.).collect::<Vec<_>>(),
            );
            near(
                &right.derivative().fixture_coordinates(),
                &expected.iter().map(|v| v / 6.).collect::<Vec<_>>(),
            );
            let mut weights = vec![0.; expected.len()];
            *weights.last_mut().unwrap() = 3.;
            let reverse = point
                .vjp(&cotangent(point.identity(), &weights).unwrap())
                .unwrap();
            near(
                &reverse.derivative().fixture_coordinates(),
                &[-2. / 3., 1. / 3.],
            );
            for (evidence, orientation) in [
                point.primal().evidence().steps()[0]
                    .primal_solve(0)
                    .unwrap(),
                left.evidence().steps()[0].derivative_solve().unwrap(),
                reverse.evidence().steps()[0].derivative_solve().unwrap(),
            ]
            .into_iter()
            .zip([
                LinearOperatorOrientation::Normal,
                LinearOperatorOrientation::Normal,
                LinearOperatorOrientation::ConjugateTransposed,
            ]) {
                assert_eq!(evidence.orientation(), orientation);
                assert_eq!(evidence.algorithm(), algorithm);
                assert_eq!(evidence.preconditioner(), preconditioner);
                assert!(evidence.true_residual_norm() <= evidence.residual_target());
            }
            let record = point.record().unwrap();
            assert_eq!(
                RecordedDifferentiableEvaluation::from_bytes(
                    &record.to_bytes().unwrap(),
                    &plan,
                    None
                )
                .unwrap(),
                record
            );
        }
    }
}

#[derive(Debug)]
struct CountedReference(AtomicUsize);
static COUNTED: CountedReference = CountedReference(AtomicUsize::new(0));
static PIECEWISE_COUNTED: CountedReference = CountedReference(AtomicUsize::new(0));
impl LinearSolverBackend for CountedReference {
    fn provider(&self) -> SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, eqiora::Diagnostic> {
        self.0.fetch_add(1, Ordering::SeqCst);
        REFERENCE_LINEAR_SOLVER.solve_with_execution(problem, plan, execution)
    }
}

#[test]
fn q1_energy_admission_preserves_the_selected_provider_and_work_budget() {
    use eqiora_backend_faer::FaerLinearSolver;
    let document = ModelDocument::compile("q1-work-controls.eqi", SOURCE).unwrap();
    let short = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::MIN,
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    let short = resolve_solver(&document, 4, short, &REFERENCE_LINEAR_SOLVER).unwrap();
    let error = short
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert!(error.message().contains("iterations"), "{error}");
    let plan = resolve(
        &document,
        4,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    )
    .unwrap();
    let error = plan
        .as_spatial()
        .unwrap()
        .run_result(None, &FaerLinearSolver)
        .unwrap_err();
    assert!(error.message().contains("provider"), "{error}");
}

#[test]
fn changed_material_coefficients_reject_before_provider_work_or_record_reacceptance() {
    let document = ModelDocument::compile("invalid-material.eqi", SOURCE).unwrap();
    let plan = resolve(
        &document,
        4,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    )
    .unwrap();
    let inputs = ["kL", "kR"].map(|name| document.parameter_ref(name).unwrap());
    let output = document.observable_ref("midpoint").unwrap().erase();
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &inputs,
        &[output],
        rules(&plan, &document, &["midpoint"], 1),
        None,
        &COUNTED,
    )
    .unwrap();
    let input_ids = inputs.each_ref().map(|input| input.id());
    let observable = document.observable_ref("midpoint").unwrap().id();
    let outputs = [observable.erase()];
    let quadratures = plan.observation_quadratures(&[observable], 1).unwrap();
    let accepted = program.evaluate(program.default_point()).unwrap();
    let unknowns = accepted.accepted_unknowns().to_vec();
    plan.as_spatial()
        .unwrap()
        .validate_differentiated_values(
            None,
            &input_ids,
            Some(&[1., 2.]),
            &outputs,
            &quadratures,
            &unknowns,
        )
        .unwrap();
    for coefficient in [0., -1., f64::NAN, f64::INFINITY] {
        let before = COUNTED.0.load(Ordering::SeqCst);
        let bound = support::real_point(&program, &[coefficient, 2.]);
        if let Ok(bound) = bound {
            assert!(program.evaluate(&bound).is_err());
        }
        assert_eq!(COUNTED.0.load(Ordering::SeqCst), before);
        let error = plan
            .as_spatial()
            .unwrap()
            .validate_differentiated_values(
                None,
                &input_ids,
                Some(&[coefficient, 2.]),
                &outputs,
                &quadratures,
                &unknowns,
            )
            .unwrap_err();
        if coefficient.is_finite() {
            assert!(error.message().contains("positive"), "{error}");
        }
    }
}

#[test]
fn an_unanchored_component_and_other_operators_do_not_acquire_spd() {
    let unanchored = SOURCE
        .replace(
            "relation fixedL on outerL {trace(u)=0;}",
            "relation freeL on outerL {normal(kL*grad(u))=0;}",
        )
        .replace(
            "relation fixedR on outerR {trace(v)=1;}",
            "relation freeR on outerR {normal(kR*grad(v))=0;}",
        );
    for source in [
        unanchored,
        SOURCE.replace("-div(kL*grad(u))=0;", "-div(kL*grad(u))+u/(1[m]*1[m])=0;"),
    ] {
        let document = ModelDocument::compile("unproved-energy.eqi", &source).unwrap();
        let general = resolve(
            &document,
            4,
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
        )
        .unwrap();
        assert_eq!(
            general.operator_properties(),
            Some(LinearOperatorProperties::General)
        );
        assert!(
            resolve(
                &document,
                4,
                LinearSolver::ConjugateGradient,
                PreconditionerPolicy::Identity
            )
            .is_err()
        );
    }
}

const PIECEWISE: &str = r#"
model Piecewise() {
 parameter log_k:array<1,2>=[0,0.6931471805599453];
 domain body=box(0,1);
 domain left=boundary(body,axis=0,side=lower); domain right=boundary(body,axis=0,side=upper);
 coordinate x:m on body from body[0];
 let conductivity:W/m/K=(if x<0.5[m] then math.exp(log_k[0]) else math.exp(log_k[1]))*1[W/m/K];
 variable temperature:K on body in h1;
 relation balance on body {-div(conductivity*grad(temperature))=0[W/m^3];}
 relation fixedL on left {trace(temperature)=300[K];}
 relation fixedR on right {trace(temperature)=301[K];}
 observable left_mean:K=integral((if x<0.5[m] then 2[1/m] else 0[1/m])*temperature,measure(body));
 observable right_mean:K=integral((if x<0.5[m] then 0[1/m] else 2[1/m])*temperature,measure(body));
 observable mean_flux:W/m^2=integral(-conductivity*partial(temperature,wrt=x)/1[m],measure(body));
 observable profile:K on body=temperature;
}
"#;

#[test]
fn piecewise_log_conductivity_uses_the_actual_quadrature_energy_and_products() {
    let document = ModelDocument::compile("piecewise-conductivity.eqi", PIECEWISE).unwrap();
    for cells in [3, 4] {
        for algorithm in [
            LinearSolver::BiConjugateGradientStabilized,
            LinearSolver::ConjugateGradient,
        ] {
            let plan = resolve(&document, cells, algorithm, PreconditionerPolicy::Jacobi).unwrap();
            assert_eq!(
                plan.operator_properties(),
                Some(LinearOperatorProperties::SymmetricPositiveDefinite)
            );
            let result = plan
                .as_spatial()
                .unwrap()
                .run_result(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            // With three cells the jump crosses the middle element. Its two
            // actual Gauss samples give kbar=3/2, so R=1/3+2/9+1/6=13/18.
            // Four aligned cells reproduce continuum resistance R=1/2+1/4.
            let expected = if cells == 3 { -18. / 13. } else { -4. / 3. };
            let observed = result
                .observe(
                    plan.model_artifact(),
                    document.observable_ref("mean_flux").unwrap().id(),
                    &rules(&plan, &document, &["mean_flux"], 2),
                    None,
                )
                .unwrap();
            assert!(
                (observed.value().real_scalar_value().unwrap().value() - expected).abs() < 2e-10
            );
            assert_eq!(
                CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
        }
    }
    let plan = resolve(
        &document,
        4,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    )
    .unwrap();
    let plan =
        ResolvedCommonPlan::from_bytes(&plan.to_bytes().unwrap(), &REFERENCE_LINEAR_SOLVER, &[])
            .unwrap();
    let inputs = [document.parameter_ref("log_k").unwrap()];
    let outputs = ["left_mean", "right_mean", "mean_flux"]
        .map(|name| document.observable_ref(name).unwrap().erase());
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &inputs,
        &outputs,
        rules(
            &plan,
            &document,
            &["left_mean", "right_mean", "mean_flux"],
            2,
        ),
        None,
        &PIECEWISE_COUNTED,
    )
    .unwrap();
    let point = program.evaluate(program.default_point()).unwrap();
    near(
        &point.primal().output().fixture_coordinates(),
        &[300. + 1. / 3., 300. + 5. / 6., -4. / 3.],
    );
    for (direction, expected) in [
        ([1., 0.], [-1. / 9., -1. / 9., -8. / 9.]),
        ([0., 1.], [1. / 9., 1. / 9., -4. / 9.]),
    ] {
        let product = point
            .jvp(&tangent(point.identity(), &direction).unwrap())
            .unwrap();
        near(&product.derivative().fixture_coordinates(), &expected);
        assert_eq!(
            product.evidence().steps()[0]
                .derivative_solve()
                .unwrap()
                .algorithm(),
            LinearSolver::ConjugateGradient
        );
    }
    let reverse = point
        .vjp(&cotangent(point.identity(), &[3., 0., 0.]).unwrap())
        .unwrap();
    near(
        &reverse.derivative().fixture_coordinates(),
        &[-1. / 3., 1. / 3.],
    );
    let record = point.record().unwrap();
    assert_eq!(
        RecordedDifferentiableEvaluation::from_bytes(&record.to_bytes().unwrap(), &plan, None)
            .unwrap(),
        record
    );
    for bad in [-1000., -744., 1000.] {
        let input = eqiora::api::DifferentiableValues::from_components(
            program.identity().inputs(),
            &[&[(bad, 0.), (2_f64.ln(), 0.)]],
        )
        .unwrap();
        let before = PIECEWISE_COUNTED.0.load(Ordering::SeqCst);
        assert!(program.evaluate(&input).is_err());
        assert_eq!(PIECEWISE_COUNTED.0.load(Ordering::SeqCst), before);
    }
}

#[test]
fn coefficient_validation_checks_every_integration_sample_not_cell_centers_or_endpoints() {
    // Two cells have centers .25/.75 and vertices 0/.5/1, all with k=+1.
    // The first actual two-point Gauss abscissa is about .10566, with k=-1.
    let source = PIECEWISE.replace(
        "if x<0.5[m] then math.exp(log_k[0]) else math.exp(log_k[1])",
        "if x<0.09[m] then 1 else if x<0.12[m] then -1 else 1",
    );
    let document = ModelDocument::compile("bad-quadrature-sample.eqi", &source).unwrap();
    let error = resolve(
        &document,
        2,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Identity,
    )
    .unwrap_err();
    assert!(error.message().contains("positive"), "{error}");
}

#[test]
fn positive_primal_diffusion_allows_a_negative_coefficient_parameter_derivative() {
    let source = SOURCE.replace("parameter kL:1=1;", "parameter p:1=0.5; let kL:1=2-p;");
    let document = ModelDocument::compile("negative-coefficient-partial.eqi", &source).unwrap();
    let plan = resolve(
        &document,
        4,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    )
    .unwrap();
    let inputs = [document.parameter_ref("p").unwrap()];
    let outputs = [document.observable_ref("midpoint").unwrap().erase()];
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &inputs,
        &outputs,
        rules(&plan, &document, &["midpoint"], 1),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let point = program.evaluate(program.default_point()).unwrap();
    // kL=3/2, kR=2 gives Tmid=4/7. dTmid/dp=8/49 because dkL/dp=-1.
    near(&point.primal().output().fixture_coordinates(), &[4. / 7.]);
    near(
        &point
            .jvp(&tangent(point.identity(), &[1.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates(),
        &[8. / 49.],
    );
    near(
        &point
            .vjp(&cotangent(point.identity(), &[3.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates(),
        &[24. / 49.],
    );
    let record = point.record().unwrap();
    assert_eq!(
        RecordedDifferentiableEvaluation::from_bytes(&record.to_bytes().unwrap(), &plan, None)
            .unwrap(),
        record
    );
}

#[test]
fn positive_subnormal_diffusion_does_not_prove_a_zero_prepared_energy() {
    assert!((-744_f64).exp() > 0.);
    let source = PIECEWISE.replace("[0,0.6931471805599453]", "[-744,-744]");
    let document = ModelDocument::compile("underflowed-energy.eqi", &source).unwrap();
    let error = resolve(
        &document,
        4,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Identity,
    )
    .unwrap_err();
    assert!(
        error.message().contains("positive prepared cell energy"),
        "{error}"
    );
}

#[test]
fn stronger_spd_admission_preserves_existing_faer_bicgstab_products_and_replay() {
    use eqiora::solver::ReductionPolicy;
    use eqiora_backend_faer::FaerLinearSolver;
    let document = ModelDocument::compile("faer-materials.eqi", SOURCE).unwrap();
    for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
        let solver = solver(LinearSolver::BiConjugateGradientStabilized, preconditioner)
            .with_reduction(ReductionPolicy::Fast);
        let plan = resolve_solver(&document, 4, solver, &FaerLinearSolver).unwrap();
        assert_eq!(
            plan.operator_properties(),
            Some(LinearOperatorProperties::SymmetricPositiveDefinite)
        );
        let plan =
            ResolvedCommonPlan::from_bytes(&plan.to_bytes().unwrap(), &FaerLinearSolver, &[])
                .unwrap();
        let result = plan
            .as_spatial()
            .unwrap()
            .run_result(None, &FaerLinearSolver)
            .unwrap();
        assert_eq!(
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
        let inputs = ["kL", "kR"].map(|name| document.parameter_ref(name).unwrap());
        let outputs = [
            document.field_ref("u").unwrap().erase(),
            document.observable_ref("midpoint").unwrap().erase(),
        ];
        let program = DifferentiableProgram::compile(
            plan.clone(),
            &inputs,
            &outputs,
            rules(&plan, &document, &["midpoint"], 1),
            None,
            &FaerLinearSolver,
        )
        .unwrap();
        let point = program.evaluate(program.default_point()).unwrap();
        near(
            &point.primal().output().fixture_coordinates(),
            &[0., 1. / 3., 2. / 3., 2. / 3.],
        );
        let forward = point
            .jvp(&tangent(point.identity(), &[1., 0.]).unwrap())
            .unwrap();
        near(
            &forward.derivative().fixture_coordinates(),
            &[0., -1. / 9., -2. / 9., -2. / 9.],
        );
        let reverse = point
            .vjp(&cotangent(point.identity(), &[0., 0., 0., 3.]).unwrap())
            .unwrap();
        near(
            &reverse.derivative().fixture_coordinates(),
            &[-2. / 3., 1. / 3.],
        );
        for (evidence, orientation) in [
            point.primal().evidence().steps()[0]
                .primal_solve(0)
                .unwrap(),
            forward.evidence().steps()[0].derivative_solve().unwrap(),
            reverse.evidence().steps()[0].derivative_solve().unwrap(),
        ]
        .into_iter()
        .zip([
            LinearOperatorOrientation::Normal,
            LinearOperatorOrientation::Normal,
            LinearOperatorOrientation::ConjugateTransposed,
        ]) {
            assert_eq!(evidence.orientation(), orientation);
            assert_eq!(evidence.solver().id(), "eqiora.faer");
            assert_eq!(
                evidence.algorithm(),
                LinearSolver::BiConjugateGradientStabilized
            );
            assert_eq!(evidence.preconditioner(), preconditioner);
            assert_eq!(evidence.reduction(), ReductionPolicy::Fast);
            assert!(evidence.true_residual_norm() <= evidence.residual_target());
        }
        let record = point.record().unwrap();
        assert_eq!(
            RecordedDifferentiableEvaluation::from_bytes(&record.to_bytes().unwrap(), &plan, None)
                .unwrap(),
            record
        );
    }
}

fn one_anchor_source() -> String {
    SOURCE
        .replace("parameter kR:1=2;", "parameter kR:1=1;")
        .replace(
            "relation fixedR on outerR {trace(v)=1;}",
            "relation freeR on outerR {normal(kR*grad(v))=0;}",
        )
}

#[test]
fn insufficient_assembly_margin_stays_unproved_without_excluding_positive_diffusion() {
    for (exponent, proved) in [(-46, true), (-47, false), (-60, false)] {
        let coefficient = 2_f64.powi(exponent);
        let source = one_anchor_source().replace(
            "parameter kL:1=1;",
            &format!("parameter kL:1={coefficient:.17e};"),
        );
        let document = ModelDocument::compile("q1-representation-margin.eqi", &source).unwrap();
        let plan = resolve(
            &document,
            2,
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
        )
        .unwrap();
        assert_eq!(
            plan.operator_properties(),
            Some(if proved {
                LinearOperatorProperties::SymmetricPositiveDefinite
            } else {
                LinearOperatorProperties::General
            })
        );
        let cg = resolve(
            &document,
            2,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Identity,
        );
        if proved {
            let cg = cg.unwrap();
            assert_eq!(
                ResolvedCommonPlan::from_bytes(
                    &cg.to_bytes().unwrap(),
                    &REFERENCE_LINEAR_SOLVER,
                    &[]
                )
                .unwrap(),
                cg
            );
        } else {
            let error = cg.unwrap_err();
            assert!(
                error.message().contains("unproved")
                    && error.message().contains("representability"),
                "{error}"
            );
        }
        // The 2^-47 matrix remains SPD; the sufficient proof simply declines.
        // With 2^-60 the anchor is actually rounded away, yielding a null mode.
        if exponent == -60 {
            assert_eq!(1. + coefficient, 1.);
        }
        if exponent == -47 {
            assert!(1. + coefficient > 1.);
        }
    }
}

static CONTRAST_COUNTED: CountedReference = CountedReference(AtomicUsize::new(0));

#[test]
fn an_spd_plan_rechecks_global_energy_margin_before_provider_or_reacceptance() {
    let document = ModelDocument::compile("q1-rebound-margin.eqi", &one_anchor_source()).unwrap();
    let plan = resolve(
        &document,
        2,
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
    )
    .unwrap();
    let inputs = ["kL", "kR"].map(|name| document.parameter_ref(name).unwrap());
    let observable = document.observable_ref("midpoint").unwrap();
    let quadratures = rules(&plan, &document, &["midpoint"], 1);
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &inputs,
        &[observable.erase()],
        quadratures.clone(),
        None,
        &CONTRAST_COUNTED,
    )
    .unwrap();
    let accepted = program.evaluate(program.default_point()).unwrap();
    let input_ids = inputs.each_ref().map(|input| input.id());
    plan.as_spatial()
        .unwrap()
        .validate_differentiated_values(
            None,
            &input_ids,
            Some(&[1., 1.]),
            &[observable.id().erase()],
            &quadratures,
            accepted.accepted_unknowns(),
        )
        .unwrap();
    let candidate = support::real_point(&program, &[2_f64.powi(-60), 1.]).unwrap();
    let before = CONTRAST_COUNTED.0.load(Ordering::SeqCst);
    let error = program.evaluate(&candidate).unwrap_err();
    assert!(
        error
            .iter()
            .any(|error| error.message().contains("representability")),
        "{error:?}"
    );
    assert_eq!(CONTRAST_COUNTED.0.load(Ordering::SeqCst), before);
    let error = plan
        .as_spatial()
        .unwrap()
        .validate_differentiated_values(
            None,
            &input_ids,
            Some(&[2_f64.powi(-60), 1.]),
            &[observable.id().erase()],
            &quadratures,
            accepted.accepted_unknowns(),
        )
        .unwrap_err();
    assert!(error.message().contains("representability"), "{error}");
}
