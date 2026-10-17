use super::*;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use eqiora_solver::{
    LinearProblem, LinearSolution, LinearSolver, ReductionPolicy, ReplicatedLinearExecution,
    SolverCapabilities, SolverProvider,
};
use std::num::NonZeroUsize;

#[derive(Debug)]
struct NoSolve;

impl LinearSolverBackend for NoSolve {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }

    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("artifact reopening must not execute a solve")
    }
}

fn fixture(name: &str, tolerance: f64) -> (ModelDocument, ResolvedCommonPlan) {
    let document = ModelDocument::compile(
        "identity.eqi",
        &format!(
            "model {name}() {{
            parameter p:m=2[m]; parameter q:s=3[s];
            variable x:m; variable y:s;
            relation system {{ x=p; y=q; }}
        }}"
        ),
    )
    .unwrap();
    let model = eqiora_artifact::ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        tolerance,
        1e-14,
        NonZeroUsize::new(16).unwrap(),
    )
    .unwrap();
    let solver = solver.with_reduction(ReductionPolicy::Fast);
    let request =
        CommonSolvePolicy::Linear(CommonLinearRequest::exact(solver, NoSolve.provider()).unwrap());
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, &[], None, &NoSolve).unwrap();
    (document, ResolvedCommonPlan::Algebraic(Box::new(plan)))
}

fn identity(document: &ModelDocument, plan: &ResolvedCommonPlan) -> DifferentiableProgramIdentity {
    DifferentiableProgramIdentity::from_plan(
        plan,
        &[
            document.parameter_ref("q").unwrap(),
            document.parameter_ref("p").unwrap(),
        ],
        &[
            document.field_ref("y").unwrap().erase(),
            document.field_ref("x").unwrap().erase(),
        ],
        None,
    )
    .unwrap()
}

#[test]
fn exact_plan_reopening_restores_ordered_typed_identity_without_solving() {
    let (document, plan) = fixture("Typed", 1e-12);
    let identity = identity(&document, &plan);
    let bytes = identity.to_bytes().unwrap();
    let plan = ResolvedCommonPlan::from_bytes(
        &plan.to_bytes().unwrap(),
        &NoSolve,
        eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES,
    )
    .unwrap();
    let reopened = DifferentiableProgramIdentity::from_bytes(&bytes, &plan, None).unwrap();
    assert_eq!(identity, reopened);
    assert_eq!(
        reopened.inputs()[0].entity(),
        document.parameter_ref("q").unwrap().id().erase()
    );
    assert_eq!(
        reopened.outputs()[0].entity(),
        document.field_ref("y").unwrap().id().erase()
    );
    assert_ne!(
        reopened.inputs()[0].mathematical_type(),
        reopened.inputs()[1].mathematical_type()
    );
    assert_eq!(bytes, reopened.to_bytes().unwrap());
    // The ordinary live owner derives precisely the same identity, but accepts
    // its default primal separately through a working numerical provider.
    let live = DifferentiableProgram::compile(
        plan,
        &[
            document.parameter_ref("q").unwrap(),
            document.parameter_ref("p").unwrap(),
        ],
        &[
            document.field_ref("y").unwrap().erase(),
            document.field_ref("x").unwrap().erase(),
        ],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    assert_eq!(live.identity(), &reopened);
}

#[test]
fn seeded_identity_reopening_requires_the_exact_initial_state() {
    let (document, plan) = fixture("Seeded", 1e-12);
    let seed = plan.as_algebraic().unwrap().initial_state(&[]).unwrap();
    let identity = DifferentiableProgramIdentity::from_plan(
        &plan,
        &[document.parameter_ref("p").unwrap()],
        &[document.field_ref("x").unwrap().erase()],
        Some(&seed),
    )
    .unwrap();
    let seed =
        CommonAlgebraicState::from_bytes(&seed.to_bytes().unwrap(), plan.as_algebraic().unwrap())
            .unwrap();
    let bytes = identity.to_bytes().unwrap();
    assert_eq!(
        DifferentiableProgramIdentity::from_bytes(&bytes, &plan, Some(&seed)).unwrap(),
        identity,
    );
    assert!(DifferentiableProgramIdentity::from_bytes(&bytes, &plan, None).is_err());
}

#[test]
fn identity_reopening_rejects_foreign_artifacts_and_invalid_selections() {
    let (document, plan) = fixture("Typed", 1e-12);
    let identity = identity(&document, &plan);
    let bytes = identity.to_bytes().unwrap();
    let (_, foreign) = fixture("Foreign", 1e-12);
    assert!(DifferentiableProgramIdentity::from_bytes(&bytes, &foreign, None).is_err());
    let (_, changed_policy) = fixture("Typed", 1e-8);
    assert!(DifferentiableProgramIdentity::from_bytes(&bytes, &changed_policy, None).is_err());
    for mutate in [
        ("inputs", vec![identity.inputs()[0].entity().to_string(); 2]),
        (
            "outputs",
            vec![identity.outputs()[0].entity().to_string(); 2],
        ),
        ("inputs", vec![identity.outputs()[0].entity().to_string()]),
        ("outputs", vec![identity.inputs()[0].entity().to_string()]),
        ("inputs", vec!["q".to_owned()]),
        ("outputs", vec![]),
    ] {
        let mut wire: WireIdentity = serde_json::from_slice(&bytes).unwrap();
        match mutate.0 {
            "inputs" => wire.inputs = mutate.1,
            _ => wire.outputs = mutate.1,
        }
        let bad = serde_json::to_vec(&wire).unwrap();
        assert!(DifferentiableProgramIdentity::from_bytes(&bad, &plan, None).is_err());
    }
    let seed = foreign.as_algebraic().unwrap().initial_state(&[]).unwrap();
    let mut wire: WireIdentity = serde_json::from_slice(&bytes).unwrap();
    wire.initial_state = Some(seed.identity().to_owned());
    assert!(
        DifferentiableProgramIdentity::from_bytes(
            &serde_json::to_vec(&wire).unwrap(),
            &plan,
            Some(&seed),
        )
        .unwrap_err()[0]
            .message()
            .contains("foreign Plan")
    );
}

#[test]
fn identity_reopening_rejects_noncanonical_truncated_and_oversized_content() {
    let (document, plan) = fixture("Typed", 1e-12);
    let bytes = identity(&document, &plan).to_bytes().unwrap();
    let mut whitespace = bytes.clone();
    whitespace.push(b' ');
    let mut unknown = bytes.clone();
    unknown.pop();
    unknown.extend_from_slice(b",\"other\":0}");
    for bad in [
        whitespace,
        unknown,
        bytes[..bytes.len() - 1].to_vec(),
        vec![b' '; MAX_BYTES + 1],
    ] {
        assert!(DifferentiableProgramIdentity::from_bytes(&bad, &plan, None).is_err());
    }
}
