use super::*;
use eqiora::{
    api::ModelDocument,
    artifact::ModelEnvelope,
    solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan},
};
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use std::num::NonZeroUsize;

#[test]
fn result_inspection_transports_exact_observations_and_rejects_stale_or_unbound_artifacts() {
    let source = "model Response(){variable z:complex<V>;relation fixed{z=math.complex(3[V],4[V]);}observable response:complex<V>=z;}";
    use eqiora::compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
    // Match the loose workspace's declared compilation namespace and logical module.
    let owner = CompilationNamespaceId::new(["editor.workspace"]).unwrap();
    let unit = ResolvedSourceUnit::new(owner.clone(), "src/response.eqi", source).unwrap();
    let input =
        ResolvedHierarchyInput::with_root_module(owner, ["response"], vec![unit], vec![]).unwrap();
    let document = ModelDocument::compile_modules(input, "Response", &[]).unwrap();
    let envelope = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
    );
    let plan = CommonAlgebraicPlan::resolve(
        &envelope,
        request,
        None,
        &[],
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let plan_bytes = String::from_utf8(result.plan().to_bytes().unwrap()).unwrap();
    let result_bytes = String::from_utf8(result.to_bytes().unwrap()).unwrap();
    let uri = "file:///workspace/response.eqi";
    let mut child = Command::new(SERVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"workspaceFolders":[{"uri":"file:///workspace","name":"workspace"}]}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"languageId":"eqiora","version":1,"text":source}}}),
        json!({"jsonrpc":"2.0","id":2,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"plan":plan_bytes,"result":result_bytes}}),
        json!({"jsonrpc":"2.0","id":3,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"result":result_bytes}}),
        json!({"jsonrpc":"2.0","id":4,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"plan":plan_bytes,"result":" ".repeat(2*1024*1024+1)}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":uri,"version":2},"contentChanges":[{"text":source.replace("3[V]","5[V]")}]}}),
        json!({"jsonrpc":"2.0","id":5,"method":"eqiora/inspect","params":{"textDocument":{"uri":uri},"plan":plan_bytes,"result":result_bytes}}),
        json!({"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}),
        json!({"jsonrpc":"2.0","method":"exit","params":null}),
    ];
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        for message in messages {
            write_packet(&mut stdin, &message);
        }
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let messages = parse_packets(&output.stdout);
    assert_eq!(
        response(&messages, 1)["result"]["capabilities"]["experimental"]["eqioraResultInspection"],
        1
    );
    assert!(
        response(&messages, 2)["error"].is_null(),
        "{}",
        response(&messages, 2)
    );
    let accepted = &response(&messages, 2)["result"]["result"];
    assert_eq!(accepted["identity"], result.identity());
    assert_eq!(accepted["modelDigest"], document.digest().unwrap());
    assert_eq!(
        accepted["observations"][0]["components"][0]["real"]["value"],
        3.0
    );
    assert_eq!(
        accepted["observations"][0]["components"][0]["imaginary"]["value"],
        4.0
    );
    for (id, gate) in [
        (3, "requires its exact numerical Plan"),
        (4, "2 MiB"),
        (5, "different exact Model"),
    ] {
        assert!(
            response(&messages, id)["error"]["message"]
                .as_str()
                .unwrap()
                .contains(gate)
        );
    }
}
