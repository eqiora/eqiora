//! A decoded form must use current Field hypotheses, even when IDs are retained.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{KernelNode, SpatialRegularity};
use eqiora_sem::KernelProgram;

#[test]
fn decoded_boundary_forms_cannot_reuse_a_stronger_field_hypothesis() {
    for (assertion, operand, accepted, rejected) in [
        (
            "h1",
            "trace(u,on=face)",
            SpatialRegularity::H1,
            SpatialRegularity::L2,
        ),
        (
            "h1",
            "trace(u,on=face)",
            SpatialRegularity::H1,
            SpatialRegularity::Unspecified,
        ),
        (
            "smooth",
            "normal(grad(u),on=face)",
            SpatialRegularity::Smooth,
            SpatialRegularity::H1,
        ),
    ] {
        let source = format!(
            r#"model M() {{
            domain body=box(0,1,0,1);
            domain face=boundary(body,axis=0,side=lower);
            variable u:1 on body in {assertion};
            relation law on body {{ u=u; }}
            form weak for law {{ test eta:1 for u in h1;
                integrate(face,trace(eta,on=face)*{operand})=0;
            }}
        }}"#
        );
        let compiled = CompiledModel::compile_selected("replay.eqi", &source, "M", &[]).unwrap();
        let original = compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection();
        let decoded = AuthoredFormulationProjection::decode(original.canonical_bytes()).unwrap();
        assert_eq!(&decoded, original);
        let (transaction, model, _) = compiled.into_parts();
        let program = |regularity| {
            let mut revised =
                Transaction::new("retain all identities and change only Field regularity");
            for op in transaction.ops() {
                revised.push(match op {
                    Op::DefineKernelNode {
                        node: KernelNode::Field(field),
                    } => Op::DefineKernelNode {
                        node: KernelNode::Field(field.clone().with_spatial_regularity(regularity)),
                    },
                    op => op.clone(),
                });
            }
            let mut store = InMemoryGraphStore::new();
            store.commit(revised).unwrap();
            KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
        };
        super::check_authored_dependence(&decoded, &program(accepted)).unwrap();
        let weaker = program(rejected);
        let error = super::check_authored_dependence(&decoded, &weaker).unwrap_err();
        assert!(
            error.message().contains("Field trace regularity"),
            "{error:?}"
        );
        // The separate spatial inspection entry must reach the same live guard
        // before dispatching its narrower vector-curl/Stokes correspondence checks.
        let error = super::check_authored_spatial_formulation(&weaker, &decoded).unwrap_err();
        assert!(
            error.message().contains("Field trace regularity"),
            "{error:?}"
        );
    }
}
