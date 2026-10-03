//! Source-to-Model execution of bounded full-coordinate constitutive maps.
use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::kernel::KernelNode;
use eqiora::sem::KernelProgram;

fn check_relations(source: &str, tolerance: f64) {
    let compiled = compile("tensor.eqi", source).unwrap().pop().unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let mut checked = 0;
    for node in program.nodes() {
        if let KernelNode::Relation(relation) = node {
            let values = program
                .evaluate_relation_operands(relation.id(), &[], &[])
                .unwrap();
            assert_eq!(values.len() % 2, 0);
            for pair in values.chunks_exact(2) {
                assert_eq!(pair[0].value_type(), pair[1].value_type());
                for (actual, expected) in pair[0]
                    .components()
                    .unwrap()
                    .zip(pair[1].components().unwrap())
                {
                    assert!(
                        (actual.0 - expected.0).abs() <= tolerance,
                        "{actual:?} != {expected:?}"
                    );
                    assert!(
                        (actual.1 - expected.1).abs() <= tolerance,
                        "{actual:?} != {expected:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn nonsymmetric_conductivity_and_complex_bilinear_contraction_execute_from_source() {
    check_relations(
        include_str!(
            "../../../verify/language/tensor-contractions/models/conductivity-complex.eqi"
        ),
        0.0,
    );
}

#[test]
fn anisotropic_rank_four_map_preserves_full_shear_coordinates_and_energy() {
    // Normal stiffnesses 10/20, coupling 3, and four shear slots 4 Pa.
    // Full tensor shear (0.01,0.01) gives both shear stresses 0.08 Pa:
    // 1/2*(0.01*0.08+0.01*0.08)=0.0008 Pa. Both slots are required.
    check_relations(
        include_str!(
            "../../../verify/language/tensor-contractions/models/anisotropic-elasticity.eqi"
        ),
        1e-15,
    );
}

#[test]
fn symmetry_deviator_and_rank_four_identity_maps_are_ordinary_compositions() {
    check_relations(
        include_str!(
            "../../../verify/language/tensor-contractions/models/projection-compositions.eqi"
        ),
        0.0,
    );
}

#[test]
fn equal_extents_do_not_identify_foreign_supports_or_nominal_spaces() {
    let source = r#"model M() {
        domain left=box(0,1,0,1);
        domain right=box(0,1,0,1);
        variable a:tensor<1,2,2> on left;
        variable b:vector<1,2> on left;
        relation r on left { component(contract(a,b,axes=((1,0),)),indices=(0,))=0; }
    }"#;
    compile("same-support.eqi", source).unwrap();
    let foreign = source.replace("b:vector<1,2> on left", "b:vector<1,2> on right");
    let errors = compile("foreign-support.eqi", &foreign).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("exact volume")),
        "{errors:?}"
    );
    let source = r#"space A=orthonormal(a,b); space B=orthonormal(c,d);
    model M() {
        variable x:coordinates<1,A>;
        variable y:coordinates<1,B>;
        relation r { contract(x,y,axes=((0,0),))=0; }
    }"#;
    let errors = compile("foreign-space.eqi", source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("exact type rule")),
        "{errors:?}"
    );
    for expression in [
        "contract([1,2],[3,4],axes=((0,0),))",
        "contract(x,y,axes=((0,0),(0,0)))",
        "component(x,indices=(2,))",
    ] {
        let source = format!(
            r#"model M() {{
            domain body=box(0,1,0,1);
            parameter x:vector<1,2>=tensor_value(frame=body,components=[1,2]);
            parameter y:vector<1,2>=tensor_value(frame=body,components=[3,4]);
            relation r {{ {expression}=0; }}
        }}"#
        );
        assert!(compile("bad-tensor.eqi", &source).is_err(), "{expression}");
    }
}
