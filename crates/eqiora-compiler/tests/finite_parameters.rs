use eqiora_compiler::CompiledModel;
use eqiora_graph::Op;
use eqiora_schema::kernel::KernelNode;

const SOURCE: &str = r#"
space A=orthonormal(a,b); space B=orthonormal(x,y,z);
space AB=product(A,B); space BA=product(B,A);
model M(){
 parameter squared:map<complex<1>,A,A>=compose(h,h);
 parameter h:map<complex<1>,A,A>=linear_map(A,A,[[0,math.complex(0,-1)],[math.complex(0,1),0]]);
 parameter conjugate:map<complex<1>,A,A>=adjoint(h);
 parameter real:map<1,A,A>=linear_map(A,A,[[2,1],[0,4]]);
 parameter inverted:map<1,A,A>=inverse(real);
 parameter singular:map<1,A,A>=linear_map(A,A,[[0,0],[0,0]]);
 parameter inactive:map<1,A,A>=if false then inverse(singular) else real;
 parameter a:coordinates<1,A>=coordinates(A,[2,3]);
 parameter b:coordinates<1,B>=coordinates(B,[5,7,11]);
 parameter product:coordinates<1,AB>=tensor_product(a,b);
 parameter permuted:coordinates<1,BA>=permute_factors(product,[1,0]);
 variable retained:coordinates<1,BA>; relation r{retained=permuted;}
}"#;

fn compile(source: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    CompiledModel::compile_selected("finite-parameters.eqi", source, "M", &[])
}

#[test]
fn static_finite_algebra_preserves_values_dependencies_and_factor_order() {
    let model = compile(SOURCE).unwrap_or_else(|errors| panic!("{errors:?}"));
    let components = |name: &str, count: usize| {
        let id = model.symbols().get(name).unwrap();
        let value = model
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Parameter(p),
                } if p.id().erase() == id => Some(p.value()),
                _ => None,
            })
            .unwrap();
        (0..count)
            .map(|i| value.component(i).unwrap())
            .collect::<Vec<_>>()
    };
    // Pauli Y is Hermitian and squares to identity.
    assert_eq!(
        components("squared", 4),
        [(1., 0.), (0., 0.), (0., 0.), (1., 0.)]
    );
    for name in ["h", "conjugate"] {
        assert_eq!(
            components(name, 4),
            [(0., 0.), (0., -1.), (0., 1.), (0., 0.)]
        );
    }
    assert_eq!(
        components("inverted", 4),
        [(0.5, 0.), (-0.125, 0.), (0., 0.), (0.25, 0.)]
    );
    assert_eq!(
        components("inactive", 4),
        [(2., 0.), (1., 0.), (0., 0.), (4., 0.)]
    );
    assert_eq!(
        components("product", 6),
        [
            (10., 0.),
            (14., 0.),
            (22., 0.),
            (15., 0.),
            (21., 0.),
            (33., 0.)
        ]
    );
    assert_eq!(
        components("permuted", 6),
        [
            (10., 0.),
            (15., 0.),
            (14., 0.),
            (21., 0.),
            (22., 0.),
            (33., 0.)
        ]
    );
}

#[test]
fn inactive_finite_branches_skip_execution_but_preserve_type_and_name_checks() {
    compile(SOURCE).unwrap();
    for invalid in [
        SOURCE.replace(
            "if false then inverse(singular)",
            "if true then inverse(singular)",
        ),
        SOURCE.replace("inverse(singular) else real", "inverse(missing) else real"),
        SOURCE.replace("inverse(singular) else real", "adjoint(a) else real"),
        SOURCE.replace("[1,0]", "[0,0]"),
        SOURCE.replace("compose(h,h)", "compose(squared,h)"),
    ] {
        assert!(compile(&invalid).is_err(), "{invalid}");
    }
}
