use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ExprNode, KernelNode};

const LAW: &str =
    "operator conductivity(input x:K,input k0:W/m/K,input a:1/K):W/m/K=k0*(1+a*x+a^2*x^2);";

#[test]
fn typed_polynomial_calls_preserve_named_order_and_live_operands() {
    let source = format!(
        "{LAW} model M(){{parameter temperature:K=20;parameter base:W/m/K=10;parameter slope:1/K=0.01;relation r{{conductivity(a=slope,x=temperature,k0=base)=12.4[W/m/K];}}}}"
    );
    let models = compile("conductivity.eqi", &source).unwrap_or_else(|errors| panic!("{errors:?}"));
    let relation = models[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Relation(value),
            } => Some(value),
            _ => None,
        })
        .unwrap();
    assert!(
        relation
            .expression()
            .definitions()
            .values()
            .all(|definition| definition
                .formals()
                .iter()
                .all(|formal| formal.scalar_domain() == Some(eqiora_core::ScalarDomain::Real)))
    );
    assert!(
        relation
            .expression()
            .nodes()
            .iter()
            .any(|node| matches!(node, ExprNode::PureOperatorApplication(_)))
    );
}

#[test]
fn scalar_composition_is_lexical_acyclic_and_dimension_checked() {
    let source = "operator squared(input x:scalar):scalar=x^2;operator law(input y:scalar):scalar=squared(x=y)+squared(x=y);model M(){relation r{law(y=3)=18;}}";
    compile("composition.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}"));
    for declaration in [
        "operator law(input x:scalar):scalar=law(x=x);",
        "operator law(input x:scalar):scalar=other(x=x);operator other(input x:scalar):scalar=law(x=x);",
        "operator law(input x:scalar):scalar=x+outside;",
        "operator law(input x:scalar):scalar=math.sin(x);",
        "operator law(input x:scalar):scalar=x^-1;",
        "operator law(input x:scalar):scalar=x^x;",
        "operator law(input x:scalar):scalar=x^1000000000;",
        "operator law(input x:scalar):scalar=rational(1,0);",
        "operator law(input x:integer):integer=x;",
        "operator law(input x:complex<1>):complex<1>=x;",
        "operator law(input x:K):m=x;",
    ] {
        assert!(
            compile(
                "invalid-operator.eqi",
                &format!("{declaration} model M(){{relation r{{1=1;}}}}")
            )
            .is_err(),
            "{declaration}"
        );
    }
}

#[test]
fn closed_quotients_admit_rational_values_and_preserve_units() {
    for body in ["x/2", "x/y", "1/(1+x*x)"] {
        let source = format!(
            "operator law(input x:1,input y:1):1={body};model M(){{parameter x:1=2;parameter y:1=3;variable z:1;relation r{{z=law(y=y,x=x);}}}}"
        );
        compile("quotient.eqi", &source).unwrap_or_else(|errors| panic!("{body}: {errors:?}"));
    }
    for (result, valid) in [("m/s", true), ("m", false)] {
        let source = format!(
            "operator rate(input distance:m,input duration:s):{result}=distance/duration;model M(){{relation r{{rate(distance=6[m],duration=2[s])=3[{result}];}}}}"
        );
        assert_eq!(compile("units.eqi", &source).is_ok(), valid);
    }
}

#[test]
fn analytic_rational_property_exposes_its_declared_first_partials() {
    let source = r#"
property contract Response(input value:1):1 {derivatives first_partials;}
property release Material:Response {
    analytic {value=1/(1+value^2);source_unit:1=1;}
    validity value>=-1 and value<=3;outside reject;branch single;
    citation org.example.manufactured;license spdx.CC0_1_0;
}
component Equation(property response:Response) {
    variable u:1;
    relation balance {response(value=u)=0.2;}
}
model Main(){instance physics:Equation(response=Material);}
"#;
    compile("rational-property.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}"));
}

#[test]
fn named_call_failures_do_not_coerce_or_capture() {
    for call in [
        "conductivity(20[K],10[W/m/K],0.01[1/K])",
        "conductivity(x=20[K],k0=10[W/m/K])",
        "conductivity(x=20[K],k0=10[W/m/K],a=0.01[1/K],extra=1)",
        "conductivity(x=20[K],x=20[K],k0=10[W/m/K],a=0.01[1/K])",
        "conductivity(x=20[m],k0=10[W/m/K],a=0.01[1/K])",
        "conductivity(x=to_integer(20),k0=10[W/m/K],a=0.01[1/K])",
        "conductivity(x=math.complex(20[K],0[K]),k0=10[W/m/K],a=0.01[1/K])",
    ] {
        assert!(
            compile(
                "invalid-call.eqi",
                &format!("{LAW} model M(){{relation r{{{call}=0[W/m/K];}}}}")
            )
            .is_err(),
            "{call}"
        );
    }
}

#[test]
fn aliases_and_named_permutations_share_the_existing_dimension_owner() {
    let source = "dimension Temperature=K;operator identity(input x:Temperature):Temperature=x;model M(){relation r{identity(x=20[K])=20[K];}}";
    compile("dimension-alias.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}"));
    let first = eqiora_lang::parse("first.eqi", &format!("{LAW} model M(){{relation r{{conductivity(x=20[K],k0=10[W/m/K],a=0.01[1/K])=12.4[W/m/K];}}}}")).into_document().unwrap();
    let second = eqiora_lang::parse("second.eqi", &format!("{LAW} model M(){{relation r{{conductivity(a=0.01[1/K],k0=10[W/m/K],x=20[K])=12.4[W/m/K];}}}}")).into_document().unwrap();
    assert_eq!(
        eqiora_compiler::source_identity::LocalSourceIdentity::from_document(&first).unwrap(),
        eqiora_compiler::source_identity::LocalSourceIdentity::from_document(&second).unwrap()
    );
}

#[test]
fn composition_work_is_bounded_before_exponential_expansion() {
    let mut source = String::from("operator level0(input x:scalar):scalar=x*x;");
    for level in 1..16 {
        source.push_str(&format!(
            "operator level{level}(input x:scalar):scalar=level{}(x=x)+level{}(x=x);",
            level - 1,
            level - 1
        ));
    }
    source.push_str("model M(){relation r{1=1;}}");
    let errors = compile("bounded-composition.eqi", &source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("node") || error.message().contains("limit")),
        "{errors:?}"
    );
    let capture = "operator f(input x:scalar):scalar=x;operator g(input f:scalar):scalar=f(x=f);";
    let errors = compile(
        "capture.eqi",
        &format!("{capture} model M(){{relation r{{1=1;}}}}"),
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("lexical formal")),
        "{errors:?}"
    );
}

#[test]
fn constant_and_dimensioned_cubic_use_exact_polynomial_multiplication() {
    for source in [
        "operator constant(input x:K):1=1;model M(){relation r{constant(x=2[K])=1;}}",
        "operator force(input x:m,input stiffness:N/m,input cubic:N/m^3):N=stiffness*x+cubic*x^3;model M(){relation r{force(x=2[m],stiffness=3[N/m],cubic=4[N/m^3])=38[N];}}",
    ] {
        compile("powers.eqi", source).unwrap_or_else(|errors| panic!("{errors:?}"));
    }
}

#[test]
fn formal_alpha_renaming_preserves_named_application_identity() {
    use eqiora_compiler::source_identity::LocalSourceIdentity;
    let source = "operator square(input x:scalar):scalar=x*x;model M(){relation r{square(x=3)=9;}}";
    let renamed = source
        .replace("input x:", "input value:")
        .replace("=x*x;", "=value*value;")
        .replace("square(x=", "square(value=");
    let identity = |source: &str| {
        LocalSourceIdentity::from_document(
            &eqiora_lang::parse("alpha.eqi", source)
                .into_document()
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(identity(source), identity(&renamed));
}

#[test]
fn zero_powers_do_not_erase_the_zero_to_zero_domain_error() {
    for body in ["x^0", "0^0"] {
        let source = format!("operator f(input x:1):1={body};model M(){{relation r{{1=1;}}}}");
        let errors = compile("zero-power.eqi", &source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("positive bounded calculus range")),
            "{errors:?}"
        );
    }
}
