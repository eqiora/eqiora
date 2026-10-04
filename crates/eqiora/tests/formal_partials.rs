//! Authored formal partials use the common typed scalar evaluator.
mod support;
use eqiora::compiler::compile;
use eqiora::graph::Op;
use eqiora::ir::ScalarOperatorIr;
use eqiora::kernel::{KernelNode, SymbolRef};

fn residuals(source: &str) -> Vec<f64> {
    let models = compile("partial.eqi", source).unwrap();
    let operations = models[0].transaction().ops();
    let parameters = operations
        .iter()
        .filter_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Parameter(parameter),
            } => Some((parameter.id(), parameter.value().clone())),
            _ => None,
        })
        .collect::<std::collections::HashMap<_, _>>();
    operations
        .iter()
        .filter_map(|operation| match operation {
            Op::DefineKernelNode {
                node: KernelNode::Relation(relation),
            } => Some(relation.expression()),
            _ => None,
        })
        .flat_map(|dag| {
            ScalarOperatorIr::lower(dag)
                .unwrap()
                .evaluate_typed(dag.roots(), &mut |symbol| match symbol {
                    SymbolRef::Parameter(id) => parameters.get(&id).cloned(),
                    _ => None,
                })
                .unwrap()
                .into_iter()
                .map(|value| value.real_scalar_value().unwrap().value())
        })
        .collect::<Vec<_>>()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| pair[0] - pair[1])
        .collect()
}

#[test]
fn authored_two_input_partials_and_composition_have_independent_analytic_values() {
    let source = include_str!("../../../verify/language/formal-partials/models/composition.eqi");
    assert_eq!(residuals(source), vec![0.; 3]);
}

#[test]
fn parameter_alias_does_not_cut_dependency_or_merge_equal_independent_inputs() {
    let source = include_str!("../../../verify/language/formal-partials/models/aliases.eqi");
    assert_eq!(residuals(source), vec![0.; 3]);
}

#[test]
fn conductivity_partial_has_conductivity_per_temperature_dimension() {
    let source = include_str!("../../../verify/language/formal-partials/models/conductivity.eqi");
    for value in residuals(source) {
        assert!(value.abs() < 1e-15);
    }
}

#[test]
fn aliases_foreign_bindings_conflicting_holding_and_implicit_solution_partials_fail() {
    for expression in [
        "partial(x*x,wrt=z)",
        "partial(x,wrt=foreign)",
        "partial(x,wrt=x,holding=(x))",
        "partial(x,wrt=x,holding=(y,y))",
        "partial(x,wrt=x,holding=(z))",
        "partial(x,wrt=v)",
        "partial(derivative(v),wrt=x)",
        "partial(math.sin(x),wrt=x)",
    ] {
        let source = format!(
            "model M() {{ parameter x:1=3; parameter y:1=3; let z=x; variable v:1; relation r {{ {expression}=0; }} }}"
        );
        assert!(compile("invalid.eqi", &source).is_err(), "{expression}");
    }
}

#[test]
fn sequential_partial_selectors_retain_distinct_live_values() {
    let source = r#"model M() {
 parameter x:1=3; parameter y:1=5;
 relation r {
   partial(x*x,wrt=x)=6;
   partial(x*y*y,wrt=y)=30;
   partial(x*x*y,wrt=x)=30;
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 3]);
}

#[test]
fn component_binding_preserves_independent_parent_parameter_directions() {
    let source = r#"
component C(parameter x:1, parameter y:1) {
 relation r {
   partial(x*x*y,wrt=x,holding=(y))=2*x*y;
   partial(x*x*y,wrt=y,holding=(x))=x*x;
 }
}
model M() {
 parameter p:1=3; parameter q:1=3;
 instance forward:C(x=p,y=q);
 instance reverse:C(x=q,y=p);
}
"#;
    assert_eq!(residuals(source), vec![0.; 4]);
    // Unequal values also expose accidental reuse across reversed bindings.
    assert_eq!(residuals(&source.replace("q:1=3", "q:1=5")), vec![0.; 4]);
}

#[test]
fn component_bindings_cannot_manufacture_independence_or_conflicting_holding() {
    for bindings in ["x=p,y=p", "x=3,y=q", "x=p+q,y=q"] {
        let source = format!(
            "component C(parameter x:1, parameter y:1) {{ relation r {{ partial(x*x*y,wrt=x,holding=(y))=0; }} }} model M() {{ parameter p:1=3; parameter q:1=3; instance c:C({bindings}); }}"
        );
        assert!(compile("dependent.eqi", &source).is_err(), "{bindings}");
    }
}

#[test]
fn shared_alias_partials_preserve_the_bounded_source_dag() {
    let source = |squarings: usize, expected: u32| {
        let mut source = String::from("model M() { parameter p:1=1; let a0=p;");
        for index in 1..=squarings {
            source.push_str(&format!("let a{index}=a{}*a{};", index - 1, index - 1));
        }
        source.push_str(&format!(
            "relation r {{ partial(a{squarings},wrt=p)={expected}; }} }}"
        ));
        source
    };
    // a7=p^128, hence a7_p=128 at p=1. Every operation is exact here.
    assert_eq!(residuals(&source(7, 128)), vec![0.0]);
    // a24=p^(2^24) exceeds the existing formal-exponent bound of 255.
    // Shared aliases must reach that declared gate without expanding a tree.
    let failure = compile("partial-bound.eqi", &source(24, 16777216)).unwrap_err();
    assert!(
        failure.iter().any(|diagnostic| diagnostic
            .message()
            .contains("formal exponent exceeds its limit")),
        "{failure:?}"
    );
}

#[test]
fn nested_model_partials_evaluate_each_hessian_entry_and_independent_actions() {
    // f=x²y+y³ gives H=[[2y,2x],[2x,6y]]. At (3,5), its
    // columns are (10,6) and (6,30), and H(2,-1)=(14,-18).
    let source = r#"model M() {
 parameter x:1=3; parameter y:1=5;
 let f=x*x*y+y*y*y;
 let xx=partial(partial(f,wrt=x),wrt=x);
 let xy=partial(partial(f,wrt=x),wrt=y);
 let yx=partial(partial(f,wrt=y),wrt=x);
 let yy=partial(partial(f,wrt=y),wrt=y);
 relation r { xx=10; xy=6; yx=6; yy=30; 2*xx-xy=14; 2*yx-yy=-18; }
}"#;
    assert_eq!(residuals(source), vec![0.; 6]);
}

#[test]
fn dimensioned_mixed_partials_and_absent_inner_selector_keep_typed_zero() {
    let source = r#"model M() {
 parameter position:m=3 [m]; parameter velocity:m/s=5 [m/s];
 let f=position*position*velocity;
 relation r {
   partial(partial(f,wrt=position),wrt=velocity)=6 [m];
   partial(partial(f,wrt=velocity),wrt=position)=6 [m];
   partial(partial(position*position,wrt=velocity),wrt=position)=0 [s];
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 3]);
}

#[test]
fn third_partials_reject_direct_alias_and_operator_composition_paths() {
    let cases = [
        "model M() { parameter x:1=3; relation r { partial(partial(partial(x*x*x,wrt=x),wrt=x),wrt=x)=6; } }",
        "model M() { parameter x:1=3; let dx=partial(x*x*x,wrt=x); let ddx=partial(dx,wrt=x); relation r { partial(ddx,wrt=x)=6; } }",
        "operator third(input x:1):1=partial(partial(partial(x*x*x,wrt=x),wrt=x),wrt=x); model M() { relation r { third(x=3)=6; } }",
        "operator first(input x:1):1=partial(x*x*x,wrt=x); operator second(input x:1):1=partial(first(x=x),wrt=x); operator third(input x:1):1=partial(second(x=x),wrt=x); model M() { relation r { third(x=3)=6; } }",
    ];
    for source in cases {
        let errors = compile("third.eqi", source).expect_err("third partial must be rejected");
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("derivative order")),
            "{errors:?}"
        );
    }
}

#[test]
fn nonsmooth_second_partials_do_not_infer_smoothness_from_the_evaluation_point() {
    // x*abs(x) is not C2 at zero. A current positive x does not turn its
    // symbolic branch into a globally admitted second derivative.
    for value in ["x*math.abs(x)", "math.max(x*x,0)"] {
        let source = format!(
            "model M() {{ parameter x:1=3; relation r {{ partial(partial({value},wrt=x),wrt=x)=2; }} }}"
        );
        assert!(compile("nonsmooth.eqi", &source).is_err(), "{value}");
    }
}

#[test]
fn differentiated_operator_formals_remain_independent_when_call_arguments_alias() {
    // g(x,y)=x*y²: g_x=y², g_y=2xy. Binding both arguments to p
    // happens after the local partial; their outer p derivatives are 2p and 4p.
    let source = r#"
operator dx(input x:1,input y:1):1=partial(x*y*y,wrt=x);
operator dy(input x:1,input y:1):1=partial(x*y*y,wrt=y);
model M() { parameter p:1=3;
 relation r {
   dx(x=p,y=p)=9;
   dy(x=p,y=p)=18;
   partial(dx(x=p,y=p),wrt=p)=6;
   partial(dy(x=p,y=p),wrt=p)=12;
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 4]);
}

#[test]
fn authored_jvp_and_hessian_actions_use_typed_ordered_blocks() {
    let source = r#"
operator forward(input x:1,input y:1,input dx:1,input dy:1):1 = jvp(x*x*y+y*y*y,[x,y],[dx,dy]);
operator hx(input x:1,input y:1,input dx:1,input dy:1):1 = jvp(partial(x*x*y+y*y*y,wrt=x),[x,y],[dx,dy]);
operator hy(input x:1,input y:1,input dx:1,input dy:1):1 = jvp(partial(x*x*y+y*y*y,wrt=y),[x,y],[dx,dy]);
operator mixed(input x:m,input v:m/s,input dx:m,input dv:m/s):m*m/s = jvp(x*v,[x,v],[dx,dv]);
model M() { relation r {
 forward(x=3,y=5,dx=2,dy=-1)=-24;
 hx(x=3,y=5,dx=2,dy=-1)=14;
 hy(x=3,y=5,dx=2,dy=-1)=-18;
 mixed(x=3 [m],v=5 [m/s],dx=2 [m],dv=-1 [m/s])=7 [m*m/s];
} }
"#;
    assert_eq!(residuals(source), vec![0.; 4]);
    for body in [
        "jvp(x*v,[x,v],[dv,dx])",
        "jvp(x*v,[x,x],[dx,dx])",
        "jvp(x*v,[x*v],[dx])",
        "jvp(x*v,[],[])",
    ] {
        let source = format!(
            "operator invalid(input x:m,input v:m/s,input dx:m,input dv:m/s):m*m/s={body}; model M() {{ relation r {{ invalid(x=3 [m],v=5 [m/s],dx=2 [m],dv=-1 [m/s])=7 [m*m/s]; }} }}"
        );
        assert!(compile("invalid-action.eqi", &source).is_err(), "{body}");
    }
}

#[test]
fn model_jvp_retains_parameter_aliases_and_heterogeneous_direction_units() {
    let source = r#"model M() {
 parameter x:m=3 [m]; parameter v:m/s=5 [m/s];
 parameter dx:m=2 [m]; parameter dv:m/s=-1 [m/s];
 let f=x*x*v;
 relation r {
   jvp(f,[x,v],[dx,dv])=51 [m*m*m/s];
   jvp(partial(f,wrt=x),[x,v],[dx,dv])=14 [m*m/s];
   jvp(partial(f,wrt=v),[x,v],[dx,dv])=12 [m*m];
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 3]);
    for replacement in [
        "jvp(f,[x,v],[dv,dx])",
        "jvp(f,[x,x],[dx,dx])",
        "jvp(f,[f],[dx])",
    ] {
        assert!(
            compile(
                "invalid-jvp.eqi",
                &source.replace("jvp(f,[x,v],[dx,dv])", replacement)
            )
            .is_err()
        );
    }
}

#[test]
fn authored_vjp_blocks_use_the_declared_real_dual_pairing() {
    let source = r#"
operator back_x(input x:m,input v:m/s,input seed:s/m^2):1/m=vjp(x*v,x,seed);
model M() {
 parameter x:m=3 [m]; parameter v:m/s=5 [m/s]; parameter seed:s/m^2=7 [s/m^2];
 let f=x*v;
 relation r {
   back_x(x=x,v=v,seed=seed)=35 [1/m];
   vjp(f,x,seed)=35 [1/m];
   vjp(f,v,seed)=21 [s/m];
   seed*jvp(f,[x,v],[2 [m],-1 [m/s]])=49;
   vjp(f,x,seed)*2 [m]+vjp(f,v,seed)*(-1 [m/s])=49;
 }
}"#;
    assert_eq!(residuals(source), vec![0.; 5]);
    let bad = source.replace("vjp(f,x,seed)=35", "vjp(f,x,1)=35");
    let errors = compile("invalid-dual.eqi", &bad).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("dual output type")),
        "{errors:?}"
    );
}

#[test]
fn manufactured_two_coordinate_mixed_partials_execute_as_spatial_observables() {
    use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope};
    use eqiora::meshing::QuadratureRule;
    use eqiora::solver::REFERENCE_LINEAR_SOLVER;
    use eqiora_numerics::CommonSpatialPolicy;

    // Differentiate f(x,y)=x²y+y³ at its independent length formals, then
    // bind those exact slots to the admitted Cartesian coordinates. This is
    // an analytic field, not two Parameters renamed as spatial coordinates.
    let operators = r#"
operator xy(input x:m,input y:m):m=partial(partial(x*x*y+y*y*y,wrt=x),wrt=y);
operator yx(input x:m,input y:m):m=partial(partial(x*x*y+y*y*y,wrt=y),wrt=x);
"#;
    let observables = r#"
  observable mixed_xy:m^3=integral(xy(x=coordinate(0),y=coordinate(1)),measure(square));
  observable mixed_yx:m^3=integral(yx(x=coordinate(0),y=coordinate(1)),measure(square));
  observable weighted_xy:m^4=integral(coordinate(0)*xy(x=coordinate(0),y=coordinate(1)),measure(square));
  observable swapped_xy:m^4=integral(coordinate(0)*xy(x=coordinate(1),y=coordinate(0)),measure(square));
"#;
    let body = support::common_scalar_plan::COMPONENT;
    let end = body.rfind('}').unwrap();
    let source = format!("{operators}{}{observables}{}", &body[..end], &body[end..]);
    let (document, plan) = support::common_scalar_plan::document_and_plan_with_source(
        CommonSpatialPolicy::Q1,
        &source,
    );
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let replay = ModelEnvelope::from_json(
        &model.canonical_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    // On the unit square, f_xy=f_yx=2x. Integrals are 1, 1,
    // integral(2x²)=2/3, and integral(2xy)=1/2 for reversed axes.
    // Two-point Gaussian quadrature is exact for these degree-two polynomials;
    // 1e-12 covers accumulated binary64 quadrature roundoff, not a PDE error.
    for (name, expected) in [
        ("mixed_xy", 1.0),
        ("mixed_yx", 1.0),
        ("weighted_xy", 2.0 / 3.0),
        ("swapped_xy", 0.5),
    ] {
        let id = document.aliases()[&format!("definition.{name}")]
            .downcast()
            .unwrap();
        let value = result
            .observe(&model, id, &observation_rules(&model, id, &quadrature))
            .unwrap();
        assert!(
            (value.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-12,
            "{name}"
        );
        assert_eq!(
            value,
            result
                .observe(&replay, id, &observation_rules(&replay, id, &quadrature))
                .unwrap()
        );
    }
}

fn observation_rules(
    model: &eqiora::artifact::ModelEnvelope,
    observable: eqiora::Id<eqiora::entity::kinds::Observable>,
    rule: &eqiora::meshing::QuadratureRule,
) -> std::collections::HashMap<
    eqiora::Id<eqiora::entity::kinds::Domain>,
    eqiora::meshing::QuadratureRule,
> {
    let (transaction, _) = model.to_transaction().unwrap();
    let domain = transaction
        .ops()
        .iter()
        .find_map(|operation| match operation {
            eqiora::graph::Op::DefineKernelNode {
                node: eqiora::kernel::KernelNode::Observable(definition),
            } if definition.id() == observable => definition.reduction().domain(),
            _ => None,
        })
        .unwrap();
    std::collections::HashMap::from([(domain, rule.clone())])
}
