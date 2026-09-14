//! Independent divergence-theorem obligations for an authored mathematical interval.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, GeometryGraph};
use std::collections::BTreeMap;

const FORM: &str = r#"
form conservative for balance {
    interval segment(a, b) on body;
    outward_flux(segment, a, -k * grad(T)) + outward_flux(segment, b, -k * grad(T)) = integrate(segment, s);
}
"#;
fn source(form: &str) -> String {
    format!(
        r#"public component M(support body:volume(ambient_dimension=1),
      parameter k:kg*m/s^3/K, parameter other_k:kg*m/s^3/K,
      parameter s:kg/m/s^3, parameter other_s:kg/m/s^3) {{
      variable T:K on body;
      variable other_T:K on body;
      law balance on body {{ flux -k * grad(T); source s; }}
      {form}
    }}"#
    )
}
fn geometry() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 8.0]).unwrap();
    graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![interval.boundaries()[0].into()]),
                ("right".into(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap()
}
fn compile(
    source: &str,
    geometry: &CanonicalGeometryV1,
) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let values = eqiora_lang::parse("values.eqi", "model V(){parameter k:1=3;parameter s:1=12;}")
        .into_document()
        .unwrap();
    let expressions = values.models()[0]
        .items()
        .iter()
        .filter_map(|item| {
            if let eqiora_lang::Item::Parameter(p) = item {
                Some(p.value())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    CompiledModel::compile_selected(
        "interval.eqi",
        source,
        "M",
        &[
            ("k", StaticBindingValue::Expression(expressions[0])),
            ("other_k", StaticBindingValue::Expression(expressions[0])),
            ("s", StaticBindingValue::Expression(expressions[1])),
            ("other_s", StaticBindingValue::Expression(expressions[1])),
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry.entity_set("body").unwrap(),
                    parent: None,
                },
            ),
        ],
    )
}
#[test]
fn ordinary_source_and_formatter_retain_quantified_interval_and_live_replay() {
    let geometry = geometry();
    let source = source(FORM);
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("interval.eqi", &source)
            .into_document()
            .unwrap(),
    );
    for source in [&source, &formatted] {
        let model = compile(source, &geometry).unwrap_or_else(|e| panic!("{e:?}"));
        let projection = model.authored_formulations().next().unwrap().projection();
        assert_eq!(projection.interval(), Some(("segment", "a", "b")));
        assert_eq!(
            projection.implication(),
            "strong-implies-interval-conservation"
        );
        assert!(projection.test_restriction().is_none());
        let decoded = AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap();
        decoded
            .check_interval(model.transaction(), &geometry)
            .unwrap();
    }
}
#[test]
fn source_mutants_reach_the_interval_obligation() {
    let geometry = geometry();
    compile(&source(FORM), &geometry).unwrap();
    for (label, mutant) in [
        (
            "duplicate lower",
            FORM.replace("segment, b,", "segment, a,"),
        ),
        (
            "duplicate upper",
            FORM.replace("segment, a,", "segment, b,"),
        ),
        (
            "reversed boundary sign",
            FORM.replace("+ outward_flux", "- outward_flux"),
        ),
        (
            "reversed physical flux",
            FORM.replace("-k * grad(T)", "k * grad(T)"),
        ),
        (
            "foreign coefficient",
            FORM.replace("-k * grad(T)", "-other_k * grad(T)"),
        ),
        (
            "foreign source",
            FORM.replace("integrate(segment, s)", "integrate(segment, other_s)"),
        ),
        (
            "reversed source",
            FORM.replace("integrate(segment, s)", "integrate(segment, -s)"),
        ),
        ("foreign Field", FORM.replace("grad(T)", "grad(other_T)")),
        (
            "whole Domain measure",
            FORM.replace("integrate(segment, s)", "integrate(body, s)"),
        ),
        ("escaped endpoint", FORM.replace("segment, a,", "body, a,")),
        (
            "unbound endpoint",
            FORM.replace("segment, a,", "segment, z,"),
        ),
        (
            "aliased endpoints",
            FORM.replace("segment(a, b)", "segment(a, a)"),
        ),
        (
            "shadowed binder",
            FORM.replace("segment(a, b)", "segment(T, b)"),
        ),
    ] {
        let errors = compile(&source(&mutant), &geometry).expect_err(label);
        assert!(
            errors.iter().any(|e| e.message().contains("interval")),
            "{label}: {errors:?}"
        );
    }
}
#[test]
fn decoded_mutants_cannot_bypass_live_source_check() {
    let geometry = geometry();
    let model = compile(&source(FORM), &geometry).unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let text = String::from_utf8(projection.canonical_bytes().to_vec()).unwrap();
    for (label, mutant) in [
        ("normal", text.replacen("\"normal\":-1", "\"normal\":1", 1)),
        (
            "endpoint",
            text.replacen("\"endpoint\":\"a\"", "\"endpoint\":\"b\"", 1),
        ),
        (
            "scope",
            text.replacen("\"interval\":\"segment\"", "\"interval\":\"body\"", 1),
        ),
        (
            "assumption",
            text.replace("every-ordered-subinterval-of-parent", "one-mesh-cell"),
        ),
        (
            "direction",
            text.replace("strong-implies-interval-conservation", "equivalent"),
        ),
    ] {
        assert_ne!(mutant, text, "{label}");
        if let Ok(decoded) = AuthoredFormulationProjection::decode(mutant.as_bytes()) {
            assert!(
                decoded
                    .check_interval(model.transaction(), &geometry)
                    .is_err(),
                "{label}"
            );
        }
    }
    let foreign = compile(
        &source(FORM).replace("flux -k * grad(T)", "flux k * grad(T)"),
        &geometry,
    );
    assert!(foreign.is_err());
}

#[test]
fn proper_asymmetric_interval_has_independently_derived_nonzero_boundary_balance() {
    // Independent polynomial: T=2*x*(8-x); T'=16-4*x;
    // j=-3*T'=12*x-48, s=12, so j(b)-j(a)=12*(b-a) for every a<b.
    // (1,3) is neither the complete Domain nor a symmetric interval.
    use eqiora_compiler::AuthoredFormExpressionV1 as F;
    let geometry = geometry();
    let model = compile(&source(FORM), &geometry).unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let coefficient = model.symbols().get("k").unwrap().ulid().to_string();
    let source_id = model.symbols().get("s").unwrap().ulid().to_string();
    fn evaluate(expr: &F, x: f64, k: &str, s: &str) -> f64 {
        match expr {
            F::Parameter { ulid } if ulid == k => 3.0,
            F::Parameter { ulid } if ulid == s => 12.0,
            F::Gradient { value } if matches!(value.as_ref(), F::Field { .. }) => 16.0 - 4.0 * x,
            F::Neg { value } => -evaluate(value, x, k, s),
            F::Mul { left, right } => evaluate(left, x, k, s) * evaluate(right, x, k, s),
            _ => panic!("unexpected oracle input {expr:?}"),
        }
    }
    let F::Add { left, right } = projection.left() else {
        panic!("boundary sum")
    };
    let mut values = Vec::new();
    for term in [left, right] {
        let F::EndpointFlux {
            endpoint,
            normal,
            flux,
            ..
        } = term.as_ref()
        else {
            panic!("endpoint")
        };
        let x = match endpoint.as_str() {
            "a" => 1.0,
            "b" => 3.0,
            _ => panic!("unbound endpoint"),
        };
        values.push(f64::from(*normal) * evaluate(flux, x, &coefficient, &source_id));
    }
    assert_eq!(values, [36.0, -12.0]);
    let F::IntervalIntegral { integrand, .. } = projection.right() else {
        panic!("interval measure")
    };
    let integral = (3.0 - 1.0) * evaluate(integrand, 0.0, &coefficient, &source_id);
    assert_eq!(integral, 24.0);
    assert_eq!(values.iter().sum::<f64>(), integral);
    for mutant in [
        -values[0] + values[1],
        values[0] - values[1],
        2.0 * values[0],
        2.0 * values[1],
        96.0,
    ] {
        assert_ne!(mutant, integral);
    }
}

#[test]
fn live_law_change_with_the_same_ids_cannot_replay_a_stale_form() {
    use eqiora_graph::{Op, Transaction};
    use eqiora_schema::kernel::{
        ConservationTerms, ExprDagBuilder, KernelNode, RelationDef, RelationMeaning,
    };
    let geometry = geometry();
    let model = compile(&source(FORM), &geometry).unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    for reverse_flux in [false, true] {
        let mut changed = Transaction::new("live changed Law");
        for op in model.transaction().ops() {
            let mut op = op.clone();
            if let Op::DefineKernelNode {
                node: KernelNode::Relation(law),
            } = &mut op
                && let RelationMeaning::Conservation(terms) = law.meaning()
            {
                let mut dag = ExprDagBuilder::from_dag(law.expression());
                let flux = if reverse_flux {
                    dag.neg(terms.flux()).unwrap()
                } else {
                    terms.flux()
                };
                let source = if reverse_flux {
                    terms.source()
                } else {
                    dag.neg(terms.source()).unwrap()
                };
                let divergence = dag.divergence(flux).unwrap();
                *law = RelationDef::conservation(
                    law.id(),
                    dag.finish([divergence, source]).unwrap(),
                    ConservationTerms::new(None, flux, source),
                )
                .unwrap();
            }
            changed.push(op);
        }
        assert!(projection.check_interval(&changed, &geometry).is_err());
    }
}

#[test]
fn native_binder_construction_uses_the_same_source_identity() {
    use eqiora_lang::{SourceAstFactory as Ast, VisibilitySyntax};
    let parsed = eqiora_lang::parse("interval.eqi", &source(FORM))
        .into_document()
        .unwrap();
    let original = &parsed.components()[0];
    let (name, relation, left, right, range) = original.formulations().next().unwrap();
    let rebuilt = Ast::component_with_form(
        VisibilitySyntax::Public,
        "M",
        original.signature().to_vec(),
        original.items().to_vec(),
        (
            name.into(),
            relation.into(),
            original.formulation_binding(name).unwrap().clone(),
        ),
        (left.clone(), right.clone(), range),
        original.range(),
    )
    .unwrap();
    let document = Ast::document(Vec::new(), Vec::new(), vec![rebuilt], Vec::new()).unwrap();
    let geometry = geometry();
    let parsed_model = compile(&source(FORM), &geometry).unwrap();
    let native_model = compile(&eqiora_lang::format(&document), &geometry).unwrap();
    assert_eq!(
        parsed_model
            .authored_formulations()
            .next()
            .unwrap()
            .projection(),
        native_model
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
    );
    let renamed = compile(&source(&FORM.replace("segment", "slice")), &geometry).unwrap();
    assert_eq!(
        parsed_model.transaction().ops(),
        renamed.transaction().ops()
    );
    assert_ne!(
        parsed_model
            .authored_formulations()
            .next()
            .unwrap()
            .source_identity(),
        renamed
            .authored_formulations()
            .next()
            .unwrap()
            .source_identity()
    );
}

#[test]
fn live_field_dimension_and_destructive_delta_are_not_typed_snapshots() {
    use eqiora_graph::{Op, Transaction};
    use eqiora_schema::kernel::{FieldDef, KernelNode};
    let geometry = geometry();
    let model = compile(&source(FORM), &geometry).unwrap();
    let projection = model.authored_formulations().next().unwrap().projection();
    let trial = model.authored_formulations().next().unwrap().trial();
    let mut changed = Transaction::new("wrong live Field dimension");
    for op in model.transaction().ops() {
        let mut op = op.clone();
        if let Op::DefineKernelNode {
            node: KernelNode::Field(field),
        } = &mut op
            && field.id() == trial
        {
            *field = FieldDef::new(
                field.id(),
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .unwrap(),
                field.role(),
            );
        }
        changed.push(op);
    }
    assert!(projection.check_interval(&changed, &geometry).is_err());
    let mut removed = Transaction::new("removed live Field");
    for op in model.transaction().ops() {
        removed.push(op.clone());
    }
    removed.push(Op::RemoveNode { id: trial.erase() });
    assert!(projection.check_interval(&removed, &geometry).is_err());
}
