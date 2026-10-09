//! Exact coordinate binders distinguish spatial axes and dimensioned product factors.
mod support;
use eqiora::api::ModelDocument;
use eqiora::compiler::StaticBindingValue;
use eqiora::kernel::AxisBounds;
use eqiora::{DimExponents, DynQuantity};

#[test]
fn position_velocity_coordinate_partials_keep_distinct_units_and_identity() {
    let interval = |time| {
        let unit = DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap();
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(0.0, unit), DynQuantity::new(4.0, unit)).unwrap(),
        )
    };
    let source = "model Phase(support position:interval(m),support velocity:interval(m/s)) {
        support phase:product(position,velocity);
        coordinate x:m on phase from position;
        coordinate v:m/s on phase from velocity;
        relation derivatives on phase {
            partial(x*v,wrt=x)=v;
            partial(x*v,wrt=v)=x;
        }
    }";
    let document = ModelDocument::compile_selected(
        "coordinate-partials.eqi",
        source,
        "Phase",
        &[("position", interval(0)), ("velocity", interval(-1))],
    )
    .unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
}

fn relation_values(
    document: &ModelDocument,
    point: &[(eqiora::RawId, usize, f64, DimExponents)],
) -> Vec<eqiora::ValueLiteral> {
    use eqiora::kernel::{KernelNode, SymbolRef};
    document
        .program()
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation.expression()),
            _ => None,
        })
        .flat_map(|dag| {
            assert!(
                dag.definitions()
                    .values()
                    .any(|definition| definition.nodes().iter().any(|node| matches!(
                        node,
                        eqiora::kernel::pure_operator::CalculusNode::Differentiated { .. }
                    )))
            );
            eqiora::ir::ScalarOperatorIr::lower(dag)
                .unwrap()
                .evaluate_typed(dag.roots(), &mut |symbol| match symbol {
                    SymbolRef::Coordinate { factor, axis, .. } => point
                        .iter()
                        .find(|(id, selected, _, _)| *id == factor.erase() && *selected == axis)
                        .and_then(|(_, _, value, dimension)| {
                            eqiora::ValueLiteral::try_from(DynQuantity::new(*value, *dimension))
                                .ok()
                        }),
                    SymbolRef::Parameter(id) => document.program().typed_value(id.erase()).cloned(),
                    _ => None,
                })
                .unwrap()
        })
        .collect()
}

#[test]
fn cartesian_polynomial_partials_have_independent_axis_values_and_units() {
    // Q0=8 kg and L=2 m. For u=Q0*((x/L)^2+3*x*y/L²),
    // u_x=Q0*(2*x+3*y)/L² and u_y=3*Q0*x/L².
    let source = include_str!("../../../verify/language/coordinate-partials/models/cartesian.eqi");
    let document = ModelDocument::compile("polynomial.eqi", source).unwrap();
    let body = document.aliases()["body"];
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let derivative = DimExponents::from_integers([1, -1, 0, 0, 0, 0, 0]).unwrap();
    for (x, y) in [(3., 5.), (1., 7.), (6., 2.)] {
        let values = relation_values(&document, &[(body, 0, x, length), (body, 1, y, length)]);
        assert_eq!(values.len(), 4);
        for (actual, expected) in [
            (
                values[0].real_scalar_value().unwrap(),
                8. * (2. * x + 3. * y) / 4.,
            ),
            (values[2].real_scalar_value().unwrap(), 24. * x / 4.),
        ] {
            assert_eq!(actual, DynQuantity::new(expected, derivative));
        }
    }
}

#[test]
fn scaled_position_velocity_polynomial_executes_distinct_factor_derivatives() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let interval = |unit| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(0., unit), DynQuantity::new(20., unit)).unwrap(),
        )
    };
    // F0=6 kg, L=2 m, V0=4 m/s; exact binary reciprocals retain
    // independent position and velocity scales in the authored polynomial.
    let source = include_str!("../../../verify/language/coordinate-partials/models/phase.eqi");
    let document = ModelDocument::compile_selected(
        "phase.eqi",
        source,
        "Phase",
        &[
            ("position", interval(length)),
            ("velocity", interval(speed)),
        ],
    )
    .unwrap();
    let position = document.aliases()["position"];
    let velocity = document.aliases()["velocity"];
    for (x, v) in [(3., 11.), (7., 2.), (1., 17.)] {
        let values = relation_values(
            &document,
            &[(position, 0, x, length), (velocity, 0, v, speed)],
        );
        assert_eq!(values.len(), 4);
        assert_eq!(
            values[0].real_scalar_value().unwrap(),
            DynQuantity::new(
                6. * v / (2. * 4.),
                DimExponents::from_integers([1, -1, 0, 0, 0, 0, 0]).unwrap()
            )
        );
        assert_eq!(
            values[2].real_scalar_value().unwrap(),
            DynQuantity::new(
                6. * x / (2. * 4.),
                DimExponents::from_integers([1, -1, 1, 0, 0, 0, 0]).unwrap()
            )
        );
    }
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
}

#[test]
fn coordinate_declarations_reject_foreign_factors_axes_and_wrong_units() {
    let prefix = "model M() { domain body=box(0,2,0,3); domain foreign=box(0,2,0,3);";
    for declaration in [
        "coordinate x:m on body from foreign[0];",
        "coordinate x:m/s on body from body[0];",
        "coordinate x:m on body from body[2];",
        "coordinate x:m on body from body;",
        "coordinate x:m on missing from body[0];",
        "coordinate x:m on body from missing[0];",
    ] {
        assert!(
            ModelDocument::compile("invalid.eqi", &format!("{prefix}{declaration}}}")).is_err(),
            "{declaration}"
        );
    }
    let source = format!(
        "{prefix} coordinate x:m on body from body[0]; coordinate y:m on foreign from foreign[0]; relation r on body {{ partial(x*x,wrt=y)=0 [m]; }} }}"
    );
    let errors = ModelDocument::compile("foreign.eqi", &source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("incompatible spatial supports")),
        "{errors:?}"
    );
}

#[test]
fn boundary_embedding_coordinates_do_not_claim_an_intrinsic_derivative() {
    let source = r#"model Boundary() {
        domain body=box(0,2,0,3);
        domain wall=boundary(body,axis=0,side=lower);
        coordinate x:m on wall from body[0];
        relation r on wall { partial(x*x,wrt=x)=0 [m]; }
    }"#;
    let errors = ModelDocument::compile("boundary.eqi", source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("intrinsic chart")),
        "{errors:?}"
    );
}

#[test]
fn component_coordinate_selectors_forward_exact_parent_support() {
    let source = r#"component Derivatives(support body:volume(ambient_dimension=2)) {
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        relation r on body { partial(x*y,wrt=x)=y; partial(x*y,wrt=y)=x; }
    }
    model M() {
        domain body=box(0,2,0,3);
        instance c:Derivatives(body=body);
    }"#;
    let document = ModelDocument::compile("component.eqi", source).unwrap();
    let body = document.aliases()["body"];
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let values = relation_values(&document, &[(body, 0, 1., length), (body, 1, 2., length)]);
    assert_eq!(values.len(), 4);
    assert_eq!(values[0], values[1]);
    assert_eq!(values[2], values[3]);
}

#[test]
fn q1_field_partials_retain_requests_and_evaluate_basis_derivatives() {
    use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope};
    use eqiora::compiler::AuthoredFormExpressionV1;
    use eqiora::kernel::{ExprNode, KernelNode};
    use eqiora::meshing::QuadratureRule;
    use eqiora::solver::REFERENCE_LINEAR_SOLVER;
    use eqiora_numerics::CommonSpatialPolicy;

    // The bilinear harmonic solution x*(3/m + offset/m + 2*y/m²) lies exactly
    // in Q1. At offset=0 its derivatives are 3/m+2*y/m² and 2*x/m²;
    // unit-square integrals are 4 m and 1 m, independently of grad.
    let body = support::common_scalar_plan::COMPONENT
        .replace("source_scale * math.sin", "0 * source_scale * math.sin")
        .replace(
            "trace(potential) - boundary_offset",
            "trace(potential) - coordinate(0)*(3 [1/m]+boundary_offset*1 [1/m]+2 [1/m^2]*coordinate(1))",
        );
    let end = body.rfind('}').unwrap();
    let declarations = r#"
        coordinate x:m on square from square[0];
        coordinate y:m on square from square[1];
        observable dx:m=integral(partial(potential,wrt=x),measure(square));
        observable dy:m=integral(partial(potential,wrt=y),measure(square));
        observable grad_x:m=integral(component(grad(potential),indices=(0,)),measure(square));
        observable grad_y:m=integral(component(grad(potential),indices=(1,)),measure(square));
        observable square_dx:m=integral(partial(potential*potential,wrt=x),measure(square));
    "#;
    let source = format!("{}{}{}", &body[..end], declarations, &body[end..]);
    let (document, plan) = support::common_scalar_plan::document_and_plan_with_source(
        CommonSpatialPolicy::Q1,
        &source,
    );
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let replay = ModelEnvelope::from_json(
        &model.canonical_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap();
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    for (name, expected) in [("dx", 4.), ("dy", 1.)] {
        let id = document.aliases()[&format!("definition.{name}")];
        let Some(KernelNode::Observable(observable)) = document.program().node(id) else {
            panic!("missing derivative Observable")
        };
        let dag = observable.expression();
        assert!(matches!(
            dag.node(dag.roots()[0]),
            Some(ExprNode::CoordinatePartial { .. })
        ));
        let projected = AuthoredFormExpressionV1::from_expression(dag, dag.roots()[0])
            .unwrap()
            .unwrap();
        assert!(matches!(
            projected,
            AuthoredFormExpressionV1::CoordinatePartial { .. }
        ));
        let form_bytes = serde_json::to_vec(&projected).unwrap();
        assert_eq!(
            projected,
            serde_json::from_slice::<AuthoredFormExpressionV1>(&form_bytes).unwrap()
        );
        let domain = observable.reduction().domain().unwrap();
        let rules = std::collections::HashMap::from([(domain, quadrature.clone())]);
        let value = result
            .observe(&model, id.downcast().unwrap(), &rules)
            .unwrap();
        let actual = value.value().real_scalar_value().unwrap();
        assert_eq!(
            actual.dim(),
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
        );
        // Q1 exactly represents this bilinear function. This bound covers only
        // the small linear solve and binary64 quadrature, not PDE truncation.
        assert!(
            (actual.value() - expected).abs() < 1e-11,
            "{name}: {actual:?}"
        );
        let grad_id = document.aliases()
            [&format!("definition.grad_{}", if name == "dx" { "x" } else { "y" })];
        let grad = result
            .observe(&model, grad_id.downcast().unwrap(), &rules)
            .unwrap();
        assert_eq!(value.value(), grad.value());
        assert_eq!(
            value,
            result
                .observe(&replay, id.downcast().unwrap(), &rules)
                .unwrap()
        );
    }
    // d(u²)/dx=2*x*(3+2*y)² on the unit square: integral =
    // integral_0^1 (9+12*y+4*y²) dy = 49/3, including the Field chain rule.
    let square_id = document.aliases()["definition.square_dx"];
    let Some(KernelNode::Observable(square)) = document.program().node(square_id) else {
        unreachable!()
    };
    let rules = std::collections::HashMap::from([(
        square.reduction().domain().unwrap(),
        quadrature.clone(),
    )]);
    let square_value = result
        .observe(&model, square_id.downcast().unwrap(), &rules)
        .unwrap();
    assert!((square_value.value().real_scalar_value().unwrap().value() - 49. / 3.).abs() < 1e-10);
    let fv = support::common_scalar_plan::plan_for_document(
        &document,
        4,
        CommonSpatialPolicy::CellCenteredTpfa,
    )
    .run_result(&REFERENCE_LINEAR_SOLVER)
    .unwrap();
    let id = document.aliases()["definition.dx"];
    let Some(KernelNode::Observable(observable)) = document.program().node(id) else {
        unreachable!()
    };
    let rules =
        std::collections::HashMap::from([(observable.reduction().domain().unwrap(), quadrature)]);
    let error = fv
        .observe(&model, id.downcast().unwrap(), &rules)
        .unwrap_err();
    assert!(error.message().contains("Q1 field space"), "{error:?}");
}

#[test]
fn coordinate_independence_is_proved_and_unknown_higher_partials_reject() {
    let source = r#"model M() {
        domain body=box(0,2,0,3);
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        parameter coefficient:kg=7[kg];
        relation independent on body {
            partial(coefficient,wrt=x)=0[kg/m];
            partial(y*y,wrt=x)=0[m];
        }
    }"#;
    let document = ModelDocument::compile("independent.eqi", source).unwrap();
    let body = document.aliases()["body"];
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let values = relation_values(&document, &[(body, 0, 1., length), (body, 1, 2., length)]);
    assert_eq!(values.len(), 4);
    for pair in values.as_chunks::<2>().0 {
        assert_eq!(pair[0], pair[1]);
    }
    for expression in [
        "partial(partial(u,wrt=x),wrt=y)",
        "partial(partial(u*u,wrt=x),wrt=y)",
    ] {
        let source = format!(
            "model M(){{domain body=box(0,2,0,3); coordinate x:m on body from body[0]; coordinate y:m on body from body[1]; variable u:1 on body; relation r on body{{ {expression}=0[1/m^2]; }} }}"
        );
        let errors = ModelDocument::compile("higher.eqi", &source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("first-order profile")),
            "{errors:?}"
        );
    }
}

#[test]
fn coordinate_partial_fingerprint_retains_axis_and_ignores_binder_spelling() {
    let source = "model M(){domain body=box(0,2,0,3); coordinate x:m on body from body[0]; variable u:1 on body; relation r on body{partial(u,wrt=x)=0[1/m];}}";
    let model = ModelDocument::compile("m.eqi", source).unwrap();
    let renamed = ModelDocument::compile(
        "renamed.eqi",
        &source
            .replace("coordinate x:", "coordinate renamed:")
            .replace("wrt=x", "wrt=renamed"),
    )
    .unwrap();
    let axis = ModelDocument::compile("axis.eqi", &source.replace("body[0]", "body[1]")).unwrap();
    assert!(model.structurally_equivalent(&renamed).unwrap());
    assert!(!model.structurally_equivalent(&axis).unwrap());
    let bytes = model.canonical_json().unwrap();
    let old = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("eqiora.model-envelope/v44", "eqiora.model-envelope/v33");
    assert_ne!(old.as_bytes(), bytes);
    assert!(ModelDocument::replay(old.as_bytes()).is_err());
}
