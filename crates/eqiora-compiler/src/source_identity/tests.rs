use eqiora_lang::{format, parse};

use super::*;

mod let_alias;
mod namespace;
mod partials;

fn document(source: &str) -> Document {
    parse("fixture.eqi", source).into_document().unwrap()
}

fn identity(source: &str) -> LocalSourceIdentity {
    LocalSourceIdentity::from_document(&document(source)).unwrap()
}

#[test]
fn pure_operator_identity_is_definition_semantic_and_order_independent() {
    let first = r#"
public operator outer(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);
private operator scale(input value: scalar): scalar
  = rational(2, 1) * component(value);
model M() { parameter p: 1 = 1; }
"#;
    let renamed_and_reordered = r#"
private operator scale(input x: scalar): scalar
  = rational(2, 1) * component(x);
public operator outer(input a: spatial[1], input b: spatial[1]): spatial[2]
  = component(a, 0) * component(b, 1);
model M() { parameter p: 1 = 1; }
"#;
    let changed_body = renamed_and_reordered.replace(
        "component(a, 0) * component(b, 1)",
        "component(b, 0) * component(a, 1)",
    );

    assert_eq!(identity(first), identity(renamed_and_reordered));
    assert_ne!(identity(first), identity(&changed_body));
}

#[test]
fn typed_operator_identity_precedes_nominal_binding_and_tracks_lexical_references() {
    let source = r#"
space A=orthonormal(a,b); space B=orthonormal(c,d);
operator retain(input value:coordinates<complex<1>,A>):coordinates<complex<1>,A>
 = component(value,0);
operator derivative(input x:1,input y:1,input z:1):1
 = partial(x*y*z,wrt=x,holding=(y,z));
model M(){parameter p:1=1;}
"#;
    let expected = identity(source);
    let renamed = source
        .replace("value", "renamed")
        .replace("input y:", "input q:")
        .replace("input z:", "input a:")
        .replace("x*y*z", "x*q*a")
        .replace("holding=(y,z)", "holding=(a,q)");
    assert_eq!(expected, identity(&renamed));
    assert_ne!(
        expected,
        identity(&source.replace("complex<1>,A", "complex<1>,B"))
    );
    assert_ne!(
        expected,
        identity(&source.replace("wrt=x,holding=(y,z)", "wrt=y,holding=(x,z)"))
    );
    let mut bound = document(source);
    let spaces =
        crate::nominal::finite_spaces("bound.eqi", &bound, &expected.namespace().unwrap(), |_| {
            None
        })
        .unwrap();
    crate::nominal::bind_finite_types("bound.eqi", &mut bound, &spaces).unwrap();
    assert_eq!(
        expected,
        LocalSourceIdentity::from_document(&bound).unwrap()
    );
    eqiora_lang::SourceAstFactory::visit_expressions(&mut bound, |_, expression| {
        if matches!(expression.kind(), ExprKind::Name(name) if name == "value") {
            *expression = eqiora_lang::SourceAstFactory::expression(
                ExprKind::Path(NamePath::from_segments(["value"], expression.range()).unwrap()),
                expression.range(),
            )
            .unwrap();
        }
    });
    assert_eq!(
        expected,
        LocalSourceIdentity::from_document(&bound).unwrap()
    );
}

#[test]
fn declaration_binding_and_anonymous_member_permutations_are_invariant() {
    let first = r#"
connector Pin {
  across voltage: 1;
  through current: A;
}
connector Heat {
  across potential: K;
  through flow: kg * m ^ 2 / (s ^ 3 * K);
}
component Pair(parameter resistance: 1 = 2, parameter scale: 1 = 3, port positive: Pin, port negative: Pin) {
  instance inner: Library.Resistor(resistance = resistance, scale = scale);
  relation law { positive.voltage - negative.voltage = 0; }
  connect positive, inner.positive, negative;
}
component Empty() {}
model circuit() {
  instance right: Pair(scale = 4, resistance = 5);
  instance left: Pair(resistance = 2, scale = 3);
  connect left.positive, right.negative, right.positive;
}
model auxiliary() {}
"#;
    let permuted = r#"
connector Heat {
  across potential: K;
  through flow: kg * m ^ 2 / (s ^ 3 * K);
}
connector Pin {
  across voltage: 1;
  through current: A;
}
component Empty() {}
component Pair(port negative: Pin, port positive: Pin, parameter scale: 1 = 3, parameter resistance: 1 = 2) {
  connect negative, positive, inner.positive;
  relation law { positive.voltage - negative.voltage = 0; }
  instance inner: Library.Resistor(scale = scale, resistance = resistance);
}
model auxiliary() {}
model circuit() {
  connect right.positive, left.positive, right.negative;
  instance left: Pair(scale = 3, resistance = 2);
  instance right: Pair(resistance = 5, scale = 4);
}
"#;

    assert_eq!(identity(first), identity(permuted));
}

#[test]
fn support_declarations_and_bindings_are_canonical_but_exact_targets_are_semantic() {
    let body_first = r#"
component C(support body: volume(ambient_dimension = 2), support wall: boundary(parent = body)) {


}
model M() {
  domain volume = box(0, 1, 0, 1);
  domain left = boundary(volume, axis = 0, side = lower);
  domain right = boundary(volume, axis = 0, side = upper);
  instance c: C(body = volume, wall = left);
}
"#;
    let permuted = r#"
component C(support wall: boundary(parent = body), support body: volume(ambient_dimension = 2)) {


}
model M() {
  instance c: C(wall = left, body = volume);
  domain right = boundary(volume, axis = 0, side = upper);
  domain left = boundary(volume, axis = 0, side = lower);
  domain volume = box(0, 1, 0, 1);
}
"#;
    let rebound = permuted.replace("wall = left", "wall = right");

    assert_eq!(identity(body_first), identity(permuted));
    assert_ne!(identity(body_first), identity(&rebound));
}

#[test]
fn field_slots_and_bindings_are_canonical_but_exact_targets_are_semantic() {
    let slot_first = r#"
component Law(variable displacement: vector<m, 2> on body, variable potential: K on body, support body: volume(ambient_dimension = 2)) {



}
model M() {
  domain body = box(0, 1, 0, 1);
  variable displacement: vector<m, 2> on body;
  variable potential: K on body; initial { potential = 0; }
  variable other: K on body; initial { other = 0; }
  instance law: Law(
body = body,
displacement = displacement,
potential = potential
  );
}
"#;
    let permuted = r#"
component Law(variable potential: K on body, variable displacement: vector<m, 2> on body, support body: volume(ambient_dimension = 2)) {



}
model M() {
  instance law: Law(
potential = potential,
displacement = displacement,
body = body
  );
  variable other: K on body; initial { other = 0; }
  variable potential: K on body; initial { potential = 0; }
  variable displacement: vector<m, 2> on body;
  domain body = box(0, 1, 0, 1);
}
"#;
    let rebound = permuted.replace("potential = potential", "potential = other");

    assert_eq!(identity(slot_first), identity(permuted));
    assert_ne!(identity(slot_first), identity(&rebound));
}

#[test]
fn complete_exterior_family_records_are_order_independent_and_exact() {
    let source = r#"
public connector MechanicalBoundary {
  trace displacement: m;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
public component SurfaceLaw(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body), port mechanical[boundary in exterior]:
MechanicalBoundary over boundary) {


  relation carrier[boundary in exterior] on boundary {
mechanical[boundary = boundary].displacement
  - mechanical[boundary = boundary].displacement = 0;
  }
}
model M() {
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);
  instance surface: SurfaceLaw(
body = body,
exterior = boundaries(x_lower, x_upper, y_lower, y_upper)
  );
  connect
surface.mechanical[boundary = x_lower],
surface.mechanical[boundary = x_upper];
}
"#;
    let permuted = source
        .replace(
            "boundaries(x_lower, x_upper, y_lower, y_upper)",
            "boundaries(y_upper, x_lower, y_lower, x_upper)",
        )
        .replace(
            "surface.mechanical[boundary = x_lower],\n    surface.mechanical[boundary = x_upper]",
            "surface.mechanical[boundary = x_upper],\n    surface.mechanical[boundary = x_lower]",
        );
    let different_selector = source.replace(
        "surface.mechanical[boundary = x_upper];",
        "surface.mechanical[boundary = y_upper];",
    );

    assert_eq!(identity(source), identity(&permuted));
    assert_ne!(identity(source), identity(&different_selector));
}

#[test]
fn complete_exterior_memberships_have_independent_source_identity_limits() {
    let document = document(
        r#"
component SurfaceLaw(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body)) {


}
model M() {
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);
  instance surface: SurfaceLaw(
body = body,
exterior = boundaries(x_lower, x_upper, y_lower, y_upper)
  );
}
"#,
    );
    let per_set = LocalSourceIdentityLimits {
        max_boundary_set_members: 3,
        ..LocalSourceIdentityLimits::default()
    };
    let total = LocalSourceIdentityLimits {
        max_total_boundary_set_memberships: 3,
        ..LocalSourceIdentityLimits::default()
    };

    assert!(LocalSourceIdentity::from_document_with_limits(&document, per_set).is_err());
    assert!(LocalSourceIdentity::from_document_with_limits(&document, total).is_err());
}

#[test]
fn parameter_support_and_field_bindings_share_one_limit() {
    let document = document(
        r#"
component Law(variable state: 1 on body, support body: volume(ambient_dimension = 1), parameter gain: 1) {


}
model M() {
  domain body = box(0, 1);
  variable state: 1 on body; initial { state = 0; }
  instance law: Law(gain = 1, body = body, state = state);
}
"#,
    );
    let limits = LocalSourceIdentityLimits {
        max_bindings_per_instance: 2,
        ..LocalSourceIdentityLimits::default()
    };
    let error = LocalSourceIdentity::from_document_with_limits(&document, limits)
        .expect_err("all three binding families share one checked budget");
    assert!(error.message().contains("binding count exceeds"));
}

#[test]
fn retired_scalar_shape_spelling_is_not_an_identity_alias() {
    let retired = eqiora_lang::parse(
        "retired.eqi",
        "model M() { variable x: 1 shape []; initial { x = 0; } relation r { x = 0; } }",
    )
    .into_document();
    assert!(retired.is_err());
}

#[test]
fn conserving_fragments_have_definition_local_equivalence_identity() {
    let component_nary = "component C() { connect a, b, c; } model Empty() {}";
    let component_chain = "component C() { connect a, b; connect b, c; } model Empty() {}";
    assert_eq!(identity(component_nary), identity(component_chain));

    let model_nary = "model M() { connect a, b, c; }";
    let model_chain = "model M() { connect a, b; connect b, c; }";
    assert_eq!(identity(model_nary), identity(model_chain));
}

#[test]
fn duplicate_conserving_fragments_are_identity_idempotent() {
    let component_once = "component C() { connect a, b; } model Empty() {}";
    let component_twice = "component C() { connect a, b; connect b, a; } model Empty() {}";
    assert_eq!(identity(component_once), identity(component_twice));

    let model_once = "model M() { connect a, b; }";
    let model_twice = "model M() { connect a, b; connect b, a; }";
    assert_eq!(identity(model_once), identity(model_twice));
}

#[test]
fn spatial_periodic_pair_identity_is_endpoint_order_invariant_but_not_conserving() {
    let lower_upper = "model M() { connect periodic lower.p, upper.p; }";
    let upper_lower = "model M() { connect periodic upper.p, lower.p; }";
    let conserving = "model M() { connect lower.p, upper.p; }";

    assert_eq!(identity(lower_upper), identity(upper_lower));
    assert_ne!(identity(lower_upper), identity(conserving));
}

#[test]
fn disjoint_conserving_and_signal_records_keep_the_legacy_bytes() {
    let document = document(
        "component C() { connect a, b; connect out -> in_b, in_a; connect c, d; } model M() { connect w, x; connect source -> sink_b, sink_a; connect y, z; }",
    );
    let limits = LocalSourceIdentityLimits::default();

    let mut component_legacy_budget = Budget::new(limits);
    let component_legacy = encode_sorted_records(
        document.components()[0].items(),
        &mut component_legacy_budget,
        encode_component_item,
    )
    .unwrap();
    let mut component_normalized_budget = Budget::new(limits);
    let component_normalized = encode_container_records(
        document.components()[0].items(),
        &mut component_normalized_budget,
        component_connection,
        encode_component_item,
        COMPONENT_CONNECTION_ITEM_TAG,
    )
    .unwrap();
    assert_eq!(component_normalized, component_legacy);

    let mut model_legacy_budget = Budget::new(limits);
    let model_legacy = encode_sorted_records(
        document.models()[0].items(),
        &mut model_legacy_budget,
        encode_model_item,
    )
    .unwrap();
    let mut model_normalized_budget = Budget::new(limits);
    let model_normalized = encode_container_records(
        document.models()[0].items(),
        &mut model_normalized_budget,
        model::model_connection,
        encode_model_item,
        MODEL_CONNECTION_ITEM_TAG,
    )
    .unwrap();
    assert_eq!(model_normalized, model_legacy);
}

#[test]
fn normalized_conserving_sets_cannot_evade_connection_member_limits() {
    let document = document("component C() { connect a, b; connect b, c; } model Empty() {}");
    let limits = LocalSourceIdentityLimits {
        max_connection_members: 2,
        ..LocalSourceIdentityLimits::default()
    };
    let diagnostic = LocalSourceIdentity::from_document_with_limits(&document, limits)
        .expect_err("the transitive set has three members");

    assert!(
        diagnostic
            .message()
            .contains("members in one normalized connection set")
    );
}

#[test]
fn signal_output_is_positional_but_inputs_are_canonical_members() {
    let base = "model m() { port o: signal output 1; port a: signal input 1; port b: signal input 1; connect o -> a, b; }";
    let inputs_permuted = "model m() { port o: signal output 1; port a: signal input 1; port b: signal input 1; connect o -> b, a; }";
    let different_output = "model m() { port o: signal output 1; port a: signal input 1; port b: signal input 1; connect a -> o, b; }";

    assert_eq!(identity(base), identity(inputs_permuted));
    assert_ne!(identity(base), identity(different_output));
}

#[test]
fn semantic_structure_and_exact_values_change_identity() {
    let base = "model m() { parameter p: 1 = 2; relation r { p + 1 = 0; } }";
    let changed_value = "model m() { parameter p: 1 = 3; relation r { p + 1 = 0; } }";
    let changed_operator = "model m() { parameter p: 1 = 2; relation r { p - 1 = 0; } }";
    let changed_activation = "model m() { clock c = periodic(1[s] / 1, phase = 0[s] / 1); parameter p: 1 = 2; relation r at c { p + 1 = 0; } }";

    assert_ne!(identity(base), identity(changed_value));
    assert_ne!(identity(base), identity(changed_operator));
    assert_ne!(identity(base), identity(changed_activation));
}

#[test]
fn canonical_identity_is_structural_not_algebraic_equivalence() {
    let folded = "model m() { parameter p: 1 = 2; relation r { p + 2 = 0; } }";
    let unfolded = "model m() { parameter p: 1 = 2; relation r { p + (1 + 1) = 0; } }";
    let multiplied_dimension = "model m() { parameter area: m * m = 1; relation r { area = 0; } }";
    let powered_dimension = "model m() { parameter area: m ^ 2 = 1; relation r { area = 0; } }";

    assert_ne!(identity(folded), identity(unfolded));
    assert_ne!(identity(multiplied_dimension), identity(powered_dimension));
}

#[test]
fn interface_visibility_defaults_ports_bindings_and_domains_are_semantic() {
    let public_default =
        "component C(parameter p: 1 = 2, input s: 1) {   } model m() { instance x: C(p = 2); }";
    let private_default =
        "component C(input s: 1) { parameter p: 1 = 2;  } model m() { instance x: C(p = 2); }";
    let required =
        "component C(parameter p: 1, input s: 1) {   } model m() { instance x: C(p = 2); }";
    let output_port =
        "component C(parameter p: 1 = 2, output s: 1) {   } model m() { instance x: C(p = 2); }";
    let changed_binding =
        "component C(parameter p: 1 = 2, input s: 1) {   } model m() { instance x: C(p = 3); }";
    assert_ne!(identity(public_default), identity(private_default));
    assert_ne!(identity(public_default), identity(required));
    assert_ne!(identity(public_default), identity(output_port));
    assert_ne!(identity(public_default), identity(changed_binding));

    let on_domain = "model m() { domain d = box(0, 1); relation r on d { 1 = 0; } }";
    let without_domain = "model m() { domain d = box(0, 1); relation r { 1 = 0; } }";
    assert_ne!(identity(on_domain), identity(without_domain));
}

#[test]
fn package_visibility_is_semantic_and_private_is_the_canonical_default() {
    let private =
        "connector Pin {\n  across voltage: 1;\n  through current: A;\n} component Resistor() {}";
    let explicit_private = "private component Resistor() {} private connector Pin {\n  across voltage: 1;\n  through current: A;\n}";
    let public_connector = "component Resistor() {} public connector Pin {\n  across voltage: 1;\n  through current: A;\n}";
    let public_component = "public component Resistor() {} connector Pin {\n  across voltage: 1;\n  through current: A;\n}";

    assert_eq!(identity(private), identity(explicit_private));
    assert_ne!(identity(private), identity(public_connector));
    assert_ne!(identity(private), identity(public_component));
    assert_ne!(identity(public_connector), identity(public_component));

    let private_model = "model Main() {}";
    let explicit_private_model = "private model Main() {}";
    let public_model = "public model Main() {}";
    assert_eq!(identity(private_model), identity(explicit_private_model));
    assert_ne!(identity(private_model), identity(public_model));

    let formatted = format(&document(public_connector));
    assert_eq!(
        identity(public_connector),
        LocalSourceIdentity::from_document(
            &parse(
                "elsewhere/library.eqi",
                &format!("// relocated\n{formatted}")
            )
            .into_document()
            .unwrap(),
        )
        .unwrap()
    );
}

#[test]
fn residual_root_order_is_preserved() {
    let first =
        "model m() { parameter a: 1 = 1; parameter b: 1 = 2; relation r { a = 0; b = 0; } }";
    let reversed =
        "model m() { parameter a: 1 = 1; parameter b: 1 = 2; relation r { b = 0; a = 0; } }";

    assert_ne!(identity(first), identity(reversed));
}

#[test]
fn negative_zero_is_normalized_to_positive_zero() {
    let positive = "model m() { parameter p: 1 = 0; }";
    let negative = "model m() { parameter p: 1 = -0; }";

    assert_eq!(identity(positive), identity(negative));
}

#[test]
fn negative_zero_has_one_source_transaction_and_model_meaning() {
    let positive = "connector Pin {\n  across potential: 1;\n  through flow: 1;\n} model m() { domain d = box(0, 1); variable x: 1 on d; initial { x = 0; } parameter p: 1 = 0; relation r on d { x + p + 0 = 0; } }";
    let negative = "connector Pin {\n  across potential: 1;\n  through flow: 1;\n} model m() { domain d = box(-0, 1); variable x: 1 on d; initial { x = -0; } parameter p: 1 = -0; relation r on d { x + p + -0 = 0; } }";

    assert_eq!(identity(positive), identity(negative));
    let mut positive = crate::compile("zero.eqi", positive).unwrap();
    let mut negative = crate::compile("zero.eqi", negative).unwrap();
    let positive = positive.remove(0);
    let negative = negative.remove(0);
    assert_eq!(positive.model(), negative.model());
    assert_eq!(positive.transaction().ops(), negative.transaction().ops());
}

#[test]
fn cartesian_coordinate_sources_preserve_exact_root_declaration_identity() {
    let parameter = "model m() { parameter extent: m = 2; parameter other: m = 2; domain body = box(-1, extent, extent, 6); relation r on body { coordinate(0) - coordinate(0) = 0; } }";
    let declarations_permuted = "model m() { domain body = box(-1, extent, extent, 6); relation r on body { coordinate(0) - coordinate(0) = 0; } parameter other: m = 2; parameter extent: m = 2; }";
    let fixed = "model m() { parameter extent: m = 2; parameter other: m = 2; domain body = box(-1, 2, 2, 6); relation r on body { coordinate(0) - coordinate(0) = 0; } }";
    let other_root = "model m() { parameter extent: m = 2; parameter other: m = 2; domain body = box(-1, other, other, 6); relation r on body { coordinate(0) - coordinate(0) = 0; } }";

    assert_eq!(identity(parameter), identity(declarations_permuted));
    assert_ne!(identity(parameter), identity(fixed));
    assert_ne!(identity(parameter), identity(other_root));
}

#[test]
fn resource_limits_fail_closed() {
    let base_document = document("model m() { parameter p: 1 = 2; relation r { p + 1 = 0; } }");
    let top_level = LocalSourceIdentityLimits {
        max_top_level_declarations: 0,
        ..LocalSourceIdentityLimits::default()
    };
    assert!(LocalSourceIdentity::from_document_with_limits(&base_document, top_level).is_err());

    let expressions = LocalSourceIdentityLimits {
        max_expression_nodes: 1,
        ..LocalSourceIdentityLimits::default()
    };
    assert!(LocalSourceIdentity::from_document_with_limits(&base_document, expressions).is_err());

    let bytes = LocalSourceIdentityLimits {
        max_canonical_bytes: 8,
        ..LocalSourceIdentityLimits::default()
    };
    assert!(LocalSourceIdentity::from_document_with_limits(&base_document, bytes).is_err());

    let intermediate = LocalSourceIdentityLimits {
        max_intermediate_bytes: 1,
        ..LocalSourceIdentityLimits::default()
    };
    assert!(LocalSourceIdentity::from_document_with_limits(&base_document, intermediate).is_err());

    let mixed_bindings = document(
        "component C(support d: volume(ambient_dimension = 1), parameter p: 1) {   } \
         model M() { domain d = box(0, 1); instance c: C(p = 1, d = d); }",
    );
    let bindings = LocalSourceIdentityLimits {
        max_bindings_per_instance: 1,
        ..LocalSourceIdentityLimits::default()
    };
    assert!(LocalSourceIdentity::from_document_with_limits(&mixed_bindings, bindings).is_err());
}

#[test]
fn initial_support_is_bound_without_pinning_unrelated_source_bytes() {
    let source = "model M() { domain body=box(0,1); state u:1 on body;
        initial on body {u=0;} relation flow on body {derivative(u)=0[1/s];} }";
    assert_eq!(identity(source), identity(&format(&document(source))));
    assert_ne!(
        identity(source),
        identity(&source.replace("initial on body", "initial"))
    );
}
