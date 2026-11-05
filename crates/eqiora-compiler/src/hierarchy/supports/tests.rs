use super::*;
use eqiora_lang::{BoundarySideSyntax, Document};

fn parse(source: &str) -> Document {
    eqiora_lang::parse("supports.eqi", source)
        .into_document()
        .expect("support-contract fixture parses")
}

fn component<'a>(document: &'a Document, name: &str) -> &'a ComponentDecl {
    document
        .components()
        .iter()
        .find(|component| component.name() == name)
        .expect("fixture component exists")
}

fn instance<'a>(document: &'a Document, name: &str) -> &'a InstanceDecl {
    document
        .models()
        .iter()
        .flat_map(ModelDecl::items)
        .find_map(|item| match item {
            Item::Instance(instance) if instance.name() == name => Some(instance),
            _ => None,
        })
        .expect("fixture instance exists")
}

fn interface(document: &Document, component_name: &str) -> SupportInterface {
    component_support_interface("supports.eqi", component(document, component_name))
        .expect("fixture component has a valid support interface")
}

fn spatial_supports(document: &Document) -> BTreeMap<String, SpatialSupport<String>> {
    let model = document.models().first().expect("fixture model exists");
    model_spatial_supports("supports.eqi", model).expect("fixture model has valid spatial supports")
}

fn cartesian_domains(document: &Document) -> BTreeMap<String, ExteriorDomain<String>> {
    let model = document.models().first().expect("fixture model exists");
    let mut domains = BTreeMap::new();
    for declaration in model.items().iter().filter_map(|item| match item {
        Item::Domain(declaration) => Some(declaration),
        _ => None,
    }) {
        if let DomainSyntax::CartesianBox(bounds) = declaration.syntax() {
            domains.insert(
                declaration.name().to_owned(),
                ExteriorDomain::Volume {
                    ambient_dimension: bounds.len(),
                },
            );
        }
    }
    for declaration in model.items().iter().filter_map(|item| match item {
        Item::Domain(declaration) => Some(declaration),
        _ => None,
    }) {
        let DomainSyntax::Boundary { parent, axis, side } = declaration.syntax() else {
            continue;
        };
        let ExteriorDomain::Volume { ambient_dimension } =
            domains.get(parent).expect("fixture boundary parent exists")
        else {
            panic!("fixture boundary parent is a volume");
        };
        domains.insert(
            declaration.name().to_owned(),
            ExteriorDomain::Boundary {
                exact_parent: parent.clone(),
                ambient_dimension: *ambient_dimension,
                axis: *axis,
                side: match side {
                    BoundarySideSyntax::Lower => BoundarySide::Lower,
                    BoundarySideSyntax::Upper => BoundarySide::Upper,
                },
            },
        );
    }
    domains
}

fn resolve_complete_exterior(
    document: &Document,
    instance_name: &str,
    budget: &mut CompleteExteriorMembershipBudget,
    resolve_forwarded: impl FnMut(&str) -> Option<ResolvedBoundarySet<String>>,
) -> Result<ResolvedSupportBindings<String>, Vec<Diagnostic>> {
    let component = component(document, "BoundaryFamily");
    let interface = interface(document, "BoundaryFamily");
    let supports = spatial_supports(document);
    let domains = cartesian_domains(document);
    resolve_instance_support_bindings(
        "supports.eqi",
        component,
        &interface,
        instance(document, instance_name),
        |target| supports.get(target).cloned(),
        |target| {
            domains
                .contains_key(target)
                .then(|| ResolvedBoundaryTarget::new(format!("flat::{target}"), target.to_owned()))
        },
        |identity| domains.get(identity).cloned(),
        resolve_forwarded,
        budget,
    )
}

fn messages(diagnostics: &[Diagnostic]) -> Vec<&str> {
    diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message())
        .collect()
}

const COMPONENT: &str = r#"
component BoundaryState(support body: volume(ambient_dimension = 2), support interface: boundary(parent = body)) {


}
"#;

const EXTERIOR_COMPONENT: &str = r#"
component BoundaryFamily(support body: volume(ambient_dimension = 2), support exterior: complete_exterior(parent = body)) {


}
"#;

const EXTERIOR_MODEL: &str = r#"
model Use() {
  domain body = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);
  instance explicit: BoundaryFamily(
    body = body,
    exterior = boundaries(y_upper, x_lower, y_lower, x_upper)
  );
  instance forwarded: BoundaryFamily(
    body = body,
    exterior = enclosing_exterior
  );
}
"#;

#[test]
fn exact_volume_and_boundary_bindings_are_accepted_independent_of_order() {
    let source = format!(
        r#"{COMPONENT}
model Use() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance forward: BoundaryState(
    body = fluid,
    interface = wall
  );
  instance reverse: BoundaryState(
    interface = wall,
    body = fluid
  );
}}
"#
    );
    let document = parse(&source);
    let component = component(&document, "BoundaryState");
    let interface = interface(&document, "BoundaryState");
    let supports = spatial_supports(&document);

    let resolve = |instance_name| {
        resolve_instance_supports(
            "supports.eqi",
            component,
            &interface,
            instance(&document, instance_name),
            |name| supports.get(name).cloned(),
        )
        .expect("exact support binding is valid")
    };
    let forward = resolve("forward");
    let reverse = resolve("reverse");

    assert_eq!(forward, reverse);
    assert_eq!(forward.get("body").map(String::as_str), Some("fluid"));
    assert_eq!(forward.get("interface").map(String::as_str), Some("wall"));
}

#[test]
fn complete_exterior_obligation_is_separate_from_singular_supports() {
    let document = parse(&format!("{EXTERIOR_COMPONENT}{EXTERIOR_MODEL}"));
    let interface = interface(&document, "BoundaryFamily");

    assert!(interface.get("exterior").is_none());
    let exterior = interface
        .complete_exterior("exterior")
        .expect("complete exterior is retained as its own obligation");
    assert_eq!(exterior.parent_slot(), "body");
    assert_eq!(exterior.ambient_dimension(), 2);
    assert_eq!(interface.complete_exteriors().count(), 1);

    let symbolic = symbolic_complete_exterior_set(
        "supports.eqi",
        "exterior",
        exterior,
        component(&document, "BoundaryFamily").range(),
    )
    .expect("definition checking retains a deferred topology obligation");
    assert_eq!(symbolic.exact_parent(), "body");
    assert!(symbolic.witness().is_err());
    assert!(matches!(
        symbolic,
        ResolvedBoundarySet::Deferred {
            ambient_dimension: 2,
            ..
        }
    ));
}

#[test]
fn explicit_complete_exterior_resolves_exact_members_and_canonical_sides() {
    let document = parse(&format!("{EXTERIOR_COMPONENT}{EXTERIOR_MODEL}"));
    let mut budget = CompleteExteriorMembershipBudget::new(CompleteExteriorLimits::default());
    let resolved = resolve_complete_exterior(&document, "explicit", &mut budget, |_| None)
        .expect("unordered exact exterior proves successfully");

    assert_eq!(budget.total_memberships(), 4);
    assert_eq!(
        resolved.singular_targets().get("body").map(String::as_str),
        Some("body")
    );
    assert!(matches!(
        resolved.singular_supports().get("body"),
        Some(SpatialSupport::Volume { dimensions: 2, .. })
    ));
    let set = resolved
        .boundary_set("exterior")
        .expect("proved exterior is retained");
    let ResolvedBoundarySet::Explicit(explicit) = set else {
        panic!("source boundaries binding remains explicitly typed");
    };
    assert_eq!(explicit.members().len(), 4);
    assert_eq!(
        explicit
            .member(&"y_upper".to_owned())
            .map(|member| member.target()),
        Some("flat::y_upper")
    );
    assert_eq!(
        explicit
            .witness()
            .members()
            .iter()
            .map(|side| (
                side.cartesian().unwrap().0,
                side.cartesian().unwrap().1,
                side.boundary().as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, BoundarySide::Lower, "x_lower"),
            (0, BoundarySide::Upper, "x_upper"),
            (1, BoundarySide::Lower, "y_lower"),
            (1, BoundarySide::Upper, "y_upper"),
        ]
    );
}

#[test]
fn forwarding_preserves_the_proved_identity_keyed_member_catalog() {
    let document = parse(&format!("{EXTERIOR_COMPONENT}{EXTERIOR_MODEL}"));
    let mut budget = CompleteExteriorMembershipBudget::new(CompleteExteriorLimits::default());
    let explicit = resolve_complete_exterior(&document, "explicit", &mut budget, |_| None)
        .expect("source exterior proves")
        .boundary_set("exterior")
        .expect("source set exists")
        .clone();
    let forwarded = resolve_complete_exterior(&document, "forwarded", &mut budget, |target| {
        (target == "enclosing_exterior").then(|| explicit.clone())
    })
    .expect("proved exterior may be forwarded");

    assert_eq!(
        budget.total_memberships(),
        4,
        "forwarding shares a proof and consumes no explicit membership budget"
    );
    let set = forwarded
        .boundary_set("exterior")
        .expect("forwarded set exists");
    let ResolvedBoundarySet::Forwarded(binding) = set else {
        panic!("forwarding remains a distinct binding kind");
    };
    assert_eq!(binding.target(), "enclosing_exterior");
    assert_eq!(binding.members(), explicit.members());
    assert_eq!(
        binding
            .member(&"x_lower".to_owned())
            .map(|member| member.target()),
        Some("flat::x_lower")
    );
    assert_eq!(binding.witness(), explicit.witness().unwrap());
}

#[test]
fn complete_exterior_memberships_have_independent_per_set_and_total_limits() {
    let document = parse(&format!("{EXTERIOR_COMPONENT}{EXTERIOR_MODEL}"));
    let mut per_set = CompleteExteriorMembershipBudget::new(CompleteExteriorLimits {
        max_members_per_set: 3,
        max_total_memberships: 100,
    });
    let diagnostics = resolve_complete_exterior(&document, "explicit", &mut per_set, |_| None)
        .expect_err("four members exceed the independent per-set limit");
    assert_eq!(per_set.total_memberships(), 0);
    assert!(
        messages(&diagnostics)
            .iter()
            .any(|message| { message.contains("has 4 members, exceeding the 3 member limit") })
    );

    let mut total = CompleteExteriorMembershipBudget::new(CompleteExteriorLimits {
        max_members_per_set: 4,
        max_total_memberships: 4,
    });
    resolve_complete_exterior(&document, "explicit", &mut total, |_| None)
        .expect("first set fits total budget");
    let diagnostics = resolve_complete_exterior(&document, "explicit", &mut total, |_| None)
        .expect_err("second set exceeds total budget");
    assert_eq!(total.total_memberships(), 4);
    assert!(messages(&diagnostics).iter().any(|message| {
        message.contains("require 8 total memberships, exceeding the 4 membership limit")
    }));
}

#[test]
fn complete_exterior_proof_failures_remain_source_located() {
    let source = format!(
        r#"{EXTERIOR_COMPONENT}
model Use() {{
  domain body = box(0, 1, 0, 1);
  domain other = box(0, 1, 0, 1);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain other_upper = boundary(other, axis = 1, side = upper);
  instance invalid: BoundaryFamily(
    body = body,
    exterior = boundaries(x_lower, x_upper, y_lower, other_upper)
  );
}}
"#
    );
    let document = parse(&source);
    let mut budget = CompleteExteriorMembershipBudget::new(CompleteExteriorLimits::default());
    let diagnostics = resolve_complete_exterior(&document, "invalid", &mut budget, |_| None)
        .expect_err("a same-dimensional side of another parent is rejected");

    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.message().contains("different exact parent"))
        .expect("wrong-parent proof diagnostic exists");
    let binding = instance(&document, "invalid")
        .bindings()
        .iter()
        .find(|binding| binding.name() == "exterior")
        .unwrap();
    let eqiora_lang::ExprKind::Call {
        arguments: eqiora_lang::CallArguments::Positional(arguments),
        ..
    } = binding.value().kind()
    else {
        panic!("set binding")
    };
    let expected_range = arguments.last().expect("wrong member exists").range();
    let span = diagnostic
        .source_span()
        .expect("proof failure retains the exact member span");
    assert_eq!(
        (span.start, span.end),
        (expected_range.start(), expected_range.end())
    );
}

#[test]
fn required_unknown_and_duplicate_bindings_fail_closed() {
    let cases = [
        (
            "missing",
            "body = fluid",
            "has no binding for required support slot `interface`",
        ),
        (
            "unknown",
            "body = fluid, interface = wall, ghost = wall",
            "`ghost` is not a public requirement",
        ),
        (
            "duplicate",
            "body = fluid, body = fluid, interface = wall",
            "duplicate binding for support slot `body`",
        ),
    ];

    for (name, bindings, expected) in cases {
        let source = format!(
            r#"{COMPONENT}
model Use() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance probe: BoundaryState({bindings});
}}
"#
        );
        let document = parse(&source);
        let component = component(&document, "BoundaryState");
        let interface = interface(&document, "BoundaryState");
        let supports = spatial_supports(&document);
        let diagnostics = if name == "unknown" {
            super::super::named_bindings::validate_names(
                "supports.eqi",
                component,
                instance(&document, "probe"),
            )
        } else {
            resolve_instance_supports(
                "supports.eqi",
                component,
                &interface,
                instance(&document, "probe"),
                |target| supports.get(target).cloned(),
            )
            .expect_err(name)
        };

        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{name}: {:?}",
            messages(&diagnostics)
        );
    }
}

#[test]
fn private_support_slots_are_rejected_at_the_definition_boundary() {
    let source =
        "component HiddenSupport() { support body: volume(ambient_dimension = 2); } model Use() {}";
    let diagnostics = eqiora_lang::parse("supports.eqi", source)
        .into_document()
        .expect_err("body support is not a signature requirement");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code() == codes::SYNTAX_ERROR)
    );
}

#[test]
fn volume_and_boundary_kind_mismatches_fail_symmetrically() {
    let cases = [
        (
            "volume bound to boundary",
            "body = wall, interface = wall",
            "support slot `body` requires volume support, found boundary",
        ),
        (
            "boundary bound to volume",
            "body = fluid, interface = fluid",
            "support slot `interface` requires boundary support, found volume",
        ),
    ];

    for (name, bindings, expected) in cases {
        let source = format!(
            r#"{COMPONENT}
model Use() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  instance probe: BoundaryState({bindings});
}}
"#
        );
        let document = parse(&source);
        let component = component(&document, "BoundaryState");
        let interface = interface(&document, "BoundaryState");
        let supports = spatial_supports(&document);
        let diagnostics = resolve_instance_supports(
            "supports.eqi",
            component,
            &interface,
            instance(&document, "probe"),
            |target| supports.get(target).cloned(),
        )
        .expect_err(name);

        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{name}: {:?}",
            messages(&diagnostics)
        );
    }
}

#[test]
fn ambient_dimension_mismatches_fail_for_volume_and_boundary_slots() {
    let cases = [
        (
            "volume dimension",
            "body = line, interface = wall",
            "volume support slot `body` requires ambient dimension 2",
        ),
        (
            "boundary dimension",
            "body = fluid, interface = point",
            "boundary support slot `interface` requires ambient dimension 2",
        ),
    ];

    for (name, bindings, expected) in cases {
        let source = format!(
            r#"{COMPONENT}
model Use() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  domain line = box(0, 1);
  domain point = boundary(line, axis = 0, side = lower);
  instance probe: BoundaryState({bindings});
}}
"#
        );
        let document = parse(&source);
        let component = component(&document, "BoundaryState");
        let interface = interface(&document, "BoundaryState");
        let supports = spatial_supports(&document);
        let diagnostics = resolve_instance_supports(
            "supports.eqi",
            component,
            &interface,
            instance(&document, "probe"),
            |target| supports.get(target).cloned(),
        )
        .expect_err(name);

        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{name}: {:?}",
            messages(&diagnostics)
        );
    }
}

#[test]
fn boundary_binding_must_share_the_exact_bound_parent() {
    let source = format!(
        r#"{COMPONENT}
model Use() {{
  domain fluid = box(0, 1, 0, 1);
  domain wall = boundary(fluid, axis = 0, side = lower);
  domain other = box(0, 1, 0, 1);
  domain other_wall = boundary(other, axis = 0, side = lower);
  instance probe: BoundaryState(
    body = fluid,
    interface = other_wall
  );
}}
"#
    );
    let document = parse(&source);
    let component = component(&document, "BoundaryState");
    let interface = interface(&document, "BoundaryState");
    let supports = spatial_supports(&document);
    let diagnostics = resolve_instance_supports(
        "supports.eqi",
        component,
        &interface,
        instance(&document, "probe"),
        |target| supports.get(target).cloned(),
    )
    .expect_err("a same-dimensional boundary of another volume is not interchangeable");

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("is not BoundaryOf its exact bound parent slot `body`")
    }));
}

#[test]
fn support_declaration_order_does_not_change_the_interface() {
    let body_first = parse(
        r#"
component C(support body: volume(ambient_dimension = 2), support wall: boundary(parent = body)) {


}
model Use() {}
"#,
    );
    let boundary_first = parse(
        r#"
component C(support wall: boundary(parent = body), support body: volume(ambient_dimension = 2)) {


}
model Use() {}
"#,
    );
    let body_first = interface(&body_first, "C");
    let boundary_first = interface(&boundary_first, "C");

    let shape = |interface: &SupportInterface| {
        interface
            .iter()
            .map(|(name, contract)| (name.to_owned(), contract.support().clone()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(shape(&body_first), shape(&boundary_first));
}
