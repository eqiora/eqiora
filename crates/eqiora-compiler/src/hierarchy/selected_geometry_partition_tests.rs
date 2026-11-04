//! Counts below follow directly from authored declarations and AST work.
use super::*;
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};

fn polygon() -> CanonicalGeometryV1 {
    // One face, one grouped selection and 100 singleton facets. The groups
    // describe the same complete frontier with different family cardinalities.
    let points = (0..100)
        .map(|index| {
            let angle = std::f64::consts::TAU * f64::from(index) / 100.;
            [angle.cos(), angle.sin()]
        })
        .collect();
    let mut names = vec![
        NamedEntitySet::new("body", 2, vec![0]),
        NamedEntitySet::new("all", 1, (0..100).collect()),
    ];
    names
        .extend((0..100).map(|index| NamedEntitySet::new(format!("facet{index}"), 1, vec![index])));
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            points,
            vec![PlanarFace::new((0..100).collect(), vec![])],
            names,
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}
fn source(reverse: bool, indexed: bool) -> String {
    let first = if reverse { "many" } else { "few" };
    let second = if reverse { "few" } else { "many" };
    let (indices, suffix) = if indexed {
        ("indexset Copies=range(2);", "[i in Copies]")
    } else {
        ("", "")
    };
    format!(
        "component Law(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){{variable value:m on body;relation law[face in exterior] on face{{trace(value)-trace(value)=0;}}}} component Wrapper(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){{instance law:Law(body=body,exterior=exterior);}} model Selected(support body:volume(ambient_dimension=2),support few:complete_exterior(parent=body),support many:complete_exterior(parent=body)){{{indices} instance first{suffix}:Wrapper(body=body,exterior={first});instance second{suffix}:Wrapper(body=body,exterior={second});}}"
    )
}
fn compile(
    geometry: &CanonicalGeometryV1,
    source: &str,
    limits: HierarchyLimits,
) -> Result<Vec<CompiledModel>, Vec<Diagnostic>> {
    let few = [geometry.entity_set("all").unwrap()];
    let many = (0..100)
        .map(|index| geometry.entity_set(&format!("facet{index}")).unwrap())
        .collect::<Vec<_>>();
    let parent = geometry.entity_set("body").unwrap();
    let bindings = [
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: parent,
                parent: None,
            },
        ),
        (
            "few",
            StaticBindingValue::CompleteExterior {
                geometry,
                parent,
                members: &few,
            },
        ),
        (
            "many",
            StaticBindingValue::CompleteExterior {
                geometry,
                parent,
                members: &many,
            },
        ),
    ];
    let document = eqiora_lang::parse("partition-budget.eqi", source)
        .into_document()
        .unwrap();
    local_document(
        "partition-budget.eqi",
        source.len(),
        document,
        Some("Selected"),
        &bindings,
        limits,
    )
}
#[test]
fn same_parameters_different_partition_cardinalities_are_accounted_in_both_orders() {
    let geometry = polygon();
    // External Domains: 1 parent + 1 grouped + 100 singleton = 102.
    // Two Law instances: 2 Field/representation pairs + 101 Relation/Activation
    // pairs = 206. Wrappers add no declarations. Total = 308.
    for reverse in [false, true] {
        let source = source(reverse, false);
        compile(
            &geometry,
            &source,
            HierarchyLimits {
                max_declarations: 308,
                ..Default::default()
            },
        )
        .unwrap_or_else(|errors| panic!("{errors:?}"));
        let errors = compile(
            &geometry,
            &source,
            HierarchyLimits {
                max_declarations: 307,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("declarations")),
            "{errors:?}"
        );
    }
}
#[test]
fn nested_indexed_occurrences_multiply_actual_exterior_work() {
    let geometry = polygon();
    // Two copies of each Wrapper/Law: 102 external + 2*206 + one IndexSet.
    for reverse in [false, true] {
        let source = source(reverse, true);
        compile(
            &geometry,
            &source,
            HierarchyLimits {
                max_declarations: 515,
                ..Default::default()
            },
        )
        .unwrap_or_else(|errors| panic!("{errors:?}"));
        assert!(
            compile(
                &geometry,
                &source,
                HierarchyLimits {
                    max_declarations: 514,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
#[test]
fn exact_identity_provenance_and_expression_limits_include_all_exterior_members() {
    let geometry = polygon();
    let source = source(false, false);
    // The declaration derivation above gives 308 semantic entries, plus the
    // root Model identity/provenance entry. Each equation has six AST nodes:
    // two (trace + name) pairs, subtraction, and RHS zero: 6*101 = 606.
    // Four support bindings at the root and four in Wrappers add eight = 614.
    for resource in 0..3 {
        let mut exact = HierarchyLimits::default();
        let mut below = exact;
        match resource {
            0 => {
                exact.identity.max_staged_identities = 309;
                below.identity.max_staged_identities = 308;
            }
            1 => {
                exact.provenance.max_entries = 309;
                below.provenance.max_entries = 308;
            }
            _ => {
                exact.max_expanded_expression_nodes = 614;
                below.max_expanded_expression_nodes = 613;
            }
        }
        compile(&geometry, &source, exact)
            .unwrap_or_else(|errors| panic!("resource {resource}: {errors:?}"));
        let errors = compile(&geometry, &source, below).unwrap_err();
        let expected = ["staged identities", "provenance", "expression"][resource];
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected)),
            "resource {resource}: {errors:?}"
        );
    }
}
#[test]
fn family_connections_are_bounded_before_any_member_is_emitted() {
    let geometry = polygon();
    let source = "connector B{trace value:1;flux flux:1;shape [];frame invariant;pairing euclidean_boundary_duality;orientation parent_outward;} component Leaf(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body),port p[b in exterior]:B over b){relation law[b in exterior] on b{p[b=b].flux=0;}} component Wrapper(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){instance child:Leaf(body=body,exterior=exterior);instance other:Leaf(body=body,exterior=exterior);connect [b in exterior] child.p[b=b],other.p[b=b];} model Selected(support body:volume(ambient_dimension=2),support few:complete_exterior(parent=body),support many:complete_exterior(parent=body)){instance first:Wrapper(body=body,exterior=few);instance second:Wrapper(body=body,exterior=many);}";
    // One forwarding connection per part: 1 + 100. These remain ordinary
    // connection fragments; changing grouping does not change their budget.
    compile(
        &geometry,
        source,
        HierarchyLimits {
            max_connections: 101,
            ..Default::default()
        },
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    assert!(
        compile(
            &geometry,
            source,
            HierarchyLimits {
                max_connections: 100,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn model_boundary_connection_families_keep_the_existing_unsupported_diagnostic() {
    let geometry = polygon();
    let source = "connector B{trace value:1;flux flux:1;shape [];frame invariant;pairing euclidean_boundary_duality;orientation parent_outward;} component Leaf(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body),port p[b in exterior]:B over b){relation law[b in exterior] on b{p[b=b].flux=0;}} model Selected(support body:volume(ambient_dimension=2),support few:complete_exterior(parent=body),support many:complete_exterior(parent=body)){instance first:Leaf(body=body,exterior=many);instance second:Leaf(body=body,exterior=many);connect [b in many] first.p[b=b],second.p[b=b];}";
    let errors = compile(&geometry, source, HierarchyLimits::default()).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("Model boundary Connection cannot declare a family binder")),
        "{errors:?}"
    );
}
