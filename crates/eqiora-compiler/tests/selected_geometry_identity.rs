use std::collections::BTreeMap;

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, GeometryGraph};
use eqiora_graph::{EdgeKind, Op};
use eqiora_schema::kernel::KernelNode;

fn rectangle(extra_name: bool) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let region = graph.rectangle([0., 2.], [0., 1.]).unwrap();
    let names = BTreeMap::from([
        ("body".to_owned(), vec![region.region().into()]),
        ("left".to_owned(), vec![region.boundaries()[0].into()]),
        ("right".to_owned(), vec![region.boundaries()[1].into()]),
        (
            (if extra_name { "other_bottom" } else { "bottom" }).to_owned(),
            vec![region.boundaries()[2].into()],
        ),
        ("top".to_owned(), vec![region.boundaries()[3].into()]),
    ]);
    graph.build(&region, &names).unwrap()
}

fn selection<'a>(
    geometry: &'a CanonicalGeometryV1,
    name: &str,
    parent: Option<&str>,
) -> StaticBindingValue<'a> {
    StaticBindingValue::GeometrySupport {
        geometry,
        selection: geometry.entity_set(name).unwrap(),
        parent: parent.map(|name| geometry.entity_set(name).unwrap()),
    }
}

#[test]
fn repeated_geometry_supports_share_domains_and_retain_both_names() {
    let geometry = rectangle(false);
    for owner in ["model", "public component"] {
        let source = format!(
            "{owner} M(support a:volume(ambient_dimension=2),support b:volume(ambient_dimension=2),support left_a:boundary(parent=a),support left_b:boundary(parent=b)) {{variable x:1 on a;variable y:1 on b;relation first on a {{x=0;}} relation second on b {{y=0;}} relation boundary on left_b {{trace(y)=0;}}}}"
        );
        let compiled = CompiledModel::compile_selected(
            "aliases.eqi",
            &source,
            "M",
            &[
                ("a", selection(&geometry, "body", None)),
                ("b", selection(&geometry, "body", None)),
                ("left_a", selection(&geometry, "left", Some("body"))),
                ("left_b", selection(&geometry, "left", Some("body"))),
            ],
        )
        .unwrap_or_else(|errors| panic!("{errors:?}"));
        let symbols = compiled.symbols();
        let body = symbols.get("a").unwrap();
        let boundary = symbols.get("left_a").unwrap();
        assert_eq!(symbols.get("b"), Some(body));
        assert_eq!(symbols.get("left_b"), Some(boundary));
        assert_ne!(body, boundary);
        assert_eq!(
            compiled
                .transaction()
                .ops()
                .iter()
                .filter(|op| matches!(
                    op,
                    Op::DefineKernelNode {
                        node: KernelNode::Domain(_)
                    }
                ))
                .count(),
            2
        );
        assert!(compiled.transaction().ops().iter().any(|op| matches!(op, Op::Connect { from, to, edge: EdgeKind::BoundaryOf } if *from == boundary && *to == body)));
    }
}

#[test]
fn equal_bounds_and_names_in_distinct_geometries_remain_distinct() {
    let first = rectangle(false);
    let second = rectangle(true);
    assert_ne!(first.digest_bytes(), second.digest_bytes());
    let source = "model M(support a:volume(ambient_dimension=2),support b:volume(ambient_dimension=2)) {variable x:1 on a;variable y:1 on b;relation first on a {x=0;} relation second on b {y=0;}}";
    let compiled = CompiledModel::compile_selected(
        "distinct.eqi",
        source,
        "M",
        &[
            ("a", selection(&first, "body", None)),
            ("b", selection(&second, "body", None)),
        ],
    )
    .unwrap();
    assert_ne!(compiled.symbols().get("a"), compiled.symbols().get("b"));
}

#[test]
fn a_boundary_cannot_borrow_a_parent_from_another_geometry_revision() {
    let first = rectangle(false);
    let second = rectangle(true);
    let source = "model M(support a:volume(ambient_dimension=2),support left:boundary(parent=a)) {variable x:1 on a;relation r on left {trace(x)=0;}}";
    assert!(
        CompiledModel::compile_selected(
            "foreign.eqi",
            source,
            "M",
            &[
                ("a", selection(&first, "body", None)),
                ("left", selection(&second, "left", Some("body"))),
            ]
        )
        .is_err()
    );
}

const EXTERIOR: &str = r#"
connector BoundaryScalar {
    trace value: 1; flux flux: 1; shape []; frame invariant;
    pairing euclidean_boundary_duality; orientation parent_outward;
}
component Leaf(support body: volume(ambient_dimension = 2),
               support exterior: complete_exterior(parent = body),
               port p[side in exterior]: BoundaryScalar over side) {
    relation law[side in exterior] on side { p[side = side].flux = 0; }
}
component Wrapper(support body: volume(ambient_dimension = 2),
                  support exterior: complete_exterior(parent = body),
                  port p[side in exterior]: BoundaryScalar over side) {
    instance child: Leaf(body = body, exterior = exterior);
    connect [side in exterior] child.p[side = side], p[side = side];
}
component Terminal(support body: volume(ambient_dimension = 2),
                   support face: boundary(parent = body),
                   port p: BoundaryScalar over face) {
    relation law on face { p.value = 0; }
}
model M(support body: volume(ambient_dimension = 2),
        support left: boundary(parent = body), support right: boundary(parent = body),
        support bottom: boundary(parent = body), support top: boundary(parent = body)) {
    instance wall: Wrapper(body = body, exterior = boundaries(left, right, bottom, top));
    instance l: Terminal(body = body, face = left);
    instance r: Terminal(body = body, face = right);
    instance b: Terminal(body = body, face = bottom);
    instance t: Terminal(body = body, face = top);
    connect wall.p[side = left], l.p;
    connect wall.p[side = right], r.p;
    connect wall.p[side = bottom], b.p;
    connect wall.p[side = top], t.p;
}
"#;

fn exterior_bindings(geometry: &CanonicalGeometryV1) -> Vec<(&str, StaticBindingValue<'_>)> {
    ["body", "left", "right", "bottom", "top"]
        .into_iter()
        .map(|name| {
            (
                name,
                selection(geometry, name, (name != "body").then_some("body")),
            )
        })
        .collect()
}

#[test]
fn exact_geometry_completes_nested_exterior_without_concrete_source_domains() {
    let geometry = rectangle(false);
    let bindings = exterior_bindings(&geometry);
    let direct = CompiledModel::compile_selected("exterior.eqi", EXTERIOR, "M", &bindings)
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let native = eqiora_lang::Module::from_document(
        eqiora_lang::parse("exterior.eqi", EXTERIOR)
            .into_document()
            .unwrap(),
    );
    let replay = eqiora_compiler::lower_module(&native, Some("M"), &bindings)
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    assert_eq!(direct.transaction().ops(), replay.transaction().ops());
    assert_eq!(direct.symbols(), replay.symbols());
    for name in ["left", "right", "bottom", "top"] {
        let boundary = direct.symbols().get(name).unwrap();
        let body = direct.symbols().get("body").unwrap();
        assert!(direct.transaction().ops().iter().any(|op| matches!(op,
            Op::Connect { from, to, edge: EdgeKind::BoundaryOf } if *from == boundary && *to == body
        )));
    }
}

#[test]
fn incomplete_and_overlapping_external_families_reach_the_completeness_gate() {
    let geometry = rectangle(false);
    for (members, expected) in [
        ("left, right, bottom", "missing Cartesian side"),
        ("left, right, bottom, bottom", "more than once"),
    ] {
        let source = EXTERIOR.replace(
            "boundaries(left, right, bottom, top)",
            &format!("boundaries({members})"),
        );
        let errors = CompiledModel::compile_selected(
            "incomplete.eqi",
            &source,
            "M",
            &exterior_bindings(&geometry),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected)),
            "{errors:?}"
        );
    }
    let mut bindings = exterior_bindings(&geometry);
    bindings[2].1 = selection(&geometry, "left", Some("body"));
    let errors =
        CompiledModel::compile_selected("overlap.eqi", EXTERIOR, "M", &bindings).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("Cartesian side (0, lower) more than once")),
        "{errors:?}"
    );
}

#[test]
fn external_family_rejects_foreign_selection_and_parent_before_topology() {
    let geometry = rectangle(false);
    let foreign = rectangle(true);
    for parent_mutant in [false, true] {
        let mut bindings = exterior_bindings(&geometry);
        bindings[1].1 = StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: if parent_mutant {
                geometry.entity_set("left").unwrap()
            } else {
                foreign.entity_set("left").unwrap()
            },
            parent: Some(if parent_mutant {
                foreign.entity_set("body").unwrap()
            } else {
                geometry.entity_set("body").unwrap()
            }),
        };
        let errors =
            CompiledModel::compile_selected("foreign.eqi", EXTERIOR, "M", &bindings).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("foreign or stale")),
            "{errors:?}"
        );
    }
}

fn selected_exterior_source() -> String {
    let source = EXTERIOR.replace("support left: boundary(parent = body), support right:", "support exterior: complete_exterior(parent = body), support left: boundary(parent = body), support right:")
        .replace("boundaries(left, right, bottom, top)", "exterior");
    source.replace("model M(", r#"public component ClosedExterior(support body: volume(ambient_dimension=2), support exterior: complete_exterior(parent=body)) {
        instance a: Wrapper(body=body, exterior=exterior);
        instance b: Wrapper(body=body, exterior=exterior);
        connect [side in exterior] a.p[side=side], b.p[side=side];
    }
    model M("#)
}

#[test]
fn direct_selected_exterior_retains_exact_members_in_source_and_native() {
    let geometry = rectangle(false);
    let source = selected_exterior_source();
    let members = ["left", "right", "bottom", "top"].map(|name| geometry.entity_set(name).unwrap());
    let reversed = members.iter().rev().copied().collect::<Vec<_>>();
    for entry in ["M", "ClosedExterior"] {
        let compile = |members| {
            let mut bindings = vec![
                ("body", selection(&geometry, "body", None)),
                (
                    "exterior",
                    StaticBindingValue::CompleteExterior {
                        geometry: &geometry,
                        members,
                        parent: geometry.entity_set("body").unwrap(),
                    },
                ),
            ];
            if entry == "M" {
                bindings.extend(
                    exterior_bindings(&geometry)
                        .into_iter()
                        .filter(|(name, _)| *name != "body"),
                );
            }
            let direct =
                CompiledModel::compile_selected("selected-exterior.eqi", &source, entry, &bindings)
                    .unwrap_or_else(|errors| panic!("{entry}: {errors:?}"));
            let native = eqiora_lang::Module::from_document(
                eqiora_lang::parse("selected-exterior.eqi", &source)
                    .into_document()
                    .unwrap(),
            );
            let replay = eqiora_compiler::lower_module(&native, Some(entry), &bindings)
                .unwrap_or_else(|errors| panic!("{entry}: {errors:?}"));
            assert_eq!(direct.transaction().ops(), replay.transaction().ops());
            assert_eq!(direct.symbols(), replay.symbols());
            direct
        };
        let direct = compile(&members);
        let permuted = compile(&reversed);
        assert_eq!(direct.transaction().ops(), permuted.transaction().ops());
        assert_eq!(direct.symbols(), permuted.symbols());
        let body = direct.symbols().get("body").unwrap();
        assert_eq!(
            direct
                .transaction()
                .ops()
                .iter()
                .filter(|op| matches!(op,
                    Op::Connect { to, edge: EdgeKind::BoundaryOf, .. } if *to == body
                ))
                .count(),
            4
        );
    }
}

#[test]
fn direct_selected_exterior_rejects_incomplete_duplicate_and_foreign_members() {
    let geometry = rectangle(false);
    let foreign = rectangle(true);
    let source = selected_exterior_source();
    let exact = |name| geometry.entity_set(name).unwrap();
    for (members, parent, expected) in [
        (
            vec![exact("left"), exact("right"), exact("bottom")],
            exact("body"),
            "missing Cartesian side",
        ),
        (
            vec![
                exact("left"),
                exact("right"),
                exact("bottom"),
                exact("bottom"),
            ],
            exact("body"),
            "more than once",
        ),
        (
            vec![exact("body"), exact("right"), exact("bottom"), exact("top")],
            exact("body"),
            "does not bind exact parent",
        ),
        (
            vec![
                foreign.entity_set("left").unwrap(),
                exact("right"),
                exact("bottom"),
                exact("top"),
            ],
            exact("body"),
            "foreign or stale",
        ),
        (
            vec![exact("left"), exact("right"), exact("bottom"), exact("top")],
            foreign.entity_set("body").unwrap(),
            "foreign or stale parent",
        ),
        (
            vec![exact("left"), exact("right"), exact("bottom"), exact("top")],
            exact("left"),
            "does not bind exact parent",
        ),
        (vec![], exact("body"), "no members"),
    ] {
        let mut bindings = vec![
            ("body", selection(&geometry, "body", None)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    members: &members,
                    parent,
                },
            ),
        ];
        bindings.extend(
            exterior_bindings(&geometry)
                .into_iter()
                .filter(|(name, _)| *name != "body"),
        );
        let errors =
            CompiledModel::compile_selected("bad-selected-exterior.eqi", &source, "M", &bindings)
                .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(expected)),
            "expected {expected}: {errors:?}"
        );
    }
}

fn resolved_exterior(
    root_source: &str,
    dependency: bool,
    native: bool,
) -> Result<eqiora_compiler::ValidatedResolvedHierarchy, Vec<eqiora_core::Diagnostic>> {
    use eqiora_compiler::{
        CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };
    let root = CompilationNamespaceId::new(["exterior_root", "1", "root-digest"]).unwrap();
    let parts = if dependency {
        CompilationNamespaceId::new(["exterior_parts", "2", "locked-digest"]).unwrap()
    } else {
        root.clone()
    };
    let prefix = if dependency {
        "exterior_parts"
    } else {
        "exterior_root"
    };
    let library = EXTERIOR
        .split("model M(")
        .next()
        .unwrap()
        .replace("component ", "public component ")
        .replace("connector ", "public connector ");
    let root_source = format!("import {prefix}.parts as lib; {root_source}")
        .replace(": Wrapper(", ": lib.Wrapper(")
        .replace(": Terminal(", ": lib.Terminal(");
    let unit = |owner, file: &str, source: &str| {
        if native {
            ResolvedSourceUnit::from_module(
                owner,
                file,
                eqiora_lang::Module::from_document(
                    eqiora_lang::parse(file, source).into_document().unwrap(),
                ),
            )
        } else {
            ResolvedSourceUnit::new(owner, file, source)
        }
        .unwrap()
    };
    let units = vec![
        unit(root.clone(), "src/main.eqi", &root_source),
        unit(parts.clone(), "src/parts.eqi", &library),
    ];
    let dependencies = if dependency {
        vec![ResolvedDependency::new(root.clone(), parts)]
    } else {
        vec![]
    };
    analyze_resolved_hierarchy(ResolvedHierarchyInput::new(root, units, dependencies))?
        .validate_definitions()
}

#[test]
fn resolved_abstract_exteriors_prove_supplied_geometry_in_local_and_locked_modules() {
    let source = &EXTERIOR[EXTERIOR.find("model M(").unwrap()..];
    let geometry = rectangle(false);
    for dependency in [false, true] {
        let checked = resolved_exterior(source, dependency, false)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        let compiled = checked
            .compile_selected("M", &exterior_bindings(&geometry))
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        let native = resolved_exterior(source, dependency, true)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        let replay = native
            .compile_selected("M", &exterior_bindings(&geometry))
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        assert_eq!(compiled.transaction().ops(), replay.transaction().ops());
        assert_eq!(compiled.symbols(), replay.symbols());
        let mut wrong_side = exterior_bindings(&geometry);
        wrong_side[2].1 = selection(&geometry, "left", Some("body"));
        let errors = checked.compile_selected("M", &wrong_side).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("more than once")),
            "{errors:?}"
        );
        assert!(checked.compile_selected("M", &[]).is_err());
        assert!(checked.compile_root("M").is_err());
    }
}

#[test]
fn abstract_exterior_analysis_keeps_static_cardinality_parent_and_duplicate_checks() {
    let source = &EXTERIOR[EXTERIOR.find("model M(").unwrap()..];
    for (mutant, expected) in [
        (source.replace("boundaries(left, right, bottom, top)", "boundaries(left, right, bottom)"), "member count"),
        (source.replace("boundaries(left, right, bottom, top)", "boundaries(left, right, bottom, bottom)"), "more than once"),
        (source.replace("support body: volume(ambient_dimension = 2),", "support other: volume(ambient_dimension = 2), support body: volume(ambient_dimension = 2),")
            .replace("support left: boundary(parent = body)", "support left: boundary(parent = other)"), "parent"),
    ] {
        let errors = resolved_exterior(&mutant, true, false).unwrap_err();
        assert!(errors.iter().any(|error| error.message().contains(expected)), "{expected}: {errors:?}");
    }
}

#[test]
fn source_named_boundary_parent_inference_preserves_transaction_identity() {
    let geometry = rectangle(false);
    let source = "model M(support body:volume(ambient_dimension=2),support side:boundary(parent=body)){variable x:1 on body;relation interior on body{x=0;}relation edge on side{trace(x)=0;}}";
    let compile = |parent| {
        CompiledModel::compile_selected(
            "boundary.eqi",
            source,
            "M",
            &[
                ("side", selection(&geometry, "left", parent)),
                ("body", selection(&geometry, "body", None)),
            ],
        )
        .unwrap()
    };
    let explicit = compile(Some("body"));
    let inferred = compile(None);
    assert_eq!(explicit.transaction().ops(), inferred.transaction().ops());
    assert_eq!(explicit.symbols(), inferred.symbols());
    for invalid in [
        vec![("side", selection(&geometry, "left", None))],
        vec![
            ("body", selection(&geometry, "right", None)),
            ("side", selection(&geometry, "left", None)),
        ],
        vec![
            ("body", selection(&geometry, "body", None)),
            ("side", selection(&geometry, "left", Some("right"))),
        ],
    ] {
        assert!(CompiledModel::compile_selected("boundary.eqi", source, "M", &invalid).is_err());
    }
    let foreign = rectangle(true);
    let invalid = [
        ("body", selection(&foreign, "body", None)),
        ("side", selection(&geometry, "left", None)),
    ];
    assert!(CompiledModel::compile_selected("boundary.eqi", source, "M", &invalid).is_err());
}
