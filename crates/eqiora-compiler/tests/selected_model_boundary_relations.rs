use std::collections::BTreeMap;

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, GeometryGraph};
use eqiora_graph::{EdgeKind, Op};

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

const SOURCE: &str = "model Selected(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){variable value:m on body;relation interior on body{value-value=0;} relation exterior_law[face in exterior] on face{trace(value)-trace(value)=0;}}";

#[test]
fn selected_model_relations_use_each_exact_exterior_member_in_source_and_native() {
    let geometry = rectangle(false);
    let members = ["left", "right", "bottom", "top"].map(|name| geometry.entity_set(name).unwrap());
    let mut previous = None;
    for members in [members.to_vec(), members.into_iter().rev().collect()] {
        let bindings = [
            ("body", selection(&geometry, "body", None)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    members: &members,
                    parent: geometry.entity_set("body").unwrap(),
                },
            ),
        ];
        let direct =
            CompiledModel::compile_selected("model-exterior.eqi", SOURCE, "Selected", &bindings)
                .unwrap_or_else(|errors| panic!("{errors:?}"));
        let native = eqiora_lang::Module::from_document(
            eqiora_lang::parse("model-exterior.eqi", SOURCE)
                .into_document()
                .unwrap(),
        );
        let replay = eqiora_compiler::lower_module(&native, Some("Selected"), &bindings)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        assert_eq!(direct.transaction().ops(), replay.transaction().ops());
        assert_eq!(direct.symbols(), replay.symbols());
        let body = direct.symbols().get("body").unwrap();
        assert_eq!(direct.transaction().ops().iter().filter(|op| matches!(op, Op::Connect { to, edge: EdgeKind::BoundaryOf, .. } if *to == body)).count(), 4);
        for side in [
            "axis=0,side=lower",
            "axis=0,side=upper",
            "axis=1,side=lower",
            "axis=1,side=upper",
        ] {
            assert!(
                direct
                    .symbols()
                    .get(&format!("exterior_law[{side}]"))
                    .is_some(),
                "{:?}",
                direct.symbols()
            );
        }
        if let Some(previous) = previous {
            assert_eq!(direct.transaction().ops(), previous);
        }
        previous = Some(direct.transaction().ops().to_vec());
    }
}

#[test]
fn malformed_model_exterior_relations_reach_exact_binding_gate() {
    let geometry = rectangle(false);
    let foreign = rectangle(true);
    let exact = |name| geometry.entity_set(name).unwrap();
    for (members, expected) in [
        (
            vec![exact("left"), exact("right"), exact("bottom")],
            "omits an entity",
        ),
        (
            vec![
                exact("left"),
                exact("right"),
                exact("bottom"),
                exact("bottom"),
            ],
            "repeats an entity",
        ),
        (
            vec![
                foreign.entity_set("left").unwrap(),
                exact("right"),
                exact("bottom"),
                exact("top"),
            ],
            "foreign or stale",
        ),
    ] {
        let bindings = [
            ("body", selection(&geometry, "body", None)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    members: &members,
                    parent: exact("body"),
                },
            ),
        ];
        let native = eqiora_lang::Module::from_document(
            eqiora_lang::parse("model-exterior.eqi", SOURCE)
                .into_document()
                .unwrap(),
        );
        for errors in [
            CompiledModel::compile_selected("model-exterior.eqi", SOURCE, "Selected", &bindings)
                .unwrap_err(),
            eqiora_compiler::lower_module(&native, Some("Selected"), &bindings).unwrap_err(),
        ] {
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains(expected)),
                "expected {expected}: {errors:?}"
            );
        }
    }
}
